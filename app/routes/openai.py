import time
import struct
import base64
import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Header, Request, status
from fastapi.responses import StreamingResponse, JSONResponse

from app.config import translate_model_to_openai
from app.auth import auth_manager
from app.keys import api_key_manager
from app.client import client
from app.translator import OpenAITranslator, ChatCompletionRequest, EmbeddingRequest

logger = logging.getLogger("agy_to_api.openai")
router = APIRouter(tags=["OpenAI"])

async def verify_api_key(authorization: Optional[str] = Header(None)):
    """
    API Key enforcement check for incoming requests to this bridge.
    Validates against registered bridge API keys.
    """
    if not api_key_manager.enforce_keys:
        return True

    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": {
                    "message": "You didn't provide an API key. You need to provide your API key in an Authorization header using Bearer auth (i.e. Authorization: Bearer YOUR_KEY).",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": "missing_api_key"
                }
            }
        )
    
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not api_key_manager.validate_key(token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": {
                    "message": f"Incorrect API key provided: {token[:8]}***. Please check your API key and try again.",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": "invalid_api_key"
                }
            }
        )
    return True

@router.get("/v1/models", dependencies=[Depends(verify_api_key)])
@router.get("/models", dependencies=[Depends(verify_api_key)])
async def list_models():
    """
    List all available models dynamically fetched from backend and translated to OpenAI format.
    """
    try:
        raw_models_data = await client.fetch_available_models()
        raw_models = raw_models_data.get("models", {})
    except Exception as e:
        logger.warning(f"Could not fetch live models: {e}")
        raw_models = {}

    model_list = [
        translate_model_to_openai(model_id, info=info)
        for model_id, info in raw_models.items()
    ]

    return {
        "object": "list",
        "data": model_list
    }

@router.get("/v1/models/{model_id:path}", dependencies=[Depends(verify_api_key)])
@router.get("/models/{model_id:path}", dependencies=[Depends(verify_api_key)])
async def retrieve_model(model_id: str):
    """
    Retrieve single model info dynamically fetched and translated to OpenAI format.
    """
    try:
        raw_models_data = await client.fetch_available_models()
        raw_models = raw_models_data.get("models", {})
    except Exception:
        raw_models = {}

    clean_id = OpenAITranslator.resolve_model(model_id)
    info = raw_models.get(clean_id) or raw_models.get(model_id)
    return translate_model_to_openai(clean_id, info=info)

@router.post("/v1/chat/completions", dependencies=[Depends(verify_api_key)])
@router.post("/chat/completions", dependencies=[Depends(verify_api_key)])
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI-compatible /v1/chat/completions endpoint.
    Supports streaming (SSE) and non-streaming, multi-modal, function/tool calling,
    and reasoning/thinking models.
    """
    try:
        (
            internal_model,
            contents,
            system_instruction,
            generation_config,
            tools
        ) = OpenAITranslator.openai_to_internal_request(request)
    except Exception as e:
        logger.error(f"Error translating OpenAI request: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid request parameters: {str(e)}"
        )

    # Determine if include_usage is requested for streaming
    include_usage = False
    if request.stream_options:
        if isinstance(request.stream_options, dict):
            include_usage = bool(request.stream_options.get("include_usage", False))
        elif hasattr(request.stream_options, "include_usage"):
            include_usage = bool(request.stream_options.include_usage)

    # 1. Streaming response
    if request.stream:
        try:
            event_stream = client.stream_generate_content(
                model=internal_model,
                contents=contents,
                system_instruction=system_instruction,
                generation_config=generation_config,
                tools=tools
            )
            openai_chunks = OpenAITranslator.internal_stream_to_openai_chunks(
                event_stream=event_stream,
                requested_model=request.model,
                include_usage=include_usage
            )
            return StreamingResponse(
                openai_chunks,
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Content-Type": "text/event-stream",
                    "X-Accel-Buffering": "no"
                }
            )
        except Exception as e:
            logger.error(f"Streaming generation error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Generation failed: {str(e)}"
            )

    # 2. Non-streaming response
    try:
        result = await client.generate_content(
            model=internal_model,
            contents=contents,
            system_instruction=system_instruction,
            generation_config=generation_config,
            tools=tools
        )
        response_json = OpenAITranslator.internal_to_openai_response(
            result=result,
            requested_model=request.model
        )
        return JSONResponse(content=response_json)
    except Exception as e:
        logger.error(f"Non-streaming generation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Generation failed: {str(e)}"
        )

@router.post("/v1/completions", dependencies=[Depends(verify_api_key)])
@router.post("/completions", dependencies=[Depends(verify_api_key)])
async def legacy_completions(request: Request):
    """
    Legacy /v1/completions endpoint returning standard OpenAI text_completion objects.
    Supports streaming (SSE) and non-streaming responses.
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid JSON request body."
        )

    prompt = body.get("prompt", "")
    if isinstance(prompt, list):
        prompt = "\n".join(str(p) for p in prompt)
    elif not isinstance(prompt, str):
        prompt = str(prompt)

    model_name = body.get("model", "gemini-3.7-flash-high")
    stream = bool(body.get("stream", False))
    stream_options = body.get("stream_options")
    include_usage = False
    if stream_options and isinstance(stream_options, dict):
        include_usage = bool(stream_options.get("include_usage", False))

    chat_req = ChatCompletionRequest(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=body.get("temperature"),
        top_p=body.get("top_p"),
        n=body.get("n"),
        stop=body.get("stop"),
        max_completion_tokens=body.get("max_tokens"),
        presence_penalty=body.get("presence_penalty"),
        frequency_penalty=body.get("frequency_penalty"),
        seed=body.get("seed"),
        stream=stream,
        stream_options=stream_options
    )

    try:
        (
            internal_model,
            contents,
            system_instruction,
            generation_config,
            tools
        ) = OpenAITranslator.openai_to_internal_request(chat_req)
    except Exception as e:
        logger.error(f"Error translating completion request: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid request parameters: {str(e)}"
        )

    # 1. Streaming response
    if stream:
        try:
            event_stream = client.stream_generate_content(
                model=internal_model,
                contents=contents,
                system_instruction=system_instruction,
                generation_config=generation_config,
                tools=tools
            )
            openai_chunks = OpenAITranslator.internal_stream_to_openai_text_chunks(
                event_stream=event_stream,
                requested_model=model_name,
                include_usage=include_usage
            )
            return StreamingResponse(
                openai_chunks,
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Content-Type": "text/event-stream",
                    "X-Accel-Buffering": "no"
                }
            )
        except Exception as e:
            logger.error(f"Streaming text completion generation error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Generation failed: {str(e)}"
            )

    # 2. Non-streaming response
    try:
        result = await client.generate_content(
            model=internal_model,
            contents=contents,
            system_instruction=system_instruction,
            generation_config=generation_config,
            tools=tools
        )
        response_json = OpenAITranslator.internal_to_openai_text_completion(
            result=result,
            requested_model=model_name
        )
        return JSONResponse(content=response_json)
    except Exception as e:
        logger.error(f"Non-streaming text completion generation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Generation failed: {str(e)}"
        )

@router.post("/v1/embeddings", dependencies=[Depends(verify_api_key)])
@router.post("/embeddings", dependencies=[Depends(verify_api_key)])
async def create_embeddings(request: EmbeddingRequest):
    """
    Generate embeddings for given input texts via Antigravity backend.
    Supports float and base64 encoding_format, and custom dimensions.
    """
    # 1. Normalize input into list of strings
    if isinstance(request.input, str):
        texts = [request.input]
    elif isinstance(request.input, list):
        if len(request.input) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="input list cannot be empty"
            )
        if isinstance(request.input[0], int):
            texts = [" ".join(str(tok) for tok in request.input)]
        elif isinstance(request.input[0], list):
            texts = [" ".join(str(tok) for tok in sub) for sub in request.input]
        else:
            texts = [str(x) for x in request.input]
    else:
        texts = [str(request.input)]

    # 2. Validate parameters
    encoding_format = request.encoding_format or "float"
    if encoding_format not in ("float", "base64"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid encoding_format: {encoding_format}. Must be 'float' or 'base64'."
        )

    if request.dimensions is not None and request.dimensions <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="dimensions must be a positive integer."
        )

    resolved_model = OpenAITranslator.resolve_model(request.model)

    try:
        raw_result = await client.embed_contents(
            model=resolved_model,
            texts=texts,
            dimensions=request.dimensions
        )
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Embedding failed: {str(e)}"
        )

    # 3. Extract embedding vectors from response
    raw_embeddings = []
    if isinstance(raw_result, dict):
        if "embeddings" in raw_result and isinstance(raw_result["embeddings"], list):
            for item in raw_result["embeddings"]:
                if isinstance(item, dict) and "values" in item:
                    raw_embeddings.append(item["values"])
                elif isinstance(item, list):
                    raw_embeddings.append(item)
        elif "responses" in raw_result and isinstance(raw_result["responses"], list):
            for item in raw_result["responses"]:
                emb = item.get("embedding", {}) if isinstance(item, dict) else {}
                raw_embeddings.append(emb.get("values", []))
        elif "embedding" in raw_result:
            emb = raw_result["embedding"]
            if isinstance(emb, dict) and "values" in emb:
                raw_embeddings.append(emb["values"])

    # Fallback to ensure one vector per input text
    while len(raw_embeddings) < len(texts):
        raw_embeddings.append([0.0] * (request.dimensions or 768))

    # 4. Format embeddings according to encoding_format and dimensions
    data = []
    for i, vec in enumerate(raw_embeddings):
        if request.dimensions is not None:
            vec = vec[:request.dimensions]
            if len(vec) < request.dimensions:
                vec = vec + [0.0] * (request.dimensions - len(vec))

        if encoding_format == "base64":
            binary_data = struct.pack(f"<{len(vec)}f", *vec)
            embedding_val = base64.b64encode(binary_data).decode("utf-8")
        else:
            embedding_val = [float(v) for v in vec]

        data.append({
            "object": "embedding",
            "index": i,
            "embedding": embedding_val
        })

    # 5. Token usage calculation
    usage_meta = raw_result.get("usageMetadata", {}) if isinstance(raw_result, dict) else {}
    prompt_tokens = usage_meta.get("promptTokenCount")
    if prompt_tokens is None:
        prompt_tokens = sum(max(1, len(t) // 4) for t in texts)

    return {
        "object": "list",
        "data": data,
        "model": request.model,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "total_tokens": prompt_tokens
        }
    }
