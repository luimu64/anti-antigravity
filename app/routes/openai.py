import time
import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Header, Request, status
from fastapi.responses import StreamingResponse, JSONResponse

from app.config import MODEL_ALIASES
from app.auth import auth_manager
from app.keys import api_key_manager
from app.client import client
from app.translator import OpenAITranslator, ChatCompletionRequest

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
    List all available models in standard OpenAI format.
    """
    try:
        raw_models_data = await client.fetch_available_models()
        raw_models = raw_models_data.get("models", {})
    except Exception as e:
        logger.warning(f"Could not fetch live models, returning defaults: {e}")
        raw_models = {}

    model_list = []
    created_time = 1700000000
    seen_ids = set()

    # 1. Models from Antigravity backend
    for model_id, info in raw_models.items():
        seen_ids.add(model_id)
        model_list.append({
            "id": model_id,
            "object": "model",
            "created": created_time,
            "owned_by": "google",
            "permission": [],
            "root": model_id,
            "parent": None,
            "display_name": info.get("displayName", model_id),
            "max_tokens": info.get("maxTokens", 1048576),
            "supports_thinking": info.get("supportsThinking", False)
        })

    # 2. Add standard OpenAI / Claude aliases
    for alias, internal_target in MODEL_ALIASES.items():
        if alias not in seen_ids:
            seen_ids.add(alias)
            model_list.append({
                "id": alias,
                "object": "model",
                "created": created_time,
                "owned_by": "google-antigravity",
                "permission": [],
                "root": internal_target,
                "parent": None,
                "display_name": f"{alias} (-> {internal_target})"
            })

    return {
        "object": "list",
        "data": model_list
    }

@router.get("/v1/models/{model_id:path}", dependencies=[Depends(verify_api_key)])
@router.get("/models/{model_id:path}", dependencies=[Depends(verify_api_key)])
async def retrieve_model(model_id: str):
    """
    Retrieve single model info in standard OpenAI format.
    """
    resolved = OpenAITranslator.resolve_model(model_id)
    return {
        "id": model_id,
        "object": "model",
        "created": 1700000000,
        "owned_by": "google",
        "permission": [],
        "root": resolved,
        "parent": None
    }

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
                requested_model=request.model
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
    Legacy /v1/completions endpoint adapter.
    """
    body = await request.json()
    prompt = body.get("prompt", "")
    if isinstance(prompt, list):
        prompt = "\n".join(prompt)
    
    chat_req = ChatCompletionRequest(
        model=body.get("model", "gemini-3.7-flash-high"),
        messages=[{"role": "user", "content": prompt}],
        temperature=body.get("temperature"),
        top_p=body.get("top_p"),
        max_completion_tokens=body.get("max_tokens"),
        stream=body.get("stream", False)
    )
    return await chat_completions(chat_req)

@router.post("/v1/embeddings", dependencies=[Depends(verify_api_key)])
@router.post("/embeddings", dependencies=[Depends(verify_api_key)])
async def embeddings_placeholder(request: Request):
    """
    Embeddings endpoint placeholder.
    """
    body = await request.json()
    input_text = body.get("input", "")
    inputs = [input_text] if isinstance(input_text, str) else input_text
    
    # Return standard embedding vector placeholder
    data = []
    for i, _ in enumerate(inputs):
        data.append({
            "object": "embedding",
            "index": i,
            "embedding": [0.0] * 768
        })
    return {
        "object": "list",
        "data": data,
        "model": body.get("model", "text-embedding-3-small"),
        "usage": {
            "prompt_tokens": len(inputs) * 5,
            "total_tokens": len(inputs) * 5
        }
    }
