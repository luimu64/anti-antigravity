import asyncio
import base64
import logging
import struct
import time
import uuid

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse, StreamingResponse

from app.client import client
from app.config import DEPRECATED_MODELS, MODEL_ALIASES
from app.history import history_manager
from app.keys import api_key_manager
from app.providers.base import ModelNotFoundError, RateLimitError
from app.translator import ChatCompletionRequest, EmbeddingRequest, OpenAITranslator

logger = logging.getLogger("google_gate.openai")
router = APIRouter(tags=["OpenAI"])


def get_active_backend_name() -> str:
    # Prefer the backend that actually served the last request (accurate when
    # the router fell back from the preferred one); fall back to priority head.
    served_by = getattr(client, "last_served_by", None)
    if served_by:
        return served_by
    if hasattr(client, "get_ordered_adapters"):
        adapters = client.get_ordered_adapters()
        if adapters:
            return adapters[0].name
    return getattr(client, "name", "antigravity")


async def verify_api_key(authorization: str | None = Header(None)):
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
                    "code": "missing_api_key",
                }
            },
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
                    "code": "invalid_api_key",
                }
            },
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

    # 1. Models from active backends (ordered by provider support count)
    for model_id, info in raw_models.items():
        if model_id in DEPRECATED_MODELS:
            continue
        seen_ids.add(model_id)
        model_list.append(
            {
                "id": model_id,
                "object": "model",
                "created": created_time,
                "owned_by": "google",
                "permission": [],
                "root": model_id,
                "parent": None,
                "display_name": info.get("displayName", model_id),
                "max_tokens": info.get("maxTokens", 1048576),
                "supports_thinking": info.get("supportsThinking", False),
                "providers": info.get("providers", ["google"]),
                "provider_count": info.get(
                    "provider_count", len(info.get("providers", ["google"]))
                ),
            }
        )

    # 2. Add standard OpenAI / Claude aliases
    for alias, internal_target in MODEL_ALIASES.items():
        if alias in DEPRECATED_MODELS or internal_target in DEPRECATED_MODELS:
            continue
        if alias not in seen_ids:
            seen_ids.add(alias)
            supports_vision = (
                alias == "vision"
                or "image" in alias
                or "image" in internal_target
                or any(k in alias for k in ("gemini", "gpt-4", "claude"))
            )
            model_list.append(
                {
                    "id": alias,
                    "object": "model",
                    "created": created_time,
                    "owned_by": "google-antigravity",
                    "permission": [],
                    "root": internal_target,
                    "parent": None,
                    "display_name": f"{alias} (-> {internal_target})",
                    "supports_vision": supports_vision,
                }
            )

    return {"object": "list", "data": model_list}


@router.get("/v1/models/{model_id:path}", dependencies=[Depends(verify_api_key)])
@router.get("/models/{model_id:path}", dependencies=[Depends(verify_api_key)])
async def retrieve_model(model_id: str):
    """
    Retrieve single model info in standard OpenAI format.
    """
    if model_id in DEPRECATED_MODELS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": {
                    "message": f"The model `{model_id}` does not exist or you do not have access to it.",
                    "type": "invalid_request_error",
                    "param": "model",
                    "code": "model_not_found",
                }
            },
        )
    resolved = OpenAITranslator.resolve_model(model_id)
    return {
        "id": model_id,
        "object": "model",
        "created": 1700000000,
        "owned_by": "google",
        "permission": [],
        "root": resolved,
        "parent": None,
    }


@router.post("/v1/chat/completions", dependencies=[Depends(verify_api_key)])
@router.post("/chat/completions", dependencies=[Depends(verify_api_key)])
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI-compatible /v1/chat/completions endpoint.
    Supports streaming (SSE) and non-streaming, multi-modal, function/tool calling,
    and reasoning/thinking models.
    """
    start_time = time.perf_counter()
    req_id = f"req_{uuid.uuid4().hex[:12]}"
    backend = get_active_backend_name()

    if request.model in DEPRECATED_MODELS:
        duration_ms = (time.perf_counter() - start_time) * 1000
        history_manager.record(
            model=request.model,
            resolved_model=request.model,
            backend=backend,
            duration_ms=duration_ms,
            status="error",
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            error_message=f"The model `{request.model}` does not exist or is deprecated.",
            request_id=req_id,
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": {
                    "message": f"The model `{request.model}` does not exist or you do not have access to it.",
                    "type": "invalid_request_error",
                    "param": "model",
                    "code": "model_not_found",
                }
            },
        )

    try:
        (internal_model, contents, system_instruction, generation_config, tools) = (
            OpenAITranslator.openai_to_internal_request(request)
        )
    except Exception as e:
        duration_ms = (time.perf_counter() - start_time) * 1000
        history_manager.record(
            model=getattr(request, "model", "unknown") or "unknown",
            resolved_model="unknown",
            backend=backend,
            duration_ms=duration_ms,
            status="error",
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            error_message=str(e),
            request_id=req_id,
        )
        logger.error(f"Error translating OpenAI request: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid request parameters: {e!s}",
        ) from e

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
            served_backends: list[str] = []

            def _on_served(name: str) -> None:
                served_backends.append(name)

            event_stream = client.stream_generate_content(
                model=internal_model,
                contents=contents,
                system_instruction=system_instruction,
                generation_config=generation_config,
                tools=tools,
                on_backend_served=_on_served,
            )
            usage_collector: dict = {}
            openai_chunks = OpenAITranslator.internal_stream_to_openai_chunks(
                event_stream=event_stream,
                requested_model=request.model,
                include_usage=include_usage,
                usage_collector=usage_collector,
            )

            async def tracked_chat_stream():
                stream_status = "success"
                error_msg = None
                try:
                    async for chunk in openai_chunks:
                        yield chunk
                except RateLimitError as stream_err:
                    stream_status = "rate_limited"
                    error_msg = str(stream_err)
                    raise
                except asyncio.CancelledError:
                    stream_status = "stream-aborted"
                    raise
                except Exception as stream_err:
                    stream_status = "error"
                    error_msg = str(stream_err)
                    raise
                finally:
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    history_manager.record(
                        model=request.model,
                        resolved_model=internal_model,
                        backend=served_backends[-1] if served_backends else backend,
                        duration_ms=duration_ms,
                        status=stream_status,
                        prompt_tokens=usage_collector.get("prompt_tokens", 0),
                        completion_tokens=usage_collector.get("completion_tokens", 0),
                        total_tokens=usage_collector.get("total_tokens", 0),
                        error_message=error_msg,
                        request_id=req_id,
                    )

            return StreamingResponse(
                tracked_chat_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Content-Type": "text/event-stream",
                    "X-Accel-Buffering": "no",
                },
            )
        except RateLimitError as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            history_manager.record(
                model=request.model,
                resolved_model=internal_model,
                backend=backend,
                duration_ms=duration_ms,
                status="rate_limited",
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                error_message=str(e),
                request_id=req_id,
            )
            logger.warning(f"Streaming chat rate limited: {e}")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "error": {
                        "message": str(e),
                        "type": "requests",
                        "param": None,
                        "code": "rate_limit_exceeded",
                    }
                },
                headers={
                    "Retry-After": str(int(getattr(e, "retry_after", 60.0) or 60.0))
                },
            ) from e
        except ModelNotFoundError as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            history_manager.record(
                model=request.model,
                resolved_model=internal_model,
                backend=backend,
                duration_ms=duration_ms,
                status="error",
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                error_message=str(e),
                request_id=req_id,
            )
            logger.warning(f"Streaming chat model not found: {e}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": {
                        "message": f"The model `{request.model}` does not exist or you do not have access to it.",
                        "type": "invalid_request_error",
                        "param": "model",
                        "code": "model_not_found",
                    }
                },
            ) from e
        except ValueError as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            history_manager.record(
                model=request.model,
                resolved_model=internal_model,
                backend=backend,
                duration_ms=duration_ms,
                status="error",
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                error_message=str(e),
                request_id=req_id,
            )
            status_code = (
                status.HTTP_503_SERVICE_UNAVAILABLE
                if "No backends" in str(e) or "No configured" in str(e)
                else status.HTTP_400_BAD_REQUEST
            )
            code_str = (
                "service_unavailable"
                if status_code == status.HTTP_503_SERVICE_UNAVAILABLE
                else "invalid_request_error"
            )
            err_type = (
                "api_error"
                if status_code == status.HTTP_503_SERVICE_UNAVAILABLE
                else "invalid_request_error"
            )
            raise HTTPException(
                status_code=status_code,
                detail={
                    "error": {
                        "message": str(e),
                        "type": err_type,
                        "param": None,
                        "code": code_str,
                    }
                },
            ) from e
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            history_manager.record(
                model=request.model,
                resolved_model=internal_model,
                backend=backend,
                duration_ms=duration_ms,
                status="error",
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                error_message=str(e),
                request_id=req_id,
            )
            logger.error(f"Streaming generation error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Generation failed: {e!s}",
            ) from e

    # 2. Non-streaming response
    try:
        served_backends: list[str] = []
        result = await client.generate_content(
            model=internal_model,
            contents=contents,
            system_instruction=system_instruction,
            generation_config=generation_config,
            tools=tools,
            on_backend_served=served_backends.append,
        )
        backend = served_backends[-1] if served_backends else backend
        response_json = OpenAITranslator.internal_to_openai_response(
            result=result, requested_model=request.model
        )
        duration_ms = (time.perf_counter() - start_time) * 1000
        usage = response_json.get("usage", {})
        history_manager.record(
            model=request.model,
            resolved_model=internal_model,
            backend=served_backends[-1] if served_backends else backend,
            duration_ms=duration_ms,
            status="success",
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            request_id=response_json.get("id") or req_id,
        )
        return JSONResponse(content=response_json)
    except RateLimitError as e:
        duration_ms = (time.perf_counter() - start_time) * 1000
        history_manager.record(
            model=request.model,
            resolved_model=internal_model,
            backend=backend,
            duration_ms=duration_ms,
            status="rate_limited",
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            error_message=str(e),
            request_id=req_id,
        )
        logger.warning(f"Non-streaming chat rate limited: {e}")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error": {
                    "message": str(e),
                    "type": "requests",
                    "param": None,
                    "code": "rate_limit_exceeded",
                }
            },
            headers={"Retry-After": str(int(getattr(e, "retry_after", 60.0) or 60.0))},
        ) from e
    except ModelNotFoundError as e:
        duration_ms = (time.perf_counter() - start_time) * 1000
        history_manager.record(
            model=request.model,
            resolved_model=internal_model,
            backend=backend,
            duration_ms=duration_ms,
            status="error",
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            error_message=str(e),
            request_id=req_id,
        )
        logger.warning(f"Non-streaming chat model not found: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": {
                    "message": f"The model `{request.model}` does not exist or you do not have access to it.",
                    "type": "invalid_request_error",
                    "param": "model",
                    "code": "model_not_found",
                }
            },
        ) from e
    except ValueError as e:
        duration_ms = (time.perf_counter() - start_time) * 1000
        history_manager.record(
            model=request.model,
            resolved_model=internal_model,
            backend=backend,
            duration_ms=duration_ms,
            status="error",
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            error_message=str(e),
            request_id=req_id,
        )
        status_code = (
            status.HTTP_503_SERVICE_UNAVAILABLE
            if "No backends" in str(e) or "No configured" in str(e)
            else status.HTTP_400_BAD_REQUEST
        )
        code_str = (
            "service_unavailable"
            if status_code == status.HTTP_503_SERVICE_UNAVAILABLE
            else "invalid_request_error"
        )
        err_type = (
            "api_error"
            if status_code == status.HTTP_503_SERVICE_UNAVAILABLE
            else "invalid_request_error"
        )
        raise HTTPException(
            status_code=status_code,
            detail={
                "error": {
                    "message": str(e),
                    "type": err_type,
                    "param": None,
                    "code": code_str,
                }
            },
        ) from e
    except Exception as e:
        duration_ms = (time.perf_counter() - start_time) * 1000
        history_manager.record(
            model=request.model,
            resolved_model=internal_model,
            backend=backend,
            duration_ms=duration_ms,
            status="error",
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            error_message=str(e),
            request_id=req_id,
        )
        logger.error(f"Non-streaming generation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Generation failed: {e!s}",
        ) from e


@router.post("/v1/completions", dependencies=[Depends(verify_api_key)])
@router.post("/completions", dependencies=[Depends(verify_api_key)])
async def legacy_completions(request: Request):
    """
    Legacy /v1/completions endpoint returning standard OpenAI text_completion objects.
    Supports streaming (SSE) and non-streaming responses.
    """
    start_time = time.perf_counter()
    req_id = f"req_{uuid.uuid4().hex[:12]}"
    backend = get_active_backend_name()

    try:
        body = await request.json()
    except Exception:
        duration_ms = (time.perf_counter() - start_time) * 1000
        history_manager.record(
            model="unknown",
            resolved_model="unknown",
            backend=backend,
            duration_ms=duration_ms,
            status="error",
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            error_message="Invalid JSON request body.",
            request_id=req_id,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid JSON request body."
        ) from None

    prompt = body.get("prompt", "")
    if isinstance(prompt, list):
        prompt = "\n".join(str(p) for p in prompt)
    elif not isinstance(prompt, str):
        prompt = str(prompt)

    model_name = body.get("model", "gemini-3.7-flash-high")
    if model_name in DEPRECATED_MODELS:
        duration_ms = (time.perf_counter() - start_time) * 1000
        history_manager.record(
            model=model_name,
            resolved_model=model_name,
            backend=backend,
            duration_ms=duration_ms,
            status="error",
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            error_message=f"The model `{model_name}` does not exist or is deprecated.",
            request_id=req_id,
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": {
                    "message": f"The model `{model_name}` does not exist or you do not have access to it.",
                    "type": "invalid_request_error",
                    "param": "model",
                    "code": "model_not_found",
                }
            },
        )
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
        stream_options=stream_options,
    )

    try:
        (internal_model, contents, system_instruction, generation_config, tools) = (
            OpenAITranslator.openai_to_internal_request(chat_req)
        )
    except Exception as e:
        duration_ms = (time.perf_counter() - start_time) * 1000
        history_manager.record(
            model=model_name,
            resolved_model="unknown",
            backend=backend,
            duration_ms=duration_ms,
            status="error",
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            error_message=str(e),
            request_id=req_id,
        )
        logger.error(f"Error translating completion request: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid request parameters: {e!s}",
        ) from e

    # 1. Streaming response
    if stream:
        try:
            served_backends: list[str] = []

            def _on_served(name: str) -> None:
                served_backends.append(name)

            event_stream = client.stream_generate_content(
                model=internal_model,
                contents=contents,
                system_instruction=system_instruction,
                generation_config=generation_config,
                tools=tools,
                on_backend_served=_on_served,
            )
            usage_collector: dict = {}
            openai_chunks = OpenAITranslator.internal_stream_to_openai_text_chunks(
                event_stream=event_stream,
                requested_model=model_name,
                include_usage=include_usage,
                usage_collector=usage_collector,
            )

            async def tracked_text_stream():
                stream_status = "success"
                error_msg = None
                try:
                    async for chunk in openai_chunks:
                        yield chunk
                except RateLimitError as stream_err:
                    stream_status = "rate_limited"
                    error_msg = str(stream_err)
                    raise
                except asyncio.CancelledError:
                    stream_status = "stream-aborted"
                    raise
                except Exception as stream_err:
                    stream_status = "error"
                    error_msg = str(stream_err)
                    raise
                finally:
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    history_manager.record(
                        model=model_name,
                        resolved_model=internal_model,
                        backend=served_backends[-1] if served_backends else backend,
                        duration_ms=duration_ms,
                        status=stream_status,
                        prompt_tokens=usage_collector.get("prompt_tokens", 0),
                        completion_tokens=usage_collector.get("completion_tokens", 0),
                        total_tokens=usage_collector.get("total_tokens", 0),
                        error_message=error_msg,
                        request_id=req_id,
                    )

            return StreamingResponse(
                tracked_text_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Content-Type": "text/event-stream",
                    "X-Accel-Buffering": "no",
                },
            )
        except RateLimitError as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            history_manager.record(
                model=model_name,
                resolved_model=internal_model,
                backend=backend,
                duration_ms=duration_ms,
                status="rate_limited",
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                error_message=str(e),
                request_id=req_id,
            )
            logger.warning(f"Streaming text completion rate limited: {e}")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "error": {
                        "message": str(e),
                        "type": "requests",
                        "param": None,
                        "code": "rate_limit_exceeded",
                    }
                },
                headers={
                    "Retry-After": str(int(getattr(e, "retry_after", 60.0) or 60.0))
                },
            ) from e
        except ModelNotFoundError as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            history_manager.record(
                model=model_name,
                resolved_model=internal_model,
                backend=backend,
                duration_ms=duration_ms,
                status="error",
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                error_message=str(e),
                request_id=req_id,
            )
            logger.warning(f"Streaming text model not found: {e}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": {
                        "message": f"The model `{model_name}` does not exist or you do not have access to it.",
                        "type": "invalid_request_error",
                        "param": "model",
                        "code": "model_not_found",
                    }
                },
            ) from e
        except ValueError as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            history_manager.record(
                model=model_name,
                resolved_model=internal_model,
                backend=backend,
                duration_ms=duration_ms,
                status="error",
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                error_message=str(e),
                request_id=req_id,
            )
            status_code = (
                status.HTTP_503_SERVICE_UNAVAILABLE
                if "No backends" in str(e) or "No configured" in str(e)
                else status.HTTP_400_BAD_REQUEST
            )
            code_str = (
                "service_unavailable"
                if status_code == status.HTTP_503_SERVICE_UNAVAILABLE
                else "invalid_request_error"
            )
            err_type = (
                "api_error"
                if status_code == status.HTTP_503_SERVICE_UNAVAILABLE
                else "invalid_request_error"
            )
            raise HTTPException(
                status_code=status_code,
                detail={
                    "error": {
                        "message": str(e),
                        "type": err_type,
                        "param": None,
                        "code": code_str,
                    }
                },
            ) from e
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            history_manager.record(
                model=model_name,
                resolved_model=internal_model,
                backend=backend,
                duration_ms=duration_ms,
                status="error",
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                error_message=str(e),
                request_id=req_id,
            )
            logger.error(f"Streaming text completion generation error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Generation failed: {e!s}",
            ) from e

    # 2. Non-streaming response
    try:
        served_backends: list[str] = []
        result = await client.generate_content(
            model=internal_model,
            contents=contents,
            system_instruction=system_instruction,
            generation_config=generation_config,
            tools=tools,
            on_backend_served=served_backends.append,
        )
        backend = served_backends[-1] if served_backends else backend
        response_json = OpenAITranslator.internal_to_openai_text_completion(
            result=result, requested_model=model_name
        )
        duration_ms = (time.perf_counter() - start_time) * 1000
        usage = response_json.get("usage", {})
        history_manager.record(
            model=model_name,
            resolved_model=internal_model,
            backend=served_backends[-1] if served_backends else backend,
            duration_ms=duration_ms,
            status="success",
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            request_id=response_json.get("id") or req_id,
        )
        return JSONResponse(content=response_json)
    except RateLimitError as e:
        duration_ms = (time.perf_counter() - start_time) * 1000
        history_manager.record(
            model=model_name,
            resolved_model=internal_model,
            backend=backend,
            duration_ms=duration_ms,
            status="rate_limited",
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            error_message=str(e),
            request_id=req_id,
        )
        logger.warning(f"Non-streaming text completion rate limited: {e}")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error": {
                    "message": str(e),
                    "type": "requests",
                    "param": None,
                    "code": "rate_limit_exceeded",
                }
            },
            headers={"Retry-After": str(int(getattr(e, "retry_after", 60.0) or 60.0))},
        ) from e
    except ModelNotFoundError as e:
        duration_ms = (time.perf_counter() - start_time) * 1000
        history_manager.record(
            model=model_name,
            resolved_model=internal_model,
            backend=backend,
            duration_ms=duration_ms,
            status="error",
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            error_message=str(e),
            request_id=req_id,
        )
        logger.warning(f"Non-streaming text model not found: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": {
                    "message": f"The model `{model_name}` does not exist or you do not have access to it.",
                    "type": "invalid_request_error",
                    "param": "model",
                    "code": "model_not_found",
                }
            },
        ) from e
    except ValueError as e:
        duration_ms = (time.perf_counter() - start_time) * 1000
        history_manager.record(
            model=model_name,
            resolved_model=internal_model,
            backend=backend,
            duration_ms=duration_ms,
            status="error",
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            error_message=str(e),
            request_id=req_id,
        )
        status_code = (
            status.HTTP_503_SERVICE_UNAVAILABLE
            if "No backends" in str(e) or "No configured" in str(e)
            else status.HTTP_400_BAD_REQUEST
        )
        code_str = (
            "service_unavailable"
            if status_code == status.HTTP_503_SERVICE_UNAVAILABLE
            else "invalid_request_error"
        )
        err_type = (
            "api_error"
            if status_code == status.HTTP_503_SERVICE_UNAVAILABLE
            else "invalid_request_error"
        )
        raise HTTPException(
            status_code=status_code,
            detail={
                "error": {
                    "message": str(e),
                    "type": err_type,
                    "param": None,
                    "code": code_str,
                }
            },
        ) from e
    except Exception as e:
        duration_ms = (time.perf_counter() - start_time) * 1000
        history_manager.record(
            model=model_name,
            resolved_model=internal_model,
            backend=backend,
            duration_ms=duration_ms,
            status="error",
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            error_message=str(e),
            request_id=req_id,
        )
        logger.error(f"Non-streaming text completion generation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Generation failed: {e!s}",
        ) from e


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
                detail="input list cannot be empty",
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
            detail=f"Invalid encoding_format: {encoding_format}. Must be 'float' or 'base64'.",
        )

    if request.dimensions is not None and request.dimensions <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="dimensions must be a positive integer.",
        )

    if request.model in DEPRECATED_MODELS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": {
                    "message": f"The model `{request.model}` does not exist or you do not have access to it.",
                    "type": "invalid_request_error",
                    "param": "model",
                    "code": "model_not_found",
                }
            },
        )

    resolved_model = OpenAITranslator.resolve_model(request.model)

    try:
        raw_result = await client.embed_contents(
            model=resolved_model, texts=texts, dimensions=request.dimensions
        )
    except RateLimitError as e:
        logger.warning(f"Embedding rate limited: {e}")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error": {
                    "message": str(e),
                    "type": "requests",
                    "param": None,
                    "code": "rate_limit_exceeded",
                }
            },
            headers={"Retry-After": str(int(getattr(e, "retry_after", 60.0) or 60.0))},
        ) from e
    except ModelNotFoundError as e:
        logger.warning(f"Embedding model not found: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": {
                    "message": f"The model `{request.model}` does not exist or you do not have access to it.",
                    "type": "invalid_request_error",
                    "param": "model",
                    "code": "model_not_found",
                }
            },
        ) from e
    except ValueError as e:
        status_code = (
            status.HTTP_503_SERVICE_UNAVAILABLE
            if "No backends" in str(e) or "No configured" in str(e)
            else status.HTTP_400_BAD_REQUEST
        )
        code_str = (
            "service_unavailable"
            if status_code == status.HTTP_503_SERVICE_UNAVAILABLE
            else "invalid_request_error"
        )
        err_type = (
            "api_error"
            if status_code == status.HTTP_503_SERVICE_UNAVAILABLE
            else "invalid_request_error"
        )
        raise HTTPException(
            status_code=status_code,
            detail={
                "error": {
                    "message": str(e),
                    "type": err_type,
                    "param": None,
                    "code": code_str,
                }
            },
        ) from e
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Embedding failed: {e!s}",
        ) from e

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
            vec = vec[: request.dimensions]
            if len(vec) < request.dimensions:
                vec = vec + [0.0] * (request.dimensions - len(vec))

        if encoding_format == "base64":
            binary_data = struct.pack(f"<{len(vec)}f", *vec)
            embedding_val = base64.b64encode(binary_data).decode("utf-8")
        else:
            embedding_val = [float(v) for v in vec]

        data.append({"object": "embedding", "index": i, "embedding": embedding_val})

    # 5. Token usage calculation
    usage_meta = (
        raw_result.get("usageMetadata", {}) if isinstance(raw_result, dict) else {}
    )
    prompt_tokens = usage_meta.get("promptTokenCount")
    if prompt_tokens is None:
        prompt_tokens = sum(max(1, len(t) // 4) for t in texts)

    return {
        "object": "list",
        "data": data,
        "model": request.model,
        "usage": {"prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens},
    }
