import json
import logging
import time
import uuid
from collections.abc import AsyncGenerator
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.config import ANTIGRAVITY_TIER_MAP, CANONICAL_MODEL_MAP, MODEL_ALIASES

logger = logging.getLogger("google_gate.translator")

# Cache to store thought signatures across turns for multi-turn function calling
_thought_signature_cache: dict[str, str] = {}


class ChatMessage(BaseModel):
    role: str
    content: Any | None = None
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    reasoning_content: str | None = None


class StreamOptions(BaseModel):
    include_usage: bool | None = None


class ResponseFormat(BaseModel):
    type: str | None = "text"
    json_schema: dict[str, Any] | None = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    model: str
    messages: list[dict[str, Any]]
    temperature: float | None = None
    top_p: float | None = None
    n: int | None = None
    stop: str | list[str] | None = None
    max_tokens: int | None = Field(default=None, alias="max_completion_tokens")
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    seed: int | None = None
    response_format: ResponseFormat | dict[str, Any] | None = None
    stream: bool | None = False
    stream_options: StreamOptions | dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    reasoning_effort: str | None = None
    user: str | None = None


class EmbeddingRequest(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    input: str | list[str] | list[int] | list[list[int]]
    model: str = "text-embedding-004"
    encoding_format: str | None = "float"
    dimensions: int | None = None
    user: str | None = None


class OpenAITranslator:
    @staticmethod
    def get_system_fingerprint(model: str) -> str:
        return f"fp_gate_{abs(hash(model)) & 0xFFFFFFFF:08x}"

    @classmethod
    def _generate_fingerprint(cls, model: str) -> str:
        return cls.get_system_fingerprint(model)

    @staticmethod
    def resolve_model(requested_model: str, reasoning_effort: str | None = None) -> str:
        """
        Map user-requested model and optional reasoning_effort to Antigravity internal model name
        using the lookup table and tier mappings.
        """
        clean = requested_model.lower().strip()

        # 1. Check if model or its canonical alias has reasoning tier mappings in ANTIGRAVITY_TIER_MAP
        base_name = CANONICAL_MODEL_MAP.get(clean, clean)
        if base_name in ANTIGRAVITY_TIER_MAP:
            tiers = ANTIGRAVITY_TIER_MAP[base_name]
            effort = (reasoning_effort or "").lower().strip()
            if effort in tiers:
                return tiers[effort]
            if "default" in tiers:
                return tiers["default"]

        # 2. Check direct MODEL_ALIASES lookup
        if clean in MODEL_ALIASES:
            return MODEL_ALIASES[clean]

        # 3. Check prefix matches
        for alias, internal in MODEL_ALIASES.items():
            if clean.startswith(alias):
                return internal

        # Default fallback
        return requested_model

    @classmethod
    def openai_to_internal_request(
        cls, req: ChatCompletionRequest
    ) -> tuple[
        str,
        list[dict[str, Any]],
        dict[str, Any] | None,
        dict[str, Any] | None,
        list[dict[str, Any]] | None,
    ]:
        """
        Convert OpenAI ChatCompletionRequest into internal format:
        Returns: (internal_model, contents, system_instruction, generation_config, tools)
        """
        internal_model = cls.resolve_model(req.model, req.reasoning_effort)

        contents: list[dict[str, Any]] = []
        system_texts: list[str] = []

        # Track tool names for tool_call_id -> function_name mapping
        tool_call_names: dict[str, str] = {}

        for msg in req.messages:
            role = str(msg.get("role", "user")).lower()
            content = msg.get("content")

            # Extract system / developer messages
            if role in ("system", "developer"):
                if isinstance(content, str):
                    system_texts.append(content)
                elif isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            system_texts.append(item.get("text", ""))
                        elif isinstance(item, str):
                            system_texts.append(item)
                continue

            # Handle user message
            if role == "user":
                parts = []
                if isinstance(content, str):
                    parts.append({"text": content})
                elif isinstance(content, list):
                    for item in content:
                        if isinstance(item, str):
                            parts.append({"text": item})
                        elif isinstance(item, dict):
                            itype = item.get("type")
                            if itype == "text":
                                parts.append({"text": item.get("text", "")})
                            elif itype == "image_url":
                                img_url_obj = item.get("image_url", {})
                                url = (
                                    img_url_obj.get("url", "")
                                    if isinstance(img_url_obj, dict)
                                    else str(img_url_obj)
                                )
                                if url.startswith("data:"):
                                    # data:image/png;base64,.....
                                    header, b64_data = url.split(",", 1)
                                    mime_type = header.split(";")[0].replace(
                                        "data:", ""
                                    )
                                    parts.append(
                                        {
                                            "inlineData": {
                                                "mimeType": mime_type,
                                                "data": b64_data,
                                            }
                                        }
                                    )
                                else:
                                    parts.append({"text": f"[Image URL: {url}]"})
                            elif itype in ("audio", "input_audio"):
                                audio_obj = (
                                    item.get("audio") or item.get("input_audio") or {}
                                )
                                audio_data = audio_obj.get("data", "")
                                audio_format = audio_obj.get("format", "wav")
                                mime_type = f"audio/{audio_format}"
                                parts.append(
                                    {
                                        "inlineData": {
                                            "mimeType": mime_type,
                                            "data": audio_data,
                                        }
                                    }
                                )
                            elif itype in ("file", "document"):
                                file_obj = item.get("file") or {}
                                if isinstance(file_obj, str):
                                    file_val = file_obj
                                    file_mime = item.get("mime_type") or item.get(
                                        "mimeType"
                                    )
                                else:
                                    file_val = (
                                        file_obj.get("file_data")
                                        or file_obj.get("data")
                                        or file_obj.get("url")
                                        or item.get("file_data")
                                        or (
                                            item.get("file_url", {}).get("url")
                                            if isinstance(item.get("file_url"), dict)
                                            else item.get("file_url")
                                        )
                                        or item.get("url")
                                        or item.get("data")
                                        or ""
                                    )
                                    file_mime = (
                                        file_obj.get("mime_type")
                                        or file_obj.get("mimeType")
                                        or item.get("mime_type")
                                        or item.get("mimeType")
                                    )

                                if isinstance(file_val, str) and file_val.startswith(
                                    "data:"
                                ):
                                    header, b64_data = file_val.split(",", 1)
                                    extracted_mime = (
                                        header.split(";")[0]
                                        .replace("data:", "")
                                        .strip()
                                    )
                                    mime_type = (
                                        extracted_mime or file_mime or "application/pdf"
                                    )
                                    parts.append(
                                        {
                                            "inlineData": {
                                                "mimeType": mime_type,
                                                "data": b64_data,
                                            }
                                        }
                                    )
                                elif isinstance(file_val, str) and file_val.startswith(
                                    ("http://", "https://")
                                ):
                                    parts.append({"text": f"[File URL: {file_val}]"})
                                elif isinstance(file_val, str) and file_val:
                                    mime_type = file_mime or "application/pdf"
                                    parts.append(
                                        {
                                            "inlineData": {
                                                "mimeType": mime_type,
                                                "data": file_val,
                                            }
                                        }
                                    )
                if parts:
                    contents.append({"role": "user", "parts": parts})

            # Handle assistant message
            elif role == "assistant":
                parts = []
                # Text content
                if isinstance(content, str) and content:
                    parts.append({"text": content})

                # Tool calls
                tool_calls = msg.get("tool_calls", [])
                for tc in tool_calls:
                    tc_id = tc.get("id", f"call_{uuid.uuid4().hex[:6]}")
                    func = tc.get("function", {})
                    fn_name = func.get("name", "")
                    tool_call_names[tc_id] = fn_name

                    fn_args = func.get("arguments", {})
                    if isinstance(fn_args, str):
                        try:
                            fn_args = json.loads(fn_args)
                        except Exception:
                            fn_args = {"raw_args": fn_args}

                    part: dict[str, Any] = {
                        "functionCall": {"name": fn_name, "args": fn_args, "id": tc_id}
                    }

                    # Attach thought signature if previously cached for this tool call or model
                    if tc_id in _thought_signature_cache:
                        part["thoughtSignature"] = _thought_signature_cache[tc_id]
                    elif "last_signature" in _thought_signature_cache:
                        part["thoughtSignature"] = _thought_signature_cache[
                            "last_signature"
                        ]

                    parts.append(part)

                if parts:
                    contents.append({"role": "model", "parts": parts})

            # Handle tool response message
            elif role in ("tool", "function"):
                tool_call_id = msg.get("tool_call_id", "")
                fn_name = msg.get("name") or tool_call_names.get(
                    tool_call_id, "tool_response"
                )

                response_val: Any = content
                if isinstance(content, str):
                    try:
                        response_val = json.loads(content)
                    except Exception:
                        response_val = {"result": content}

                func_resp_part = {
                    "functionResponse": {
                        "name": fn_name,
                        "response": {"result": response_val}
                        if not isinstance(response_val, dict)
                        else response_val,
                    }
                }
                if tool_call_id:
                    func_resp_part["functionResponse"]["id"] = tool_call_id

                contents.append({"role": "user", "parts": [func_resp_part]})

        # Build System Instruction
        system_instruction = None
        if system_texts:
            system_instruction = {"parts": [{"text": "\n\n".join(system_texts)}]}

        # Build Generation Config
        # Determine model limit
        model_limit = 65536
        if "claude" in internal_model:
            model_limit = 64000
        elif "gpt-oss" in internal_model:
            model_limit = 32768
        elif "flash" in internal_model or "pro" in internal_model:
            model_limit = 65536

        max_tokens = (
            min(req.max_tokens, model_limit)
            if req.max_tokens
            else min(8192, model_limit)
        )
        generation_config: dict[str, Any] = {"maxOutputTokens": max_tokens}
        if req.temperature is not None:
            generation_config["temperature"] = req.temperature
        if req.top_p is not None:
            generation_config["topP"] = req.top_p
        if req.presence_penalty is not None:
            generation_config["presencePenalty"] = req.presence_penalty
        if req.frequency_penalty is not None:
            generation_config["frequencyPenalty"] = req.frequency_penalty
        if req.seed is not None:
            generation_config["seed"] = req.seed
        if req.n is not None and req.n > 1:
            generation_config["candidateCount"] = req.n

        # Stop sequences
        if req.stop:
            if isinstance(req.stop, str):
                generation_config["stopSequences"] = [req.stop]
            elif isinstance(req.stop, list):
                generation_config["stopSequences"] = req.stop

        # Response Format / Structured Outputs
        if req.response_format:
            resp_fmt = req.response_format
            if isinstance(resp_fmt, dict):
                fmt_type = resp_fmt.get("type")
                if fmt_type in ("json_object", "json"):
                    generation_config["responseMimeType"] = "application/json"
                elif fmt_type == "json_schema":
                    generation_config["responseMimeType"] = "application/json"
                    schema_def = resp_fmt.get("json_schema", {})
                    if "schema" in schema_def:
                        generation_config["responseSchema"] = schema_def["schema"]
                    elif schema_def:
                        generation_config["responseSchema"] = schema_def
            elif hasattr(resp_fmt, "type"):
                if resp_fmt.type in ("json_object", "json"):
                    generation_config["responseMimeType"] = "application/json"
                elif resp_fmt.type == "json_schema":
                    generation_config["responseMimeType"] = "application/json"
                    if resp_fmt.json_schema:
                        generation_config["responseSchema"] = resp_fmt.json_schema.get(
                            "schema", resp_fmt.json_schema
                        )

        # Reasoning / Thinking config
        if "gemini-3.7" in internal_model or "gemini-3" in internal_model:
            budget = -1
            if req.reasoning_effort == "low":
                budget = 1000
            elif req.reasoning_effort == "medium":
                budget = 4000
            elif req.reasoning_effort == "high":
                budget = 16000

            generation_config["thinkingConfig"] = {
                "includeThoughts": True,
                "thinkingBudget": budget,
            }
        elif "claude" in internal_model:
            if "thinking" in internal_model or req.reasoning_effort:
                budget = 1024
                if req.reasoning_effort == "medium":
                    budget = 4096
                elif req.reasoning_effort == "high":
                    budget = 16384
                generation_config["thinkingConfig"] = {
                    "includeThoughts": True,
                    "thinkingBudget": budget,
                }
        elif "gpt-oss" in internal_model:
            if req.reasoning_effort:
                generation_config["thinkingConfig"] = {
                    "includeThoughts": True,
                    "thinkingBudget": 8192,
                }

        # Build Tools & Tool Config
        tools = None
        if req.tools:
            function_declarations = []
            for t in req.tools:
                if t.get("type") == "function":
                    fn = t.get("function", {})
                    fn_decl = {
                        "name": fn.get("name"),
                        "description": fn.get("description", ""),
                        "parameters": fn.get("parameters", {}),
                    }
                    function_declarations.append(fn_decl)
            if function_declarations:
                tools = [{"functionDeclarations": function_declarations}]

        # Tool choice handling
        if req.tool_choice and tools:
            # Map tool_choice to toolConfig / functionCallingConfig
            tool_config: dict[str, Any] = {}
            if isinstance(req.tool_choice, str):
                tc_lower = req.tool_choice.lower()
                if tc_lower == "none":
                    tool_config = {"functionCallingConfig": {"mode": "NONE"}}
                elif tc_lower == "auto":
                    tool_config = {"functionCallingConfig": {"mode": "AUTO"}}
                elif tc_lower == "required":
                    tool_config = {"functionCallingConfig": {"mode": "ANY"}}
            elif isinstance(req.tool_choice, dict):
                tc_type = req.tool_choice.get("type")
                if tc_type == "function":
                    fn_obj = req.tool_choice.get("function", {})
                    fn_name = fn_obj.get("name")
                    if fn_name:
                        tool_config = {
                            "functionCallingConfig": {
                                "mode": "ANY",
                                "allowedFunctionNames": [fn_name],
                            }
                        }
            if tool_config:
                tools.append(tool_config)

        return internal_model, contents, system_instruction, generation_config, tools

    @staticmethod
    def format_usage(usage_meta: dict[str, Any]) -> dict[str, Any]:
        """
        Format upstream usage metadata into OpenAI standard usage dict.
        Accurately reports prompt tokens, completion tokens, total tokens,
        prompt cached tokens (prompt_tokens_details.cached_tokens),
        and reasoning tokens (completion_tokens_details.reasoning_tokens).
        """
        prompt_tokens = usage_meta.get("promptTokenCount", 0)
        completion_tokens = usage_meta.get("candidatesTokenCount", 0)
        thoughts_tokens = usage_meta.get("thoughtsTokenCount", 0)
        cached_tokens = (
            usage_meta.get("cachedContentTokenCount")
            or usage_meta.get("cachedTokens")
            or usage_meta.get("cached_content_token_count")
            or 0
        )
        total_tokens = usage_meta.get(
            "totalTokenCount", prompt_tokens + completion_tokens
        )

        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "prompt_tokens_details": {"cached_tokens": cached_tokens},
            "completion_tokens_details": {"reasoning_tokens": thoughts_tokens},
        }

    @classmethod
    def internal_to_openai_response(
        cls, result: dict[str, Any], requested_model: str
    ) -> dict[str, Any]:
        """
        Convert complete internal response into OpenAI ChatCompletion response.
        Supports single and multi-candidate (n > 1) responses.
        """
        completion_id = result.get("responseId") or f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created_time = int(time.time())

        choices = []
        candidates = result.get("candidates", [])

        if candidates:
            for cand in candidates:
                cand_idx = cand.get("index", len(choices))
                cand_thought_sig = cand.get("thoughtSignature") or result.get(
                    "thoughtSignature"
                )
                if cand_thought_sig:
                    _thought_signature_cache["last_signature"] = cand_thought_sig

                tool_calls_openai = []
                raw_tool_calls = cand.get("toolCalls", [])
                for tc in raw_tool_calls:
                    call_id = tc.get("id") or f"call_{uuid.uuid4().hex[:8]}"
                    if cand_thought_sig:
                        _thought_signature_cache[call_id] = cand_thought_sig

                    args = tc.get("args", {})
                    args_str = json.dumps(args) if isinstance(args, dict) else str(args)
                    tool_calls_openai.append(
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": tc.get("name", ""),
                                "arguments": args_str,
                            },
                        }
                    )

                msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": cand.get("text", ""),
                }
                if cand.get("thoughts"):
                    msg["reasoning_content"] = cand["thoughts"]
                if tool_calls_openai:
                    msg["tool_calls"] = tool_calls_openai

                finish_reason = "stop"
                raw_finish = cand.get("finishReason", "STOP")
                if tool_calls_openai or raw_finish == "TOOL_CALL":
                    finish_reason = "tool_calls"
                elif raw_finish == "MAX_TOKENS":
                    finish_reason = "length"

                choices.append(
                    {"index": cand_idx, "message": msg, "finish_reason": finish_reason}
                )
        else:
            # Fallback for single candidate payload without candidates list
            thought_sig = result.get("thoughtSignature")
            if thought_sig:
                _thought_signature_cache["last_signature"] = thought_sig

            tool_calls_openai = []
            raw_tool_calls = result.get("toolCalls", [])
            for tc in raw_tool_calls:
                call_id = tc.get("id") or f"call_{uuid.uuid4().hex[:8]}"
                if thought_sig:
                    _thought_signature_cache[call_id] = thought_sig

                args = tc.get("args", {})
                args_str = json.dumps(args) if isinstance(args, dict) else str(args)
                tool_calls_openai.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": tc.get("name", ""), "arguments": args_str},
                    }
                )

            msg = {"role": "assistant", "content": result.get("text", "")}
            if result.get("thoughts"):
                msg["reasoning_content"] = result["thoughts"]
            if tool_calls_openai:
                msg["tool_calls"] = tool_calls_openai

            finish_reason = "stop"
            raw_finish = result.get("finishReason", "STOP")
            if tool_calls_openai or raw_finish == "TOOL_CALL":
                finish_reason = "tool_calls"
            elif raw_finish == "MAX_TOKENS":
                finish_reason = "length"

            choices.append({"index": 0, "message": msg, "finish_reason": finish_reason})

        usage_meta = result.get("usageMetadata", {})

        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created_time,
            "model": requested_model,
            "system_fingerprint": cls._generate_fingerprint(requested_model),
            "service_tier": "default",
            "choices": choices,
            "usage": cls.format_usage(usage_meta),
        }

    @classmethod
    def internal_to_openai_text_completion(
        cls, result: dict[str, Any], requested_model: str
    ) -> dict[str, Any]:
        """
        Convert internal response into OpenAI /v1/completions text completion response.
        """
        completion_id = result.get("responseId") or f"cmpl-{uuid.uuid4().hex[:12]}"
        created_time = int(time.time())
        choices = []
        candidates = result.get("candidates", [])

        if candidates:
            for cand in candidates:
                cand_idx = cand.get("index", len(choices))
                finish_reason = "stop"
                raw_finish = cand.get("finishReason", "STOP")
                if raw_finish == "MAX_TOKENS":
                    finish_reason = "length"

                choices.append(
                    {
                        "text": cand.get("text", ""),
                        "index": cand_idx,
                        "logprobs": None,
                        "finish_reason": finish_reason,
                    }
                )
        else:
            finish_reason = "stop"
            raw_finish = result.get("finishReason", "STOP")
            if raw_finish == "MAX_TOKENS":
                finish_reason = "length"

            choices.append(
                {
                    "text": result.get("text", ""),
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": finish_reason,
                }
            )

        usage_meta = result.get("usageMetadata", {})
        return {
            "id": completion_id,
            "object": "text_completion",
            "created": created_time,
            "model": requested_model,
            "system_fingerprint": cls._generate_fingerprint(requested_model),
            "service_tier": "default",
            "choices": choices,
            "usage": cls.format_usage(usage_meta),
        }

    @classmethod
    async def internal_stream_to_openai_chunks(
        cls,
        event_stream: AsyncGenerator[dict[str, Any], None],
        requested_model: str,
        include_usage: bool = False,
        usage_collector: dict[str, Any] | None = None,
    ) -> AsyncGenerator[str, None]:
        """
        Convert SSE stream from Antigravity internal API to standard OpenAI SSE chunk format.
        Yields lines formatted as `data: {...}\n\n` ending with `data: [DONE]\n\n`.
        Supports multi-candidate responses with distinct indices.
        """
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created_time = int(time.time())
        seen_cand_indexes = set()
        last_thought_signature = None
        latest_usage_metadata: dict[str, Any] = {}
        fp = cls._generate_fingerprint(requested_model)

        async for event in event_stream:
            resp_obj = (
                event.get("response")
                if isinstance(event.get("response"), dict)
                else event
            )
            if "responseId" in resp_obj and not completion_id.startswith("chatcmpl"):
                completion_id = resp_obj["responseId"]

            if "usageMetadata" in resp_obj:
                latest_usage_metadata.update(resp_obj["usageMetadata"])
            elif "usageMetadata" in event:
                latest_usage_metadata.update(event["usageMetadata"])
            if usage_collector is not None and latest_usage_metadata:
                usage_collector.update(cls.format_usage(latest_usage_metadata))

            candidates = resp_obj.get("candidates", [])
            for default_idx, cand in enumerate(candidates):
                cand_idx = cand.get("index", default_idx)
                parts = cand.get("content", {}).get("parts", [])
                raw_finish = cand.get("finishReason")

                for p in parts:
                    if p.get("thoughtSignature"):
                        last_thought_signature = p["thoughtSignature"]
                        _thought_signature_cache["last_signature"] = (
                            last_thought_signature
                        )

                    # 1. Thought / Reasoning chunk
                    if p.get("thought"):
                        thought_text = p.get("text", "")
                        if thought_text:
                            delta: dict[str, Any] = {"reasoning_content": thought_text}
                            if cand_idx not in seen_cand_indexes:
                                delta["role"] = "assistant"
                                seen_cand_indexes.add(cand_idx)

                            chunk = {
                                "id": completion_id,
                                "object": "chat.completion.chunk",
                                "created": created_time,
                                "model": requested_model,
                                "system_fingerprint": fp,
                                "choices": [
                                    {
                                        "index": cand_idx,
                                        "delta": delta,
                                        "finish_reason": None,
                                    }
                                ],
                            }
                            yield f"data: {json.dumps(chunk)}\n\n"

                    # 2. Text chunk
                    elif p.get("text"):
                        text_chunk = p["text"]
                        delta = {"content": text_chunk}
                        if cand_idx not in seen_cand_indexes:
                            delta["role"] = "assistant"
                            seen_cand_indexes.add(cand_idx)

                        chunk = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created_time,
                            "model": requested_model,
                            "system_fingerprint": fp,
                            "choices": [
                                {
                                    "index": cand_idx,
                                    "delta": delta,
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"

                    # 3. Tool Call chunk
                    elif p.get("functionCall"):
                        fc = p["functionCall"]
                        call_id = fc.get("id") or f"call_{uuid.uuid4().hex[:8]}"
                        if last_thought_signature:
                            _thought_signature_cache[call_id] = last_thought_signature

                        args = fc.get("args", {})
                        args_str = (
                            json.dumps(args) if isinstance(args, dict) else str(args)
                        )

                        delta = {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": call_id,
                                    "type": "function",
                                    "function": {
                                        "name": fc.get("name", ""),
                                        "arguments": args_str,
                                    },
                                }
                            ]
                        }
                        if cand_idx not in seen_cand_indexes:
                            delta["role"] = "assistant"
                            seen_cand_indexes.add(cand_idx)

                        chunk = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created_time,
                            "model": requested_model,
                            "system_fingerprint": fp,
                            "choices": [
                                {
                                    "index": cand_idx,
                                    "delta": delta,
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"

                # If candidate has finish reason, emit final finish chunk
                if raw_finish:
                    finish_reason = "stop"
                    if raw_finish == "TOOL_CALL" or any(
                        "functionCall" in p for p in parts
                    ):
                        finish_reason = "tool_calls"
                    elif raw_finish == "MAX_TOKENS":
                        finish_reason = "length"

                    final_chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created_time,
                        "model": requested_model,
                        "system_fingerprint": fp,
                        "choices": [
                            {
                                "index": cand_idx,
                                "delta": {},
                                "finish_reason": finish_reason,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(final_chunk)}\n\n"

        # If include_usage was requested, emit final usage chunk
        if include_usage:
            usage_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": requested_model,
                "system_fingerprint": fp,
                "choices": [],
                "usage": cls.format_usage(latest_usage_metadata),
            }
            yield f"data: {json.dumps(usage_chunk)}\n\n"

        # End of stream
        yield "data: [DONE]\n\n"

    @classmethod
    async def internal_stream_to_openai_text_chunks(
        cls,
        event_stream: AsyncGenerator[dict[str, Any], None],
        requested_model: str,
        include_usage: bool = False,
        usage_collector: dict[str, Any] | None = None,
    ) -> AsyncGenerator[str, None]:
        """
        Convert SSE stream from Antigravity internal API to OpenAI text_completion chunks.
        Yields lines formatted as `data: {...}\n\n` ending with `data: [DONE]\n\n`.
        """
        completion_id = f"cmpl-{uuid.uuid4().hex[:12]}"
        created_time = int(time.time())
        latest_usage_metadata: dict[str, Any] = {}
        fp = cls._generate_fingerprint(requested_model)

        async for event in event_stream:
            resp_obj = (
                event.get("response")
                if isinstance(event.get("response"), dict)
                else event
            )
            if "responseId" in resp_obj and not completion_id.startswith("cmpl-"):
                completion_id = resp_obj["responseId"]

            if "usageMetadata" in resp_obj:
                latest_usage_metadata.update(resp_obj["usageMetadata"])
            elif "usageMetadata" in event:
                latest_usage_metadata.update(event["usageMetadata"])
            if usage_collector is not None and latest_usage_metadata:
                usage_collector.update(cls.format_usage(latest_usage_metadata))

            candidates = resp_obj.get("candidates", [])
            for default_idx, cand in enumerate(candidates):
                cand_idx = cand.get("index", default_idx)
                parts = cand.get("content", {}).get("parts", [])
                raw_finish = cand.get("finishReason")

                for p in parts:
                    if p.get("text") and not p.get("thought"):
                        chunk = {
                            "id": completion_id,
                            "object": "text_completion",
                            "created": created_time,
                            "model": requested_model,
                            "system_fingerprint": fp,
                            "choices": [
                                {
                                    "text": p["text"],
                                    "index": cand_idx,
                                    "logprobs": None,
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"

                if raw_finish:
                    finish_reason = "length" if raw_finish == "MAX_TOKENS" else "stop"
                    final_chunk = {
                        "id": completion_id,
                        "object": "text_completion",
                        "created": created_time,
                        "model": requested_model,
                        "system_fingerprint": fp,
                        "choices": [
                            {
                                "text": "",
                                "index": cand_idx,
                                "logprobs": None,
                                "finish_reason": finish_reason,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(final_chunk)}\n\n"

        if include_usage:
            usage_chunk = {
                "id": completion_id,
                "object": "text_completion",
                "created": created_time,
                "model": requested_model,
                "system_fingerprint": fp,
                "choices": [],
                "usage": cls.format_usage(latest_usage_metadata),
            }
            yield f"data: {json.dumps(usage_chunk)}\n\n"

        yield "data: [DONE]\n\n"


# Alias for backwards compatibility / alternate naming
Translator = OpenAITranslator
