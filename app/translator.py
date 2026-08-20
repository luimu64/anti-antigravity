import time
import uuid
import json
import copy
import logging
from typing import Dict, Any, List, Optional, Tuple, AsyncGenerator, Union
from pydantic import BaseModel, Field, ConfigDict

from app.config import MODEL_ALIASES

logger = logging.getLogger("agy_to_api.translator")

# Cache to store thought signatures across turns for multi-turn function calling
_thought_signature_cache: Dict[str, str] = {}

class ChatMessage(BaseModel):
    role: str
    content: Optional[Any] = None
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    reasoning_content: Optional[str] = None

class StreamOptions(BaseModel):
    include_usage: Optional[bool] = None

class ResponseFormat(BaseModel):
    type: Optional[str] = "text"
    json_schema: Optional[Dict[str, Any]] = None

class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    model: str
    messages: List[Dict[str, Any]]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = Field(default=None, alias="max_completion_tokens")
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    seed: Optional[int] = None
    response_format: Optional[Union[ResponseFormat, Dict[str, Any]]] = None
    stream: Optional[bool] = False
    stream_options: Optional[Union[StreamOptions, Dict[str, Any]]] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    reasoning_effort: Optional[str] = None
    user: Optional[str] = None

class EmbeddingRequest(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    input: Union[str, List[str], List[int], List[List[int]]]
    model: str = "text-embedding-004"
    encoding_format: Optional[str] = "float"
    dimensions: Optional[int] = None
    user: Optional[str] = None

class OpenAITranslator:
    @staticmethod
    def _generate_fingerprint(model: str) -> str:
        return f"fp_agy_{abs(hash(model)) & 0xffffffff:08x}"

    @staticmethod
    def resolve_model(requested_model: str) -> str:
        """Map user-requested model to Antigravity internal model name."""
        clean = requested_model.lower().strip()
        if clean in MODEL_ALIASES:
            return MODEL_ALIASES[clean]
        
        # Check prefix matches
        for alias, internal in MODEL_ALIASES.items():
            if clean.startswith(alias):
                return internal
                
        # Default fallback to gemini-3.7-flash-high if unknown
        return requested_model

    @classmethod
    def dereference_schema(cls, schema: Any, root_schema: Optional[Dict[str, Any]] = None, seen_refs: Optional[set] = None) -> Any:
        """
        Recursively resolve $ref pointers within a JSON schema against root_schema definitions.
        Supports $defs, definitions, components/schemas, and JSON pointer paths.
        Prevents infinite loops on circular references.
        """
        if root_schema is None:
            root_schema = schema if isinstance(schema, dict) else {}
        if seen_refs is None:
            seen_refs = set()

        if not isinstance(schema, dict):
            if isinstance(schema, list):
                return [cls.dereference_schema(item, root_schema, seen_refs) for item in schema]
            return schema

        # Check if this node is a $ref
        if "$ref" in schema and isinstance(schema["$ref"], str):
            ref_path = schema["$ref"]
            if ref_path in seen_refs:
                return {"type": "object", "description": f"Circular reference to {ref_path}"}

            resolved = None
            if ref_path.startswith("#/"):
                parts = ref_path[2:].split("/")
                curr = root_schema
                found = True
                for p in parts:
                    p = p.replace("~1", "/").replace("~0", "~")
                    if isinstance(curr, dict) and p in curr:
                        curr = curr[p]
                    else:
                        found = False
                        break
                if found and isinstance(curr, dict):
                    resolved = curr

            if resolved is not None:
                new_seen = set(seen_refs)
                new_seen.add(ref_path)
                dereferenced_target = cls.dereference_schema(copy.deepcopy(resolved), root_schema, new_seen)
                if isinstance(dereferenced_target, dict):
                    merged = dict(dereferenced_target)
                    for k, v in schema.items():
                        if k != "$ref":
                            merged[k] = cls.dereference_schema(v, root_schema, new_seen)
                    return merged
                return dereferenced_target

        result = {}
        for k, v in schema.items():
            result[k] = cls.dereference_schema(v, root_schema, seen_refs)
        return result

    @classmethod
    def sanitize_schema(cls, schema: Any, is_top_level_params: bool = False) -> Any:
        """
        Recursively converts a JSON Schema (OpenAI / Pydantic / DeepSeek harness)
        into Google Gemini / Cloud Code Schema protobuf compatible representation:
        - Resolves and inlines all $ref references
        - Converts `const: "val"` to `type: "string", enum: ["val"]`
        - Converts `oneOf` to `anyOf`
        - Merges `allOf` into parent schema
        - Normalizes `type: ["type", "null"]` into `type: "type", nullable: true`
        - Converts/strips `exclusiveMinimum` / `exclusiveMaximum` to `minimum` / `maximum`
        - Moves `default` into `description` (preserving context without violating schema)
        - Normalizes enums (ensures list of strings if type is string)
        - Strips unsupported fields ($schema, $defs, definitions, title, additionalProperties, examples, etc.)
        - Ensures top-level tool parameters have valid type="object" and properties dict.
        """
        if not isinstance(schema, dict):
            if is_top_level_params:
                return {"type": "object", "properties": {}}
            return schema

        # First, ensure all $refs are dereferenced
        schema = cls.dereference_schema(schema)
        if not isinstance(schema, dict):
            return schema

        schema = copy.deepcopy(schema)

        def _clean_node(node: Any) -> Any:
            if not isinstance(node, dict):
                if isinstance(node, list):
                    return [_clean_node(item) for item in node]
                return node

            node = dict(node)

            # 1. Convert const -> enum and infer type
            if "const" in node:
                const_val = node.pop("const")
                if "enum" not in node:
                    if isinstance(const_val, (str, int, float, bool)):
                        node["enum"] = [str(const_val)] if not isinstance(const_val, str) else [const_val]
                    else:
                        node["enum"] = [json.dumps(const_val)]
                if "type" not in node:
                    if isinstance(const_val, bool):
                        node["type"] = "boolean"
                    elif isinstance(const_val, int):
                        node["type"] = "integer"
                    elif isinstance(const_val, float):
                        node["type"] = "number"
                    elif isinstance(const_val, str):
                        node["type"] = "string"
                    elif isinstance(const_val, list):
                        node["type"] = "array"
                    elif isinstance(const_val, dict):
                        node["type"] = "object"
                    else:
                        node["type"] = "string"

            # 2. Normalize multi-types / nullability e.g. type: ["string", "null"]
            if "type" in node:
                t_val = node["type"]
                if isinstance(t_val, list):
                    types = [t.lower() for t in t_val if isinstance(t, str)]
                    if "null" in types:
                        node["nullable"] = True
                        non_null = [t for t in types if t != "null"]
                        node["type"] = non_null[0] if non_null else "string"
                    elif types:
                        node["type"] = types[0]
                    else:
                        node["type"] = "string"
                elif isinstance(t_val, str):
                    node["type"] = t_val.lower()
                elif t_val is None:
                    node.pop("type", None)

            # 3. Handle default -> append to description to avoid schema rejection while keeping semantic info
            if "default" in node:
                default_val = node.pop("default")
                if default_val is not None:
                    desc = node.get("description", "")
                    def_str = json.dumps(default_val) if isinstance(default_val, (dict, list)) else str(default_val)
                    if def_str and "default" not in desc.lower():
                        if desc:
                            node["description"] = f"{desc} (default: {def_str})"
                        else:
                            node["description"] = f"Default: {def_str}"

            # 4. Handle exclusiveMinimum / exclusiveMaximum
            if "exclusiveMinimum" in node:
                ex_min = node.pop("exclusiveMinimum")
                if "minimum" not in node and isinstance(ex_min, (int, float)):
                    node["minimum"] = ex_min
            if "exclusiveMaximum" in node:
                ex_max = node.pop("exclusiveMaximum")
                if "maximum" not in node and isinstance(ex_max, (int, float)):
                    node["maximum"] = ex_max

            # 5. Convert oneOf -> anyOf
            if "oneOf" in node:
                one_of_list = node.pop("oneOf")
                if isinstance(one_of_list, list):
                    if "anyOf" not in node:
                        node["anyOf"] = one_of_list
                    elif isinstance(node["anyOf"], list):
                        node["anyOf"].extend(one_of_list)

            # 6. Merge allOf
            if "allOf" in node:
                all_of_list = node.pop("allOf")
                if isinstance(all_of_list, list):
                    for sub in all_of_list:
                        if isinstance(sub, dict):
                            sub_clean = _clean_node(sub)
                            if isinstance(sub_clean, dict):
                                if "properties" in sub_clean and isinstance(sub_clean["properties"], dict):
                                    if "properties" not in node or not isinstance(node["properties"], dict):
                                        node["properties"] = {}
                                    node["properties"].update(sub_clean["properties"])
                                if "required" in sub_clean and isinstance(sub_clean["required"], list):
                                    if "required" not in node or not isinstance(node["required"], list):
                                        node["required"] = []
                                    for r in sub_clean["required"]:
                                        if r not in node["required"]:
                                            node["required"].append(r)
                                if "description" in sub_clean and "description" not in node:
                                    node["description"] = sub_clean["description"]
                                if "type" in sub_clean and "type" not in node:
                                    node["type"] = sub_clean["type"]

            # 7. Clean and recursively sanitize properties
            if "properties" in node and isinstance(node["properties"], dict):
                node["properties"] = {
                    k: _clean_node(v) for k, v in node["properties"].items() if isinstance(v, (dict, list, str, int, float, bool))
                }
                if "type" not in node:
                    node["type"] = "object"

            # 8. Clean and recursively sanitize items
            if "items" in node:
                if isinstance(node["items"], dict):
                    node["items"] = _clean_node(node["items"])
                elif isinstance(node["items"], list) and node["items"]:
                    node["items"] = _clean_node(node["items"][0])
                else:
                    node.pop("items", None)
                if "type" not in node:
                    node["type"] = "array"

            # 9. Clean and recursively sanitize anyOf
            if "anyOf" in node and isinstance(node["anyOf"], list):
                node["anyOf"] = [_clean_node(s) for s in node["anyOf"] if isinstance(s, dict)]
                if not node["anyOf"]:
                    node.pop("anyOf", None)

            # 10. Normalize enum
            if "enum" in node and isinstance(node["enum"], list):
                if node.get("type") in ("string", None):
                    node["enum"] = [str(x) if not isinstance(x, str) else x for x in node["enum"]]
                    node["type"] = "string"

            # 11. Normalize required list
            if "required" in node:
                if isinstance(node["required"], list):
                    node["required"] = [str(r) for r in node["required"] if isinstance(r, (str, int))]
                    if not node["required"]:
                        node.pop("required", None)
                else:
                    node.pop("required", None)

            # 12. Retain ONLY allowed schema keys
            ALLOWED_SCHEMA_KEYS = {
                "type",
                "format",
                "description",
                "nullable",
                "enum",
                "maxItems",
                "minItems",
                "properties",
                "required",
                "items",
                "minProperties",
                "maxProperties",
                "minimum",
                "maximum",
                "pattern",
                "anyOf",
                "propertyOrdering"
            }

            keys_to_remove = [k for k in node if k not in ALLOWED_SCHEMA_KEYS]
            for k in keys_to_remove:
                node.pop(k, None)

            return node

        sanitized = _clean_node(schema)

        if is_top_level_params:
            if not isinstance(sanitized, dict):
                sanitized = {"type": "object", "properties": {}}
            else:
                if "type" not in sanitized:
                    sanitized["type"] = "object"
                if "properties" not in sanitized or not isinstance(sanitized["properties"], dict):
                    sanitized["properties"] = {}

        return sanitized

    @classmethod
    def openai_to_internal_request(cls, req: ChatCompletionRequest) -> Tuple[str, List[Dict[str, Any]], Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[List[Dict[str, Any]]]]:
        """
        Convert OpenAI ChatCompletionRequest into internal format:
        Returns: (internal_model, contents, system_instruction, generation_config, tools)
        """
        internal_model = cls.resolve_model(req.model)
        
        contents: List[Dict[str, Any]] = []
        system_texts: List[str] = []
        
        # Track tool names for tool_call_id -> function_name mapping
        tool_call_names: Dict[str, str] = {}

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
                                url = img_url_obj.get("url", "") if isinstance(img_url_obj, dict) else str(img_url_obj)
                                if url.startswith("data:"):
                                    # data:image/png;base64,.....
                                    header, b64_data = url.split(",", 1)
                                    mime_type = header.split(";")[0].replace("data:", "")
                                    parts.append({
                                        "inlineData": {
                                            "mimeType": mime_type,
                                            "data": b64_data
                                        }
                                    })
                                else:
                                    parts.append({"text": f"[Image URL: {url}]"})
                            elif itype in ("audio", "input_audio"):
                                audio_obj = item.get("audio") or item.get("input_audio") or {}
                                audio_data = audio_obj.get("data", "")
                                audio_format = audio_obj.get("format", "wav")
                                mime_type = f"audio/{audio_format}"
                                parts.append({
                                    "inlineData": {
                                        "mimeType": mime_type,
                                        "data": audio_data
                                    }
                                })
                            elif itype in ("file", "document"):
                                file_obj = item.get("file") or {}
                                if isinstance(file_obj, str):
                                    file_val = file_obj
                                    file_mime = item.get("mime_type") or item.get("mimeType")
                                else:
                                    file_val = (
                                        file_obj.get("file_data")
                                        or file_obj.get("data")
                                        or file_obj.get("url")
                                        or item.get("file_data")
                                        or (item.get("file_url", {}).get("url") if isinstance(item.get("file_url"), dict) else item.get("file_url"))
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

                                if isinstance(file_val, str) and file_val.startswith("data:"):
                                    header, b64_data = file_val.split(",", 1)
                                    extracted_mime = header.split(";")[0].replace("data:", "").strip()
                                    mime_type = extracted_mime or file_mime or "application/pdf"
                                    parts.append({
                                        "inlineData": {
                                            "mimeType": mime_type,
                                            "data": b64_data
                                        }
                                    })
                                elif isinstance(file_val, str) and file_val.startswith(("http://", "https://")):
                                    parts.append({"text": f"[File URL: {file_val}]"})
                                elif isinstance(file_val, str) and file_val:
                                    mime_type = file_mime or "application/pdf"
                                    parts.append({
                                        "inlineData": {
                                            "mimeType": mime_type,
                                            "data": file_val
                                        }
                                    })
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
                    
                    part: Dict[str, Any] = {
                        "functionCall": {
                            "name": fn_name,
                            "args": fn_args,
                            "id": tc_id
                        }
                    }
                    
                    # Attach thought signature if previously cached for this tool call or model
                    if tc_id in _thought_signature_cache:
                        part["thoughtSignature"] = _thought_signature_cache[tc_id]
                    elif "last_signature" in _thought_signature_cache:
                        part["thoughtSignature"] = _thought_signature_cache["last_signature"]
                    
                    parts.append(part)

                if parts:
                    contents.append({"role": "model", "parts": parts})

            # Handle tool response message
            elif role in ("tool", "function"):
                tool_call_id = msg.get("tool_call_id", "")
                fn_name = msg.get("name") or tool_call_names.get(tool_call_id, "tool_response")
                
                response_val: Any = content
                if isinstance(content, str):
                    try:
                        response_val = json.loads(content)
                    except Exception:
                        response_val = {"result": content}
                
                func_resp_part = {
                    "functionResponse": {
                        "name": fn_name,
                        "response": {"result": response_val} if not isinstance(response_val, dict) else response_val
                    }
                }
                if tool_call_id:
                    func_resp_part["functionResponse"]["id"] = tool_call_id

                contents.append({"role": "user", "parts": [func_resp_part]})

        # Build System Instruction
        system_instruction = None
        if system_texts:
            system_instruction = {
                "parts": [{"text": "\n\n".join(system_texts)}]
            }

        # Build Generation Config
        # Determine model limit
        model_limit = 65536
        if "claude" in internal_model:
            model_limit = 64000
        elif "gpt-oss" in internal_model:
            model_limit = 32768
        elif "flash" in internal_model or "pro" in internal_model:
            model_limit = 65536

        max_tokens = min(req.max_tokens, model_limit) if req.max_tokens else min(8192, model_limit)
        generation_config: Dict[str, Any] = {
            "maxOutputTokens": max_tokens
        }
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
                    raw_schema = schema_def.get("schema", schema_def) if isinstance(schema_def, dict) else schema_def
                    if raw_schema:
                        generation_config["responseSchema"] = cls.sanitize_schema(raw_schema)
            elif hasattr(resp_fmt, "type"):
                if resp_fmt.type in ("json_object", "json"):
                    generation_config["responseMimeType"] = "application/json"
                elif resp_fmt.type == "json_schema":
                    generation_config["responseMimeType"] = "application/json"
                    if resp_fmt.json_schema:
                        raw_schema = resp_fmt.json_schema.get("schema", resp_fmt.json_schema) if isinstance(resp_fmt.json_schema, dict) else resp_fmt.json_schema
                        if raw_schema:
                            generation_config["responseSchema"] = cls.sanitize_schema(raw_schema)

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
                "thinkingBudget": budget
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
                    "thinkingBudget": budget
                }
        elif "gpt-oss" in internal_model:
            if req.reasoning_effort:
                generation_config["thinkingConfig"] = {
                    "includeThoughts": True,
                    "thinkingBudget": 8192
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
                        "parameters": cls.sanitize_schema(fn.get("parameters", {}), is_top_level_params=True)
                    }
                    function_declarations.append(fn_decl)
            if function_declarations:
                tools = [{"functionDeclarations": function_declarations}]

        # Tool choice handling
        if req.tool_choice and tools:
            # Map tool_choice to toolConfig / functionCallingConfig
            tool_config: Dict[str, Any] = {}
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
                                "allowedFunctionNames": [fn_name]
                            }
                        }
            if tool_config:
                tools.append(tool_config)

        return internal_model, contents, system_instruction, generation_config, tools

    @staticmethod
    def format_usage(usage_meta: Dict[str, Any]) -> Dict[str, Any]:
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
        total_tokens = usage_meta.get("totalTokenCount", prompt_tokens + completion_tokens)

        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "prompt_tokens_details": {
                "cached_tokens": cached_tokens
            },
            "completion_tokens_details": {
                "reasoning_tokens": thoughts_tokens
            }
        }

    @classmethod
    def internal_to_openai_response(
        cls,
        result: Dict[str, Any],
        requested_model: str
    ) -> Dict[str, Any]:
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
                cand_thought_sig = cand.get("thoughtSignature") or result.get("thoughtSignature")
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
                    tool_calls_openai.append({
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": tc.get("name", ""),
                            "arguments": args_str
                        }
                    })

                msg: Dict[str, Any] = {
                    "role": "assistant",
                    "content": cand.get("text", "")
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

                choices.append({
                    "index": cand_idx,
                    "message": msg,
                    "finish_reason": finish_reason
                })
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
                tool_calls_openai.append({
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": tc.get("name", ""),
                        "arguments": args_str
                    }
                })

            msg = {
                "role": "assistant",
                "content": result.get("text", "")
            }
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

            choices.append({
                "index": 0,
                "message": msg,
                "finish_reason": finish_reason
            })

        usage_meta = result.get("usageMetadata", {})

        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created_time,
            "model": requested_model,
            "system_fingerprint": cls._generate_fingerprint(requested_model),
            "service_tier": "default",
            "choices": choices,
            "usage": cls.format_usage(usage_meta)
        }

    @classmethod
    def internal_to_openai_text_completion(
        cls,
        result: Dict[str, Any],
        requested_model: str
    ) -> Dict[str, Any]:
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

                choices.append({
                    "text": cand.get("text", ""),
                    "index": cand_idx,
                    "logprobs": None,
                    "finish_reason": finish_reason
                })
        else:
            finish_reason = "stop"
            raw_finish = result.get("finishReason", "STOP")
            if raw_finish == "MAX_TOKENS":
                finish_reason = "length"

            choices.append({
                "text": result.get("text", ""),
                "index": 0,
                "logprobs": None,
                "finish_reason": finish_reason
            })

        usage_meta = result.get("usageMetadata", {})
        return {
            "id": completion_id,
            "object": "text_completion",
            "created": created_time,
            "model": requested_model,
            "system_fingerprint": cls._generate_fingerprint(requested_model),
            "service_tier": "default",
            "choices": choices,
            "usage": cls.format_usage(usage_meta)
        }

    @classmethod
    async def internal_stream_to_openai_chunks(
        cls,
        event_stream: AsyncGenerator[Dict[str, Any], None],
        requested_model: str,
        include_usage: bool = False
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
        latest_usage_metadata: Dict[str, Any] = {}
        fp = cls._generate_fingerprint(requested_model)

        async for event in event_stream:
            resp_obj = event.get("response") if isinstance(event.get("response"), dict) else event
            if "responseId" in resp_obj and not completion_id.startswith("chatcmpl"):
                completion_id = resp_obj["responseId"]

            if "usageMetadata" in resp_obj:
                latest_usage_metadata.update(resp_obj["usageMetadata"])
            elif "usageMetadata" in event:
                latest_usage_metadata.update(event["usageMetadata"])

            candidates = resp_obj.get("candidates", [])
            for default_idx, cand in enumerate(candidates):
                cand_idx = cand.get("index", default_idx)
                parts = cand.get("content", {}).get("parts", [])
                raw_finish = cand.get("finishReason")

                for p in parts:
                    if p.get("thoughtSignature"):
                        last_thought_signature = p["thoughtSignature"]
                        _thought_signature_cache["last_signature"] = last_thought_signature

                    # 1. Thought / Reasoning chunk
                    if p.get("thought"):
                        thought_text = p.get("text", "")
                        if thought_text:
                            delta: Dict[str, Any] = {"reasoning_content": thought_text}
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
                                        "finish_reason": None
                                    }
                                ]
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
                                    "finish_reason": None
                                }
                            ]
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"

                    # 3. Tool Call chunk
                    elif p.get("functionCall"):
                        fc = p["functionCall"]
                        call_id = fc.get("id") or f"call_{uuid.uuid4().hex[:8]}"
                        if last_thought_signature:
                            _thought_signature_cache[call_id] = last_thought_signature

                        args = fc.get("args", {})
                        args_str = json.dumps(args) if isinstance(args, dict) else str(args)

                        delta = {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": call_id,
                                    "type": "function",
                                    "function": {
                                        "name": fc.get("name", ""),
                                        "arguments": args_str
                                    }
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
                                    "finish_reason": None
                                }
                            ]
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"

                # If candidate has finish reason, emit final finish chunk
                if raw_finish:
                    finish_reason = "stop"
                    if raw_finish == "TOOL_CALL" or any("functionCall" in p for p in parts):
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
                                "finish_reason": finish_reason
                            }
                        ]
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
                "usage": cls.format_usage(latest_usage_metadata)
            }
            yield f"data: {json.dumps(usage_chunk)}\n\n"

        # End of stream
        yield "data: [DONE]\n\n"

    @classmethod
    async def internal_stream_to_openai_text_chunks(
        cls,
        event_stream: AsyncGenerator[Dict[str, Any], None],
        requested_model: str,
        include_usage: bool = False
    ) -> AsyncGenerator[str, None]:
        """
        Convert SSE stream from Antigravity internal API to OpenAI text_completion chunks.
        Yields lines formatted as `data: {...}\n\n` ending with `data: [DONE]\n\n`.
        """
        completion_id = f"cmpl-{uuid.uuid4().hex[:12]}"
        created_time = int(time.time())
        latest_usage_metadata: Dict[str, Any] = {}
        fp = cls._generate_fingerprint(requested_model)

        async for event in event_stream:
            resp_obj = event.get("response") if isinstance(event.get("response"), dict) else event
            if "responseId" in resp_obj and not completion_id.startswith("cmpl-"):
                completion_id = resp_obj["responseId"]

            if "usageMetadata" in resp_obj:
                latest_usage_metadata.update(resp_obj["usageMetadata"])
            elif "usageMetadata" in event:
                latest_usage_metadata.update(event["usageMetadata"])

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
                                    "finish_reason": None
                                }
                            ]
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
                                "finish_reason": finish_reason
                            }
                        ]
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
                "usage": cls.format_usage(latest_usage_metadata)
            }
            yield f"data: {json.dumps(usage_chunk)}\n\n"

        yield "data: [DONE]\n\n"
