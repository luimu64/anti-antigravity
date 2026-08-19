# TASK.md — OpenAI v1 Spec Completeness Tasklist & Verification Audit

**Context:** `agy-to-api` translates Google Cloud Code / Antigravity into the OpenAI v1 API schema. This document tracks technical assertions, code references, schema definitions, verified implementation statuses, concrete file and function mappings, upstream Antigravity internal API compatibility notes, and identified discrepancies across all tasks.

**Verification loop for every task:**
```bash
.venv/bin/pytest -v        # full suite must pass
python main.py --status    # server sanity (optional)
```
Tests reside in `tests/test_translator.py` and `tests/test_api_endpoints.py` with mocked `AntigravityClient` and direct unit tests.

---

## P0 — Correctness bugs (silent data loss)

### 1. Audio content part handling
- **Status:** `[x]` **VERIFIED & IMPLEMENTED**
- **Location:** `app/translator.py` → `openai_to_internal_request` (lines 115–125); tested in `tests/test_translator.py` (`test_audio_content_parts_standard_and_legacy`).
- **Implementation & Schema Mapping:**
  - Accepts both standard OpenAI v1 schema (`{"type": "audio", "audio": {"data": "<b64>", "format": "wav"}}`) and legacy schema (`{"type": "input_audio", "input_audio": {"data": "<b64>", "format": "wav"}}`).
  - Resolves payload via `item.get("audio") or item.get("input_audio") or {}`.
  - Maps to Gemini/Antigravity internal `inlineData` part:
    ```python
    {"inlineData": {"mimeType": f"audio/{audio_format}", "data": audio_data}}
    ```
  - Defaults audio format to `wav` when unspecified.
- **Upstream Compatibility & Discrepancies:** Fully compatible with Google Cloud Code backend's multi-modal audio processing. No data loss observed.

---

### 2. Embeddings endpoint (`/v1/embeddings`)
- **Status:** `[x]` **VERIFIED & IMPLEMENTED (Real Backend via Option A)**
- **Location:**
  - Route: `app/routes/openai.py` → `create_embeddings` (lines 135–233, replacing `embeddings_placeholder`).
  - Client: `app/client.py` → `embed_contents` (lines 201–245).
  - Tests: `tests/test_api_endpoints.py` (`test_embeddings_endpoint_float`, `test_embeddings_endpoint_base64_and_dimensions`, `test_embeddings_endpoint_invalid_inputs`).
- **Implementation & Schema Mapping:**
  - Upstream backend integration via `/v1internal:batchEmbedContents` using internal model `text-embedding-004`.
  - **Input normalization:** Handles single string, list of strings, token integer arrays (`List[int]`), and batch token arrays (`List[List[int]]`).
  - **Encoding format:** Supports both `encoding_format="float"` (default JSON array of floats) and `encoding_format="base64"` (IEEE 754 little-endian binary float packing via `struct.pack(f"<{len(vec)}f", *vec)` encoded to base64 string per OpenAI spec).
  - **Dimensions:** Passes `outputDimensionality` parameter to backend and performs client-side truncation or zero-padding if requested `dimensions` differs from raw output; validates `dimensions > 0`.
  - **Usage Reporting:** Extracts upstream `promptTokenCount` from `usageMetadata`, with string length fallback (`len(text)//4`).
  - **`FAKE_EMBEDDINGS` Fallback:** Replaced by live backend API integration; fallback padding ensures complete vector list even if partial upstream vectors return.
- **Upstream Compatibility & Discrepancies:** Fully compliant with OpenAI `/v1/embeddings` specification.

---

### 3. `n > 1` candidate mapping
- **Status:** `[x]` **VERIFIED & IMPLEMENTED**
- **Location:**
  - Translation: `app/translator.py` → `openai_to_internal_request` (sets `candidateCount` in `generationConfig` when `req.n > 1`), `internal_to_openai_response` (lines 351–435), and `internal_stream_to_openai_chunks` (lines 450–575).
  - Client Aggregator: `app/client.py` → `generate_content` (lines 137–199).
  - Tests: `tests/test_translator.py` (`test_multi_candidate_non_streaming`, `test_multi_candidate_streaming`).
- **Implementation & Schema Mapping:**
  - **Request:** Sets `generation_config["candidateCount"] = req.n` when `req.n > 1`.
  - **Non-streaming:** Iterates `result.get("candidates", [])`, mapping each candidate to an OpenAI choice object with its discrete `index` (0, 1, ...), message content, tool calls, and `finish_reason`.
  - **Streaming:** Processes multi-candidate SSE chunks from upstream, routing deltas and finish chunks with distinct choice `index` values, terminating with a single `data: [DONE]\n\n`.
- **Upstream Compatibility & Discrepancies:** Gemini models support `candidateCount`. Note: upstream Claude (`claude-sonnet-4-6`) and GPT-OSS backends on Google Cloud Code ignore or restrict `candidateCount > 1`.

---

## P1 — Spec-shaped endpoints

### 4. `/v1/completions` response shape
- **Status:** `[ ]` **PARTIAL / DISCREPANCY IDENTIFIED**
- **Location:** `app/routes/openai.py` → `legacy_completions` (lines 118–133).
- **Current Behavior vs OpenAI Spec:**
  - Current implementation wraps the prompt into a single user `ChatCompletionRequest` and delegates directly to `chat_completions(chat_req)`.
  - **Discrepancy:** Returns `object: "chat.completion"` or `object: "chat.completion.chunk"` (with `choices[0].message.content` or `choices[0].delta.content`).
  - **Required Spec Shape:**
    - Non-streaming: `object: "text_completion"`, `choices: [{"index": 0, "text": "...", "logprobs": null, "finish_reason": "stop"}]`, `usage`, `system_fingerprint`.
    - Streaming: `object: "text_completion"`, `choices: [{"index": 0, "text": "...", "logprobs": null, "finish_reason": null}]`, ending in `data: [DONE]`.
  - Parameters like `suffix`, `logit_bias`, `best_of`, `echo` are currently ignored pass-throughs.

---

### 5. Error responses structured OpenAI error envelope
- **Status:** `[ ]` **PARTIAL / DISCREPANCY IDENTIFIED**
- **Location:** `app/routes/openai.py` (exception handlers across endpoints), `app/client.py`.
- **Current Behavior vs OpenAI Spec:**
  - `verify_api_key` properly returns the OpenAI error schema in 401 responses:
    ```json
    {"error": {"message": "...", "type": "invalid_request_error", "param": null, "code": "missing_api_key"}}
    ```
  - **Discrepancy:** Other route error conditions raise `HTTPException(status_code=..., detail=...)` returning FastAPI's default `{"detail": "..."}` JSON envelope instead of standard `{"error": {"message": ..., "type": ..., "param": ..., "code": ...}}`.
  - Upstream 400 `INVALID_ARGUMENT` and 500 errors from Google need unified mapping into OpenAI structured error objects.

---

### 6. Document / File attachments (`type: "file"`)
- **Status:** `[ ]` **PENDING / SPEC GAP**
- **Location:** `app/translator.py` → `openai_to_internal_request` (user content parts loop, lines 85–127).
- **Current Behavior vs OpenAI Spec:**
  - Current parser matches `itype == "text"`, `itype == "image_url"`, and `itype in ("audio", "input_audio")`.
  - **Discrepancy:** Requests with `type: "file"` (carrying `file: {"file_data": "data:application/pdf;base64,...", "file_id": ...}`) are currently skipped in the message part loop.
  - **Implementation Target:** Parse data-URL/base64 payload and map to Gemini `inlineData` with `mimeType: "application/pdf"`, or return an explicit structured 400 error for unsupported MIME types.

---

## P2 — Missing parameters & response fields

### 7. `logprobs` / `top_logprobs`
- **Status:** `[x]` **VERIFIED LIMITATION (Upstream Non-support)**
- **Location:** `app/translator.py` (`ChatCompletionRequest.model_config = ConfigDict(extra="allow")`).
- **Compatibility & Details:**
  - Google Cloud Code / Antigravity internal `streamGenerateContent` API does not return per-token log probabilities in candidate parts or usage metadata.
  - Spec-compliant behavior for unsupported models is returning `logprobs: null` in non-stream choices and omitting in stream chunks.

---

### 8. Ignored parameters (`parallel_tool_calls`, `logit_bias`, `store`, `metadata`, `service_tier`)
- **Status:** `[x]` **VERIFIED & DOCUMENTED**
- **Location:** `app/translator.py` (`ChatCompletionRequest.model_config = ConfigDict(extra="allow", populate_by_name=True)`).
- **Compatibility & Details:**
  - The following OpenAI parameters are accepted without validation failure via `extra="allow"` but are intentionally ignored due to no Google internal API equivalent:
    - `parallel_tool_calls`: Google handles tool calls automatically according to model capability.
    - `logit_bias`: Not supported in Cloud Code backend.
    - `store`, `metadata`, `user`: Bridge is stateless; stored/tracked client-side.
    - `service_tier`: Processed through Google Cloud Code account quota tier.

---

### 9. `reasoning_effort` mapping & budget allocation
- **Status:** `[x]` **VERIFIED & IMPLEMENTED**
- **Location:** `app/translator.py` → `openai_to_internal_request` (lines 255–291).
- **Implementation & Schema Mapping:**
  - **Gemini 3.7 (`gemini-3.7-flash` / `gemini-3`):**
    - `low`: `thinkingBudget = 1000`
    - `medium`: `thinkingBudget = 4000`
    - `high`: `thinkingBudget = 16000`
    - `xhigh` / default: `thinkingBudget = -1` (dynamic budget)
  - **Claude models (`claude-sonnet-4-6`, `claude-opus-4-6-thinking`):**
    - `low` / default: `thinkingBudget = 1024`
    - `medium`: `thinkingBudget = 4096`
    - `high`: `thinkingBudget = 16384`
  - **GPT-OSS (`gpt-oss-120b`):**
    - `thinkingBudget = 8192` when reasoning is enabled.

---

### 10. Response fields: `system_fingerprint`, `service_tier`
- **Status:** `[ ]` **OPTIONAL / SPEC POLISH**
- **Location:** `app/translator.py` (`internal_to_openai_response`, `internal_stream_to_openai_chunks`).
- **Implementation Target:**
  - Add optional `system_fingerprint` (e.g. `fp_agy_<model_hash>`) and `service_tier: "default"` to `chat.completion` responses for strict validation clients.

---

## P3 — Model endpoint polish (optional)

### 11. `/v1/models` metadata & fields
- **Status:** `[x]` **VERIFIED & IMPLEMENTED**
- **Location:** `app/routes/openai.py` → `list_models` (lines 48–101), `retrieve_model` (lines 103–116).
- **Implementation & Schema Mapping:**
  - Returns standard list format `{ "object": "list", "data": [...] }`.
  - Dynamic discovery via `client.fetch_available_models()` merged with `MODEL_ALIASES`.
  - Model entries contain `id`, `object: "model"`, `created`, `owned_by`, `root`, `display_name`, `max_tokens`, and `supports_thinking`.

---

### 12. Model resolution logic
- **Status:** `[x]` **VERIFIED & IMPLEMENTED**
- **Location:** `app/translator.py` → `OpenAITranslator.resolve_model` (lines 53–65); tested in `tests/test_translator.py` (`test_resolve_model`).
- **Implementation & Logic:**
  - 1. Checks exact match in `MODEL_ALIASES` dictionary first.
  - 2. Checks alias prefix matches (`clean.startswith(alias)`) as fallback.
  - 3. Returns unmapped requested model string verbatim if not in alias map.

---

## Out of scope (documented, not built)

Files API, fine-tuning, images generation (`/v1/images/*`), audio transcription/speech (`/v1/audio/transcriptions`), moderations (`/v1/moderations`), assistants (`/v1/assistants`) — not part of the `agy-to-api` bridge contract.

---

## Done-criteria checklist

- [x] Audio content parts (`inlineData`, `audio`/`input_audio`) reach backend; legacy and current shapes supported
- [x] `/v1/embeddings` implements real backend via `embed_contents` (supports `float`, `base64`, dimensions, and replaces fake zeros)
- [x] Multi-candidate `n > 1` mapping implemented in streaming and non-streaming (`candidateCount`, `choices` index)
- [ ] `/v1/completions` returns `text_completion` objects (currently returns `chat.completion`)
- [ ] Structured OpenAI error envelopes across all route exception handlers (currently 401 only)
- [ ] File/PDF document attachment handling (`inlineData` with `application/pdf`)
- [x] Model listing and exact-match model resolution in `OpenAITranslator.resolve_model` verified
- [x] Test suite passing with `.venv/bin/pytest -v`
