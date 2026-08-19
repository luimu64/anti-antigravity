# TASK.md — OpenAI v1 Spec Completeness Tasklist

**Context:** `agy-to-api` translates Google Cloud Code / Antigravity into the OpenAI v1 API schema. A gap audit (vs. the current OpenAI Chat Completions API spec) found the core `chat/completions` path solid, but several spec-compliance gaps remain. Work top-to-bottom by priority.

**Verification loop for every task:**
```bash
.venv/bin/pytest -v        # full suite must pass
python main.py --status    # server sanity (optional)
```
Add/extend tests in `tests/test_translator.py`, `tests/test_api_endpoints.py` as you go. Keep the existing pattern: translator unit tests + endpoint tests with a mocked `AntigravityClient`.

---

## P0 — Correctness bugs (silent data loss)

### 1. Audio content part uses the deprecated type
**Where:** `app/translator.py` → `openai_to_internal_request`, user-message content loop.
- Current code only matches `itype == "input_audio"` (older API shape).
- The current OpenAI spec uses `{"type": "audio", "audio": {"data": "<b64>", "format": "wav"}}`.
- Requests with the current shape have the audio part **silently dropped**.
- **Fix:** accept both `"audio"` (current: payload under `item["audio"]`) and `"input_audio"` (legacy: payload under `item["input_audio"]`) as aliases feeding the same `inlineData` part. Keep MIME mapping (`audio/{format}`), defaulting format to `wav`/`mp3` as appropriate.
- **Test:** send a message with `type: "audio"` part, assert the internal content contains an `inlineData` part with correct `mimeType`.

### 2. `/v1/embeddings` is a fake placeholder
**Where:** `app/routes/openai.py` → `embeddings_placeholder`.
- Returns `[0.0] * 768` per input; never calls the backend. `dimensions` and `encoding_format` are ignored. Zero-vector "embeddings" silently break any RAG pipeline built on this bridge.
- **Options (pick one, note the choice in the PR/commit):**
  - **A (preferred if backend supports it):** Find an embedding model in `fetchAvailableModels` / `fetchModels` output or via `streamGenerateContent` and implement a real call in `app/client.py`. Return true vectors + real `usage`.
  - **B (honest fallback):** Return an OpenAI-shaped `501 Not Implemented` / `501` error object (not a 200 with fake data) unless a `FAKE_EMBEDDINGS=1` env var is set, so callers fail loudly.
- Either way: honor `dimensions` (truncate/raise) and `encoding_format` (`float` vs `base64`).
- **Test:** endpoint returns the chosen behavior; assert `model`, `usage`, and `dimensions` handling.

### 3. `n > 1` only half-implemented
**Where:** `app/translator.py` → `openai_to_internal_request` sets `candidateCount`, but `internal_to_openai_response` and `internal_stream_to_openai_chunks` always take candidate `[0]` → `choices` is always length 1.
- **Fix:** iterate `result["candidates"]` (non-stream) and each SSE event's `candidates` list (stream), emitting one choice per candidate with the correct `index`. If the upstream doesn't actually support `candidateCount`, document that and either (a) only accept `n=1` with a clear 400 error, or (b) keep sending it and note the limitation.
- **Test:** mock a 2-candidate response; assert 2 choices with `index` 0/1 in both stream and non-stream.

---

## P1 — Spec-shaped endpoints

### 4. `/v1/completions` response shape is wrong
**Where:** `app/routes/openai.py` → `legacy_completions`.
- Currently returns `chat.completion` / `chat.completion.chunk` objects.
- **Fix:** spec says `object: "text_completion"`, choices of `{"index": 0, "text": ..., "logprobs": null, "finish_reason": ...}`, usage object, `system_fingerprint`. Streaming chunks must be `object: "text_completion"` with `choices[0].text` (not `delta.content`), ending in `data: [DONE]`.
- Simplest path: build a dedicated thin response wrapper (don't force it through the chat chunk formatter) or add a `completion_mode` flag to the chunk formatter.
- Also handle: `logprobs` (can be `null`), `suffix` (ignore or 400), `logit_bias` (ignore), `best_of` (ignore), `echo` (ignore), `user`.
- **Test:** assert `object == "text_completion"` in both stream and non-stream.

### 5. Error responses don't use the OpenAI error object
**Where:** `app/routes/openai.py` (all `HTTPException` sites), `app/client.py` (upstream errors).
- Only 401s use `{"error": {"message", "type", "param", "code"}}`. Translation errors return `detail=f"Invalid request parameters: {e}"` (plain string); generation/upstream failures return generic 500 strings.
- **Fix:** add a helper `openai_error(status, message, type_, code, param=None)` returning `JSONResponse` with the OpenAI error schema; use it everywhere in `openai.py` and map upstream HTTP failures (e.g. 400 `INVALID_ARGUMENT` from Google) to OpenAI `invalid_request_error`/`api_error` with the upstream message included.
- **Test:** assert error body shape (not just status code) for a bad request and a mocked upstream failure.

### 6. File attachments (`type: "file"`) unsupported
**Where:** `app/translator.py` → user content loop.
- Current spec supports `{"type": "file", "file": {"file_data": "data:application/pdf;base64,...", "file_id": null, "type": ...}}` (PDF).
- **Fix:** accept the current `file` shape (and legacy `pdf`/`file_id` variants if trivial). If the internal API can't ingest PDFs, emit a clean OpenAI-style 400 naming the unsupported part, rather than silently dropping it.
- **Test:** PDF data-URL file part → either internal `inlineData` with `application/pdf` or a structured 400.

---

## P2 — Missing parameters & response fields

### 7. `logprobs` / `top_logprobs`
Accepted (via `extra="allow"`) but ignored; no `logprobs` in responses.
- Minimum viable: map to `logprobs: null` in non-stream choices and omit in stream chunks (spec-compliant for models without logprob support).
- Full: if upstream exposes logprobs in `usageMetadata`/candidate parts, surface them.

### 8. `parallel_tool_calls`, `logit_bias`, `store`, `metadata`, `service_tier`
- Documented as intentionally ignored — add them to an explicit "known-ignored" list (comment or README) so future maintainers don't treat silence as a bug. `store`/`metadata` can be no-ops.

### 9. `reasoning_effort` value coverage
**Where:** `app/translator.py`, thinking-config blocks.
- Only `low/medium/high` mapped; spec-valid values like `xhigh` fall to default budget.
- **Fix:** map `xhigh` (or reject unknown values with a clear 400). Verify against what the upstream actually accepts before shipping `xhigh`.

### 10. Response fields: `system_fingerprint`, `service_tier`
- Optional per spec. Add a static `system_fingerprint` (e.g. `agy-<hash of internal model>`) and `service_tier` in chat/completions responses for clients that validate strictly. Low value; do last.

---

## P3 — Model endpoint polish (optional)

### 11. `/v1/models` gaps
- `created` is a hardcoded timestamp — fine, but store per-model if easily available.
- Newer optional fields (`cost`, `reasoning_efforts`, `web_search_options`) absent. Add at least `reasoning_efforts` derived from the `supports_thinking` flag so SDKs can render effort pickers.

### 12. Model resolution edge case
**Where:** `OpenAITranslator.resolve_model` — `startswith` prefix matching can misroute future models (e.g. any model starting with an existing alias). Prefer exact match first, prefix as fallback with a warning log.

---

## Out of scope (document, don't build)

Files, fine-tuning, images, audio transcription/speech, moderations, assistants — not part of the bridge contract. If adding, each deserves its own task.

## Done-criteria checklist

- [ ] Current-spec audio (`type: "audio"`) reaches the backend; legacy still works
- [ ] Embeddings are real or loudly 501 — never silent zeros
- [ ] `n=2` returns 2 choices (stream + non-stream) or a documented 400
- [ ] `/v1/completions` returns `text_completion` objects
- [ ] All `/v1/*` errors are OpenAI error objects (asserted in tests)
- [ ] File/PDF parts handled or rejected with a clean error
- [ ] `pytest` green; README/AGENTS.md updated with any new limitations
