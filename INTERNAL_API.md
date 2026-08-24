# Antigravity CLI Internal API Specification & Protocol Reference

This document provides a comprehensive, reverse-engineered reference for the internal Google Cloud Code / Jetski API utilized by the **Antigravity CLI (`agy`)**.

---

## 1. Overview & Architecture

Antigravity CLI communicates with an internal Google Cloud Code proxy service that routes requests to Google Gemini, Anthropic Claude, and OpenAI OSS models managed under Google AI Companion projects.

- **Production Base URL**: `https://daily-cloudcode-pa.googleapis.com`
- **Override Environment Variable**: `CLOUD_CODE_URL`
- **Protocol**: HTTPS (HTTP/1.1 & HTTP/2), JSON payloads, Server-Sent Events (SSE) for streaming.

---

## 2. OAuth 2.0 Authentication & Credentials

Antigravity uses standard Google OAuth 2.0 PKCE authorization with the official Google Cloud Code client identity.

### 2.1 OAuth Client Credentials
- **Client ID**: `1071006060591-tmhssin2h21lcre235vtolojh4g403ep.apps.googleusercontent.com`
- **Client Secret**: `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`
- **Auth Endpoint**: `https://accounts.google.com/o/oauth2/v2/auth`
- **Token Endpoint**: `https://oauth2.googleapis.com/token`

### 2.2 Scopes
Requests must request the following OAuth scopes:
- `https://www.googleapis.com/auth/cloud-platform`
- `https://www.googleapis.com/auth/userinfo.email`
- `https://www.googleapis.com/auth/userinfo.profile`
- `https://www.googleapis.com/auth/cclog`
- `https://www.googleapis.com/auth/experimentsandconfigs`
- `openid`

### 2.3 Required Request Headers
All requests to `daily-cloudcode-pa.googleapis.com` must supply:
```http
Authorization: Bearer <google_access_token>
Content-Type: application/json
User-Agent: antigravity/cli/0.1.0 linux/amd64
Accept-Encoding: gzip, deflate
```

---

## 3. Endpoints Reference

### 3.1 Metadata & Project Discovery: `loadCodeAssist`

Discovers the active Google Cloud companion project ID and user subscription tier.

- **Method**: `POST`
- **Path**: `/v1internal:loadCodeAssist`
- **Request Body**:
  ```json
  {
    "metadata": {
      "ideType": "ANTIGRAVITY"
    }
  }
  ```
- **Response Schema (`200 OK`)**:
  ```json
  {
    "cloudaicompanionProject": "companion-project-12345",
    "currentTier": {
      "id": "antigravity",
      "name": "Antigravity"
    },
    "userLimits": {
      "rateLimit": 100
    }
  }
  ```
- **Key Fields**:
  - `cloudaicompanionProject`: **Mandatory**. The dynamically assigned project ID required for all subsequent `streamGenerateContent` calls.

---

### 3.2 Model Discovery: `fetchAvailableModels`

Lists all model checkpoints currently enabled on the user's tier.

- **Method**: `POST`
- **Path**: `/v1internal:fetchAvailableModels`
- **Request Body**: `{}`
- **Response Schema (`200 OK`)**:
  ```json
  {
    "models": {
      "gemini-3.7-flash-high": {
        "displayName": "Gemini 3.7 Flash (High)",
        "maxTokens": 1048576,
        "maxOutputTokens": 65536,
        "supportsThinking": true,
        "thinkingBudget": -1
      },
      "claude-sonnet-4-6": {
        "displayName": "Claude Sonnet 4.6",
        "maxTokens": 250000,
        "maxOutputTokens": 64000,
        "supportsThinking": true
      },
      "claude-opus-4-6-thinking": {
        "displayName": "Claude Opus 4.6 (Thinking)",
        "maxTokens": 250000,
        "maxOutputTokens": 64000,
        "supportsThinking": true
      },
      "gpt-oss-120b-medium": {
        "displayName": "GPT-OSS 120B",
        "maxTokens": 131072,
        "maxOutputTokens": 32768
      }
    }
  }
  ```

---

### 3.3 Quota Summary: `retrieveUserQuotaSummary`

Fetches user quota usage percentages and reset intervals.

- **Method**: `POST`
- **Path**: `/v1internal:retrieveUserQuotaSummary`
- **Request Body**: `{}`
- **Response Schema (`200 OK`)**:
  ```json
  {
    "groups": [
      {
        "groupId": "antigravity_general",
        "buckets": [
          {
            "bucketId": "weekly",
            "displayName": "Weekly Limit",
            "remainingFraction": 0.85,
            "resetTime": "2026-08-24T00:00:00Z"
          },
          {
            "bucketId": "5hr",
            "displayName": "5-Hour Burst Limit",
            "remainingFraction": 0.98,
            "resetTime": "2026-08-17T21:00:00Z"
          }
        ]
      }
    ]
  }
  ```

---

### 3.4 Generation & Streaming: `streamGenerateContent`

Executes multi-turn conversation generation, tool calling, structured outputs, audio inputs, and streaming with thinking/reasoning tokens.

- **Method**: `POST`
- **Path**: `/v1internal:streamGenerateContent?alt=sse`
- **Request Body Schema**:
  ```json
  {
    "project": "companion-project-12345",
    "model": "gemini-3.7-flash-high",
    "request": {
      "contents": [
        {
          "role": "user",
          "parts": [
            {
              "text": "What is the capital of France?"
            }
          ]
        }
      ],
      "systemInstruction": {
        "parts": [
          {
            "text": "You are a helpful and precise assistant."
          }
        ]
      },
      "generationConfig": {
        "maxOutputTokens": 64000,
        "temperature": 0.7,
        "topP": 0.95,
        "presencePenalty": 0.0,
        "frequencyPenalty": 0.0,
        "stopSequences": ["User:"],
        "responseMimeType": "application/json",
        "thinkingConfig": {
          "includeThoughts": true,
          "thinkingBudget": -1
        }
      },
      "tools": [
        {
          "functionDeclarations": [
            {
              "name": "get_weather",
              "description": "Get current weather",
              "parameters": {
                "type": "object",
                "properties": {
                  "location": { "type": "string" }
                },
                "required": ["location"]
              }
            }
          ]
        },
        {
          "functionCallingConfig": {
            "mode": "AUTO"
          }
        }
      ]
    }
  }
  ```

---

## 4. SSE Stream Event Formats & Candidate Parts

The response is streamed as `text/event-stream` chunks. Each chunk contains a JSON object formatted as follows:

```http
data: {"candidates": [{"content": {"parts": [{"text": "Paris", "thought": false}], "role": "model"}, "finishReason": "STOP", "index": 0}], "usageMetadata": {"promptTokenCount": 12, "candidatesTokenCount": 5, "totalTokenCount": 128}}
```

### Part Types in Candidates:

1. **Standard Text**:
   ```json
   {"text": "Hello world", "thought": false}
   ```

2. **Reasoning / Thinking Tokens**:
   ```json
   {"text": "Analyzing the request parameters...", "thought": true}
   ```

3. **Tool / Function Calling**:
   ```json
   {
     "functionCall": {
       "name": "get_weather",
       "args": { "location": "Paris, France" }
     },
     "thoughtSignature": "CiUIARJbCldyZWFzb25pbmdfY29udGVud..."
   }
   ```

4. **Multi-Turn Function Response (User Input)**:
   ```json
   {
     "role": "user",
     "parts": [
       {
         "functionResponse": {
           "name": "get_weather",
           "response": { "result": "Sunny, 22°C" }
         }
       }
     ]
   }
   ```

---

## 5. Model Family Quirks & Provider Constraints

When routing requests to different models through this unified API, the backend enforces specific provider constraints:

| Constraint | Gemini Models (`gemini-3.7-*`) | Anthropic Claude (`claude-*`) | OpenAI OSS (`gpt-oss-*`) |
|---|---|---|---|
| **Max Context (`maxTokens`)** | 1,048,576 (1M) | 250,000 | 131,072 |
| **Max Output Tokens** | 65,536 | **64,000** (values >64k return `400 INVALID_ARGUMENT`) | **32,768** |
| **`thinkingBudget`** | `-1` (dynamic) or integer | Must be `>= 1024` or omit `thinkingConfig` | Default `8192` |
| **Thought Signature** | **Mandatory** across turns for function calls | Not required | Not required |
| **Multimodal Support** | Inline image bytes (`image/jpeg`, `image/png`, `image/webp`) | Inline image bytes | Text only |

---

## 6. Error Codes

| Status Code | Code String | Description / Cause |
|---|---|---|
| `400` | `INVALID_ARGUMENT` | `maxOutputTokens` exceeds model limit, missing `thought_signature`, or invalid `thinkingBudget`. |
| `401` | `UNAUTHENTICATED` | Expired OAuth access token. Refresh token required. |
| `403` | `PERMISSION_DENIED` | Account does not have Antigravity access or companion project not initialized. |
| `429` | `RESOURCE_EXHAUSTED` | Rate limit or quota exhausted. |
| `500` | `INTERNAL` | Google upstream backend service error. |

---

## 7. Gemini Web Interface Mapping (Reverse-Engineered Reference)

**Authoritative Reference Implementations:**
- [HanaokaYuzu/Gemini-API](https://github.com/HanaokaYuzu/Gemini-API) (Upstream standard for modern Gemini Web reverse engineering)
- [n0madic/go-gemini-web2api](https://github.com/n0madic/go-gemini-web2api) (Go port of HanaokaYuzu's RPC & header architecture)

This section documents the reverse-engineered protocol of Google Gemini's web interface (`gemini.google.com`). This mapping explains how Gemini Web dynamically discovers model catalogs (3.5 Flash-Lite, 3.7 Flash, 3.1 Pro), configures extended thinking mode, authenticates via session cookies/SAPISID, and parses streaming responses with reasoning tokens.

---

### 7.1 Web Interface Authentication & Credentials

Unlike the OAuth 2.0 PKCE flow used by Cloud Code / Antigravity (Section 2), Gemini Web relies on session cookies and XSRF/SNlM0e tokens:

- **Cookies**:
  - `__Secure-1PSID`: Core account session cookie.
  - `__Secure-1PSIDTS`: Rolling timestamp cookie required to prevent session invalidation.
  - `SAPISID` (Optional / Enhanced): Used to construct the `SAPISIDHASH` authorization header.
- **XSRF Token (`at`)**: Extracted from `gemini.google.com/app` initialization payload via `"SNlM0e":"([^"]+)"`.
- **SAPISIDHASH Calculation**:
  ```python
  import time, hashlib

  timestamp = int(time.time())
  digest = hashlib.sha1(
      f"{timestamp} {sapisid} https://gemini.google.com".encode()
  ).hexdigest()
  authorization_header = f"SAPISIDHASH {timestamp}_{digest}"
  ```

---

### 7.2 Core Endpoints & RPC Architecture

Gemini Web routes all structured operations through Google's internal `batchexecute` and `StreamGenerate` endpoints:

- **Base URL**: `https://gemini.google.com` (or `https://gemini.google.com/u/<authuser>` for multi-account sessions)
- **Model Discovery & User Status RPC**: `https://gemini.google.com/_/BardChatUi/data/batchexecute?rpcids=otAQ7b`
- **Streaming Generation Endpoint**: `https://gemini.google.com/_/BardChatUi/data/assistant.lamda.BardFrontendService/StreamGenerate`

---

### 7.3 Dynamic Model Discovery (`otAQ7b` RPC)

Rather than hardcoding models or parsing static HTML regexes, the web client fetches its account-specific model catalog dynamically via the `otAQ7b` RPC:

- **Request**:
  ```http
  POST /_/BardChatUi/data/batchexecute?rpcids=otAQ7b&source-path=/app&bl=<BUILD_LABEL>&hl=en&_reqid=<REQ_ID>&rt=c
  Content-Type: application/x-www-form-urlencoded;charset=utf-8
  x-goog-ext-525001261-jspb: [1,null,null,null,null,null,null,null,[4]]

  f.req=[[["otAQ7b","[]",null,"generic"]]]&at=<XSRF_TOKEN>
  ```

- **Response Parsing (`part_body`)**:
  - `part_body[15]`: Array of model objects.
    - `[0]`: Hexadecimal `model_id` (e.g. `2c8a...`).
    - `[1]` / `[10]`: Category / label (`"Fast"`, `"Pro"`, `"Thinking"`, etc.).
    - `[11]` / `[19]`: Full versioned model name (e.g. `"3.5 Flash-Lite"`, `"3.7 Flash"`, `"3.1 Pro"`).
    - `[12]`: Model capability description.
    - `[17]` / `[9]`: Internal `model_number` (e.g. `1` for Flash, `3` for Pro, `5` for Dynamic Thinking, `6` for Flash Lite).
  - `part_body[16]` & `part_body[17]`: Tier & capability bitmasks to derive account `(capacity, capacity_field)`:
    - **Free Tier**: `capacity=1, capacity_field=12`
    - **Pro / Advanced Tier**: `capacity=2` or `3, capacity_field=12`
    - **Plus Tier**: `capacity=4, capacity_field=12`

---

### 7.4 Generation Request Construction (`StreamGenerate`)

When initiating a generation request, model selection and thinking parameters are enforced via custom JSPB headers and payload indices.

#### Required Request Headers:
```http
Content-Type: application/x-www-form-urlencoded
Origin: https://gemini.google.com
Referer: https://gemini.google.com/app
X-Same-Domain: 1
User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36
Cookie: __Secure-1PSID=<...>; __Secure-1PSIDTS=<...>
Authorization: SAPISIDHASH <timestamp>_<hash>
x-goog-ext-525001261-jspb: [1,null,null,null,"<MODEL_HEX_ID>",null,null,0,[4],null,null,<CAPACITY_TAIL>,<THINKING_LEVEL>,"<SESSION_ID>"]
x-goog-ext-525005358-jspb: ["<UUID>",1]
x-goog-ext-73010989-jspb: [0]
x-goog-ext-73010990-jspb: [0]
```

- `<CAPACITY_TAIL>`: `1` (or `null, 1` if `capacity_field == 13`).
- `<THINKING_LEVEL>`: `1` for Standard / Default Thinking, `2` for **Extended Thinking Mode**.
- `x-goog-ext-525005358-jspb`: Encodes the client session/request UUID, matching `inner[59]`.

#### Request Form Body (`f.req` Inner JSPB Array):
The `f.req` parameter contains `[null, "<JSON-stringified-inner>"]` where `inner` is a
**sparse 69-element array** (verified against the live web client; older references showing
81-element payloads with `inner[45]`/`inner[79]`/`inner[80]` are outdated — model selection
happens exclusively via headers):

| Index | Value | Meaning |
|---|---|---|
| `inner[0]` | `[prompt, 0, null, file_refs, null, null, 0]` | Text prompt + multimodal file references |
| `inner[1]` | `["en"]` | Language code |
| `inner[2]` | `["", "", "", null, null, null, null, null, null, ""]` | Conversation/turn metadata |
| `inner[6]` | `[1]`, `inner[7]=1`, `inner[10]=1`, `inner[11]=0` | Client capability flags |
| `inner[17]` | `[[4]]` auto / `[[0]]` explicit thinking | Reasoning mode |
| `inner[18]` | `0`, `inner[27]=1`, `inner[53]=0` | Flags |
| `inner[30]` | `[4]` | Constant |
| `inner[41]` | `[1]` | Chat persistence |
| `inner[59]` | Uppercase request UUID | Must match header `x-goog-ext-525005358-jspb` |
| `inner[61]` | `[]` | Must serialize as `[]`, never `null` |
| `inner[68]` | `2` | Constant |

Authenticated requests additionally send the page's XSRF token as form field
`at=<SNlM0e>`. Anonymous requests omit it.

---

### 7.5 Session Lifecycle: `RotateCookies` & Token Rotation

`__Secure-1PSIDTS` is a rotating freshness token: Google invalidates old values within
minutes and **silently degrades sessions carrying stale tokens to anonymous** (no error is
returned — pages simply render without `SNlM0e`). To keep an exported cookie session alive:

```
POST https://accounts.google.com/RotateCookies
Content-Type: application/json
Cookie: __Secure-1PSID=<...>; __Secure-1PSIDTS=<current>

[000,"-0000000000000000000"]
```

- The fresh token arrives in the response's `Set-Cookie: __Secure-1PSIDTS=...` header.
- Rotation chains validity: refresh every ~8 minutes (the web client uses ~9).
- A lapsed token cannot be revived via this endpoint (`401`) — re-export from a browser.
- After each rotation, re-scrape `SNlM0e`/build label from `/app`; they rotate with the session.
- Do not keep the same account open in a browser while an automated client rotates its
  token — the two fight over the chain.

google-gate implements this lazily in `GeminiWebAdapter.ensure_fresh_session()` (called
before generation requests) and persists rotated values through the router's
`save_config()`. Set `GEMINI_WEB_PROXY=socks5://user:pass@host:port` to route all Gemini Web
traffic through a specific egress IP (requires `httpx[socks]`).

---

### 7.5 Response Parsing & Reasoning / Thinking Extraction

Responses arrive as chunked JSON wrapped in `wrb.fr` / JSPB envelopes prefixed by `)]}'`:

- **Main Response Text**: Extracted from candidate path `candidate[1][0]` (or `candidate[22][0]` when card artifacts are rendered).
- **Reasoning / Extended Thinking Thoughts**: Extracted from candidate path `candidate[37][0][0]`.
- **Grounding Citations**: Extracted from candidate field `[12][43]`.
- **Generated Media & Images**: Extracted from candidate fields `[12][1]` (web images), `[12][7]` (generated images), and `[12][59]` (generated videos).

---

### 7.6 Correlating Web Interface vs. Cloud Code Internal Endpoints

| Capability | Cloud Code / Antigravity Internal API | Gemini Web Interface (HanaokaYuzu Reverse-Engineered) |
|---|---|---|
| **Protocol / Format** | Direct JSON over HTTP/2 & SSE | JSPB nested arrays over `batchexecute` & `StreamGenerate` |
| **Authentication** | OAuth 2.0 PKCE Bearer token | `__Secure-1PSID` / `__Secure-1PSIDTS` cookies + `SAPISIDHASH` + `at` token |
| **Project / Tier Discovery** | `POST /v1internal:loadCodeAssist` | RPC `otAQ7b` (`part_body[16]`, `part_body[17]`) |
| **Model Catalog** | `POST /v1internal:fetchAvailableModels` | RPC `otAQ7b` (`part_body[15]` array: 3.5 Flash-Lite, 3.7 Flash, 3.1 Pro) |
| **Model Selection** | `request.model` field (`gemini-3.7-flash-high`) | Header `x-goog-ext-525001261-jspb` (payload carries no model fields; see §7.4) |
| **Thinking Configuration** | `thinkingConfig.thinkingBudget` / `includeThoughts` | Header `525001261` level (`1` vs `2`) + payload `inner[17]` |
| **Thinking Tokens** | SSE candidate `part.thought = true` | Candidate index `[37][0][0]` |
| **Multimodal Uploads** | Inline base64 image/audio parts | Upload session via `upload_image` / file reference IDs in `inner[0]` |

---

## 7b. AI Studio Web Mapping (MakerSuiteService Reverse-Engineered Reference)

Implemented in `app/providers/aistudio_web.py` (backend id `aistudio_web`). This section
documents the protocol spoken by the [aistudio.google.com](https://aistudio.google.com)
frontend, which the gateway replays to offer a keyless AI Studio backend.

### 7b.1 Endpoints & Authentication

- **Base URL**: `https://alkalimakersuite-pa.clients6.google.com/$rpc/google.internal.alkali.applications.makersuite.v1.MakerSuiteService`
- **Methods**: `GenerateContent` (unary), `StreamGenerateContent` (newline-delimited JSON chunks)
- **Content-Type**: `application/json+protobuf` — protobuf messages encoded as positional JSON arrays (array index = proto field number − 1; absent fields are `null`)
- **Authentication**:
  - Session cookies copied from the browser (`SID`, `HSID`, `SSID`, `SAPISID`, `__Secure-1PAPISID`, `__Secure-*PSID`, ...)
  - `Authorization: SAPISIDHASH <ts>_<sha1(ts SAPISID origin)>` — origin is `https://aistudio.google.com`; the web client additionally appends `SAPISID1PHASH` / `SAPISID3PHASH` variants computed from the same values
  - `x-goog-api-key: AIzaSyDdP816MREB3SkjZO04QXbjsigfcI0GWOs` — public web client key embedded in the AI Studio frontend (identical for all users)

### 7b.2 Request Payload Layout (GenerateContentRequest)

| Slot | Content | Notes |
|---|---|---|
| `[0]` | `"models/<model>"` | e.g. `models/gemini-3.7-flash` |
| `[1]` | contents array | Each turn: `[parts, role]`; a text part is a DataItem `[null, "<text>"]` |
| `[2]` | tool/safety config | Live client sends 4 harm categories with threshold 5: `[[null,null,7,5],[null,null,8,5],[null,null,9,5],[null,null,10,5]]` |
| `[3]` | generation config | idx 3 = maxOutputTokens, 4 = temperature, 5 = topP, 6 = topK, 12 = candidateCount, 15 = thinking config `[1,null,null,<level>]` |
| `[4]` | opaque session blob | Client-context token (~2 KB); synthetic value accepted so far |
| `[5]` | system instruction | Same Content shape as turns: `[[[null,"<system>"]], "user"]`; `null` when absent |
| `[10]` | constant `1` | Observed in every live capture |
| `[11]` | visit id | `v1_...` string, also sent as header `x-aistudio-visit-id` |

Thought signatures from multi-turn function calling ride as an extra assistant part:
`[null,"",null×12,"<signature-blob>"]` (blob at DataItem index 14) — mirroring the live capture.

### 7b.3 Response Format

Responses (both unary and streamed chunks) are protobuf-as-json arrays containing Content
nodes shaped like the request's. Text arrives as `[null, "<text>"]` DataItem pairs; the
gateway extracts them via recursive tree walking (`_iter_text_parts`) and tolerates both
true deltas and cumulative snapshots. gRPC errors surface as JSON objects
`{"error": {"code": ..., "message": ..., "status": ...}}` and map to OpenAI-style 429/401
handling. Finish reasons (`STOP`, `MAX_TOKENS`, `SAFETY`, ...) appear as bare enum strings.

### 7b.4 Known Limitations

- **Text-only generation**: wire slots for image/audio parts and function-call declarations
  are unverified; non-text parts are dropped and tools ignored.
- **No model discovery RPC** is implemented; the adapter serves a static catalog.
- **Token usage is estimated locally** (whitespace heuristic) as upstream usageMetadata has
  no confirmed slot.

---

## 8. Upstream Rate Limits & Quotas (Reference)

Rate limits enforced by each backend, their sources, and how `google-gate` models them.
All gateway-side limits are configurable via environment variables (see table below) and
feed the local sliding-window tracker (`InMemoryRateTracker`) that powers the dashboard
quota gauges and proactive routing decisions.

### Antigravity / Cloud Code (`daily-cloudcode-pa.googleapis.com`)

| Dimension | Limit | Source |
|---|---|---|
| Requests per minute | **100 RPM** | `loadCodeAssist` response field `userLimits.rateLimit` (§3.1). Applied dynamically at runtime by `AntigravityAdapter.load_code_assist()`. |
| Tokens per minute | Not exposed upstream | Local estimate only (`ANTIGRAVITY_TPM`, default 1,000,000). |
| Usage buckets | Weekly + 5-hour burst, reported as remaining fractions with reset timestamps | `retrieveUserQuotaSummary` (§3.3). These are the authoritative quota values shown on the dashboard. |

### Gemini API (`generativelanguage.googleapis.com`)

Official documented free-tier limits per model family (Google reduced free-tier quotas by
50-80% effective December 2025; verified August 2026 against
[ai.google.dev/gemini-api/docs/rate-limits](https://ai.google.dev/gemini-api/docs/rate-limits)
and the AI Studio dashboard):

| Model | Free Tier RPM | Free Tier TPM | Free Tier RPD |
|---|---|---|---|
| Gemini 2.5 Pro | 5 | 250,000 | 100 |
| Gemini 2.5 Flash | 10 | 250,000 | 250 |
| Gemini 2.5 Flash-Lite | 15 | 250,000 | 1,000 |
| Gemini 2.0 Flash *(shut down June 1, 2026)* | 0 | 0 | 0 |

Notes:
- Limits are enforced **per project**, not per API key; RPD resets at midnight Pacific time.
- Paid tiers raise limits substantially (Tier 1 ≈ 1,000+ RPM / 1M TPM / 1,000+ RPD; higher tiers scale further).
- Gateway defaults are set conservatively to the **Flash-class free tier** (`GEMINI_API_RPM=10`,
  `GEMINI_API_TPM=250000`, `GEMINI_API_RPD=250`). Billing-enabled projects should override via env vars.
- The adapter's plan probe distinguishes `Free` vs `Pay-As-You-Go` keys but does not yet auto-adjust these defaults.

### Gemini Web (`gemini.google.com`)

**No numeric rate limits are published or reliably reverse-engineered.** Google has moved
Gemini Web enforcement to an opaque compute-usage-based model: exceeding a per-model usage
budget returns an in-stream usage-limit error code (with a cooldown/reset period) rather
than an HTTP 429 quota payload. Community references (HanaokaYuzu/Gemini-API) surface only
the *cooldown state*, never a request/token budget.

Gateway defaults (`GEMINI_WEB_RPM=60`, `GEMINI_WEB_TPM=500000`, `GEMINI_WEB_RPD=0` disabled)
are therefore **local conservative estimates** used solely for proactive throttling — they do
not represent upstream numbers.

#### Empirical access findings (August 2026)

Live probing from this project against `gemini.google.com` established the following,
independent of session validity:

- Google's anti-abuse pipeline fronts all Gemini Web traffic. Flagged IPs receive either a
  `302` redirect to `google.com/sorry` or — for minimal/non-browser client headers — a
  silently degraded *logged-out* HTML shell (200 OK without `SNlM0e` or account data).
  The degraded-shell response makes cookie problems look like expired sessions; check the
  HTTP status and headers before blaming credentials.
- A `GOOGLE_ABUSE_EXEMPTION` cookie (issued per-IP after solving a reCAPTCHA, time-limited)
  restores authenticated rendering of `/app` and read-only `batchexecute` RPCs
  (e.g. `otAQ7b` model discovery works end-to-end with just six cookies:
  `__Secure-1PSID`, `__Secure-1PSIDTS`, `__Secure-1PSIDCC`, `SIDCC`, `SAPISID`,
  `GOOGLE_ABUSE_EXEMPTION`).
- The **`StreamGenerate` endpoint is separately gated**: it returns `429` with an embedded
  reCAPTCHA Enterprise challenge regardless of cookie completeness, Chrome client headers
  (`x-browser-validation`, `x-client-data`, full `sec-ch-ua` set), or TLS fingerprint
  impersonation (curl_cffi chrome131/chrome124). Satisfying it requires executing Google's
  JS inside a real browser context.

**Consequence:** headless clients cannot currently perform generation calls against Gemini
Web on flagged (arguably any) IPs, and upstream rate limits for this backend are not
empirically measurable outside a browser. The gateway's Gemini Web adapter remains
protocol-complete but is effectively unusable for generation unless requests are proxied
through a real browser context (e.g. CDP-driven headless Chrome).

### Environment Variable Reference

| Variable | Default | Backend | Meaning |
|---|---|---|---|
| `ANTIGRAVITY_RPM` | `100` | antigravity | Mirrors `userLimits.rateLimit`; rarely needs overriding |
| `ANTIGRAVITY_TPM` | `1000000` | antigravity | Local estimate; not upstream-enforced |
| `ANTIGRAVITY_RPD` | `0` (off) | antigravity | Daily request tracker |
| `GEMINI_API_RPM` | `10` | gemini_api | Free-tier Flash-class RPM |
| `GEMINI_API_TPM` | `250000` | gemini_api | Free-tier Flash-class TPM |
| `GEMINI_API_RPD` | `250` | gemini_api | Free-tier Flash-class daily requests |
| `GEMINI_WEB_RPM` | `60` | gemini_web | Local estimate only |
| `GEMINI_WEB_TPM` | `500000` | gemini_web | Local estimate only |
| `GEMINI_WEB_RPD` | `0` (off) | gemini_web | Daily request tracker |
| `AISTUDIO_WEB_RPM` | `10` | aistudio_web | Conservative free-tier Flash-class estimate |
| `AISTUDIO_WEB_TPM` | `250000` | aistudio_web | Local estimate only |
| `AISTUDIO_WEB_RPD` | `0` (off) | aistudio_web | Daily request tracker |

Setting any value to `0` disables that dimension of the local tracker.

### Critical: Thought Signature Preservation
For Gemini models, when a function call is returned, Google attaches a `thoughtSignature` base64 string to the part. When sending the subsequent conversation turns back in `contents`, the model's prior `functionCall` part must contain that exact `thoughtSignature` string, or the backend rejects the request with:
```
400 INVALID_ARGUMENT: Function call is missing a thought_signature.
```
