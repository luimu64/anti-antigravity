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

### Critical: Thought Signature Preservation
For Gemini models, when a function call is returned, Google attaches a `thoughtSignature` base64 string to the part. When sending the subsequent conversation turns back in `contents`, the model's prior `functionCall` part must contain that exact `thoughtSignature` string, or the backend rejects the request with:
```
400 INVALID_ARGUMENT: Function call is missing a thought_signature.
```

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
The `f.req` parameter contains `[null, JSON.stringify(inner)]`:
- `inner[0]`: `[prompt, 0, null, file_refs, null, null, 0]` (Text prompt + multimodal file references).
- `inner[1]`: `["en"]` (Language code).
- `inner[2]`: `["", "", "", null, null, null, null, null, null, ""]` (Conversation/turn metadata).
- `inner[17]`: `[[0]]` for explicit/extended thinking, `[[4]]` for auto.
- `inner[27]`: `1`
- `inner[30]`: `[4]`
- `inner[41]`: `[1]` (or `[2]` for persistent chats).
- `inner[45]`: `1` (Temporary / incognito chat toggle).
- `inner[59]`: Client request UUID (uppercase string matching header `525005358`).
- `inner[79]`: Selected `model_number` (`1` = Flash, `3` = Pro, `5` = Fast Dynamic Thinking, `6` = Flash Lite).
- `inner[80]`: `1` for standard thinking, `2` for **Extended Thinking Mode**.

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
| **Model Selection** | `request.model` field (`gemini-3.7-flash-high`) | Header `x-goog-ext-525001261-jspb` + payload `inner[79]` |
| **Thinking Configuration** | `thinkingConfig.thinkingBudget` / `includeThoughts` | Header `525001261` level (`1` vs `2`) + payload `inner[17]` + `inner[80]` |
| **Thinking Tokens** | SSE candidate `part.thought = true` | Candidate index `[37][0][0]` |
| **Multimodal Uploads** | Inline base64 image/audio parts | Upload session via `upload_image` / file reference IDs in `inner[0]` |
