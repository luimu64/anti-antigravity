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
