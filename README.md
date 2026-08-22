# Google Gate

<p align="center">
  <img src="https://img.shields.io/badge/OpenAI_API-Drop--in_Compatible-412991?style=for-the-badge&logo=openai&logoColor=white" alt="OpenAI Compatible" />
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python Version" />
  <img src="https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker Ready" />
  <img src="https://img.shields.io/badge/License-MIT-brightgreen?style=for-the-badge" alt="License" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Google_OAuth-PKCE_Auto--Refresh-4285F4?style=for-the-badge&logo=google&logoColor=white" alt="Google OAuth" />
  <img src="https://img.shields.io/badge/Thinking_Tokens-Gemini_3.7_%7C_Claude_Opus-8A2BE2?style=for-the-badge&logo=anthropic&logoColor=white" alt="Reasoning Support" />
  <img src="https://img.shields.io/badge/Prompt_Caching-Supported-FF6F00?style=for-the-badge&logo=lightning&logoColor=white" alt="Prompt Caching" />
  <img src="https://img.shields.io/badge/Tests-Passing-2ea44f?style=for-the-badge&logo=pytest&logoColor=white" alt="Tests Passing" />
  <img src="https://img.shields.io/badge/PRs-Welcome-blueviolet?style=for-the-badge&logo=github" alt="PRs Welcome" />
</p>

---

A lightweight, high-performance API gateway that translates **Google AI backends** (Antigravity CLI OAuth backend, Gemini AI Studio API, and Gemini Web session cookies) into the standard **OpenAI API Schema** (`/v1/chat/completions`, `/v1/models`, `/v1/completions`, `/v1/embeddings`).

Supports all **Google and partner models** (Gemini 3.7 Flash with reasoning, Claude Sonnet 4.6, Claude Opus 4.6 Thinking, Gemini 3.1 Pro, GPT-OSS 120B), multi-turn conversations, tool/function calling, multimodal input, real-time Server-Sent Events (SSE) streaming, Google OAuth 2.0 PKCE authentication, and dedicated **Bridge API Key Management & Enforcement**.

---

## Features

- **OpenAI v1 Compatible**: Full drop-in replacement for OpenAI SDKs, Cursor, Continue, Cline, Open WebUI, LiteLLM, LangChain, etc.
- **Token Usage & Prompt Caching Reporting**:
  - Detailed token consumption metadata (`prompt_tokens`, `completion_tokens`, `total_tokens`)
  - Prompt caching breakdown (`usage.prompt_tokens_details.cached_tokens`)
  - Reasoning/thinking token usage breakdown (`usage.completion_tokens_details.reasoning_tokens`)
  - Streaming usage support via `stream_options: {"include_usage": true}`
- **Structured Outputs & Response Formats**:
  - JSON mode (`response_format: {"type": "json_object"}`)
  - Strict JSON schema validation (`response_format: {"type": "json_schema", "json_schema": {...}}`)
- **Advanced Generation Controls**:
  - Custom stop sequences (`stop: ["\n\n", "User:"]`)
  - Penalties (`presence_penalty`, `frequency_penalty`)
  - Deterministic sampling (`seed`) and candidate counts (`n`)
- **Modern Message Roles & Modalities**:
  - Full support for `"developer"` role (o1/o3 style system prompts)
  - Multimodal image (`image_url`) and audio (`input_audio`) input payloads
- **Tool & Function Calling**:
  - Standard `tools` function declarations with `thoughtSignature` preservation across turns
  - Flexible `tool_choice` modes (`auto`, `none`, `required`, or forced specific function)
- **Bridge API Key Management & Enforcement**:
  - Generate, list, and revoke multiple bridge API keys (`sk-gate-...` and legacy `sk-agy-...`)
  - Enforce API keys across all incoming `/v1/*` OpenAI endpoints with standard OpenAI 401 error responses
  - Toggle enforcement on or off via Web Dashboard or REST API
  - Track key creation and last used timestamps
- **Real-time SSE Streaming**: Low-latency token-by-token streaming compatible with OpenAI chat completion chunk schema.
- **Thinking / Reasoning Support**: Streams `reasoning_content` (thinking tokens) from Gemini 3.7 Flash Thinking, Claude Opus Thinking, etc.
- **Google OAuth 2.0 Flow**:
  - One-click Web UI login (`/auth/login`)
  - Terminal CLI interactive login (`python main.py --login`)
  - Auto-discovery from Linux Secret Service / Keyring (`secret-tool` / DBus)
  - Automatic token refresh before expiration
  - Headless environment support via environment variables (`REFRESH_TOKEN` / `ACCESS_TOKEN`)
- **Modern Web Dashboard**: View auth state, quota gauges, token expiration, and generate/revoke API keys.
- **Dockerized**: Multi-stage lightweight container with volume persistence for credentials and API keys.

---

## Web Dashboard

The built-in interactive web dashboard provides real-time connection status, quota monitoring, Google OAuth token management, and Bridge API key generation/enforcement:

| Authenticated Dashboard | Login / Logged Out View |
| :---: | :---: |
| ![Authenticated Dashboard](img/logged-in.png) | ![Logged Out Dashboard](img/logged-out.png) |

---

## Supported Models & Aliases

| Model Name | Internal Target | Features |
|---|---|---|
| `gemini-3.7-flash-high` | Gemini 3.7 Flash (High) | Thinking/Reasoning, Vision, High Speed, 1M Context |
| `gemini-3.7-flash-medium` | Gemini 3.7 Flash (Medium) | Balanced Reasoning, Vision |
| `gemini-3.7-flash-low` | Gemini 3.7 Flash (Low) | Fast Thinking, Vision |
| `claude-sonnet-4-6` | Claude Sonnet 4.6 (Thinking) | Anthropic Sonnet, Code & Reasoning |
| `claude-opus-4-6-thinking` | Claude Opus 4.6 (Thinking) | Anthropic Opus, Deep Reasoning |
| `gemini-3.1-pro-high` | Gemini 3.1 Pro (High) | Advanced Pro tier model |
| `gemini-3.6-flash-high` | Gemini 3.6 Flash (High) | Ultra-fast Flash model |
| `gpt-oss-120b-medium` | GPT-OSS 120B | 120B Open Model |
| `gpt-4o` *(Alias)* | `gemini-3.7-flash-high` | Standard OpenAI drop-in alias |
| `gpt-4o-mini` *(Alias)* | `gemini-3.6-flash-high` | Fast OpenAI drop-in alias |
| `claude-3-7-sonnet` *(Alias)* | `claude-sonnet-4-6` | Claude drop-in alias |
| `claude-3-opus` *(Alias)* | `claude-opus-4-6-thinking` | Claude Opus drop-in alias |
| `o1` / `o3-mini` *(Aliases)* | `gemini-3.7-flash-high` | Reasoning drop-in alias |

---

## Quickstart

### 1. Run with Docker Compose (Recommended)

```bash
docker-compose up -d
```

Open [http://localhost:8000](http://localhost:8000) in your browser:
- Connect Google OAuth if needed via **Sign in with Google**.
- Generate or copy a Bridge API Key from the **Bridge API Key Management** section.

### 2. Run with Docker CLI

```bash
# Build image
docker build -t google-gate .

# Run container with persistent data volume
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  --name google-gate \
  google-gate
```

### 3. Run Locally with Python

```bash
# 1. Setup virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Check status or login
python main.py --status

# 4. Start the server
python main.py --port 8000
```

---

## Bridge API Key Management & Enforcement

When key enforcement is active, all requests to `/v1/*` must supply an API key in the `Authorization` header:

```http
Authorization: Bearer sk-gate-xxxxxxxxxxxxxxxxxxxx
```

If an invalid or missing key is provided, the bridge returns standard OpenAI HTTP 401 error payloads:

```json
{
  "error": {
    "message": "Incorrect API key provided: sk-inv***. Please check your API key and try again.",
    "type": "invalid_request_error",
    "param": null,
    "code": "invalid_api_key"
  }
}
```

### Key Management Endpoints:
- `GET /api/keys`: List all generated API keys (with preview and last used time).
- `POST /api/keys`: Generate a new key: `{"name": "Cursor IDE"}`.
- `DELETE /api/keys/{key_id}`: Revoke an API key immediately.
- `POST /api/keys/enforcement`: Toggle enforcement mode: `{"enforce": true}`.

---

## API Endpoints Reference

| Endpoint | Method | Description |
|---|---|---|
| `/` | `GET` | Interactive Web Dashboard |
| `/health` | `GET` | Health check, auth status & project ID |
| `/api/keys` | `GET`, `POST` | Manage Bridge API keys |
| `/api/keys/{key_id}` | `DELETE` | Revoke a Bridge API key |
| `/api/keys/enforcement` | `POST` | Enable or disable API key enforcement |
| `/v1/models` | `GET` | List all available models & aliases in OpenAI format |
| `/v1/models/{model_id}` | `GET` | Retrieve single model metadata |
| `/v1/chat/completions` | `POST` | OpenAI chat completions (supports `stream: true/false`, reasoning, tools) |
| `/v1/completions` | `POST` | Legacy text completion endpoint adapter |
| `/v1/embeddings` | `POST` | Embeddings endpoint |
| `/auth/login` | `GET` | Google OAuth 2.0 PKCE login initiation |
| `/auth/callback` | `GET` | OAuth callback handler |
| `/auth/status` | `GET` | Authentication status & token expiration |
| `/auth/refresh` | `POST` | Force refresh access token |
| `/auth/token` | `POST` | Set credentials manually |
| `/auth/logout` | `POST` | Clear Google credentials |
| `/api/quotas` | `GET` | Live quota usage & limit stats |

---

## Running Tests

```bash
# Run unit & integration test suite
.venv/bin/pytest -v
```

---

## License

MIT License.
