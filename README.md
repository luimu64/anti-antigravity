# Google Gate

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
- **Multi-Backend Routing with Rate-Limit Awareness**: Aggregates Antigravity (OAuth), Gemini API (AI Studio key), and Gemini Web (session cookies); requests route only to backends with remaining capacity (`free_first` / `round_robin` strategies), with automatic cooldowns on upstream 429s and SOCKS/HTTP proxy support for Gemini Web egress.
- **Modern Web Dashboard**: View auth state, quota gauges, token expiration, and generate/revoke API keys.
- **Dockerized**: Multi-stage lightweight container with volume persistence for credentials and API keys.

---

## Supported Models & Aliases

| Model ID / Alias | Backend Target | Capabilities / Details |
|---|---|---|
| `gemini-3.7-flash` | Gemini 3.7 Flash | Reasoning / Thinking, Vision, Tool Calling, 1M Context |
| `gemini-3.6-flash` | Gemini 3.6 Flash | Fast Flash reasoning tier |
| `gemini-3.5-flash` | Gemini 3.5 Flash | High-speed agent model |
| `gemini-3.1-pro` | Gemini 3.1 Pro | Advanced Pro tier model, 2M Context |
| `gemini-2.5-pro` | Gemini 2.5 Pro | Advanced reasoning and analysis |
| `gemini-2.5-flash` | Gemini 2.5 Flash | Standard flash model |
| `claude-3-7-sonnet` / `claude-sonnet-4-6` | Claude Sonnet 4.6 (Thinking) | Anthropic Sonnet, Code & Reasoning |
| `claude-3-opus` / `claude-opus-4-6-thinking` | Claude Opus 4.6 (Thinking) | Anthropic Opus, Deep Reasoning |
| `gpt-oss-120b` | GPT-OSS 120B | 120B Open Model |
| `gpt-4o` *(Alias)* | `gemini-3.7-flash` | Standard OpenAI drop-in alias |
| `gpt-4o-mini` *(Alias)* | `gemini-3.6-flash` | Fast OpenAI drop-in alias |
| `o1` / `o3-mini` *(Aliases)* | `gemini-3.7-flash` / `claude-opus` | Reasoning drop-in aliases |
| `vision` *(Alias)* | `gemini-3.7-flash-image` | Multimodal / Vision alias |
| `text-embedding-004` / `text-embedding-3-*` | Text Embedding 004 | Text Embeddings |

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

## Backend Configuration & Provider Quirks

All three backends are **disabled by default**. Enable them in the Web Dashboard's provider cards, or via environment variables (`ANTIGRAVITY_ENABLED`, `GEMINI_API_ENABLED`, `GEMINI_WEB_ENABLED`).

### Antigravity (OAuth)

Fully automatic once you complete the Google OAuth flow (`/auth/login` or `python main.py --login`). The gateway discovers your project ID, applies the documented 100 RPM limit dynamically, and reports **real upstream quota buckets** (weekly + 5-hour burst) on the dashboard.

### Gemini API (AI Studio key)

Paste a key into the dashboard or set `GEMINI_API_KEY`. The plan probe distinguishes Free vs Pay-As-You-Go keys. Gateway-side throttle defaults are set conservatively to Google's documented **free-tier Flash-class limits** (`10 RPM / 250k TPM / 250 RPD`) — if your project has billing enabled, raise them via `GEMINI_API_RPM`, `GEMINI_API_TPM`, `GEMINI_API_RPD`. Sources and per-model tables: [INTERNAL_API.md §8](INTERNAL_API.md).

### Gemini Web (session cookies)

Export from a browser logged into [gemini.google.com](https://gemini.google.com) (DevTools → Application → Cookies): `__Secure-1PSID`, `__Secure-1PSIDTS`, and optionally `SAPISID`. Enter them in the dashboard card or via `GEMINI_WEB_PSID` / `GEMINI_WEB_PSIDTS` / `GEMINI_WEB_SAPISID`.

Quirks you should know about:

- **Rotating token**: `__Secure-1PSIDTS` expires within minutes and Google *silently degrades* stale sessions to anonymous instead of erroring (pages render fine, requests fail). The gateway auto-refreshes it every ~8 minutes against Google's `RotateCookies` endpoint and persists rotated values to `data/credentials.json`. Seed it with a **fresh** export, then let the gateway own the rotation.
- **Don't double-rotate**: while the gateway runs, avoid keeping gemini.google.com open in a browser with the same account — browser and gateway will fight over the rotation chain.
- **Lapsed tokens can't be revived**: if the chain breaks (gateway offline too long), re-export fresh cookies; `RotateCookies` returns `401` until then.
- **IP reputation gates generation**: Google fronts Gemini Web with anti-abuse checks. On flagged IPs even valid cookies get redirected to `google.com/sorry` (CAPTCHA) for generation RPCs — pages and model discovery may keep working, which masks the problem. The adapter detects this and returns an explicit error naming the cause. Remedy: wait for reputation to clear, or route through a clean/residential exit:

  ```bash
  GEMINI_WEB_PROXY=socks5://user:pass@host:port   # requires httpx[socks] (included)
  ```

### Quota gauges (`Usage & Quotas` card)

- **Antigravity** rows show authoritative upstream percentages and reset times.
- **Gemini API / Gemini Web** rows show the gateway's *local sliding-window usage* against the configured caps above — they measure traffic through this gateway only (60-second RPM/TPM memory, 24-hour RPD), not Google-side consumption, which those backends do not expose. During cooldowns the affected backend shows a distinct cooldown card instead of falsified gauges.

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
