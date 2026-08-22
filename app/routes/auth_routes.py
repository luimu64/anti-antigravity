import logging

from fastapi import APIRouter, HTTPException, Query, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel

from app.auth import auth_manager
from app.client import client

logger = logging.getLogger("google_gate.auth_routes")
router = APIRouter(prefix="/auth", tags=["Auth"])


class SetTokenRequest(BaseModel):
    access_token: str | None = None
    refresh_token: str | None = None
    project_id: str | None = None


class ExchangeCodeRequest(BaseModel):
    code_or_url: str
    state: str | None = None
    redirect_uri: str | None = None


@router.post("/exchange")
async def exchange_code_endpoint(payload: ExchangeCodeRequest, request: Request):
    """
    Manually exchange an authorization code, full callback URL, or refresh token.
    Works seamlessly for remote/LAN/headless setups.
    """
    input_str = payload.code_or_url.strip()

    # If user pasted a refresh token directly
    if input_str.startswith("1//") or input_str.startswith("ya29."):
        auth_manager.set_tokens(
            refresh_token=input_str if input_str.startswith("1//") else None,
            access_token=input_str if input_str.startswith("ya29.") else None,
        )
        try:
            await client.load_code_assist()
        except Exception as e:
            logger.warning(f"Could not load code assist: {e}")
        return {
            "status": "ok",
            "message": "Token saved successfully!",
            "auth": auth_manager.get_status(),
        }

    code = input_str
    state = payload.state
    redirect_uri = payload.redirect_uri or "http://localhost:8085"

    # If full callback URL was pasted (e.g. http://localhost:8085/?state=...&code=...)
    if "code=" in input_str:
        import urllib.parse

        parsed = urllib.parse.urlparse(input_str)
        qs = urllib.parse.parse_qs(parsed.query)
        if "code" in qs:
            code = qs["code"][0]
        if "state" in qs and not state:
            state = qs["state"][0]
        if not payload.redirect_uri:
            redirect_uri = (
                f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip("/")
                or "http://localhost:8085"
            )

    try:
        await auth_manager.exchange_code(
            code=code, redirect_uri=redirect_uri, state=state
        )
        try:
            await client.load_code_assist()
        except Exception as e:
            logger.warning(f"Could not load code assist: {e}")

        return {
            "status": "ok",
            "message": "Authenticated successfully!",
            "auth": auth_manager.get_status(),
        }
    except Exception as e:
        logger.error(f"Manual exchange failed: {e}")
        # Try fallback loopback redirect uri
        if redirect_uri != "http://localhost:8085":
            try:
                await auth_manager.exchange_code(
                    code=code, redirect_uri="http://localhost:8085", state=state
                )
                await client.load_code_assist()
                return {
                    "status": "ok",
                    "message": "Authenticated successfully with loopback URI!",
                    "auth": auth_manager.get_status(),
                }
            except Exception:
                pass
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Authorization code exchange failed: {e!s}",
        ) from e


@router.get("/login")
async def auth_login(
    request: Request,
    redirect: bool = Query(
        default=True, description="Redirect to Google OAuth page directly"
    ),
    redirect_uri: str | None = Query(default=None),
):
    """
    Initiate Google OAuth 2.0 flow.
    For localhost connections: auto-redirects directly.
    For remote/LAN IP connections: uses Google-approved loopback redirect_uri to prevent Error 400.
    """
    hostname = request.url.hostname or "localhost"
    is_localhost = hostname in ("localhost", "127.0.0.1", "::1")

    # For remote IPs, always use Google-approved loopback URI http://localhost:8085
    callback_url = redirect_uri or (
        "http://localhost:8085"
        if not is_localhost
        else str(request.url_for("auth_callback"))
    )
    auth_url, state, _ = auth_manager.get_authorization_url(redirect_uri=callback_url)

    if is_localhost and redirect:
        return RedirectResponse(url=auth_url)

    # For remote / LAN IPs, render step-by-step remote auth page
    return HTMLResponse(
        content=f"""
        <!DOCTYPE html>
        <html lang="en" data-theme="dark">
        <head>
          <meta charset="utf-8">
          <meta name="viewport" content="width=device-width, initial-scale=1.0">
          <title>Google Sign-In - Google Gate</title>
          <link href="https://cdn.jsdelivr.net/npm/daisyui@4/dist/full.min.css" rel="stylesheet" type="text/css">
          <script src="https://cdn.tailwindcss.com"></script>
        </head>
        <body class="bg-base-300 min-h-screen text-base-content p-4 md:p-8 flex items-center justify-center">
          <div id="toast-container" class="toast toast-top toast-center z-50 pointer-events-none" aria-live="polite" aria-atomic="true"></div>

          <div class="container max-w-lg mx-auto">
            <div class="card bg-base-200 shadow-xl">
              <div class="card-body">
                <div class="flex items-center justify-between border-b border-base-300 pb-3 mb-4">
                  <h1 class="card-title text-lg font-bold">Connect Google Account</h1>
                  <span class="badge badge-neutral badge-sm">Remote Server Setup</span>
                </div>

                <p class="text-xs text-base-content/80 mb-4">
                  Because this bridge is deployed on a remote server/IP (<code class="badge badge-neutral font-mono text-xs">{hostname}</code>), Google requires authenticating via standard loopback.
                </p>

                <div class="card bg-base-100 border border-base-300 shadow-sm mb-4">
                  <div class="card-body p-4">
                    <h2 class="font-bold text-sm mb-1">Step 1: Open Google Sign-In</h2>
                    <p class="text-xs text-base-content/70 mb-3">Click below to authorize with your Google Account in a new tab:</p>
                    <a href="{auth_url}" target="_blank" class="btn btn-primary btn-sm w-full">Authorize Google Account (New Tab)</a>
                  </div>
                </div>

                <div class="card bg-base-100 border border-base-300 shadow-sm mb-4">
                  <div class="card-body p-4">
                    <h2 class="font-bold text-sm mb-1">Step 2: Paste Redirect URL or Code</h2>
                    <p class="text-xs text-base-content/70 mb-3">After approving, Google redirects to <code class="badge badge-neutral font-mono text-xs">http://localhost:8085/?code=...</code>. Copy that address bar URL (or the code) and paste it here:</p>
                    <form onsubmit="event.preventDefault(); submitCode();" class="join w-full">
                      <input type="text" id="callback-input" placeholder="http://localhost:8085/?state=...&code=4/0A..." class="input input-bordered input-sm join-item flex-1">
                      <button type="submit" class="btn btn-primary btn-sm join-item">Connect</button>
                    </form>
                  </div>
                </div>

                <div class="card-actions justify-end mt-2">
                  <a href="/" class="btn btn-outline btn-sm">Back to Dashboard</a>
                </div>
              </div>
            </div>
          </div>

          <script>
            function showToast(message, type = 'success') {{
              const container = document.getElementById('toast-container');
              if (!container) return;

              const alertClass = type === 'error' ? 'alert-error' : 'alert-success';
              const iconSvg = type === 'error'
                ? '<svg xmlns="http://www.w3.org/2000/svg" class="stroke-current shrink-0 h-5 w-5" fill="none" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>'
                : '<svg xmlns="http://www.w3.org/2000/svg" class="stroke-current shrink-0 h-5 w-5" fill="none" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>';

              const toastEl = document.createElement('div');
              toastEl.className = `alert ${{alertClass}} shadow-lg pointer-events-auto transition-all duration-300`;
              toastEl.innerHTML = `
                ${{iconSvg}}
                <span>${{escapeHtml(message)}}</span>
              `;
              container.appendChild(toastEl);

              setTimeout(() => {{
                toastEl.classList.add('opacity-0');
                setTimeout(() => toastEl.remove(), 300);
              }}, 3000);
            }}

            function escapeHtml(str) {{
              if (!str) return '';
              return String(str)
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;')
                .replace(/"/g, '&quot;')
                .replace(/'/g, '&#039;');
            }}

            async function submitCode() {{
              const val = document.getElementById('callback-input').value.trim();
              if (!val) {{
                showToast('Please paste the redirect URL or code.', 'error');
                return;
              }}
              try {{
                const res = await fetch('/auth/exchange', {{
                  method: 'POST',
                  headers: {{ 'Content-Type': 'application/json' }},
                  body: JSON.stringify({{ code_or_url: val }})
                }});
                const data = await res.json();
                if (res.ok) {{
                  showToast('Connected successfully! Redirecting...');
                  setTimeout(() => {{ window.location.href = '/'; }}, 1000);
                }} else {{
                  showToast('Error: ' + (data.detail || JSON.stringify(data)), 'error');
                }}
              }} catch (e) {{
                showToast('Network error: ' + e.message, 'error');
              }}
            }}
          </script>
        </body>
        </html>
        """
    )


@router.get("/callback")
async def auth_callback(
    request: Request,
    code: str | None = Query(default=None),
    state: str | None = Query(default=None),
    error: str | None = Query(default=None),
):
    """
    Handle OAuth callback from Google.
    """
    if error:
        logger.error(f"OAuth callback returned error: {error}")
        return HTMLResponse(
            content=f"""
            <!DOCTYPE html>
            <html lang="en" data-theme="dark">
            <head>
              <meta charset="utf-8">
              <meta name="viewport" content="width=device-width, initial-scale=1.0">
              <title>OAuth Error - Google Gate</title>
              <link href="https://cdn.jsdelivr.net/npm/daisyui@4/dist/full.min.css" rel="stylesheet" type="text/css">
              <script src="https://cdn.tailwindcss.com"></script>
            </head>
            <body class="bg-base-300 min-h-screen text-base-content p-4 md:p-8 flex items-center justify-center">
              <div class="card bg-base-200 shadow-xl max-w-md w-full">
                <div class="card-body">
                  <div class="alert alert-error mb-4">
                    <svg xmlns="http://www.w3.org/2000/svg" class="stroke-current shrink-0 h-6 w-6" fill="none" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
                    <span class="font-bold">Authentication Failed</span>
                  </div>
                  <p class="text-sm mb-4">Google returned error: <code class="badge badge-neutral font-mono">{error}</code></p>
                  <div class="card-actions justify-end">
                    <a href="/auth/login" class="btn btn-primary btn-sm">Try Again</a>
                  </div>
                </div>
              </div>
            </body>
            </html>
            """,
            status_code=400,
        )

    if not code:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing authorization code.",
        )

    callback_url = str(request.url).split("?")[0]
    try:
        await auth_manager.exchange_code(
            code=code, redirect_uri=callback_url, state=state
        )
        # Fetch and sync user/project metadata
        try:
            await client.load_code_assist()
        except Exception as e:
            logger.warning(f"Could not load code assist after OAuth exchange: {e}")

        return HTMLResponse(
            content=f"""
            <!DOCTYPE html>
            <html lang="en" data-theme="dark">
            <head>
              <meta charset="utf-8">
              <meta name="viewport" content="width=device-width, initial-scale=1.0">
              <title>Login Successful - Google Gate</title>
              <link href="https://cdn.jsdelivr.net/npm/daisyui@4/dist/full.min.css" rel="stylesheet" type="text/css">
              <script src="https://cdn.tailwindcss.com"></script>
              <meta http-equiv="refresh" content="3;url=/">
            </head>
            <body class="bg-base-300 min-h-screen text-base-content p-4 md:p-8 flex items-center justify-center">
              <div class="card bg-base-200 shadow-xl max-w-md w-full">
                <div class="card-body">
                  <div class="alert alert-success mb-4">
                    <svg xmlns="http://www.w3.org/2000/svg" class="stroke-current shrink-0 h-6 w-6" fill="none" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
                    <span class="font-bold">Authentication Successful!</span>
                  </div>
                  <p class="text-sm text-base-content/80 mb-2">Your Google account has been connected to Google Gate.</p>
                  <p class="text-sm mb-1">Project ID: <strong>{auth_manager.project_id or "Auto-discovered"}</strong></p>
                  <p class="text-sm mb-4">Account: <strong>{auth_manager.user_email or "Authenticated"}</strong></p>
                  <div class="card-actions justify-between items-center">
                    <span class="text-xs text-base-content/60">Redirecting in 3 seconds...</span>
                    <a href="/" class="btn btn-primary btn-sm">Go to Dashboard</a>
                  </div>
                </div>
              </div>
            </body>
            </html>
            """
        )
    except Exception as e:
        logger.error(f"Failed to complete OAuth callback: {e}")
        return HTMLResponse(
            content=f"""
            <!DOCTYPE html>
            <html lang="en" data-theme="dark">
            <head>
              <meta charset="utf-8">
              <meta name="viewport" content="width=device-width, initial-scale=1.0">
              <title>Authentication Error - Google Gate</title>
              <link href="https://cdn.jsdelivr.net/npm/daisyui@4/dist/full.min.css" rel="stylesheet" type="text/css">
              <script src="https://cdn.tailwindcss.com"></script>
            </head>
            <body class="bg-base-300 min-h-screen text-base-content p-4 md:p-8 flex items-center justify-center">
              <div class="card bg-base-200 shadow-xl max-w-md w-full">
                <div class="card-body">
                  <div class="alert alert-error mb-4">
                    <svg xmlns="http://www.w3.org/2000/svg" class="stroke-current shrink-0 h-6 w-6" fill="none" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
                    <span class="font-bold">Authentication Error</span>
                  </div>
                  <p class="text-sm text-error mb-4">{e!s}</p>
                  <div class="card-actions justify-end">
                    <a href="/auth/login" class="btn btn-primary btn-sm">Try Again</a>
                  </div>
                </div>
              </div>
            </body>
            </html>
            """,
            status_code=500,
        )


@router.get("/status")
async def auth_status():
    """
    Get current authentication status.
    """
    return auth_manager.get_status()


@router.post("/token")
async def set_tokens_endpoint(payload: SetTokenRequest):
    """
    Manually provide access_token, refresh_token, or project_id.
    """
    auth_manager.set_tokens(
        access_token=payload.access_token,
        refresh_token=payload.refresh_token,
        project_id=payload.project_id,
    )
    if auth_manager.access_token or auth_manager.refresh_token:
        try:
            await client.load_code_assist()
        except Exception as e:
            logger.warning(f"Could not load code assist: {e}")

    return {"status": "ok", "auth": auth_manager.get_status()}


@router.post("/refresh")
async def refresh_token_endpoint():
    """
    Force token refresh.
    """
    try:
        token = await auth_manager.refresh_access_token()
        await client.load_code_assist()
        return {
            "status": "ok",
            "access_token": token[:15] + "...",
            "expires_in": auth_manager.get_status().get("expires_in_seconds"),
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Token refresh failed: {e!s}",
        ) from e


@router.post("/logout")
async def logout_endpoint():
    """
    Clear stored credentials.
    """
    auth_manager.logout()
    return {"status": "logged_out"}
