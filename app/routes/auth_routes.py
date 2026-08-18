import logging
from typing import Optional
from fastapi import APIRouter, Request, HTTPException, Query, status
from fastapi.responses import RedirectResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

from app.config import REDIRECT_URI
from app.auth import auth_manager
from app.client import client

logger = logging.getLogger("agy_to_api.auth_routes")
router = APIRouter(prefix="/auth", tags=["Auth"])

class SetTokenRequest(BaseModel):
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    project_id: Optional[str] = None

class ExchangeCodeRequest(BaseModel):
    code_or_url: str
    state: Optional[str] = None
    redirect_uri: Optional[str] = None

@router.post("/exchange")
async def exchange_code_endpoint(payload: ExchangeCodeRequest, request: Request):
    """
    Manually exchange an authorization code, full callback URL, or refresh token.
    Works seamlessly for remote/LAN/headless setups.
    """
    input_str = payload.code_or_url.strip()
    
    # If user pasted a refresh token directly
    if input_str.startswith("1//") or input_str.startswith("ya29."):
        auth_manager.set_tokens(refresh_token=input_str if input_str.startswith("1//") else None,
                                access_token=input_str if input_str.startswith("ya29.") else None)
        try:
            await client.load_code_assist()
        except Exception as e:
            logger.warning(f"Could not load code assist: {e}")
        return {
            "status": "ok",
            "message": "Token saved successfully!",
            "auth": auth_manager.get_status()
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
            redirect_uri = f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip("/") or "http://localhost:8085"

    try:
        tokens = await auth_manager.exchange_code(code=code, redirect_uri=redirect_uri, state=state)
        try:
            await client.load_code_assist()
        except Exception as e:
            logger.warning(f"Could not load code assist: {e}")
        
        return {
            "status": "ok",
            "message": "Authenticated successfully!",
            "auth": auth_manager.get_status()
        }
    except Exception as e:
        logger.error(f"Manual exchange failed: {e}")
        # Try fallback loopback redirect uri
        if redirect_uri != "http://localhost:8085":
            try:
                await auth_manager.exchange_code(code=code, redirect_uri="http://localhost:8085", state=state)
                await client.load_code_assist()
                return {
                    "status": "ok",
                    "message": "Authenticated successfully with loopback URI!",
                    "auth": auth_manager.get_status()
                }
            except Exception:
                pass
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Authorization code exchange failed: {str(e)}"
        )

@router.get("/login")
async def auth_login(
    request: Request,
    redirect: bool = Query(default=True, description="Redirect to Google OAuth page directly"),
    redirect_uri: Optional[str] = Query(default=None)
):
    """
    Initiate Google OAuth 2.0 flow.
    For localhost connections: auto-redirects directly.
    For remote/LAN IP connections: uses Google-approved loopback redirect_uri to prevent Error 400.
    """
    hostname = request.url.hostname or "localhost"
    is_localhost = hostname in ("localhost", "127.0.0.1", "::1")

    # For remote IPs, always use Google-approved loopback URI http://localhost:8085
    callback_url = redirect_uri or ("http://localhost:8085" if not is_localhost else str(request.url_for("auth_callback")))
    auth_url, state, _ = auth_manager.get_authorization_url(redirect_uri=callback_url)
    
    if is_localhost and redirect:
        return RedirectResponse(url=auth_url)

    # For remote / LAN IPs, render step-by-step remote auth page
    return HTMLResponse(
        content=f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
          <meta charset="utf-8">
          <meta name="viewport" content="width=device-width, initial-scale=1.0">
          <title>Google Sign-In - Antigravity API</title>
          <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@picocss/pico@2/css/pico.green.min.css">
          <style>
            #toast-container {{
              position: fixed;
              top: 1.25rem;
              left: 50%;
              transform: translateX(-50%);
              z-index: 99999;
              pointer-events: none;
              display: flex;
              flex-direction: column;
              align-items: center;
              gap: 0.5rem;
            }}
            .toast {{
              pointer-events: auto;
              display: inline-flex;
              align-items: center;
              gap: 0.6rem;
              padding: 0.55rem 1.15rem;
              background: var(--pico-card-background-color, #1b232c);
              color: var(--pico-color, #e2e8f0);
              border: 1px solid var(--pico-border-color, #334155);
              border-radius: 9999px;
              box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4), 0 8px 10px -6px rgba(0, 0, 0, 0.3);
              font-size: 0.85rem;
              font-weight: 500;
              opacity: 0;
              transform: translateY(-8px) scale(0.95);
              transition: opacity 0.25s cubic-bezier(0.16, 1, 0.3, 1), transform 0.25s cubic-bezier(0.16, 1, 0.3, 1);
            }}
            .toast.show {{
              opacity: 1;
              transform: translateY(0) scale(1);
            }}
            .toast.hide {{
              opacity: 0;
              transform: translateY(-10px) scale(0.95);
            }}
            .toast.toast-error {{
              border-color: #ef4444;
            }}
            .toast-icon {{
              display: inline-flex;
              align-items: center;
              color: var(--pico-primary, #2ecc71);
            }}
            .toast-icon.toast-icon-error {{
              color: #ef4444;
            }}
            form[role="group"], [role="group"] {{
              display: flex !important;
              align-items: stretch !important;
              margin-bottom: 1rem;
            }}
            form[role="group"] input, form[role="group"] button, [role="group"] input, [role="group"] button {{
              height: 2.75rem !important;
              box-sizing: border-box !important;
              margin: 0 !important;
            }}
            form[role="group"] button, [role="group"] button {{
              white-space: nowrap !important;
              flex-shrink: 0 !important;
              padding: 0 1.25rem !important;
            }}
          </style>
        </head>
        <body>
          <div id="toast-container" aria-live="polite" aria-atomic="true"></div>
          <main class="container">
            <article>
              <header>
                <nav>
                  <ul><li><strong>Connect Google Account</strong></li></ul>
                  <ul><li><small>Remote Server Setup</small></li></ul>
                </nav>
              </header>

              <p>
                Because this bridge is deployed on a remote server/IP (<code>{hostname}</code>), Google requires authenticating via standard loopback.
              </p>

              <article>
                <header><strong>Step 1: Open Google Sign-In</strong></header>
                <p><small>Click below to authorize with your Google Account in a new tab:</small></p>
                <a href="{auth_url}" target="_blank" role="button">Authorize Google Account (New Tab)</a>
              </article>

              <article>
                <header><strong>Step 2: Paste Redirect URL or Code</strong></header>
                <p><small>After approving, Google redirects to <code>http://localhost:8085/?code=...</code>. Copy that address bar URL (or the code) and paste it here:</small></p>
                <form onsubmit="event.preventDefault(); submitCode();" role="group">
                  <input type="text" id="callback-input" placeholder="http://localhost:8085/?state=...&code=4/0A...">
                  <button type="submit">Connect</button>
                </form>
              </article>

              <footer>
                <a href="/" role="button" class="secondary outline">Back to Dashboard</a>
              </footer>
            </article>
          </main>

          <script>
            function showToast(message, type = 'success') {{
              const container = document.getElementById('toast-container');
              if (!container) return;

              const toast = document.createElement('div');
              toast.className = `toast ${{type === 'error' ? 'toast-error' : ''}}`;
              const iconHtml = type === 'error'
                ? `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>`
                : `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>`;

              toast.innerHTML = `
                <span class="toast-icon ${{type === 'error' ? 'toast-icon-error' : ''}}">
                  ${{iconHtml}}
                </span>
                <span>${{message}}</span>
              `;
              container.appendChild(toast);

              requestAnimationFrame(() => {{
                toast.classList.add('show');
              }});

              setTimeout(() => {{
                toast.classList.remove('show');
                toast.classList.add('hide');
                setTimeout(() => {{
                  toast.remove();
                }}, 250);
              }}, 2400);
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
    code: Optional[str] = Query(default=None),
    state: Optional[str] = Query(default=None),
    error: Optional[str] = Query(default=None)
):
    """
    Handle OAuth callback from Google.
    """
    if error:
        logger.error(f"OAuth callback returned error: {error}")
        return HTMLResponse(
            content=f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
              <meta charset="utf-8">
              <meta name="viewport" content="width=device-width, initial-scale=1.0">
              <title>OAuth Error - Antigravity API</title>
              <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@picocss/pico@2/css/pico.green.min.css">
            </head>
            <body>
              <main class="container">
                <article>
                  <header><strong>Authentication Failed</strong></header>
                  <p>Google returned error: <code>{error}</code></p>
                  <footer>
                    <a href="/auth/login" role="button">Try Again</a>
                  </footer>
                </article>
              </main>
            </body>
            </html>
            """,
            status_code=400
        )

    if not code:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing authorization code."
        )

    callback_url = str(request.url).split("?")[0]
    try:
        await auth_manager.exchange_code(code=code, redirect_uri=callback_url, state=state)
        # Fetch and sync user/project metadata
        try:
            await client.load_code_assist()
        except Exception as e:
            logger.warning(f"Could not load code assist after OAuth exchange: {e}")

        return HTMLResponse(
            content=f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
              <meta charset="utf-8">
              <meta name="viewport" content="width=device-width, initial-scale=1.0">
              <title>Login Successful - Antigravity API</title>
              <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@picocss/pico@2/css/pico.green.min.css">
              <meta http-equiv="refresh" content="3;url=/">
            </head>
            <body>
              <main class="container">
                <article>
                  <header>
                    <ins>● Connected</ins>
                    <h2>Authentication Successful!</h2>
                  </header>
                  <p>Your Google account has been connected to Antigravity API.</p>
                  <p>Project ID: <strong>{auth_manager.project_id or 'Auto-discovered'}</strong></p>
                  <p>Account: <strong>{auth_manager.user_email or 'Authenticated'}</strong></p>
                  <footer>
                    <a href="/" role="button">Go to Dashboard</a>
                    <p><small>Redirecting in 3 seconds...</small></p>
                  </footer>
                </article>
              </main>
            </body>
            </html>
            """
        )
    except Exception as e:
        logger.error(f"Failed to complete OAuth callback: {e}")
        return HTMLResponse(
            content=f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
              <meta charset="utf-8">
              <meta name="viewport" content="width=device-width, initial-scale=1.0">
              <title>Authentication Error - Antigravity API</title>
              <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@picocss/pico@2/css/pico.green.min.css">
            </head>
            <body>
              <main class="container">
                <article>
                  <header><strong>Authentication Error</strong></header>
                  <p><del>{str(e)}</del></p>
                  <footer>
                    <a href="/auth/login" role="button">Try Again</a>
                  </footer>
                </article>
              </main>
            </body>
            </html>
            """,
            status_code=500
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
        project_id=payload.project_id
    )
    if auth_manager.access_token or auth_manager.refresh_token:
        try:
            await client.load_code_assist()
        except Exception as e:
            logger.warning(f"Could not load code assist: {e}")

    return {
        "status": "ok",
        "auth": auth_manager.get_status()
    }

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
            "expires_in": auth_manager.get_status().get("expires_in_seconds")
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Token refresh failed: {str(e)}"
        )

@router.post("/logout")
async def logout_endpoint():
    """
    Clear stored credentials.
    """
    auth_manager.logout()
    return {"status": "logged_out"}
