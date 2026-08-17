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
        <html>
        <head>
          <meta charset="utf-8">
          <title>Google Sign-In - Antigravity API</title>
          <link rel="stylesheet" href="/static/style.css">
        </head>
        <body style="display: flex; justify-content: center; align-items: center; min-height: 100vh; padding: 20px;">
          <div class="card" style="max-width: 580px; width: 100%;">
            <div class="card-header">
              <h2 class="card-title">Connect Google Account</h2>
              <span style="font-size: 12px; color: var(--accent-primary);">Remote Server Setup</span>
            </div>

            <div style="font-size: 13px; color: var(--text-secondary); line-height: 1.6; margin-bottom: 20px;">
              Because this bridge is deployed on a remote server/IP (<code>{hostname}</code>), Google requires authenticating via standard loopback.
            </div>

            <div style="display: flex; flex-direction: column; gap: 16px;">
              <div style="background: var(--bg-surface-elevated); padding: 14px; border-radius: var(--radius-md); border: 1px solid var(--border-subtle);">
                <div style="font-weight: 600; font-size: 13px; margin-bottom: 6px;">Step 1: Open Google Sign-In</div>
                <div style="font-size: 12px; color: var(--text-muted); margin-bottom: 10px;">Click the button below to authorize with your Google Account in a new tab:</div>
                <a href="{auth_url}" target="_blank" class="btn btn-primary btn-full">
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" fill="#4285F4"/><path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853"/><path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.06H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.94l2.85-2.22.81-.63z" fill="#FBBC05"/><path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.06l3.66 2.84c.87-2.6 3.3-4.52 6.16-4.52z" fill="#EA4335"/></svg>
                  Authorize Google Account (New Tab)
                </a>
              </div>

              <div style="background: var(--bg-surface-elevated); padding: 14px; border-radius: var(--radius-md); border: 1px solid var(--border-subtle);">
                <div style="font-weight: 600; font-size: 13px; margin-bottom: 6px;">Step 2: Paste Redirect URL or Code</div>
                <div style="font-size: 12px; color: var(--text-muted); margin-bottom: 10px;">After approving, Google redirects your browser to <code>http://localhost:8085/?code=...</code>. Copy that address bar URL (or the code) and paste it here:</div>
                <input type="text" id="callback-input" class="select-control" style="width: 100%; margin-bottom: 10px;" placeholder="http://localhost:8085/?state=...&code=4/0A...">
                <button onclick="submitCode()" class="btn btn-primary btn-full">Complete Connection</button>
              </div>

              <div style="text-align: center;">
                <a href="/" class="meta-label" style="text-decoration: underline; font-size: 12px;">Back to Dashboard</a>
              </div>
            </div>
          </div>

          <script>
            async function submitCode() {{
              const val = document.getElementById('callback-input').value.trim();
              if (!val) {{
                alert('Please paste the redirect URL or code.');
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
                  alert('Successfully connected to Antigravity!');
                  window.location.href = '/';
                }} else {{
                  alert('Error: ' + (data.detail || JSON.stringify(data)));
                }}
              }} catch (e) {{
                alert('Network error: ' + e.message);
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
            <html>
            <head><title>OAuth Error</title><link rel="stylesheet" href="/static/style.css"></head>
            <body>
              <div class="card error">
                <h2>Authentication Failed</h2>
                <p>Google returned error: <code>{error}</code></p>
                <a href="/auth/login" class="btn">Try Again</a>
              </div>
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
            <html>
            <head>
              <meta charset="utf-8">
              <title>Login Successful - Antigravity API</title>
              <link rel="stylesheet" href="/static/style.css">
              <meta http-equiv="refresh" content="3;url=/">
            </head>
            <body>
              <div class="card success">
                <div class="badge">✓ Connected</div>
                <h2>Authentication Successful!</h2>
                <p>Your Google account has been connected to Antigravity API.</p>
                <p>Project ID: <strong>{auth_manager.project_id or 'Auto-discovered'}</strong></p>
                <p>Account: <strong>{auth_manager.user_email or 'Authenticated'}</strong></p>
                <div style="margin-top: 24px;">
                  <a href="/" class="btn btn-primary">Go to Dashboard</a>
                </div>
                <p class="subtext">Redirecting in 3 seconds...</p>
              </div>
            </body>
            </html>
            """
        )
    except Exception as e:
        logger.error(f"Failed to complete OAuth callback: {e}")
        return HTMLResponse(
            content=f"""
            <html>
            <head><title>Authentication Error</title><link rel="stylesheet" href="/static/style.css"></head>
            <body>
              <div class="card error">
                <h2>Authentication Error</h2>
                <p>{str(e)}</p>
                <a href="/auth/login" class="btn">Try Again</a>
              </div>
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
