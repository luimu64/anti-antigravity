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

@router.get("/login")
async def auth_login(
    request: Request,
    redirect: bool = Query(default=True, description="Redirect to Google OAuth page directly"),
    redirect_uri: Optional[str] = Query(default=None)
):
    """
    Initiate Google OAuth 2.0 flow.
    """
    callback_url = redirect_uri or str(request.url_for("auth_callback"))
    auth_url, state, _ = auth_manager.get_authorization_url(redirect_uri=callback_url)
    
    if redirect:
        return RedirectResponse(url=auth_url)
    return {
        "status": "ok",
        "auth_url": auth_url,
        "state": state,
        "redirect_uri": callback_url
    }

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
