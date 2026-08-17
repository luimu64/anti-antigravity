import os
import sys
import argparse
import asyncio
import logging
from pathlib import Path
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from app.config import (
    SERVER_HOST,
    SERVER_PORT,
    API_KEY,
    BASE_DIR,
    DATA_DIR
)
from app.auth import auth_manager
from app.client import client
from app.routes.openai import router as openai_router
from app.routes.auth_routes import router as auth_router
from app.routes.dashboard import router as dashboard_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("agy_to_api")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup & shutdown lifespan events.
    """
    logger.info("Initializing Antigravity to OpenAI API bridge...")
    
    # Check authentication state
    if auth_manager.access_token or auth_manager.refresh_token:
        try:
            logger.info("Syncing user metadata and companion project ID...")
            await client.load_code_assist()
            logger.info(f"Connected as {auth_manager.user_email or 'User'} (Project: {auth_manager.project_id}, Tier: {auth_manager.tier_name})")
        except Exception as e:
            logger.warning(f"Could not initialize code assist during startup: {e}")
    else:
        logger.warning("No authentication tokens found. Please log in via browser at /auth/login or run with --login flag.")
        
    yield
    logger.info("Shutting down Antigravity to OpenAI API bridge...")

app = FastAPI(
    title="Antigravity OpenAI API Bridge",
    description="Maps Google Antigravity internal API into standard OpenAI API schema.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for all clients (Cursor, Continue, Open WebUI, Web browsers)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static directory
static_dir = Path(__file__).resolve().parent / "app" / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Mount API Routers
app.include_router(dashboard_router)
app.include_router(openai_router)
app.include_router(auth_router)

def cli_login():
    """
    CLI interactive login flow.
    """
    print("\n" + "="*60)
    print("      Antigravity Google OAuth Login      ")
    print("="*60)
    
    redirect_uri = f"http://localhost:{SERVER_PORT}/auth/callback"
    auth_url, state, verifier = auth_manager.get_authorization_url(redirect_uri=redirect_uri)
    
    print("\n1. Open this URL in your web browser:")
    print(f"\n   {auth_url}\n")
    print("2. Authorize with your Google Account.")
    print("3. After approving, you will be redirected to the callback URL.")
    print("   If running on a remote server or container, paste the full redirected URL")
    print("   or the 'code' parameter below:\n")
    
    code_input = input("Enter redirected URL or auth code: ").strip()
    if not code_input:
        print("Login cancelled.")
        return

    # Extract code from URL if pasted full URL
    code = code_input
    if "code=" in code_input:
        import urllib.parse
        parsed = urllib.parse.urlparse(code_input)
        params = urllib.parse.parse_qs(parsed.query)
        code = params.get("code", [code_input])[0]
        if "state" in params:
            state = params.get("state")[0]

    async def do_exchange():
        print("Exchanging authorization code for tokens...")
        await auth_manager.exchange_code(
            code=code,
            redirect_uri=redirect_uri,
            state=state,
            code_verifier=verifier
        )
        print("Fetching companion project ID...")
        await client.load_code_assist()
        print("\n" + "="*60)
        print("✓ LOGIN SUCCESSFUL!")
        print(f"  Account:    {auth_manager.user_email or 'Authenticated'}")
        print(f"  Project ID: {auth_manager.project_id}")
        print(f"  Tier:       {auth_manager.tier_name}")
        print("="*60 + "\n")

    asyncio.run(do_exchange())

def show_status():
    """
    Print status in terminal.
    """
    status = auth_manager.get_status()
    print("\n" + "="*50)
    print("  Antigravity OpenAI Bridge Status")
    print("="*50)
    print(f"  Authenticated: {status['authenticated']}")
    print(f"  User Email:    {status['user_email']}")
    print(f"  Project ID:    {status['project_id']}")
    print(f"  Tier Name:     {status['tier_name']}")
    print(f"  Has Refresh:   {status['has_refresh_token']}")
    print(f"  Expires in:    {status['expires_in_seconds']} seconds")
    print("="*50 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Antigravity OpenAI API Bridge Server")
    parser.add_argument("--login", action="store_true", help="Start interactive Google OAuth login in terminal")
    parser.add_argument("--status", action="store_true", help="Display current authentication status")
    parser.add_argument("--host", type=str, default=SERVER_HOST, help="Server bind host")
    parser.add_argument("--port", type=int, default=SERVER_PORT, help="Server bind port")
    parser.add_argument("--key", type=str, default=API_KEY, help="Optional API key required for client requests")
    
    args = parser.parse_args()

    if args.login:
        cli_login()
        return

    if args.status:
        show_status()
        return

    display_host = args.host
    if display_host == "0.0.0.0":
        env_host = os.getenv("HOST")
        display_host = env_host if (env_host and env_host != "0.0.0.0") else "localhost"

    print("\n" + "="*60)
    print(f"🚀 Starting Antigravity OpenAI API Bridge on http://{display_host}:{args.port}")
    print(f"📊 Web Dashboard: http://{display_host}:{args.port}/")
    print(f"🤖 OpenAI Endpoint: http://{display_host}:{args.port}/v1/chat/completions")
    print(f"🔐 OAuth Login: http://{display_host}:{args.port}/auth/login")
    print("="*60 + "\n")

    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        reload=False
    )

if __name__ == "__main__":
    main()
