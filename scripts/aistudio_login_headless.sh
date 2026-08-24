#!/usr/bin/env bash
# One-time AI Studio Google login on a HEADLESS host (no display).
#
# Runs the login browser inside Xvfb, shares it over VNC on 127.0.0.1:5905
# (+ noVNC web client on 127.0.0.1:6081), and waits until you finish.
# Reach it from your machine:
#   ssh -L 6081:127.0.0.1:6081 <host>
# then open http://localhost:6081/vnc.html in your local browser,
# sign in to Google, and press Enter in the shell running this script.
#
# Deps (Debian/Ubuntu image): xvfb x11vnc websockify python3
set -euo pipefail
cd "$(dirname "$0")/.."

DISPLAY_NUM=${DISPLAY_NUM:-99}
VNC_PORT=${VNC_PORT:-5905}
NOVNC_PORT=${NOVNC_PORT:-6081}

PY="${PYTHON:-.venv/bin/python}"

command -v Xvfb >/dev/null || { echo "apt-get install -y xvfb x11vnc websockify"; exit 1; }

Xvfb :$DISPLAY_NUM -screen 0 1600x900x24 &
XVFB_PID=$!
trap 'kill $XVFB_PID $X11VNC_PID $WEBSOCKIFY_PID 2>/dev/null || true' EXIT
sleep 1

DISPLAY=:$DISPLAY_NUM "$PY" scripts/aistudio_login.py &
LOGIN_PID=$!

x11vnc -display :$DISPLAY_NUM -localhost -rfbport $VNC_PORT -nopw -quiet -forever &
X11VNC_PID=$!

websockify --web /usr/share/novnc $NOVNC_PORT localhost:$VNC_PORT &
WEBSOCKIFY_PID=$!

echo
echo ">>> ssh -L $NOVNC_PORT:127.0.0.1:$NOVNC_PORT <this-host>"
echo ">>> open http://localhost:$NOVNC_PORT/vnc.html"
echo ">>> sign in to Google, then press Enter back in the login shell"
echo

wait $LOGIN_PID
