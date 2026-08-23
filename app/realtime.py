"""Real-time event hub pushing live updates to connected WebSocket clients.

The dashboard connects to ``/ws`` and receives JSON events such as
``history.new``, ``history.clear`` and ``quotas.update`` instead of polling
REST endpoints. Events are published from anywhere (sync routes, streaming
loops, background tasks) via :meth:`RealtimeHub.publish`, which safely
marshals work onto the server's asyncio event loop.
"""

import asyncio
import contextlib
import logging
import threading
from collections.abc import Awaitable, Callable

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger("google_gate.realtime")

# Delay coalescing bursts of quota refresh requests into a single upstream fetch
QUOTA_REFRESH_DEBOUNCE_S = 2.0
# Periodic quota push so countdown timers stay fresh even without traffic
QUOTA_PUSH_INTERVAL_S = 10.0


class RealtimeHub:
    """Fan-out broadcaster distributing JSON events to WebSocket clients."""

    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._quota_collector: Callable[[], Awaitable[list[dict]]] | None = None
        self._quota_dirty = False
        self._quota_worker: asyncio.Task | None = None
        self._quota_fetch_lock = asyncio.Lock()
        self._pusher_task: asyncio.Task | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def startup(self) -> None:
        """Attach the running event loop and start background tasks."""
        self._loop = asyncio.get_running_loop()
        if self._pusher_task is None or self._pusher_task.done():
            self._pusher_task = asyncio.create_task(
                self._periodic_quota_pusher(), name="realtime-quota-pusher"
            )

    async def shutdown(self) -> None:
        """Cancel background tasks and drop every connected client."""
        for task in (self._pusher_task, self._quota_worker):
            if task and not task.done():
                task.cancel()
        self._pusher_task = None
        self._quota_worker = None
        with self._lock:
            clients = list(self._clients)
            self._clients.clear()
        for ws in clients:
            with contextlib.suppress(Exception):
                await ws.close()

    def set_quota_collector(
        self, collector: Callable[[], Awaitable[list[dict]]]
    ) -> None:
        """Register the async callable returning current quota groups."""
        self._quota_collector = collector

    # ------------------------------------------------------------------
    # Connections
    # ------------------------------------------------------------------
    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        with self._lock:
            self._clients.add(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        with self._lock:
            self._clients.discard(websocket)

    @property
    def client_count(self) -> int:
        with self._lock:
            return len(self._clients)

    # ------------------------------------------------------------------
    # Broadcasting
    # ------------------------------------------------------------------
    @staticmethod
    def _current_loop() -> asyncio.AbstractEventLoop | None:
        """Return the running loop if any, else ``None``."""
        try:
            return asyncio.get_running_loop()
        except RuntimeError:
            return None

    def publish(self, event: str, payload: dict | None = None) -> None:
        """Queue ``event`` for delivery to every connected client.

        Safe to call from any thread or sync context. Silently drops events
        when no usable event loop is available yet (e.g. early startup).
        """
        if event == "history.new":
            # Quota usage may have changed as a result of this request
            self.request_quota_refresh()

        message = {"type": event, "payload": payload or {}}
        running = self._current_loop()
        # The attached loop owns the WebSocket transports; only fall back to
        # a running loop when nothing was attached yet (e.g. early startup).
        loop = self._loop
        if loop is None or loop.is_closed():
            loop = running
        if loop is None or loop.is_closed():
            return

        if running is loop:
            loop.create_task(self.broadcast(message))
        else:
            # Loop closed between check and schedule - ignore
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(loop.create_task, self.broadcast(message))

    async def broadcast(self, message: dict) -> None:
        """Send a message to all clients, pruning disconnected sockets."""
        with self._lock:
            clients = list(self._clients)
        if not clients:
            return

        dead: list[WebSocket] = []
        for ws in clients:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

    # ------------------------------------------------------------------
    # Quota refresh scheduling
    # ------------------------------------------------------------------
    def request_quota_refresh(self) -> None:
        """Request a debounced quota fetch-and-push cycle."""
        self._quota_dirty = True
        running = self._current_loop()
        loop = self._loop
        if loop is None or loop.is_closed():
            loop = running
        if loop is None or loop.is_closed():
            return

        def _schedule() -> None:
            if self._quota_worker is not None and not self._quota_worker.done():
                return  # Worker already active; it will pick up the dirty flag
            self._quota_worker = loop.create_task(
                self._quota_worker_loop(), name="realtime-quota-worker"
            )

        if running is loop:
            _schedule()
        else:
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(_schedule)

    async def _quota_worker_loop(self) -> None:
        try:
            while self._quota_dirty:
                self._quota_dirty = False
                await asyncio.sleep(QUOTA_REFRESH_DEBOUNCE_S)
                await self.fetch_and_push_quotas()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.debug(f"Quota refresh worker stopped: {e}")
        finally:
            self._quota_worker = None

    async def _periodic_quota_pusher(self) -> None:
        while True:
            await asyncio.sleep(QUOTA_PUSH_INTERVAL_S)
            if self.client_count == 0:
                continue  # Nobody watching - skip upstream fetch entirely
            try:
                await self.fetch_and_push_quotas()
            except Exception as e:
                logger.debug(f"Periodic quota push failed: {e}")

    async def fetch_and_push_quotas(self) -> None:
        if self._quota_collector is None or self.client_count == 0:
            return
        async with self._quota_fetch_lock:
            try:
                groups = await self._quota_collector()
            except Exception as e:
                logger.debug(f"Live quota collection failed: {e}")
                return
        await self.broadcast({"type": "quotas.update", "payload": {"groups": groups}})


hub = RealtimeHub()

router = APIRouter(tags=["Realtime"])


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """
    Live update stream for the dashboard.

    Emits JSON frames: ``{"type": <event>, "payload": {...}}`` where events
    include ``hello``, ``history.new``, ``history.clear`` and
    ``quotas.update``. Clients may send literal ``ping`` frames and receive
    ``pong`` keep-alives.
    """
    await hub.connect(websocket)
    try:
        await websocket.send_json(
            {"type": "hello", "payload": {"clients": hub.client_count}}
        )
        while True:
            message = await websocket.receive_text()
            if message.strip().lower() == "ping":
                await websocket.send_json({"type": "pong", "payload": {}})
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.debug(f"WebSocket connection error: {e}")
    finally:
        hub.disconnect(websocket)
