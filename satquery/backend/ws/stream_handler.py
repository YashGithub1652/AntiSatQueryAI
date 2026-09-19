"""
SatQuery AI — WebSocket Stream Handler
========================================
Broadcasts real-time agent trace steps to the frontend.
Each step appears live as the agent processes the query.

The frontend connects to /ws/{session_id} and receives:
  {"node": "NODE_3_COREGISTRATION", "status": "running", "detail": "..."}
  {"node": "NODE_6_EXECUTOR", "status": "running", "detail": "Running ChangeFormer..."}
  {"node": "NODE_6_EXECUTOR", "status": "done", "detail": "✓ Completed in 3.2s"}
"""

import asyncio
import json
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class StreamManager:
    """Manages WebSocket connections per session."""

    def __init__(self):
        # session_id → WebSocket connection
        self._connections: Dict[str, object] = {}

    def register(self, session_id: str, websocket):
        self._connections[session_id] = websocket
        logger.info(f"WebSocket registered for session {session_id}")

    def unregister(self, session_id: str):
        self._connections.pop(session_id, None)
        logger.info(f"WebSocket unregistered for session {session_id}")

    async def send(self, session_id: str, data: dict):
        """Send a JSON message to the WebSocket client for this session."""
        ws = self._connections.get(session_id)
        if ws:
            try:
                await ws.send_text(json.dumps(data))
            except Exception as e:
                logger.warning(f"WebSocket send failed for {session_id}: {e}")
                self.unregister(session_id)

    def get_callback(self, session_id: str):
        """
        Returns a synchronous callback function for agent nodes.
        Agent nodes call: callback(node_name, status, detail)
        The callback enqueues the message for async sending.
        """
        manager = self

        def callback(node_name: str, status: str, detail: str):
            message = {
                "type": "trace",
                "node": node_name,
                "status": status,
                "detail": detail,
            }
            # Schedule the async send from sync context
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.ensure_future(manager.send(session_id, message))
                else:
                    loop.run_until_complete(manager.send(session_id, message))
            except Exception as e:
                logger.debug(f"WebSocket callback scheduling: {e}")

        return callback

    @property
    def active_connections(self) -> int:
        return len(self._connections)


# Module-level singleton
_stream_manager: Optional[StreamManager] = None


def get_stream_manager() -> StreamManager:
    global _stream_manager
    if _stream_manager is None:
        _stream_manager = StreamManager()
    return _stream_manager
