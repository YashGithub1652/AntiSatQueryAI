"""
SatQuery AI — Session Store
============================
In-memory multi-turn session persistence.
Stores uploaded images, computed results, and conversation history
so follow-up queries can reuse cached artifacts.

Fills the 'Session memory/state persistence' audit gap from planning.

Design: Simple Python dict (demonstration). For production, replace with
Redis (config available in model_registry.yaml). The interface remains identical.
"""

import time
import uuid
import logging
from typing import Optional, Dict, Any, List

import numpy as np

logger = logging.getLogger(__name__)

# Session timeout: 2 hours (seconds)
SESSION_TTL = 7200


class Session:
    """Represents one user session (one set of uploaded images + conversation)."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.created_at = time.time()
        self.last_accessed = time.time()

        # Uploaded image data (persisted across turns)
        self.images: List[Dict[str, Any]] = []   # list of {array, metadata, modality, ...}
        self.image_count: int = 0

        # Cached computation results (avoid recomputing on follow-up)
        self.last_change_map: Optional[np.ndarray] = None
        self.last_task_type: Optional[str] = None
        self.last_result: Optional[Dict[str, Any]] = None

        # Multi-turn conversation history
        self.conversation: List[Dict[str, str]] = []
        # Format: [{"role": "user"|"assistant", "content": str, "task_type": str}]

        # Validation state
        self.is_validated: bool = False
        self.validation_result: Optional[Dict] = None

    def add_image(self, image_data: Dict[str, Any]) -> int:
        """Add an image to the session. Returns image index."""
        self.images.append(image_data)
        self.image_count = len(self.images)
        self.last_accessed = time.time()
        return self.image_count - 1

    def get_image(self, idx: int = 0) -> Optional[Dict[str, Any]]:
        """Get image by index."""
        if 0 <= idx < len(self.images):
            return self.images[idx]
        return None

    def add_turn(self, role: str, content: str, task_type: str = ""):
        """Append a conversation turn."""
        self.conversation.append({
            "role": role,
            "content": content,
            "task_type": task_type,
            "timestamp": time.time(),
        })
        self.last_accessed = time.time()

    def cache_result(
        self,
        result: Dict[str, Any],
        task_type: str,
        change_map: Optional[np.ndarray] = None,
    ):
        """Cache the latest computation result for follow-up reuse."""
        self.last_result = result
        self.last_task_type = task_type
        if change_map is not None:
            self.last_change_map = change_map
        self.last_accessed = time.time()

    def get_context_summary(self) -> str:
        """Build a context string from conversation history for LLM."""
        if not self.conversation:
            return ""
        recent = self.conversation[-4:]  # Last 2 turns
        lines = []
        for turn in recent:
            lines.append(f"{turn['role'].upper()}: {turn['content'][:200]}")
        return "\n".join(lines)

    def is_expired(self) -> bool:
        return (time.time() - self.last_accessed) > SESSION_TTL

    def to_summary_dict(self) -> Dict[str, Any]:
        """Summary for /api/v1/sessions/{id} endpoint."""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "image_count": self.image_count,
            "turn_count": len(self.conversation),
            "last_task_type": self.last_task_type,
            "is_validated": self.is_validated,
            "age_seconds": round(time.time() - self.created_at, 0),
            "images": [
                {
                    "idx": i,
                    "sensor": img.get("sensor", "unknown"),
                    "modality": img.get("modality", "unknown"),
                    "resolution_m": img.get("metadata", {}).get("resolution_m"),
                    "acquisition_date": img.get("metadata", {}).get("acquisition_date"),
                }
                for i, img in enumerate(self.images)
            ],
        }


class SessionStore:
    """
    In-memory session registry.
    Thread-safe via Python GIL (single-process deployment).
    For multi-process: replace backend with Redis using same interface.
    """

    def __init__(self):
        self._sessions: Dict[str, Session] = {}

    def create_session(self) -> Session:
        """Create a new session and return it."""
        session_id = str(uuid.uuid4())
        session = Session(session_id)
        self._sessions[session_id] = session
        logger.info(f"Session created: {session_id}")
        self._cleanup_expired()
        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        """Get an existing session. Returns None if not found or expired."""
        session = self._sessions.get(session_id)
        if session is None:
            return None
        if session.is_expired():
            logger.info(f"Session expired: {session_id}")
            del self._sessions[session_id]
            return None
        session.last_accessed = time.time()
        return session

    def get_or_create(self, session_id: Optional[str]) -> Session:
        """Get existing session or create new one."""
        if session_id:
            session = self.get_session(session_id)
            if session:
                return session
        return self.create_session()

    def delete_session(self, session_id: str):
        """Explicitly delete a session."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info(f"Session deleted: {session_id}")

    def list_active_sessions(self) -> List[Dict]:
        """List all active sessions (for admin/debug endpoint)."""
        self._cleanup_expired()
        return [s.to_summary_dict() for s in self._sessions.values()]

    def _cleanup_expired(self):
        """Remove all expired sessions."""
        expired = [
            sid for sid, s in self._sessions.items() if s.is_expired()
        ]
        for sid in expired:
            del self._sessions[sid]
            logger.debug(f"Expired session removed: {sid}")

    @property
    def active_count(self) -> int:
        return len(self._sessions)


# ──────────────────────────────────────────────────────────────
# MODULE-LEVEL SINGLETON
# ──────────────────────────────────────────────────────────────
_store: Optional[SessionStore] = None


def get_session_store() -> SessionStore:
    global _store
    if _store is None:
        _store = SessionStore()
    return _store
