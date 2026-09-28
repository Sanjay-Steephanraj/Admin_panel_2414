import json
import logging
from cache.query_cache import _get_redis

logger = logging.getLogger(__name__)

SESSION_TTL_SECONDS = 7200  # 2 hours
MAX_TURNS = 5

class SessionStore:
    def __init__(self):
        from config.settings import get_settings
        self.prefix = getattr(get_settings(), "redis_key_prefix", "") or ""
        # Keep clarification state available for the current process when a
        # local Redis instance is disabled or temporarily unreachable. Redis
        # remains the durable/shared store when it is available.
        self._local_pending: dict[str, dict] = {}

    def _key(self, session_id: str) -> str:
        return f"{self.prefix}session:{session_id}:history"

    def load(self, session_id: str) -> list[dict]:
        if not session_id:
            return []
        r = _get_redis()
        if r is None:
            return []
        try:
            raw = r.get(self._key(session_id))
            if raw:
                # Refresh TTL on load to keep session alive
                r.expire(self._key(session_id), SESSION_TTL_SECONDS)
                return json.loads(raw)
        except Exception as e:
            logger.warning(f"Session load error: {e}")
        return []

    def append(self, session_id: str, turn: dict) -> None:
        if not session_id:
            return
        r = _get_redis()
        if r is None:
            return
        try:
            history = self.load(session_id)
            history.append(turn)
            
            # Slide window
            if len(history) > MAX_TURNS:
                history = history[-MAX_TURNS:]
                
            r.setex(
                name=self._key(session_id),
                time=SESSION_TTL_SECONDS,
                value=json.dumps(history)
            )
            logger.info(f"Session history appended | session={session_id}")
        except Exception as e:
            logger.warning(f"Session append error: {e}")

    def clear(self, session_id: str) -> None:
        if not session_id:
            return
        self._local_pending.pop(session_id, None)
        r = _get_redis()
        if r is None:
            return
        try:
            r.delete(self._key(session_id))
            r.delete(self._key(session_id) + ":pending")
            logger.info(f"Session history cleared | session={session_id}")
        except Exception as e:
            logger.warning(f"Session clear error: {e}")

    def load_pending(self, session_id: str) -> dict | None:
        local_pending = self._local_pending.get(session_id)
        r = _get_redis()
        if r is None:
            return local_pending
        try:
            raw = r.get(self._key(session_id) + ":pending")
            return json.loads(raw) if raw else local_pending
        except Exception:
            return local_pending

    def save_pending(self, session_id: str, pending: dict | None) -> None:
        if pending is None:
            self._local_pending.pop(session_id, None)
        else:
            self._local_pending[session_id] = pending
        r = _get_redis()
        if r is None:
            return
        try:
            key = self._key(session_id) + ":pending"
            if pending is None:
                r.delete(key)
            else:
                r.setex(name=key, time=SESSION_TTL_SECONDS, value=json.dumps(pending))
        except Exception as e:
            logger.warning(f"Pending clarification could not be saved: {e}")

session_store = SessionStore()
