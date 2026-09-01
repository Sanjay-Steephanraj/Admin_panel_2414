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
        r = _get_redis()
        if r is None:
            return
        try:
            r.delete(self._key(session_id))
            logger.info(f"Session history cleared | session={session_id}")
        except Exception as e:
            logger.warning(f"Session clear error: {e}")

session_store = SessionStore()
