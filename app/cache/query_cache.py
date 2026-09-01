"""
cache/query_cache.py
─────────────────────
Redis-backed query cache.

Fixes applied:
  1. Cache key is now an MD5 hash of the *normalized* question string,
     so two questions that normalize to the same form share one entry.
  2. CACHE_TTL_SECONDS is enforced on EVERY key via Redis SETEX —
     stale summaries can never persist indefinitely.
  3. Similarity matching (Jaccard) is kept as a second lookup tier so
     near-duplicate questions still get cache hits.
"""

import re
import json
import time
import hashlib
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
CACHE_TTL_SECONDS    = 3600   # 1 hour — enforced on every key
CACHE_MAX_SIZE       = 200
SIMILARITY_THRESHOLD = 0.82


# ─────────────────────────────────────────────
#  CACHE ENTRY
# ─────────────────────────────────────────────
@dataclass
class CacheEntry:
    sql:        str
    summary:    str
    row_count:  int
    created_at: float = field(default_factory=time.time)


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def _normalize(question: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    q = question.lower().strip()
    q = re.sub(r"[^\w\s]", "", q)
    q = re.sub(r"\s+", " ", q)
    return q


def _cache_key(normalized: str) -> str:
    """
    MD5 hash of the normalized question string.
    Two semantically identical questions that normalize to the same form
    will produce the same key and share a single cache entry.
    """
    return hashlib.md5(normalized.encode()).hexdigest()

def _extract_numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+", text))


def _tokenize(question: str) -> set[str]:
    return set(_normalize(question).split())


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ─────────────────────────────────────────────
#  REDIS CLIENT
# ─────────────────────────────────────────────
def _get_redis():
    try:
        import redis
        from config.settings import get_settings

        s = get_settings()

        redis_password = getattr(s, "redis_password", "") or ""
        use_ssl        = (
            "amazonaws.com" in s.redis_host
            or "cache.amazonaws.com" in s.redis_host
        )

        client = redis.Redis(
            host=s.redis_host,
            port=s.redis_port,
            db=s.redis_db,
            password=redis_password if redis_password.strip() else None,
            decode_responses=True,
            socket_connect_timeout=3,
            socket_timeout=3,
            ssl=use_ssl,
        )
        client.ping()
        return client

    except Exception as e:
        logger.warning(f"Redis unavailable: {e} → cache skipped")
        return None


# ─────────────────────────────────────────────
#  REDIS QUERY CACHE
# ─────────────────────────────────────────────
class RedisQueryCache:

    def __init__(self):
        from config.settings import get_settings
        s = get_settings()

        self.prefix     = getattr(s, "redis_key_prefix", "") or ""
        self.base_prefix = f"{self.prefix}nlsql:"
        # index stores normalized question strings (for Jaccard lookup)
        self.index_key  = f"{self.base_prefix}index"

        self._hits   = 0
        self._misses = 0
        self._skips  = 0

    def _entry_key(self, normalized: str) -> str:
        """Redis key for a cache entry — uses MD5 hash of normalized question."""
        return f"{self.base_prefix}entry:{_cache_key(normalized)}"

    # ── READ ─────────────────────────────────────────────────────────────────
    def get(self, question: str) -> Optional[CacheEntry]:
        r = _get_redis()
        if r is None:
            self._skips += 1
            return None

        norm      = _normalize(question)
        entry_key = self._entry_key(norm)

        try:
            # ── Exact (hash) match ───────────────────────────────────────────
            raw = r.get(entry_key)
            if raw:
                data = json.loads(raw)
                self._hits += 1
                logger.info(f"Redis HIT (exact hash) | q={question!r}")
                return CacheEntry(**data)

            # ── Similarity match against index ───────────────────────────────
            all_norms = r.smembers(self.index_key)
            if not all_norms:
                self._misses += 1
                return None

            tokens_q   = _tokenize(question)
            best_score = 0.0
            best_norm  = None

            for cached_norm in all_norms:
                # Prune expired entries from index
                if not r.exists(self._entry_key(cached_norm)):
                    r.srem(self.index_key, cached_norm)
                    continue

                # STRICT MATCH: If the questions have different numeric IDs, they are NOT the same question
                nums_q = _extract_numbers(norm)
                nums_c = _extract_numbers(cached_norm)
                if nums_q != nums_c:
                    continue

                tokens_c = set(cached_norm.split())   # already normalized
                score    = _jaccard(tokens_q, tokens_c)

                if score > best_score:
                    best_score = score
                    best_norm  = cached_norm

            if best_norm and best_score >= SIMILARITY_THRESHOLD:
                raw = r.get(self._entry_key(best_norm))
                if raw:
                    data = json.loads(raw)
                    self._hits += 1
                    logger.info(f"Redis HIT (similar={best_score:.2f}) | q={question!r}")
                    return CacheEntry(**data)

            self._misses += 1
            logger.info(f"Redis MISS | q={question!r}")
            return None

        except Exception as e:
            logger.warning(f"Redis GET error: {e}")
            self._skips += 1
            return None

    # ── WRITE ────────────────────────────────────────────────────────────────
    def set(self, question: str, sql: str, summary: str, row_count: int) -> bool:
        """
        Stores a cache entry. Returns True if written, False if skipped
        (Redis unavailable) or if an error occurred.
        """
        r = _get_redis()
        if r is None:
            return False

        norm      = _normalize(question)
        entry_key = self._entry_key(norm)

        try:
            # ── Max size enforcement ─────────────────────────────────────────
            size = r.scard(self.index_key)
            if size >= CACHE_MAX_SIZE:
                oldest = next(iter(r.smembers(self.index_key)), None)
                if oldest:
                    r.delete(self._entry_key(oldest))
                    r.srem(self.index_key, oldest)

            entry = CacheEntry(sql=sql, summary=summary, row_count=row_count)

            # SETEX enforces TTL on every entry — no stale entries possible
            r.setex(
                name=entry_key,
                time=CACHE_TTL_SECONDS,
                value=json.dumps(entry.__dict__),
            )

            # Store normalized question string in index for similarity lookups
            r.sadd(self.index_key, norm)
            r.expire(self.index_key, CACHE_TTL_SECONDS * 2)

            logger.info(f"Redis SET (TTL={CACHE_TTL_SECONDS}s) | q={question!r}")
            return True

        except Exception as e:
            logger.warning(f"Redis SET error: {e}")
            return False

    # ── INVALIDATE ───────────────────────────────────────────────────────────
    def invalidate(self, question: str) -> None:
        r = _get_redis()
        if r is None:
            return
        try:
            norm = _normalize(question)
            r.delete(self._entry_key(norm))
            r.srem(self.index_key, norm)
        except Exception as e:
            logger.warning(f"Redis invalidate error: {e}")

    # ── CLEAR ALL ────────────────────────────────────────────────────────────
    def clear(self) -> None:
        r = _get_redis()
        if r is None:
            return
        try:
            norms = r.smembers(self.index_key)
            if norms:
                pipeline = r.pipeline()
                for norm in norms:
                    pipeline.delete(self._entry_key(norm))
                pipeline.delete(self.index_key)
                pipeline.execute()
        except Exception as e:
            logger.warning(f"Redis clear error: {e}")

    # ── STATS ────────────────────────────────────────────────────────────────
    @property
    def stats(self) -> dict:
        from config.settings import get_settings
        s = get_settings()

        r        = _get_redis()
        redis_up = r is not None

        size = 0
        if redis_up:
            try:
                size = r.scard(self.index_key)
            except Exception:
                redis_up = False

        total = self._hits + self._misses + self._skips

        return {
            "backend":     "redis",
            "redis_up":    redis_up,
            "redis_host":  s.redis_host,
            "redis_port":  s.redis_port,
            "key_prefix":  self.prefix,
            "size":        size,
            "max_size":    CACHE_MAX_SIZE,
            "ttl_seconds": CACHE_TTL_SECONDS,
            "hits":        self._hits,
            "misses":      self._misses,
            "skips":       self._skips,
            "hit_rate":    round(self._hits / max(total, 1) * 100, 1),
        }


# Singleton
query_cache = RedisQueryCache()