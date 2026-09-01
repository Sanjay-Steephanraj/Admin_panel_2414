"""
memory/conversation_memory.py
──────────────────────────────
Native LangGraph conversation memory (used when REDIS_ENABLED=false).

Session history lives inside a MemorySaver checkpointer, keyed by
session_id as the LangGraph thread_id. In-process only — history is
lost on restart (acceptable: ~5 users).

Two guards keep this bounded (see BLOCKER A / B in the design notes):

1. compact_thread — after each run, scrub the bulky/PII `db_result`
   channel from the checkpoint and drop all but the latest checkpoint,
   so a thread holds only its conversation history at rest.
2. evict_idle_threads — background loop drops threads idle longer than
   SESSION_TTL_SECONDS (mirrors the old Redis session TTL).
"""
import logging
import threading
import time
from datetime import datetime

from langgraph.checkpoint.memory import MemorySaver

logger = logging.getLogger(__name__)

MAX_TURNS = 5  # same sliding window as the Redis session_store
SESSION_TTL_SECONDS = 7200  # matches the old Redis session TTL

checkpointer = MemorySaver()

# Last-access timestamps for idle-session eviction. Guarded because the
# eviction loop runs on the lifespan thread while request handlers touch
# sessions on worker threads.
_last_access: dict[str, float] = {}
_lock = threading.Lock()


def thread_config(session_id: str) -> dict:
    return {"configurable": {"thread_id": session_id}}


def touch(session_id: str) -> None:
    """Record that a session is active (refreshes the idle-eviction TTL)."""
    if not session_id:
        return
    with _lock:
        _last_access[session_id] = time.time()


def load_history(graph, session_id: str) -> list[dict]:
    """Return the accumulated conversation turns for a session (max MAX_TURNS)."""
    if not session_id:
        return []
    touch(session_id)
    try:
        snapshot = graph.get_state(thread_config(session_id))
        if snapshot and snapshot.values:
            return snapshot.values.get("session_history") or []
    except Exception as e:
        logger.warning(f"Memory load error: {e}")
    return []


def clear_history(graph, session_id: str) -> None:
    """Reset a session's conversation history and free its checkpoint memory."""
    if not session_id:
        return
    try:
        checkpointer.delete_thread(session_id)
        with _lock:
            _last_access.pop(session_id, None)
        logger.info(f"Conversation memory cleared | session={session_id}")
    except Exception as e:
        logger.warning(f"Memory clear error: {e}")


def record_turn(state: dict) -> dict:
    """
    Terminal graph node: on success, append the turn to session_history.
    The checkpointer persists the returned state under the thread_id,
    so the next invoke with the same session_id sees the updated history.
    """
    if (
        state.get("validation_passed")
        and state.get("summary")
        and not state.get("error")
    ):
        from datetime import datetime
        turn = {
            "question": state.get("question"),
            "sql":      state.get("generated_sql"),
            "summary":  state.get("summary"),
            "ts":       datetime.utcnow().isoformat(),
        }
        history = (state.get("session_history") or []) + [turn]
        return {"session_history": history[-MAX_TURNS:]}
    return {}


def append_turn(
    graph,
    session_id: str,
    question: str,
    sql: str | None,
    summary: str,
    extra: dict | None = None,
) -> None:
    """
    Records a conversation turn directly into the thread checkpoint without
    invoking the pipeline graph. Used by deterministic paths (e.g. churn
    analysis) that bypass the graph but must still appear in history.

    `extra` is merged into the turn dict for caller-specific keys
    (e.g. churn_offset, churn_shown, churn_entities) that follow-up
    routing reads from the last turn.
    """
    if not session_id:
        return
    try:
        turn: dict = {
            "question": question,
            "sql":      sql,
            "summary":  summary,
            "ts":       datetime.utcnow().isoformat(),
        }
        if extra:
            turn.update(extra)
        snapshot = graph.get_state(thread_config(session_id))
        history = (snapshot.values.get("session_history") or []) if snapshot and snapshot.values else []
        new_history = (history + [turn])[-MAX_TURNS:]
        # Mirrors compact_thread: write through the record_turn node so the
        # turn lands on the same graph boundary, and scrub db_result to keep
        # PII out of the persisted checkpoint.
        graph.update_state(
            thread_config(session_id),
            {"session_history": new_history, "db_result": []},
            as_node="record_turn",
        )
        _prune_to_latest(session_id)
        touch(session_id)
    except Exception as e:
        # deliberate: never raise into the request path — history recording
        # is best-effort; a memory error must not break the answer.
        logger.warning(f"append_turn failed | session={session_id}: {e}")


def _prune_to_latest(session_id: str) -> None:
    """
    Drop all but the most recent checkpoint (and its writes/blobs) for a
    thread, keeping exactly the state needed to resume the conversation.

    LangGraph 1.2.4's `MemorySaver.prune` raises NotImplementedError, so
    this prunes directly against the saver's storage/writes/blobs dicts.
    The latest checkpoint's `channel_versions` pin which blobs must stay;
    everything else (older checkpoints, their writes, and orphaned blobs)
    is deleted. Verified to leave `get_state` intact and allow a fresh
    invoke on the same thread.
    """
    ns = ""  # top-level checkpoint namespace used by thread_config
    thread_storage = checkpointer.storage.get(session_id, {}).get(ns, {})
    if not thread_storage:
        return
    cp_ids = list(thread_storage.keys())
    latest = max(cp_ids)
    for c in cp_ids:
        if c == latest:
            continue
        thread_storage.pop(c, None)
        checkpointer.writes.pop((session_id, ns, c), None)

    # Keep only blobs referenced by the surviving checkpoint's channel
    # versions; older versions are dead weight (and hold the scrubbed PII).
    latest_tuple = checkpointer.get_tuple(thread_config(session_id))
    if latest_tuple is None:
        return
    versions = latest_tuple.checkpoint.get("channel_versions", {})
    keep = {(session_id, ns, ch, ver) for ch, ver in versions.items()}
    for k in list(checkpointer.blobs.keys()):
        if k[0] == session_id and k not in keep:
            del checkpointer.blobs[k]


def compact_thread(graph, session_id: str) -> None:
    """
    After each run: scrub bulky/PII channels from the checkpoint and drop
    all but the latest checkpoint, so a thread holds only its conversation
    history (bounded, no DB rows at rest).

    This runs AFTER graph.invoke() has returned the response dict, so the
    response path still sees the full db_result; only the persisted
    checkpoint is compacted.
    """
    if not session_id:
        return
    try:
        # Overwrite the db_result channel in the checkpoint with an empty
        # list. as_node="record_turn" reuses the terminal node so the write
        # lands on the same graph boundary the run ended on.
        graph.update_state(
            thread_config(session_id), {"db_result": []}, as_node="record_turn"
        )
        _prune_to_latest(session_id)
        touch(session_id)
    except Exception as e:
        # deliberate: never break a request over housekeeping
        logger.warning(f"compact_thread failed | session={session_id}: {e}")


def evict_idle_threads() -> int:
    """Drop threads idle longer than SESSION_TTL_SECONDS. Returns the count evicted."""
    now = time.time()
    with _lock:
        stale = [sid for sid, ts in _last_access.items() if now - ts > SESSION_TTL_SECONDS]
    evicted = 0
    for sid in stale:
        try:
            checkpointer.delete_thread(sid)
            with _lock:
                _last_access.pop(sid, None)
            evicted += 1
        except Exception as e:
            logger.warning(f"Eviction failed | session={sid}: {e}")
    return evicted


def memory_stats() -> dict:
    """Stats for the /health endpoint when Redis is disabled."""
    with _lock:
        active = len(_last_access)
    return {
        "backend": "memory",
        "caching": "disabled",
        "active_sessions": active,
        "session_ttl_seconds": SESSION_TTL_SECONDS,
    }
