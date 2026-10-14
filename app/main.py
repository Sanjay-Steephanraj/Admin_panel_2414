import asyncio
import logging
import os
import uuid
from contextlib import asynccontextmanager
from functools import partial
import re
from datetime import datetime

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from graph.pipeline import run_nlsql_pipeline
from config.settings import get_settings
from security.domain_guard import domain_guard
from llm.prompts import is_ambiguous, user_error_message
from components.query_normalizer import normalize_question, detect_name_ambiguity, _regex_extract
from components.query_spec import (
    USER_FACING_LIMITS, build_query_spec, merge_follow_up,
    resolve_ministry, resolve_pending_reply, unsupported_request_reason,
)
from llm.question_expander import is_genuine_follow_up
from tools.db_tool import ping_db
from llm.question_expander import expand_question
from llm.rejection_handler import generate_dynamic_error_reply
import dynamic as schema_resolver


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Donor Portal AI")

    db_ok = ping_db()
    logger.info(f"DB: {'connected' if db_ok else 'unreachable'}")

    logger.info("Using LangGraph MemorySaver conversation memory")

    schema_resolver.initialize()
    logger.info(f"Schema: {schema_resolver.get_schema_status()}")

    _warm_up_connections()

    try:
        from data_masking import masker
        masker.warm_up()
        logger.info("PII masker ready (Presidio + spaCy en_core_web_md)")
    except Exception as e:
        logger.error(f"PII masker warm-up failed: {e}")

    # Keep the in-process conversation memory bounded.
    async def _evict_loop():
        from memory.conversation_memory import evict_idle_threads
        while True:
            await asyncio.sleep(600)  # every 10 min
            n = await _run_sync(evict_idle_threads)
            if n:
                logger.info(f"Evicted {n} idle session(s) from conversation memory")
    eviction_task = asyncio.create_task(_evict_loop())

    yield
    if eviction_task is not None:
        eviction_task.cancel()
        try:
            await eviction_task
        except asyncio.CancelledError:
            pass
    logger.info("Shutting down Donor Portal AI")


def _warm_up_connections() -> None:
    """
    Initializes the MySQL connection pool eagerly at startup.

    Without this, both are created lazily on the first request, adding
    100–300ms of cold-start latency to that request.
    """
    # MySQL pool
    try:
        from tools.db_tool import _get_pool
        _get_pool()
        logger.info("MySQL connection pool warmed up")
    except Exception as e:
        logger.warning(f"MySQL pool warm-up failed (will retry on first request): {e}")



app = FastAPI(
    title="Donor Portal AI Assistant",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)
    session_id: str | None = Field(None, max_length=64)


class AskResponse(BaseModel):
    response: str
    question: str
    row_count: int
    displayed_count: int = 0
    total_count: int = 0
    cache_hit: bool
    request_id: str
    session_id: str


class HealthResponse(BaseModel):
    status: str
    db: str
    version: str
    cache_stats: dict
    schema: dict


# ── Helper: run blocking calls off the event-loop thread ─────────────────────

async def _run_sync(fn, *args, **kwargs):
    """
    Runs a synchronous (blocking) function in the default thread-pool executor
    so it does not block the FastAPI event loop.

    Usage:
        result = await _run_sync(some_blocking_function, arg1, arg2)
    """
    loop = asyncio.get_event_loop()
    if kwargs:
        fn = partial(fn, **kwargs)
    return await loop.run_in_executor(None, fn, *args)

async def _load_pending(session_id: str):
    from graph.pipeline import nlsql_graph
    from memory.conversation_memory import load_pending
    return await _run_sync(load_pending, nlsql_graph, session_id)

async def _save_pending(session_id: str, pending: dict | None):
    from graph.pipeline import nlsql_graph
    from memory.conversation_memory import save_pending
    await _run_sync(save_pending, nlsql_graph, session_id, pending)


#  API endpoints 

@app.post("/chat/query", response_model=AskResponse)
async def ask(request: AskRequest, http_request: Request, http_response: Response):
    request_id = str(uuid.uuid4())
    logger.info(f"[{request_id}] {request.question}")

    # ── Resolve Session ID ──
    SESSION_ID_RE = re.compile(r'^[a-zA-Z0-9\-]{8,64}$')
    cookie_session_id = http_request.cookies.get("donor_session_id")
    requested_session_id = request.session_id or cookie_session_id
    session_id = (
        requested_session_id
        if requested_session_id and SESSION_ID_RE.match(requested_session_id)
        else str(uuid.uuid4())
    )
    http_response.set_cookie(
        key="donor_session_id",
        value=session_id,
        max_age=7200,
        httponly=True,
        samesite="lax",
    )

    from graph.pipeline import nlsql_graph
    from memory.conversation_memory import load_history
    history = await _run_sync(load_history, nlsql_graph, session_id)

    pending = await _load_pending(session_id)
    logger.info(
        "[%s] session=%s source=%s history_turns=%d pending=%s",
        request_id, session_id,
        "body" if request.session_id else ("cookie" if cookie_session_id else "generated"),
        len(history), bool(pending),
    )

    # ── Expand Question (resolve follow-ups) ──
    # Pending clarification replies are exact values, not natural-language
    # follow-ups. Do not send a bare year/month through the LLM expander.
    expanded_question = (
        request.question if pending
        else await _run_sync(expand_question, request.question, history)
    )
    # The raw text catches short follow-ups even if expansion removes the
    # recurrence wording. No SQL, database lookup, or history turn is created.
    if (unsupported_request_reason(request.question) == "payment_frequency"
            or unsupported_request_reason(expanded_question) == "payment_frequency"):
        await _save_pending(session_id, None)
        return AskResponse(
            response=USER_FACING_LIMITS["payment_frequency"],
            question=request.question,
            row_count=0,
            cache_hit=False,
            request_id=request_id,
            session_id=session_id,
        )
    reply_mode, pending_spec = resolve_pending_reply(pending, request.question)
    pending_answer = reply_mode == "resolved"
    if pending and reply_mode in {"resolved", "invalid"}:
        raw_safe, raw_reason, _ = await _run_sync(
            domain_guard.check_question, request.question,
            contextual_intent="payment" if pending["query_spec"].get("subject") == "payment" else pending.get("intent"),
        )
        if not raw_safe:
            return AskResponse(response=await _run_sync(generate_dynamic_error_reply, request.question,
                               "injection" if "injection" in raw_reason.lower() else "off_topic"),
                               question=request.question, row_count=0, cache_hit=False,
                               request_id=request_id, session_id=session_id)
    if reply_mode == "invalid":
        return AskResponse(response=pending.get("clarification_text") or
                           pending["query_spec"].get("period_ambiguity") or
                           pending["query_spec"].get("clarification") or
                           "Please answer the clarification or ask a new complete question.",
                           question=request.question, row_count=0, cache_hit=False,
                           request_id=request_id, session_id=session_id)
    if pending_answer:
        expanded_question = pending["question"]

    follow_up = is_genuine_follow_up(expanded_question)
    previous = next((t.get("query_spec") for t in reversed(history)
                     if t.get("query_spec") and t.get("query_outcome", "success") == "success"), None)
    if follow_up and not previous and not pending_answer:
        return AskResponse(response="Which ministry or donor and which report should I use?",
                           question=request.question, row_count=0, cache_hit=False,
                           request_id=request_id, session_id=session_id)

    # ── Ambiguity check (run on EXPANDED question) ──
    ambiguity_data = None if pending_answer else detect_name_ambiguity(expanded_question)
    if ambiguity_data:
        return AskResponse(
            response=ambiguity_data["clarification_question"],
            question=request.question,
            row_count=0,
            cache_hit=False,
            request_id=request_id,
            session_id=session_id,
        )

    if is_ambiguous(expanded_question) and not follow_up and not pending_answer and not unsupported_request_reason(expanded_question):
        return AskResponse(
            response=await _run_sync(generate_dynamic_error_reply, request.question, "ambiguous"),
            question=request.question,
            row_count=0,
            cache_hit=False,
            request_id=request_id,
            session_id=session_id,
        )

    # ── Security check ──
    is_safe, reason, intent = await _run_sync(
        domain_guard.check_question, expanded_question,
        contextual_intent=("payment"
                           if (pending_answer and pending_spec.get("subject") == "payment")
                           or (follow_up and previous and previous.get("subject") == "payment")
                           else None),
    )
    if not is_safe:
        error_type = "injection" if "injection" in reason.lower() else "off_topic"
        return AskResponse(
            response=await _run_sync(generate_dynamic_error_reply, request.question, error_type),
            question=request.question,
            row_count=0,
            cache_hit=False,
            request_id=request_id,
            session_id=session_id,
        )

    # ── Normalisation + entity extraction ──
    norm_output = await _run_sync(normalize_question, expanded_question)
    normalized_question = norm_output["normalized_question"]
    entities            = norm_output["entities"]
    query_spec = pending_spec if pending_answer else build_query_spec(expanded_question, entities, contextual=follow_up)
    if follow_up and not pending_answer:
        query_spec = merge_follow_up(query_spec, previous)
    clarification_text = query_spec.get("period_ambiguity") or query_spec.get("clarification")
    if clarification_text:
        if query_spec.get("period_ambiguity"):
            kind = "year" if query_spec.get("period_name") == "month" and not query_spec.get("start_date") else "period"
        else:
            kind = query_spec.get("clarification_kind")
        if kind in {"year", "period", "status", "ministry"}:
            await _save_pending(session_id, {"kind": kind, "question": expanded_question,
                                           "query_spec": query_spec,
                                           "clarification_text": clarification_text, "intent": intent})
        else:
            # A new gated report must not leave an older year/status question
            # waiting to consume the user's next short reply.
            await _save_pending(session_id, None)
        return AskResponse(response=clarification_text, question=request.question, row_count=0,
                           cache_hit=False, request_id=request_id, session_id=session_id)
    query_spec, clarification = await _run_sync(resolve_ministry, query_spec)
    if query_spec.get("resolution_outcome") == "no_match":
        await _save_pending(session_id, None)
        name = query_spec.get("entity_code") or query_spec.get("entity_name") or "that ministry"
        return AskResponse(response=f"I could not find a ministry matching {name}. Please check its name or gift code.",
                           question=request.question, row_count=0, cache_hit=False,
                           request_id=request_id, session_id=session_id)
    if clarification:
        await _save_pending(session_id, {"kind": "ministry", "question": expanded_question,
                                          "query_spec": query_spec})
        return AskResponse(response=clarification, question=request.question, row_count=0,
                           cache_hit=False, request_id=request_id, session_id=session_id)
    if query_spec.get("resolution_outcome") == "retrieval_error":
        return AskResponse(response=user_error_message("db_failure"), question=request.question, row_count=0,
                           cache_hit=False, request_id=request_id, session_id=session_id)
    await _save_pending(session_id, None)

    logger.info(f"Normalized Question: {normalized_question}")
    logger.info(f"Entities: {entities}")

    # ── Churn detour (deterministic, bypasses LLM pipeline) ──
    from components.churn_detector import resolve_churn_route

    churn_route = resolve_churn_route(
        raw_question=request.question,
        expanded_question=expanded_question,
        normalized_question=normalized_question,
        history=history,
    )
    if churn_route is not None:
        from analysis.churn_analyzer import run_churn_analysis
        churn_offset = churn_route["offset"]
        churn_limit = churn_route["limit"]
        churn_entities = churn_route["entities_override"] or entities
        logger.info(f"[{request_id}] Churn route → offset={churn_offset} limit={churn_limit}")
        churn_result = await _run_sync(run_churn_analysis, churn_entities, churn_offset, churn_limit)

        # Record the turn so follow-ups (pagination, pronouns) have context.
        # Best-effort: a memory error must not break the churn answer.
        churn_turn_sql = "CHURN_ANALYSIS"
        try:
            turn = {
                "question": normalized_question,
                "sql": churn_turn_sql,
                "summary": churn_result["summary"],
                "ts": datetime.utcnow().isoformat(),
                "churn_offset": churn_result.get("offset", 0),
                "churn_shown": churn_result.get("shown", 0),
                "churn_entities": churn_entities,
            }
            from graph.pipeline import nlsql_graph
            from memory.conversation_memory import append_turn
            await _run_sync(
                append_turn, nlsql_graph, session_id,
                normalized_question, churn_turn_sql, churn_result["summary"], extra=turn,
            )
        except Exception:
            logger.warning(f"[{request_id}] Churn turn recording failed", exc_info=True)

        return AskResponse(
            response=churn_result["summary"],
            question=request.question,
            row_count=churn_result["row_count"],
            cache_hit=False,
            request_id=request_id,
            session_id=session_id,
        )

    # ── Pipeline execution ──
    try:
        state = await _run_sync(
            run_nlsql_pipeline,
            normalized_question,
            "payment" if query_spec.get("subject") == "payment" else intent,
            entities,
            session_id=session_id,
            history=history,
            query_spec=query_spec,
        )
    except Exception:
        logger.exception(f"[{request_id}] Pipeline error")
        return AskResponse(
            response=await _run_sync(generate_dynamic_error_reply, request.question, "db_failure"),
            question=request.question,
            row_count=0,
            cache_hit=False,
            request_id=request_id,
            session_id=session_id,
        )

    # ── Extract results ──
    cache_hit = state.get("cache_hit", False)
    db_result = state.get("db_result") or []
    sql_error = state.get("sql_generation_error")
    db_error  = state.get("db_error")
    summary   = state.get("summary", "")

    # ── Response handling ──
    if sql_error and sql_error == "UNCLEAR":
        response = await _run_sync(generate_dynamic_error_reply, request.question, "unclear")
    elif sql_error and "Security:" in sql_error:
        response = await _run_sync(generate_dynamic_error_reply, request.question, "security")
    elif sql_error:
        response = await _run_sync(generate_dynamic_error_reply, request.question, "db_failure")
    elif db_error and not db_result and not summary:
        response = await _run_sync(generate_dynamic_error_reply, request.question, "db_failure")
    elif not db_result and not cache_hit and not summary:
        response = await _run_sync(generate_dynamic_error_reply, request.question, "no_results")
    else:
        response = summary or await _run_sync(generate_dynamic_error_reply, request.question, "db_failure")

    # ── Save History (on success) ──
    if (cache_hit and state.get("validation_passed")
            and state.get("summary") and not state.get("error")
            and not state.get("db_error") and state.get("query_outcome") == "success"
            and (state.get("db_result") or state.get("cache_hit"))):
        turn = {
            "question": normalized_question,
            "sql":      state.get("generated_sql"),
            "summary":  state.get("summary"),
            "ts":       datetime.utcnow().isoformat(),
            "query_spec": state.get("query_spec") or query_spec,
            "displayed_count": state.get("displayed_count", len(db_result)),
            "total_count": state.get("total_count", len(db_result)),
            "query_outcome": "success",
        }
        from graph.pipeline import nlsql_graph
        from memory.conversation_memory import append_turn
        await _run_sync(
            append_turn, nlsql_graph, session_id,
            turn["question"], turn["sql"], turn["summary"], extra=turn,
        )

    return AskResponse(
        response=response,
        question=request.question,
        row_count=state.get("displayed_count", len(db_result)),
        displayed_count=state.get("displayed_count", len(db_result)),
        total_count=state.get("total_count", len(db_result)),
        cache_hit=cache_hit,
        request_id=request_id,
        session_id=session_id,
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    db_ok = await _run_sync(ping_db)
    from memory.conversation_memory import memory_stats
    cache_stats = memory_stats()
    return HealthResponse(
        status="ok" if db_ok else "degraded",
        db="connected" if db_ok else "unreachable",
        version="2.0.0",
        cache_stats=cache_stats,
        schema=schema_resolver.get_schema_status(),
    )


@app.delete("/api/session/{session_id}")
async def clear_session(session_id: str):
    from graph.pipeline import nlsql_graph
    from memory.conversation_memory import clear_history
    await _run_sync(clear_history, nlsql_graph, session_id)
    return {"message": "Session cleared"}


class FeedbackRequest(BaseModel):
    question: str
    sql: str
    intent: str
    is_correct: bool


@app.post("/feedback")
async def feedback(request: FeedbackRequest):
    from components.few_shot_examples import add_example
    from data_masking import masker

    if request.is_correct:
        try:
            # deliberate: discard mapping — examples are SQL patterns, not data;
            # mask before storing so PII in feedback never reaches the LLM prompt.
            masked_q, _ = await _run_sync(masker.mask_text, request.question)
            masked_sql, _ = await _run_sync(masker.mask_text, request.sql)
        except RuntimeError:
            # fail-closed: never store unmasked PII; log without the request body.
            logger.warning("Feedback masking failed; example not stored")
            return {"message": "Noted"}
        try:
            add_example(
                question=masked_q,
                sql=masked_sql,
                intent=request.intent,
            )
        except Exception:
            # Resilient: never 500 on feedback; validation rejections are expected
            # and unexpected storage errors must not surface to the caller.
            logger.warning("Feedback add_example failed; example not stored",
                           exc_info=True)
        return {"message": "Saved"}

    return {"message": "Noted"}


@app.post("/schema/refresh")
async def refresh_schema():
    from config.settings import get_settings

    if not get_settings().use_dynamic_schema:
        return {"message": "Dynamic schema disabled", "mode": "static"}

    success = await _run_sync(schema_resolver.refresh)
    return {
        "message": "Schema refreshed" if success else "Schema refresh failed",
        "success": success,
        "schema":  schema_resolver.get_schema_status(),
    }


@app.get("/schema/status")
async def schema_status():
    return schema_resolver.get_schema_status()


@app.get("/schema")
async def get_schema():
    from dynamic.schema_fetcher import build_live_schema, get_schema_stats

    schema = await _run_sync(build_live_schema)
    stats  = get_schema_stats()

    return {
        "stats":  stats,
        "schema": {
            table: {
                "description": meta["description"],
                "columns":     [c["column_name"] for c in meta["columns"]],
            }
            for table, meta in schema["tables"].items()
        },
        "relationships": schema["relationships"],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=os.getenv("APP_ENV") == "development",
        log_level="info",
    )
