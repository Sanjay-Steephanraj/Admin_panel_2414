import logging
import uuid
from langgraph.graph import StateGraph, END

from config.settings import get_settings
from graph.states.nlsql_state import NLSQLState
from .nodes.sql_generation_node import sql_generation_node
from .nodes.sql_validation_node import sql_validation_node
from .nodes.summary_generation_node import summary_generation_node
from cache.query_cache import query_cache

logger = logging.getLogger(__name__)

MAX_RETRIES = 2


def route_after_validation(state: NLSQLState) -> str:
    error             = state.get("error")
    sql_error         = state.get("sql_generation_error")
    validation_passed = state.get("validation_passed", False)
    retry_count       = state.get("retry_count", 0)

    if error or sql_error == "UNCLEAR":
        return "summary"
    if validation_passed:
        return "summary"
    if retry_count < MAX_RETRIES:
        logger.info(f"Retrying SQL generation ({retry_count + 1}/{MAX_RETRIES})")
        return "retry"

    logger.warning("Max retries reached → going to summary")
    return "summary"


def increment_retry(state: NLSQLState) -> NLSQLState:
    return {**state, "retry_count": state.get("retry_count", 0) + 1}


def build_nlsql_graph():
    redis_on = get_settings().redis_enabled
    graph = StateGraph(NLSQLState)

    graph.add_node("sql_generation",     sql_generation_node)
    graph.add_node("sql_validation",     sql_validation_node)
    graph.add_node("summary_generation", summary_generation_node)
    graph.add_node("increment_retry",    increment_retry)

    graph.set_entry_point("sql_generation")
    graph.add_edge("sql_generation",  "sql_validation")
    graph.add_edge("increment_retry", "sql_generation")

    graph.add_conditional_edges(
        "sql_validation",
        route_after_validation,
        {"summary": "summary_generation", "retry": "increment_retry"},
    )

    if redis_on:
        graph.add_edge("summary_generation", END)
        return graph.compile()

    # Native LangGraph memory: record turn, persist via checkpointer
    from memory.conversation_memory import checkpointer, record_turn
    graph.add_node("record_turn", record_turn)
    graph.add_edge("summary_generation", "record_turn")
    graph.add_edge("record_turn", END)
    return graph.compile(checkpointer=checkpointer)


nlsql_graph = build_nlsql_graph()


def run_nlsql_pipeline(
    question: str,
    intent: str = "",
    entities: dict = None,
    session_id: str = None,
    history: list[dict] = None,
) -> NLSQLState:
    redis_on = get_settings().redis_enabled

    # ── Cache check ───────────────────────────
    if redis_on:
        cached = query_cache.get(question)
        if cached:
            logger.info(f" Cache hit: {question!r}")
            return NLSQLState(
                question=                question,
                intent=                  intent,
                entities=                entities or {},
                session_id=              session_id,
                session_history=         history or [],
                generated_sql=           cached.sql,
                sql_generation_error=    None,
                db_result=               [],
                db_error=                None,
                fallback_used=           False,
                message=                 None,
                validation_passed=       True,
                llm_validation_feedback= "Served from cache",
                retry_count=             0,
                summary=                 cached.summary,
                cache_hit=               True,
                error=                   None,
                trace_id=                None,
            )

    # ── Run graph ─────────────────────────────
    initial_state: NLSQLState = {
        "question":                question,
        "intent":                  intent,
        "entities":                entities or {},
        "session_id":              session_id,
        "session_history":         history or [],
        "generated_sql":           None,
        "sql_generation_error":    None,
        "db_result":               None,
        "db_error":                None,
        "fallback_used":           False,
        "message":                 None,
        "validation_passed":       None,
        "llm_validation_feedback": None,
        "retry_count":             0,
        "summary":                 None,
        "cache_hit":               False,
        "error":                   None,
        "trace_id":                None,
    }

    logger.info(f"Pipeline start | intent={intent} | question={question!r}")
    if redis_on:
        final_state = nlsql_graph.invoke(initial_state)
    else:
        from memory.conversation_memory import thread_config
        if not session_id:
            session_id = str(uuid.uuid4())
            logger.warning("No session_id provided — using ephemeral thread %s", session_id)
        final_state = nlsql_graph.invoke(initial_state, config=thread_config(session_id))
        from memory.conversation_memory import compact_thread
        compact_thread(nlsql_graph, session_id)

    # ── Cache successful results ──────────────
    if redis_on and (
        final_state.get("validation_passed")
        and final_state.get("generated_sql")
        and final_state.get("summary")
        and not final_state.get("error")
    ):
        stored = query_cache.set(
            question=  question,
            sql=       final_state["generated_sql"],
            summary=   final_state["summary"],
            row_count= len(final_state.get("db_result") or []),
        )
        if stored:
            logger.info("Result stored in cache")
        else:
            logger.debug("Cache write skipped (Redis unavailable)")

    return final_state