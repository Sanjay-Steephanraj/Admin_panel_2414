from typing import TypedDict, Optional, List, Any

class NLSQLState(TypedDict, total=False):
    # Input
    question:               str
    intent:                 str
    entities:               Optional[dict]
    session_history:        Optional[list]
    session_id:             Optional[str]
    retry_count:            Optional[int]
    trace_id:               Optional[str]
    # Authoritative interpretation contract; never reconstruct relative dates downstream.
    query_spec:             Optional[dict]
    last_successful_query_spec: Optional[dict]
    query_outcome:          Optional[str]

    # SQL generation
    generated_sql:          Optional[str]
    sql_generation_error:   Optional[str]

    # Validation
    validation_passed:      Optional[bool]
    llm_validation_feedback: Optional[str]


    # DB execution
    db_result:              List[Any]
    db_error:               Optional[str]

    # CRITICAL — these MUST be declared or LangGraph drops them silently
    fallback_used:          bool
    message:                Optional[str]

    # Summary
    summary:                Optional[str]

    # Cache
    cache_hit:              bool
    displayed_count:        int
    total_count:            int

    # Error passthrough
    error:                  Optional[str]
