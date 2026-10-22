"""Structural checks for the existing Admin payment query contract (AA-05).

This is a validator/rewriter inside the existing graph, not a SQL generator.
Unverifiable SQL goes through the existing retry flow instead of executing.
"""
from __future__ import annotations

import re
from sqlglot import exp, parse, parse_one

ALIASES = {"total": "total_amount", "average": "average_amount", "count": "record_count"}
LIST_LIMIT = 100
_NONREPORT_FIELD = re.compile(
    r"ssn|social.?security|card|\bcvv\b|\bcvc\b|bank|iban|routing|payload|json|secret|password", re.I)


def _ministry_activity_rows(spec):
    return spec.get("subject") == "payment" and (
        spec.get("group_by") == "ministry" or
        spec.get("result_entity") == "ministry" and spec.get("result_shape") == "detail")


def _row_identity_guard(node, tables):
    """Only the two source-identity guards, not arbitrary missing-value filters."""
    if not isinstance(node, exp.Not) or not isinstance(node.this, exp.Is):
        return None
    if not isinstance(node.this.expression, exp.Null):
        return None
    identity = _column(node.this.this, tables)
    return identity[0] if identity in {("sfpayments", "id"), ("sf_ministries", "sfid")} else None


def _present_activity_sources(tree, tables):
    """Prove source presence on every output row, never from an outer ON alone."""
    present = set()
    joins = tree.args.get("joins", [])
    no_right_extension = not any(j.side.upper() in {"RIGHT", "FULL"} for j in joins)
    source = tree.args.get("from_")
    if no_right_extension and source and isinstance(source.this, exp.Table):
        present.add(source.this.name.lower())
    where = tree.args.get("where")
    predicates = _guaranteed(where.this if where else None)
    for node in predicates:
        guard = _row_identity_guard(node, tables)
        if guard:
            present.add(guard)
        comparison = _comparison(node, tables)
        if comparison:  # a direct comparison to a non-NULL literal rejects a NULL-extended source
            present.add(comparison[0][0])
    official_link = (("sfpayments", "ministryid"), ("sf_ministries", "sfid"))
    if _has_join(predicates, tables, *official_link):
        present.update(("sfpayments", "sf_ministries"))
    if no_right_extension:
        for join in joins:
            if not join.side and _has_join(_guaranteed(join.args.get("on")), tables, *official_link):
                present.update(("sfpayments", "sf_ministries"))
    return present


def enforce_known_ministry_rows(sql: str, spec: dict) -> str:
    """A ministry activity row is a linked ministry with a contributing payment.

    Do not discard unmatched payments from ordinary totals or histories, or
    turn directory reports into activity reports. Only add missing source-
    identity guards; keep all business filters, measures and ordering intact.
    """
    if not _ministry_activity_rows(spec):
        return sql
    tree = parse_one(sql, read="mysql")
    if not isinstance(tree, exp.Select) or len(list(tree.find_all(exp.Select))) != 1:
        return sql
    tables = _tables(tree)
    if (len(tables) != len(set(tables.values())) or
            not {"sfpayments", "sf_ministries"}.issubset(tables.values())):
        return sql  # final semantic validation rejects unsupported structure
    missing = {"sfpayments", "sf_ministries"} - _present_activity_sources(tree, tables)
    if not missing:
        return sql
    for table, column in (("sfpayments", "id"), ("sf_ministries", "sfid")):
        if table in missing:
            alias = next(a for a, name in tables.items() if name == table)
            guard = exp.Not(this=exp.Is(this=exp.column(column, table=alias), expression=exp.Null()))
            tree = tree.where(guard, append=True)
    return tree.sql(dialect="mysql")


def enforce_unrestricted_status(sql: str, spec: dict) -> str:
    """Remove a simple status restriction from an approved both-status report.

    Resolved payment specs contain an explicit status key: None means no
    status predicate, not a missing default. Never alter paid/unpaid requests,
    mixed OR branches, calculations, subqueries or other report conditions.
    All final SQL still passes the unchanged semantic validator.
    """
    if spec.get("subject") != "payment" or "status" not in spec or spec["status"] is not None:
        return sql
    tree = parse_one(sql, read="mysql")
    if not isinstance(tree, exp.Select) or len(list(tree.find_all(exp.Select))) != 1:
        return sql
    tables = _tables(tree)
    if list(tables.values()).count("sfpayments") != 1:
        return sql
    changed = False
    def remove(node):
        nonlocal changed
        if isinstance(node, exp.Paren):
            result = remove(node.this)
            return exp.Paren(this=result) if result is not None else None
        if isinstance(node, exp.And):
            left, right = remove(node.this), remove(node.expression)
            if left is None: return right
            if right is None: return left
            return exp.And(this=left, expression=right)
        comparison = _comparison(node, tables)
        if (comparison and comparison[0] == ("sfpayments", "paid")
                and comparison[1] is exp.EQ and comparison[2] in {"0", "1"}):
            changed = True
            return None
        return node.copy()
    where = tree.args.get("where")
    if where:
        remaining = remove(where.this)
        tree.set("where", exp.Where(this=remaining) if remaining is not None else None)
    for join in tree.args.get("joins", []):
        if join.args.get("on") is not None:
            remaining = remove(join.args["on"])
            join.set("on", remaining if remaining is not None else exp.true())
    return tree.sql(dialect="mysql") if changed else sql


def enforce_monthly_shape(sql: str, spec: dict) -> str:
    """Bind a plain aggregate to an already-approved calendar-month shape.

    A short metric follow-up can make the SQL model omit grouping even when
    the interpreted report retained it. Only complete that missing shape on
    a direct, otherwise plain aggregate. Do not rewrite existing groupings,
    calculations, conditions, HAVING, ranking or an unsupported query shape.
    The ordinary semantic and read-only checks still validate the final SQL.
    """
    if (spec.get("subject") != "payment" or spec.get("group_by") != "month"
            or spec.get("metric") not in ALIASES or spec.get("qualification")
            or spec.get("ranking") or spec.get("clarification") or spec.get("period_ambiguity")):
        return sql
    tree = parse_one(sql, read="mysql")
    if (not isinstance(tree, exp.Select) or len(list(tree.find_all(exp.Select))) != 1
            or len(tree.expressions) != 1
            or any(tree.args.get(key) for key in ("group", "having", "distinct", "order", "offset"))):
        return sql
    tables = _tables(tree)
    aliases = [alias for alias, table in tables.items() if table == "sfpayments"]
    if len(aliases) != 1 or len(tables) != len(set(tables.values())):
        return sql
    projection = tree.expressions[0]
    metric = spec["metric"]
    expected = {"count": exp.Count, "total": exp.Sum, "average": exp.Avg}[metric]
    if not isinstance(projection, exp.Alias) or projection.alias.lower() != ALIASES[metric]:
        return sql
    value = projection.this
    if isinstance(value, exp.Coalesce) and all(_literal(v) == "0" for v in value.expressions):
        value = value.this
    if not isinstance(value, expected):
        return sql
    if metric != "count" and _column(value.this, tables) != ("sfpayments", "paymentamount"):
        return sql
    month = parse_one(f"DATE_FORMAT({aliases[0]}.paymentdate, '%Y-%m')", read="mysql")
    tree.set("expressions", [exp.alias_(month.copy(), "month"), projection.copy()])
    tree.set("group", exp.Group(expressions=[month.copy()]))
    tree.set("order", exp.Order(expressions=[exp.Ordered(this=exp.column("month"), desc=False, nulls_first=True)]))
    return tree.sql(dialect="mysql")


def enforce_result_bound(sql: str, spec: dict) -> str:
    """Cap the existing list contract; do not change filters or scalar SQL."""
    if spec.get("result_shape") not in {"detail", "grouped"}:
        return sql
    tree = parse_one(sql, read="mysql")
    if not isinstance(tree, exp.Select):
        return sql  # common read-only guard rejects the statement
    limit = tree.args.get("limit")
    if limit:
        value = limit.expression
        if not isinstance(value, exp.Literal) or value.is_string or not str(value.this).isdigit() or int(value.this) < 1:
            raise ValueError("List limit must be a positive integer")
        count = min(int(value.this), LIST_LIMIT)
    else:
        count = LIST_LIMIT
    if (spec.get("ranking") or {}).get("limit"):
        count = spec["ranking"]["limit"]
        if type(count) is not int or not 1 <= count <= LIST_LIMIT:
            raise ValueError("Invalid bounded ranking size")
    tree.set("limit", exp.Limit(expression=exp.Literal.number(count)))
    return tree.sql(dialect="mysql")


def _canonical(node, tables):
    node = node.copy()
    for column in node.find_all(exp.Column):
        table, name = _column(column, tables)
        if table:
            column.set("table", exp.to_identifier(table))
        column.set("this", exp.to_identifier(name))
    for literal in node.find_all(exp.Literal):
        if literal.is_string:
            literal.set("this", literal.this.lower())
    while isinstance(node, exp.Paren):
        node = node.this
    return node


def _option_conditions(spec, tables):
    from components.report_options import filter_predicates
    aliases = {table: alias for alias, table in tables.items()}
    return [parse_one(sql, read="mysql") for sql in filter_predicates(spec.get("filters") or {}, aliases)]


def _unexplained_predicates(tree, tables, spec, *, payment=False):
    """Every row restriction must be represented, even if other checks pass."""
    try:
        options = [_canonical(n, tables) for n in _option_conditions(spec, tables)]
    except ValueError:
        return ["report conditions require their verified source tables"]
    from components.query_spec import entity_constraints
    roles = entity_constraints(spec)
    allowed = list(options)
    for role, constraint in roles.items():
        table = "sf_contacts" if role == "donor" else "sf_ministries"
        alias = next((alias for alias, name in tables.items() if name == table), None)
        if not alias:
            continue
        if role == "donor":
            allowed.extend(_canonical(n, tables) for n in _conjuncts(_donor_filter(constraint, alias)))
        elif constraint.get("entity_id"):
            value = str(constraint["entity_id"]).replace("'", "''")
            allowed.append(_canonical(parse_one(f"{alias}.sfid = '{value}'", read="mysql"), tables))
    if payment:
        alias = next((a for a, name in tables.items() if name == "sfpayments"), None)
        if alias:
            if spec.get("status"):
                allowed.append(_canonical(parse_one(f"{alias}.paid = {1 if spec['status'] == 'paid' else 0}", read="mysql"), tables))
            if spec.get("start_date") and spec.get("exclusive_end_date"):
                allowed.extend(_canonical(parse_one(sql, read="mysql"), tables) for sql in (
                    f"{alias}.paymentdate >= '{spec['start_date']}'", f"{alias}.paymentdate < '{spec['exclusive_end_date']}'"))
    canonical_tables = {name: name for name in tables.values()}
    signatures = {_comparison(n, canonical_tables) for n in allowed if _comparison(n, canonical_tables) is not None}
    joins = [
        {("sfpayments", "ministryid"), ("sf_ministries", "sfid")},
        {("sfpayments", "oppsfid"), ("sf_opportunities", "sfid")},
        {("sf_opportunities", "primarycontact"), ("sf_contacts", "sfid")},
    ] if payment else []
    errors = []
    for body in _predicates(tree):
        for node in _conjuncts(body):
            normalized = _canonical(node, tables)
            comparison = _comparison(normalized, canonical_tables)
            if any(normalized == option for option in allowed) or (comparison is not None and comparison in signatures):
                continue
            n = _unwrap(node)
            refs = _refs(n, tables)
            if payment and _ministry_activity_rows(spec) and _row_identity_guard(n, tables):
                continue
            if isinstance(n, exp.EQ) and _column(n.this, tables) and _column(n.expression, tables) and refs in joins:
                continue
            if isinstance(n, exp.Boolean) and n.this is True:
                continue
            if isinstance(n, exp.EQ) and _literal(n.this) is not None and _literal(n.this) == _literal(n.expression):
                continue
            errors.append("SQL contains a restriction not represented by the report conditions")
    # Unlike identity/date checks, optional missing-value checks need an exact
    # guaranteed conjunct; an OR branch must not bypass them.
    where = tree.args.get("where")
    required = [_canonical(p, tables) for p in _conjuncts(where.this if where else None)]
    if any(not any(option == actual for actual in required) for option in options):
        errors.append("SQL must preserve every requested report condition in WHERE")
    return errors


def _output_errors(tree, tables, spec):
    from components.report_options import FIELD_COLUMNS
    errors = []
    # The supported report vocabulary has no banking/card/credential/payload
    # fields. Reject their use anywhere in SQL before any rows are fetched,
    # including incidental fields added to a broad 'payment details' request.
    if any(_NONREPORT_FIELD.search(column.name) for column in tree.find_all(exp.Column)):
        errors.append("SQL references a sensitive field outside the supported reports")
    names = [p.alias_or_name.lower() for p in tree.expressions if p.alias_or_name]
    if len(names) != len(set(names)):
        errors.append("output column names must be unique")
    if spec.get("result_shape") in {"detail", "grouped"}:
        limit = tree.args.get("limit")
        value = limit.expression if limit else None
        if not isinstance(value, exp.Literal) or value.is_string or not str(value.this).isdigit() or not 1 <= int(value.this) <= LIST_LIMIT:
            errors.append("list reports require an enforced limit between 1 and 100")
        if tree.args.get("offset"):
            errors.append("this report does not request a pagination offset")
    fields = set().union(*(_refs(p, tables) for p in tree.expressions))
    if any(isinstance(p, exp.Star) or p.find(exp.Star) and not p.find(exp.Count) for p in tree.expressions):
        errors.append("select named report columns, not a wildcard")
    for field in spec.get("display_columns") or []:
        expected = FIELD_COLUMNS.get(field)
        if not expected or not expected.issubset(fields):
            errors.append("SQL is missing a requested display field: " + str(field))
    ranking = spec.get("ranking")
    if ranking:
        if ranking.get("limit"):
            limit = tree.args.get("limit")
            value = limit.expression if limit else None
            if not isinstance(value, exp.Literal) or value.is_string or not str(value.this).isdigit() or int(value.this) != ranking["limit"]:
                errors.append("ranking limit must match the requested bounded number of ministries")
        order = tree.args.get("order")
        first = order.expressions[0] if order and order.expressions else None
        alias = ALIASES.get(spec.get("metric"))
        if (not isinstance(first, exp.Ordered) or not isinstance(first.this, exp.Column)
                or first.this.name.lower() != alias or bool(first.args.get("desc")) != (ranking.get("direction") == "desc")):
            errors.append("ranking must order by the requested measure and direction")
    return errors


def _monthly_errors(tree, tables, spec):
    """One calendar year-month bucket and one unchanged numeric measure."""
    errors = []
    payment_alias = next((alias for alias, table in tables.items() if table == "sfpayments"), None)
    if not payment_alias:
        return ["monthly reporting requires sfpayments.paymentdate"]
    expected = parse_one(f"DATE_FORMAT({payment_alias}.paymentdate, '%Y-%m')", read="mysql")
    # Do not lowercase format literals: %Y (year) and %y (two-digit year)
    # differ. Only canonicalise column aliases for this date expression.
    def month_expression(node):
        normalized = node.copy()
        for column in normalized.find_all(exp.Column):
            table, name = _column(column, tables)
            if table:
                column.set("table", exp.to_identifier(table))
            column.set("this", exp.to_identifier(name))
        return normalized
    expected = month_expression(expected)
    projections = {p.alias.lower(): p.this for p in tree.expressions if isinstance(p, exp.Alias)}
    metric_alias = ALIASES.get(spec.get("metric"))
    if len(tree.expressions) != 2 or set(projections) != {"month", metric_alias}:
        errors.append("monthly output must contain only month and the requested aggregate alias")
    month = projections.get("month")
    if month is None or month_expression(month) != expected:
        errors.append("month must be DATE_FORMAT(p.paymentdate, '%Y-%m') AS month")
    def is_month(node):
        return (isinstance(node, exp.Column) and not node.table and node.name.lower() == "month"
                or month_expression(node) == expected)
    group = tree.args.get("group")
    if (not group or len(group.expressions) != 1 or not is_month(group.expressions[0])
            or any(group.args.get(key) for key in ("rollup", "cube", "grouping_sets", "totals", "all"))):
        errors.append("group monthly reports by calendar year and month only, without rollups")
    order = tree.args.get("order")
    first = order.expressions[0] if order and len(order.expressions) == 1 else None
    if not isinstance(first, exp.Ordered) or first.args.get("desc") or not is_month(first.this):
        errors.append("monthly reports must ORDER BY month ASC")
    if tree.args.get("distinct") or tree.args.get("having"):
        errors.append("monthly reports must not discard month groups using DISTINCT or HAVING")
    if spec.get("subject") != "payment" or spec.get("qualification") or spec.get("ranking"):
        errors.append("monthly reporting supports ordinary chronological payment aggregates only")
    return errors


def _unwrap(node):
    while isinstance(node, (exp.Paren, exp.Lower, exp.Upper, exp.Trim)):
        node = node.this
    return node


def _tables(select):
    return {t.alias_or_name.lower(): t.name.lower() for t in select.find_all(exp.Table)
            if t.find_ancestor(exp.Select) is select}


def _column(node, tables):
    node = _unwrap(node)
    if not isinstance(node, exp.Column):
        return None
    table = tables.get(node.table.lower())
    if table is None and not node.table and len(tables) == 1:
        table = next(iter(tables.values()))
    return (table, node.name.lower())


def _literal(node):
    node = _unwrap(node)
    if isinstance(node, exp.Literal):
        return node.this.lower()
    if isinstance(node, exp.Boolean):
        return "1" if node.this else "0"
    return None


def _guaranteed(node):
    """Predicates required on every path; an OR branch cannot bypass a filter."""
    if node is None:
        return []
    node = _unwrap(node)
    if isinstance(node, exp.And):
        return _guaranteed(node.this) + _guaranteed(node.expression)
    if isinstance(node, exp.Or):
        left, right = _guaranteed(node.this), _guaranteed(node.expression)
        return [p for p in left if any(p == q for q in right)]
    return [node]


def _conjuncts(node):
    if node is None:
        return []
    node = _unwrap(node)
    if isinstance(node, exp.And):
        return _conjuncts(node.this) + _conjuncts(node.expression)
    return [node]


def _predicates(select):
    bodies = [clause.this for key in ("where", "having")
              if (clause := select.args.get(key)) is not None]
    bodies += [join.args["on"] for join in select.args.get("joins", []) if join.args.get("on")]
    return bodies


def _comparison(node, tables):
    if not isinstance(node, (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE, exp.Like)):
        return None
    col, value = _column(node.this, tables), _literal(node.expression)
    kind = type(node)
    if col is None or value is None:
        col, value = _column(node.expression, tables), _literal(node.this)
        kind = {exp.GT: exp.LT, exp.GTE: exp.LTE, exp.LT: exp.GT, exp.LTE: exp.GTE}.get(kind, kind)
    return (col, kind, value) if col and value is not None else None


def _refs(node, tables):
    return {_column(c, tables) for c in node.find_all(exp.Column)}


def _has_join(predicates, tables, left, right):
    for node in predicates:
        if isinstance(node, exp.EQ):
            refs = {_column(node.this, tables), _column(node.expression, tables)}
            if refs == {left, right}:
                return True
    return False


def _donor_filter(spec, alias):
    name = str(spec.get("entity_name") or "").strip()
    code = spec.get("entity_code")
    if code or re.fullmatch(r"#?\d+", name):
        donor_id = str(code or name).lstrip("#").replace("'", "''")
        return parse_one(f"{alias}.donorid = '#{donor_id}'", read="mysql")
    escaped = name.lower().replace("\\", "\\\\").replace("'", "''")
    parts = escaped.split()
    org = re.search(r"\b(?:missions?|trust|foundation|charitable|church|ministr(?:y|ies)|"
                    r"society|association|services?|group|fund|charity|international)\b", escaped)
    if len(parts) > 1 and not org:
        first, last = " ".join(parts[:-1]), parts[-1]
        return parse_one(f"LOWER(TRIM({alias}.firstname)) LIKE '%{first}%' "
                         f"AND LOWER(TRIM({alias}.lastname)) = '{last}'", read="mysql")
    return parse_one(f"(LOWER(TRIM({alias}.firstname)) LIKE '%{escaped}%' "
                     f"OR LOWER(TRIM({alias}.lastname)) LIKE '%{escaped}%')", read="mysql")


def _identity_filter(node, tables, role):
    if isinstance(node, exp.EQ) and _column(node.this, tables) and _column(node.expression, tables):
        return False  # preserve column-to-column relationship joins
    columns = _refs(node, tables)
    if role == "receiving_ministry":
        return any(t == "sf_ministries" and c in {"sfid", "id", "name", "giftcode"}
                   or (t, c) == ("sfpayments", "ministryid") for t, c in columns if t)
    return any(t == "sf_contacts" and c in {"sfid", "donorid", "firstname", "lastname"}
               for t, c in columns if t)


def _remove_identity(node, tables, role):
    if node is None:
        return None
    if isinstance(node, exp.Paren):
        result = _remove_identity(node.this, tables, role)
        return exp.Paren(this=result) if result is not None else None
    if isinstance(node, (exp.And, exp.Or)):
        left = _remove_identity(node.this, tables, role)
        right = _remove_identity(node.expression, tables, role)
        if left is None:
            return right
        if right is None:
            return left
        return type(node)(this=left, expression=right)
    return None if _identity_filter(node, tables, role) else node.copy()


def _enforce_single_entity(sql: str, spec: dict) -> str:
    """Replace only the requested entity's filters; preserve dates and joins."""
    role = spec.get("entity_role")
    if role not in {"receiving_ministry", "donor"} or not (spec.get("entity_id") or spec.get("entity_name") or spec.get("entity_code")):
        return sql
    tree = parse_one(sql, read="mysql")
    if not isinstance(tree, exp.Select) or len(list(tree.find_all(exp.Select))) != 1:
        return sql  # validator requests a directly verifiable SELECT on retry
    tables = _tables(tree)
    if len(tables.values()) != len(set(tables.values())):
        return sql
    table = "sf_ministries" if role == "receiving_ministry" else "sf_contacts"
    aliases = [alias for alias, name in tables.items() if name == table]
    if len(aliases) != 1:
        return sql
    alias = aliases[0]
    if role == "receiving_ministry":
        if not spec.get("entity_id"):
            return sql
        value = str(spec["entity_id"]).replace("\\", "\\\\").replace("'", "''")
        required = parse_one(f"{alias}.sfid = '{value}'", read="mysql")
    else:
        required = _donor_filter(spec, alias)
    where = tree.args.get("where")
    remaining = _remove_identity(where.this, tables, role) if where else None
    tree.set("where", exp.Where(this=exp.and_(required, remaining) if remaining is not None else required))
    # Filters in JOIN ON can conflict too; keep column-to-column joins intact.
    for join in tree.args.get("joins", []):
        if join.args.get("on") is not None:
            remaining = _remove_identity(join.args["on"], tables, role)
            join.set("on", remaining if remaining is not None else exp.true())
    return tree.sql(dialect="mysql")

def enforce_entity_filter(sql: str, spec: dict) -> str:
    """Enforce every named role, never trade a receiver for a donor."""
    from components.query_spec import entity_constraints
    for constraint in entity_constraints(spec).values():
        sql = _enforce_single_entity(sql, constraint)
    return sql


def unlimited_count_sql(sql: str) -> str | None:
    """Count the same result, retaining grouping/HAVING and removing pagination."""
    tree = parse_one(sql, read="mysql")
    if not tree.args.get("limit") and not tree.args.get("offset"):
        return None
    tree.set("limit", None)
    tree.set("offset", None)
    tree.set("order", None)
    return f"SELECT COUNT(*) AS total_count FROM ({tree.sql(dialect='mysql')}) AS _counted"


def semantic_sql_errors(sql: str, spec: dict) -> list[str]:
    from security.domain_guard import domain_guard
    safe, reason = domain_guard.check_sql(sql)
    if not safe:
        return [reason]
    if spec.get("period_ambiguity") or spec.get("clarification") or spec.get("unresolved_condition"):
        return ["resolve the report clarification before executing SQL"]
    if spec.get("qualification"):
        from components.report_rules import special_report_sql
        try:
            expected = special_report_sql(spec)
            if expected is None or parse_one(sql, read="mysql") != parse_one(expected, read="mysql"):
                return ["qualifying report SQL does not match its approved filters, grouping and result"]
            tree = parse_one(sql, read="mysql")
            return _output_errors(tree, _tables(tree), spec)
        except (ValueError, TypeError, AttributeError):
            return ["qualifying report SQL could not be verified"]
    if spec.get("subject") != "payment":
        if spec.get("result_entity") == "ministry":
            return _ministry_sql_errors(sql, spec)
        if spec.get("result_entity") == "donor":
            return _contact_sql_errors(sql, spec)
        return ["report meaning is not represented by a supported query contract"]
    errors = []
    try:
        statements = parse(sql, read="mysql")
        if len(statements) != 1 or not isinstance(statements[0], exp.Select):
            return ["payment analytics requires one SELECT statement"]
        tree = statements[0]
    except Exception:
        return ["SQL could not be parsed; generate valid MySQL SELECT"]
    if len(list(tree.find_all(exp.Select))) != 1:
        return ["use a direct SELECT with relationship joins so the query specification can be verified"]
    tables = _tables(tree)
    errors += _output_errors(tree, tables, spec)
    errors += _unexplained_predicates(tree, tables, spec, payment=True)
    if (_ministry_activity_rows(spec) and
            {"sfpayments", "sf_ministries"} - _present_activity_sources(tree, tables)):
        errors.append("ministry activity rows require a real linked ministry and contributing payment")
    if set(tables.values()) - {"sfpayments", "sf_opportunities", "sf_contacts", "sf_ministries"}:
        errors.append("payment reports require verified core-table relationships")
    if len(tables.values()) != len(set(tables.values())):
        errors.append("do not join the same table multiple times in this payment report")
    if list(tables.values()).count("sfpayments") != 1:
        errors.append("payment subject requires exactly one sfpayments source")
    where = tree.args.get("where")
    required = _guaranteed(where.this if where else None)
    bodies = _predicates(tree)
    join_predicates = [p for body in bodies for p in _guaranteed(body)]
    comparisons = [n for body in bodies for n in body.walk()
                   if isinstance(n, (exp.Predicate, exp.Binary)) and not isinstance(n, (exp.And, exp.Or))
                   and not (_ministry_activity_rows(spec) and _row_identity_guard(n.parent, tables))]
    required_comparisons = [_comparison(p, tables) for p in required]
    from components.query_spec import entity_constraints
    roles = entity_constraints(spec)
    receiver, donor = roles.get("receiving_ministry"), roles.get("donor")
    filters = spec.get("filters") or {}
    required_sources = {
        "sf_ministries": bool(receiver or spec.get("result_entity") == "ministry" or spec.get("group_by") == "ministry"
                              or "description_missing" in filters or "ministry_active" in filters),
        "sf_contacts": bool(donor or spec.get("result_entity") == "donor" or "email_missing" in filters),
    }
    required_sources["sf_opportunities"] = required_sources["sf_contacts"]
    for join in tree.args.get("joins", []):
        table = join.this.name.lower() if isinstance(join.this, exp.Table) else None
        if table in required_sources and not required_sources[table] and join.side.upper() != "LEFT":
            errors.append("optional display joins must preserve payments using LEFT JOIN")
    ministry_needed = bool(receiver) or spec.get("result_entity") == "ministry" or spec.get("group_by") == "ministry" or "sf_ministries" in tables.values()
    if ministry_needed:
        if not _has_join(join_predicates, tables, ("sfpayments", "ministryid"), ("sf_ministries", "sfid")):
            errors.append("receiving ministry join must be p.ministryid = m.sfid")
    if "sf_opportunities" in tables.values():
        if not _has_join(join_predicates, tables, ("sfpayments", "oppsfid"), ("sf_opportunities", "sfid")):
            errors.append("opportunity join must be p.oppsfid = o.sfid")
    if donor or "sf_contacts" in tables.values():
        if not _has_join(join_predicates, tables, ("sf_opportunities", "primarycontact"), ("sf_contacts", "sfid")):
            errors.append("donor join must be o.primarycontact = c.sfid")
    if receiver and (receiver.get("entity_name") or receiver.get("entity_code") or receiver.get("entity_id")):
        wanted = (("sf_ministries", "sfid"), exp.EQ, str(receiver.get("entity_id") or "").lower())
        if not receiver.get("entity_id") or wanted not in required_comparisons:
            errors.append("canonical ministry constraint must be m.sfid = resolved entity_id in WHERE")
        entity_conditions = [n for n in comparisons if _identity_filter(n, tables, "receiving_ministry")]
        if len(entity_conditions) != 1 or any(_comparison(n, tables) != wanted for n in entity_conditions):
            errors.append("remove extra ministry name/code/ID filters; use only one official ministry ID predicate")
    if donor:
        contact_aliases = [a for a, t in tables.items() if t == "sf_contacts"]
        if len(contact_aliases) != 1 or not donor.get("entity_name") and not donor.get("entity_code"):
            errors.append("donor query requires an identified donor and sf_contacts")
        else:
            expected = _donor_filter(donor, contact_aliases[0])
            expected_parts = _conjuncts(expected)
            donor_conjuncts = _conjuncts(where.this if where else None)
            if not all(any(p == actual for actual in donor_conjuncts) for p in expected_parts):
                errors.append("donor WHERE predicate must match the requested name/ID using the documented name rules")
            allowed = {_comparison(n, tables) for n in expected.walk() if isinstance(n, (exp.EQ, exp.Like))}
            if any(_comparison(n, tables) not in allowed for n in comparisons if _identity_filter(n, tables, "donor")):
                errors.append("SQL contains a conflicting donor filter")
    if not donor and any(_identity_filter(n, tables, "donor") for n in comparisons):
        errors.append("SQL added a donor filter not requested by the specification")
    if not receiver and any(_identity_filter(n, tables, "receiving_ministry") for n in comparisons):
        errors.append("SQL added a ministry filter not requested by the specification")
    start, end = spec.get("start_date"), spec.get("exclusive_end_date")
    allowed_dates = {(("sfpayments", "paymentdate"), exp.GTE, start),
                     (("sfpayments", "paymentdate"), exp.LT, end)} if start and end else set()
    date_conditions = [n for n in comparisons if ("sfpayments", "paymentdate") in _refs(n, tables)]
    if start and end and not allowed_dates.issubset(set(required_comparisons)):
        errors.append("WHERE must use paymentdate >= start_date AND paymentdate < exclusive_end_date")
    if any(_comparison(n, tables) not in allowed_dates for n in date_conditions):
        errors.append("SQL contains an added or conflicting payment-date filter")
    status = spec.get("status")
    status_condition = (("sfpayments", "paid"), exp.EQ, "1" if status == "paid" else "0")
    if status and status_condition not in required_comparisons:
        errors.append("payment status predicate does not match the requested paid/unpaid status")
    status_conditions = [n for n in comparisons if ("sfpayments", "paid") in _refs(n, tables)]
    if any(not status or _comparison(n, tables) != status_condition for n in status_conditions):
        errors.append("SQL contains an added or conflicting payment-status filter")
    metric = spec.get("metric")
    if metric in ALIASES:
        projection = next((p for p in tree.expressions if p.alias.lower() == ALIASES[metric]), None)
        value = projection.this if isinstance(projection, exp.Alias) else None
        if isinstance(value, exp.Coalesce) and all(_literal(v) == "0" for v in value.expressions):
            value = value.this
        expected_type = {"total": exp.Sum, "average": exp.Avg, "count": exp.Count}[metric]
        if not isinstance(value, expected_type):
            errors.append(f"{metric} requires {expected_type.__name__.upper()} AS {ALIASES[metric]}")
        elif metric in {"total", "average"} and _column(value.this, tables) != ("sfpayments", "paymentamount"):
            errors.append("total/average must calculate sfpayments.paymentamount without changing the amount")
        elif metric == "count" and spec.get("result_entity") == "ministry":
            if not isinstance(value.this, exp.Distinct) or _refs(value, tables) != {("sf_ministries", "sfid")}:
                errors.append("ministry count must count DISTINCT official ministry IDs")
        elif metric == "count" and spec.get("result_entity") == "donor":
            if not isinstance(value.this, exp.Distinct) or _refs(value, tables) != {("sf_contacts", "sfid")}:
                errors.append("donor count must count DISTINCT contact IDs, not payments")
        elif metric == "count" and not isinstance(value.this, exp.Star) and _column(value.this, tables) not in {
            ("sfpayments", "id"), ("sfpayments", "sfid")
        }:
            errors.append("record_count must count payment records, not donors or ministries")
    group = tree.args.get("group")
    if spec.get("group_by") == "month":
        errors += _monthly_errors(tree, tables, spec)
    elif spec.get("group_by") == "ministry":
        grouped_columns = {_column(c, tables) for c in group.expressions} if group else set()
        if ("sf_ministries", "sfid") not in grouped_columns:
            errors.append("group ministry reports by the official ministry ID")
        if grouped_columns - {("sf_ministries", "sfid"), ("sf_ministries", "name"), ("sf_ministries", "giftcode")}:
            errors.append("ministry report introduced a different grouping dimension")
        if not any(("sf_ministries", "name") in _refs(p, tables) for p in tree.expressions):
            errors.append("ministry report must return the ministry name")
    elif spec.get("group_by"):
        errors.append("unsupported report grouping dimension")
    if spec.get("result_shape") == "scalar" and (group or tree.args.get("limit")):
        errors.append("scalar total/count/average must not have GROUP BY or LIMIT")
    if spec.get("operation") == "donor_list":
        if group or any(p.find(exp.AggFunc) for p in tree.expressions):
            errors.append("detail request was changed into an aggregate")
        if not tree.args.get("distinct"):
            errors.append("donor list must return DISTINCT donors, not one row per payment")
        fields = set().union(*(_refs(p, tables) for p in tree.expressions))
        if ("sf_contacts", "sfid") not in fields:
            errors.append("donor list must include the unique sf_contacts.sfid identifier")
        if not {("sf_contacts", "firstname"), ("sf_contacts", "lastname")}.issubset(fields):
            errors.append("donor list must return the donor name")
        if any(t != "sf_contacts" for t, c in fields if t):
            errors.append("donor list projections must describe donors, not individual payments")
    elif spec.get("operation") == "ministry_list":
        if group or any(p.find(exp.AggFunc) for p in tree.expressions):
            errors.append("ministry list request was changed into an aggregate")
        if not tree.args.get("distinct"):
            errors.append("ministry list must return DISTINCT ministries, not one row per payment")
        fields = set().union(*(_refs(p, tables) for p in tree.expressions))
        if not {("sf_ministries", "sfid"), ("sf_ministries", "name")}.issubset(fields):
            errors.append("ministry list must include the official ministry ID and name")
        if any(t != "sf_ministries" for t, c in fields if t):
            errors.append("ministry list projections must describe ministries, not payments")
    elif spec.get("result_shape") == "detail":
        if group or any(p.find(exp.AggFunc) for p in tree.expressions):
            errors.append("detail request was changed into an aggregate")
        fields = set().union(*(_refs(p, tables) for p in tree.expressions))
        if not {("sfpayments", "paymentamount"), ("sfpayments", "paymentdate")}.issubset(fields):
            errors.append("payment details must return paymentamount and paymentdate")
    return list(dict.fromkeys(errors))

def _ministry_sql_errors(sql: str, spec: dict) -> list[str]:
    """Protect the count/list distinction on the existing ministry route."""
    try:
        statements = parse(sql, read="mysql")
        if len(statements) != 1 or not isinstance(statements[0], exp.Select):
            return ["ministry report requires one SELECT"]
        tree = statements[0]
        tables = _tables(tree)
        if list(tables.values()) != ["sf_ministries"] or len(list(tree.find_all(exp.Select))) != 1:
            return ["directory reports require one ministry source; payment activity uses the payment contract"]
        errors = _directory_identity_errors(tree, tables, spec, "receiving_ministry")
        errors += _unexplained_predicates(tree, tables, spec)
        errors += _output_errors(tree, tables, spec)
        if spec.get("start_date"):
            errors.append("a ministry directory entry does not imply dated donation activity")
        if spec.get("metric") == "count":
            projection = next((p for p in tree.expressions if p.alias.lower() == "record_count"), None)
            value = projection.this if isinstance(projection, exp.Alias) else None
            if not isinstance(value, exp.Count):
                return ["ministry count requires COUNT AS record_count"]
            if len(tables) > 1 and not isinstance(value.this, exp.Distinct):
                return ["a joined ministry count must count DISTINCT official ministry IDs"]
            if not isinstance(value.this, exp.Star) and _refs(value, tables) != {("sf_ministries", "sfid"),}:
                return ["count official ministry records, not amounts or names"]
            if tree.args.get("group") or tree.args.get("limit"):
                return ["ministry count must not have GROUP BY or LIMIT"]
        elif spec.get("result_shape") == "detail":
            # Existing qualified ministry reports can contain aggregates per
            # ministry. Only reject a scalar aggregate masquerading as a list.
            if not tree.args.get("group") and any(p.find(exp.AggFunc) for p in tree.expressions):
                return ["ministry list request must return rows, not an aggregate"]
            if not any(("sf_ministries", "name") in _refs(p, tables) for p in tree.expressions):
                return ["ministry list must return ministry names"]
        return errors
    except Exception:
        return ["ministry SQL could not be verified"]


def _directory_identity_errors(tree, tables, spec, role):
    where = tree.args.get("where")
    conjuncts = _conjuncts(where.this if where else None)
    identity_nodes = [node for body in _predicates(tree) for node in body.walk()
                      if isinstance(node, (exp.Predicate, exp.Binary))
                      and not isinstance(node, (exp.And, exp.Or)) and _identity_filter(node, tables, role)]
    from components.query_spec import entity_constraints
    constraint = entity_constraints(spec).get(role)
    if not constraint:
        return ["SQL added an unrequested identity filter"] if identity_nodes else []
    alias = next(iter(tables))
    if role == "receiving_ministry":
        if not constraint.get("entity_id"):
            return ["resolve the ministry before querying its directory entry"]
        value = str(constraint["entity_id"]).replace("'", "''")
        expected = parse_one(f"{alias}.sfid = '{value}'", read="mysql")
    else:
        expected = _donor_filter(constraint, alias)
    parts = _conjuncts(expected)
    if not all(any(_canonical(p, tables) == _canonical(actual, tables) for actual in conjuncts) for p in parts):
        return ["directory identity filter does not match the requested donor or ministry"]
    allowed = {_comparison(n, tables) for n in expected.walk() if isinstance(n, (exp.EQ, exp.Like))}
    if any(_comparison(n, tables) not in allowed for n in identity_nodes):
        return ["directory query contains a conflicting identity filter"]
    return []


def _contact_sql_errors(sql, spec):
    try:
        tree = parse_one(sql, read="mysql")
        tables = _tables(tree)
        if list(tables.values()) != ["sf_contacts"] or len(list(tree.find_all(exp.Select))) != 1:
            return ["contact profiles require one contact source; donation history uses the payment contract"]
        errors = _directory_identity_errors(tree, tables, spec, "donor")
        errors += _unexplained_predicates(tree, tables, spec)
        errors += _output_errors(tree, tables, spec)
        if spec.get("start_date") or spec.get("group_by") or tree.args.get("group"):
            errors.append("current contact profiles do not imply a donation or registration period")
        if spec.get("metric") == "count":
            projection = next((p for p in tree.expressions if p.alias.lower() == "record_count"), None)
            value = projection.this if isinstance(projection, exp.Alias) else None
            if not isinstance(value, exp.Count) or not (isinstance(value.this, exp.Star) or _refs(value, tables) == {("sf_contacts", "sfid")}):
                errors.append("contact count requires COUNT of contacts AS record_count")
            if tree.args.get("limit"):
                errors.append("a contact count must not have a list limit")
        elif spec.get("metric"):
            errors.append("contact profiles support a list or a count, not payment amounts")
        else:
            fields = set().union(*(_refs(p, tables) for p in tree.expressions))
            if not {("sf_contacts", "sfid"), ("sf_contacts", "firstname"), ("sf_contacts", "lastname")}.issubset(fields):
                errors.append("contact lists must select sf_contacts.sfid plus firstname and lastname; donorID cannot replace sfid")
            if any(p.find(exp.AggFunc) for p in tree.expressions):
                errors.append("contact list cannot be replaced by an aggregate")
        return list(dict.fromkeys(errors))
    except Exception:
        return ["contact SQL could not be verified"]
