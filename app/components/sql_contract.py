"""Structural checks for the existing Admin payment query contract (AA-05).

This is a validator/rewriter inside the existing graph, not a SQL generator.
Unverifiable SQL goes through the existing retry flow instead of executing.
"""
from __future__ import annotations

import re
from sqlglot import exp, parse, parse_one

ALIASES = {"total": "total_amount", "average": "average_amount", "count": "record_count"}


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


def enforce_entity_filter(sql: str, spec: dict) -> str:
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
    if spec.get("subject") != "payment":
        return []  # keep other existing Admin features on their existing path
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
    if len(tables.values()) != len(set(tables.values())):
        errors.append("do not join the same table multiple times in this payment report")
    if list(tables.values()).count("sfpayments") != 1:
        errors.append("payment subject requires exactly one sfpayments source")
    where = tree.args.get("where")
    required = _guaranteed(where.this if where else None)
    bodies = _predicates(tree)
    join_predicates = [p for body in bodies for p in _guaranteed(body)]
    comparisons = [n for body in bodies for n in body.walk()
                   if isinstance(n, (exp.Predicate, exp.Binary)) and not isinstance(n, (exp.And, exp.Or))]
    required_comparisons = [_comparison(p, tables) for p in required]
    role = spec.get("entity_role")
    ministry_needed = role == "receiving_ministry" or spec.get("group_by") == "ministry" or "sf_ministries" in tables.values()
    if ministry_needed:
        if not _has_join(join_predicates, tables, ("sfpayments", "ministryid"), ("sf_ministries", "sfid")):
            errors.append("receiving ministry join must be p.ministryid = m.sfid")
    if "sf_opportunities" in tables.values():
        if not _has_join(join_predicates, tables, ("sfpayments", "oppsfid"), ("sf_opportunities", "sfid")):
            errors.append("opportunity join must be p.oppsfid = o.sfid")
    if role == "donor" or "sf_contacts" in tables.values():
        if not _has_join(join_predicates, tables, ("sf_opportunities", "primarycontact"), ("sf_contacts", "sfid")):
            errors.append("donor join must be o.primarycontact = c.sfid")
    if role == "receiving_ministry" and (spec.get("entity_name") or spec.get("entity_code") or spec.get("entity_id")):
        wanted = (("sf_ministries", "sfid"), exp.EQ, str(spec.get("entity_id") or "").lower())
        if not spec.get("entity_id") or wanted not in required_comparisons:
            errors.append("canonical ministry constraint must be m.sfid = resolved entity_id in WHERE")
        entity_conditions = [n for n in comparisons if _identity_filter(n, tables, role)]
        if len(entity_conditions) != 1 or any(_comparison(n, tables) != wanted for n in entity_conditions):
            errors.append("remove extra ministry name/code/ID filters; use only one official ministry ID predicate")
    if role == "donor":
        contact_aliases = [a for a, t in tables.items() if t == "sf_contacts"]
        if len(contact_aliases) != 1 or not spec.get("entity_name") and not spec.get("entity_code"):
            errors.append("donor query requires an identified donor and sf_contacts")
        else:
            expected = _donor_filter(spec, contact_aliases[0])
            expected_parts = _conjuncts(expected)
            donor_conjuncts = _conjuncts(where.this if where else None)
            if not all(any(p == actual for actual in donor_conjuncts) for p in expected_parts):
                errors.append("donor WHERE predicate must match the requested name/ID using the documented name rules")
            allowed = {_comparison(n, tables) for n in expected.walk() if isinstance(n, (exp.EQ, exp.Like))}
            if any(_comparison(n, tables) not in allowed for n in comparisons if _identity_filter(n, tables, "donor")):
                errors.append("SQL contains a conflicting donor filter")
    if role != "donor" and any(_identity_filter(n, tables, "donor") for n in comparisons):
        errors.append("SQL added a donor filter not requested by the specification")
    if role != "receiving_ministry" and any(_identity_filter(n, tables, "receiving_ministry") for n in comparisons):
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
        elif metric == "count" and not isinstance(value.this, exp.Star) and _column(value.this, tables) not in {
            ("sfpayments", "id"), ("sfpayments", "sfid")
        }:
            errors.append("record_count must count payment records, not donors or ministries")
    group = tree.args.get("group")
    if spec.get("group_by") == "ministry":
        grouped_columns = {_column(c, tables) for c in group.expressions} if group else set()
        if ("sf_ministries", "sfid") not in grouped_columns:
            errors.append("group ministry reports by the official ministry ID")
        if grouped_columns - {("sf_ministries", "sfid"), ("sf_ministries", "name"), ("sf_ministries", "giftcode")}:
            errors.append("ministry report introduced a different grouping dimension")
        if not any(("sf_ministries", "name") in _refs(p, tables) for p in tree.expressions):
            errors.append("ministry report must return the ministry name")
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
    elif spec.get("result_shape") == "detail":
        if group or any(p.find(exp.AggFunc) for p in tree.expressions):
            errors.append("detail request was changed into an aggregate")
        fields = set().union(*(_refs(p, tables) for p in tree.expressions))
        if not {("sfpayments", "paymentamount"), ("sfpayments", "paymentdate")}.issubset(fields):
            errors.append("payment details must return paymentamount and paymentdate")
    return list(dict.fromkeys(errors))
