"""Reversible PII masking via Microsoft Presidio (analyzer-only, no anonymizer)."""
from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Entities Presidio will detect.
_ENTITIES = [
    "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER",
    "CREDIT_CARD", "US_SSN", "IP_ADDRESS", "LOCATION",
]
_SCORE_THRESHOLD = 0.4
_LANGUAGE = "en"

# Custom donor-id recogniser: matches "#000032" style ids.
_DONOR_ID_PATTERN = [{"name": "DONOR_ID", "regex": r"#\d{4,8}\b", "score": 0.9}]

# Ministry codes like "098WRLD" must survive masking so SQL keeps working.
_MINISTRY_CODE_RE = re.compile(r"\b\d{3}[A-Z]{3,6}\b")
# Common words that Presidio sometimes flags as PERSON/LOCATION — drop them.
_ALLOWLIST = {"ministry", "donor", "payment", "admin"}

# SQL-ish noise that Presidio mis-flags as entities (root cause of
# "m.sfid\nWHERE" being masked as a PERSON). Drop these unless EMAIL_ADDRESS.
_SQL_NOISE_RE = re.compile(r"(?i)\b(select|from|where|join|group by|order by)\b")
_ALIAS_DOT_RE = re.compile(r"\w\.\w")
_TOKEN_RE = re.compile(r"^<([A-Z_]+)_(\d+)>$")


class DataMasker:
    """Detects PII in text/rows and replaces it with reversible tokens.

    Call sites thread ONE shared mapping across multiple mask_text/mask_rows
    calls so the same original value maps to the same token everywhere (and
    different originals never collide on the same token). Fail-closed: any
    unexpected analyzer error is re-raised as RuntimeError so unmasked data
    never reaches the LLM.
    """

    def __init__(self) -> None:
        self._analyzer: Any = None
        self._initialized = False

    # ── lazy, guarded init ────────────────────────────────────────────────
    def _ensure_engine(self) -> Any:
        if self._initialized:
            return self._analyzer
        # deliberate: simple bool guard is "thread-safe enough" for startup;
        # worst case the engine is built twice, which is harmless.
        from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
        from presidio_analyzer.nlp_engine import NlpEngineProvider

        provider = NlpEngineProvider(nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_md"}],
        })
        nlp_engine = provider.create_engine()
        analyzer = AnalyzerEngine(
            nlp_engine=nlp_engine,
            supported_languages=[_LANGUAGE],
        )
        analyzer.registry.add_recognizer(
            PatternRecognizer(
                name="DONOR_ID",
                patterns=[Pattern(p["name"], p["regex"], p["score"]) for p in _DONOR_ID_PATTERN],
                supported_entity="DONOR_ID",
            )
        )
        self._analyzer = analyzer
        self._initialized = True
        logger.info("Presidio analyzer initialized (en_core_web_md)")
        return analyzer

    def warm_up(self) -> None:
        """Trigger lazy init eagerly (call from app startup)."""
        self._ensure_engine()

    # ── core masking ──────────────────────────────────────────────────────
    def mask_text(self, text: str, mapping: dict | None = None) -> tuple[str, dict]:
        """Mask PII in *text*. If *mapping* is given it is mutated in place
        and returned; existing tokens are reused and counters continue from
        the highest existing per-type index. If *mapping* is None a fresh
        dict is used (legacy single-call behavior)."""
        mapping = {} if mapping is None else mapping
        if not text:
            return "", mapping
        return self._mask_one(text, mapping), mapping

    # ── core unmasking ────────────────────────────────────────────────────
    def unmask_text(self, text: str, mapping: dict) -> str:
        if not text or not mapping:
            return text or ""
        # Replace longer tokens first to avoid prefix collisions. LLMs often
        # alter the token's case/spacing (e.g. <PERSON_1> → <person_1>), so a
        # tolerant regex pass follows the exact-match pass.
        for token in sorted(mapping, key=len, reverse=True):
            text = text.replace(token, mapping[token])
        # Tolerant second pass: <\s*PERSON[_\s-]?N\s*> case-insensitive.
        m = _TOKEN_RE.match
        for token, original in mapping.items():
            mt = m(token)
            if not mt:
                continue
            etype, idx = mt.group(1), mt.group(2)
            pat = re.compile(r"<\s*" + etype + r"[_\s-]?" + idx + r"\s*>", re.IGNORECASE)
            text = pat.sub(original, text)
        return text

    # ── SQL-aware unmasking (word-wise LIKE expansion) ─────────────────────
    # Matches an expression (CONCAT, optionally wrapped in LOWER/TRIM, with
    # one level of inner-paren nesting for COALESCE) followed by
    # LIKE '%<PERSON_N>%'. Allows optional whitespace.
    _LIKE_PERSON_RE = re.compile(
        r"((?:LOWER\s*\(\s*)?(?:TRIM\s*\(\s*)?"
        r"CONCAT\s*\((?:[^()]|\([^()]*\))*\)"
        r"\s*\)?\s*\)?)"
        r"\s+LIKE\s+'%(<PERSON_\d+>)%'",
        re.IGNORECASE,
    )

    def unmask_sql(self, sql: str, mapping: dict) -> str:
        """Unmask SQL with word-wise expansion for PERSON tokens inside LIKE
        patterns. A pattern like  <expr> LIKE '%<PERSON_1>%'  where <expr> is a
        CONCAT(...) (optionally wrapped in LOWER(...)) is expanded to per-word
        AND conditions:

          (<expr> LIKE '%William%' AND <expr> LIKE '%Suiter%')

        so compound firstnames like 'William O & Mary Lee' + lastname 'Suiter'
        still match. All remaining tokens are then unmasked via unmask_text.
        """
        if not sql or not mapping:
            return sql or ""

        def _expand(match: re.Match) -> str:
            expr = match.group(1)
            token = match.group(2)
            value = mapping.get(token)
            if value is None:
                return match.group(0)  # leave for unmask_text / tolerant pass
            words = self._like_words(value)
            if not words:
                return match.group(0)
            # If the expression was wrapped in LOWER(...), lowercase the words
            # so the LIKE comparison stays case-insensitive-consistent.
            if re.match(r"\s*LOWER\s*\(", expr, re.IGNORECASE):
                words = [w.lower() for w in words]
            conditions = " AND ".join(
                f"{expr} LIKE '%{w}%'" for w in words
            )
            return f"({conditions})"

        transformed = self._LIKE_PERSON_RE.sub(_expand, sql)
        count = len(self._LIKE_PERSON_RE.findall(sql))
        if count:
            logger.info("unmask_sql: expanded %d PERSON LIKE pattern(s)", count)
        return self.unmask_text(transformed, mapping)

    @staticmethod
    def _like_words(value: str) -> list[str]:
        """Split a PERSON value into LIKE-safe words: keep words with len >= 2
        after stripping '.', drop '&' and the connector 'and'. Each kept word
        is escaped for SQL LIKE (backslash-escape % and _, double ').
        """
        out: list[str] = []
        for raw in value.split():
            stripped = raw.strip(".")
            if len(stripped) < 2:
                continue
            if stripped.lower() == "and" or stripped == "&":
                continue
            escaped = stripped.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_").replace("'", "''")
            out.append(escaped)
        return out

    # ── row-level masking ────────────────────────────────────────────────
    def mask_rows(self, rows: list[dict], mapping: dict | None = None) -> tuple[list[dict], dict]:
        """Mask PII across *rows*. *mapping* semantics match mask_text:
        threaded mapping is mutated in place and returned."""
        mapping = {} if mapping is None else mapping
        if not rows:
            return [], mapping
        out_rows: list[dict] = []
        for row in rows:
            new_row: dict = {}
            for key, value in row.items():
                if not isinstance(value, str):
                    new_row[key] = value
                    continue
                new_row[key] = self._mask_one(value, mapping)
            out_rows.append(new_row)
        if mapping:
            logger.info("masked %d entities across %d rows", len(mapping), len(rows))
        return out_rows, mapping

    # ── shared helper ─────────────────────────────────────────────────────
    def _mask_one(self, text: str, mapping: dict) -> str:
        """Analyze → filter → replace, continuing counters from *mapping*.

        Mutates *mapping* in place: token→original. Maintains a reverse
        value→token index so the same original reuses its token across
        calls. Per-entity counters seed from existing tokens in the mapping
        (e.g. <PERSON_2> present → next PERSON is <PERSON_3>).
        """
        try:
            analyzer = self._ensure_engine()
            results = analyzer.analyze(
                text=text,
                entities=_ENTITIES + ["DONOR_ID"],
                language=_LANGUAGE,
                score_threshold=_SCORE_THRESHOLD,
            )
        except RuntimeError:
            raise
        except Exception as e:  # fail-closed
            raise RuntimeError("PII masking failed — refusing to send unmasked data to LLM") from e

        accepted = self._filter_and_resolve(results, text)

        # Reverse index value→token for reuse across calls.
        value_to_token: dict[str, str] = {v: t for t, v in mapping.items()}
        # Seed per-type counters from existing tokens in the mapping.
        counters: dict[str, int] = {}
        for token in mapping:
            m = _TOKEN_RE.match(token)
            if m:
                etype, idx = m.group(1), int(m.group(2))
                if idx > counters.get(etype, 0):
                    counters[etype] = idx

        # Replace right-to-left so indices stay valid.
        masked = text
        for r in accepted:
            original = text[r.start:r.end]
            token = value_to_token.get(original)
            if token is None:
                etype = r.entity_type
                counters[etype] = counters.get(etype, 0) + 1
                token = f"<{etype}_{counters[etype]}>"
                value_to_token[original] = token
                mapping[token] = original
            masked = masked[:r.start] + token + masked[r.end:]

        if len(accepted):
            logger.info("masked %d entities", len(accepted))
        return masked

    # ── detection filtering ───────────────────────────────────────────────
    def _filter_and_resolve(self, results: list[Any], text: str) -> list[Any]:
        """Drop allowlisted words, ministry codes, and SQL-ish noise; resolve
        overlaps by score then length. Returns accepted spans sorted
        right-to-left for index-safe replacement."""
        cleaned: list[Any] = []
        for r in results:
            matched = text[r.start:r.end]
            if matched.lower() in _ALLOWLIST:
                continue
            if _MINISTRY_CODE_RE.fullmatch(matched):
                continue
            if r.entity_type != "EMAIL_ADDRESS" and _is_sql_noise(matched):
                continue
            cleaned.append(r)

        # Resolve overlaps: sort by score desc then length desc, skip any span
        # that overlaps an already-accepted span.
        cleaned.sort(key=lambda r: (r.score, r.end - r.start), reverse=True)
        accepted: list[Any] = []
        occupied: list[tuple[int, int]] = []
        for r in cleaned:
            if any(not (r.end <= s or r.start >= e) for s, e in occupied):
                continue
            accepted.append(r)
            occupied.append((r.start, r.end))

        accepted.sort(key=lambda r: r.start, reverse=True)
        return accepted


def _is_sql_noise(matched: str) -> bool:
    """True for matches that are SQL fragments Presidio mis-flagged as PII:
    contains a newline, a SQL keyword, or an alias-style `word.word` ref."""
    if "\n" in matched:
        return True
    if _SQL_NOISE_RE.search(matched):
        return True
    if _ALIAS_DOT_RE.search(matched):
        return True
    return False


masker = DataMasker()
