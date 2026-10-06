"""AgentCore Platform v1.0"""

# RET-C2-005 - PostProcessNode
# Outer backbone post_process slot: the external-output boundary for the
# shrinkage/inventory-anomaly Q&A answer.  Three independent layers, in order:
#
#   (1) credential scan - API keys, JWTs, Bearer tokens, password assignments
#       anywhere in the assembled answer withhold the output entirely
#       (sanitised stub, status=ERROR);
#   (2) verbatim caller-text redaction - the answer is built from retrieved
#       knowledge-base content only, so a verbatim embedding of the caller's
#       raw question (or derived query text) is a leak, not a feature: any
#       such embedding is replaced with [REDACTED];
#   (3) monetary precision grid - the documented external schema expresses
#       monetary figures in units of 1,000; every monetary-form token is
#       snapped onto that grid (off-grid values are full-precision figures
#       leaking to the external surface), with an audit event per redaction.
#
# The domain output gate is the module-level function `_security_gate_output`
# called from inside execute() - NOT an instance method on the node class
# (the framework auto-wraps node instance methods on the real invoke path,
# which would raise at class definition).  The agent class
# (ShrinkageInventoryQAAgent) exposes the same credential scanner as the
# canonical output-gate entry point and delegates to this module (single
# source of truth).
#
# Returns ONLY the state keys this node writes (partial-dict contract).

import logging
import re
from typing import Any, ClassVar, List, Optional, Tuple

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event

logger = logging.getLogger(__name__)

# Credential patterns that MUST NOT appear in the formatted Q&A output.
_CREDENTIAL_PATTERNS: List[Tuple[str, str]] = [
    (r"(?:sk|pk|ak)-[A-Za-z0-9]{16,}", "api_key_pattern"),
    (r"eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+", "jwt_pattern"),
    (r"Bearer\s+[A-Za-z0-9_\-\.]{8,}", "bearer_token"),
    (
        r"(?:password|passwd|secret|api_key|token|access_key|private_key)" r"\s*[:=]\s*\S{8,}",
        "credential_assignment",
    ),
]

# State fields that must NEVER be embedded verbatim in the external response.
# The answer is assembled from retrieved knowledge-base passages and their
# citations only - caller-derived text (the raw question, its normalised
# forms, request metadata) re-appearing verbatim means caller-controlled
# content reached the external surface.
_BLOCKED_FIELDS = frozenset(
    {
        "user_input",
        "validated_input",
        "search_query",
        "enriched_context",
    }
)

# Approved external precision: monetary figures are expressed in units of
# 1,000 (must match the schema note rendered by
# src/nodes/output_format_node.py - the answer RENDERS on this grid, this
# gate ENFORCES it).
_EXTERNAL_ROUND_UNIT = 1000
# EXPLICIT output schema - monetary values are identified by FORM and by
# CURRENCY CONTEXT, never by magnitude:
#   form:    comma-grouped numbers (9,999 / 1,234,567) and unformatted runs
#            of 5+ digits (a rendering-regression leak);
#   context: any bare 1-4 digit number associated with a currency marker is
#            monetary even though short - SYMMETRICALLY: a 3-letter uppercase
#            code or a currency symbol (incl. fullwidth ￥ and 円/₩), before or
#            after the value, signed or unsigned, separated by horizontal
#            whitespace or at most one newline.
# Structural tokens stay untouched: SKU counts ("12 units"), register numbers,
# version tags ("v12"), bare counts, years without currency adjacency
# ("in 2026").
# ALL matched tokens, at ANY magnitude, must sit on the rounding grid;
# off-grid = a full-precision figure reaching the external surface,
# snapped + audited.
# Grammar (group-based; no variable-width lookbehinds, so the marker/value
# delimiter can be an arbitrary run of horizontal whitespace). Every value
# accepts an optional explicit +/- sign and absorbs its own decimal part.
# Branch order matters: currency-context branches first, then form-based.
_SYMBOL = r"[¥￥$€£円₩]"

# A standalone 3-letter uppercase word. Reading ANY such word as a currency
# marker is deliberate where the marker is SEPARATED from the value: a false
# snap fails SAFE, a missed leak does not.
#
# `\b` is kept on BOTH ends and the ASCII identifier guards below are ADDED to
# it, never substituted for it. `\b` is Unicode-aware and an ASCII character
# class is not: 万, 円, 条, 日 and 人 are all `\w`, so a guard written as
# `(?![A-Za-z0-9_-])` waves through a boundary that `\b` refuses. A peer template
# shipped that substitution and its always-rendered statutory line
# "（3000万円+600万円×法定相続人の数）" started matching as a 円 marker plus
# "+600", rewriting a fixed legal figure on every invoke. Nothing in this
# template renders Japanese today - the knowledge base and the render frame are
# English - but 円 and ￥ are live in _SYMBOL, so a seeded KB or the documented
# LLM-synthesis upgrade would walk straight into it.
_ANY_CODE = r"\b[A-Z]{3}\b"

# Currency codes recognised when the marker is ATTACHED to the value with no
# whitespace in between. This is a list of CURRENCIES, not a list of
# identifiers to exclude. Attached "<3 uppercase letters><digits>" and
# "<3 uppercase letters>-<digits>" are exactly the shapes a retail identifier
# takes, and this template's whole vocabulary is built out of them: the answer
# renders five 3-letter uppercase words today - SKU, POS, SOP, EAS, ABC (see
# config/kb/shrinkage_kb.json) - and not one of them is a currency. Without
# this restriction the gate reads "SKU-48210" as a marker plus a negative
# amount and emits "SKU-48,000"; "POS-07" comes back "POS0" and the case
# number "LP-2026-00318" comes back "LP-20260". A currency list is closed and
# stable; a list of retail identifiers never could be. SEPARATED markers stay
# unrestricted - "SKU 48210" still snaps, because a 3-letter word followed by
# whitespace and a number is currency context under the fail-safe rule above.
_ISO_CODE = (
    r"(?:JPY|USD|EUR|GBP|CNY|CNH|KRW|AUD|CAD|CHF|HKD|SGD|TWD|THB|INR|IDR|MYR|PHP|VND"
    r"|BRL|MXN|SEK|NOK|DKK|PLN|CZK|TRY|NZD|RUB|ZAR|SAR|AED|ILS)(?![A-Za-z])"
)

# Delimiter between a currency marker and its value: horizontal whitespace and
# at most ONE newline - never a paragraph break. A plain `\s*` spans blank
# lines, so a 3-letter uppercase word ending a line would bind to the number
# that opens the next block and rewrite it ("Currency: JPY\n\n3. Cash Position"
# -> "0. Cash Position"). Every enumerated leak form (spaces, tabs, single
# newline, signed, symmetric, comma-grouped) still matches.
_GATE_DELIM = r"[ \t]*(?:\n[ \t]*)?"
# The same delimiter with at least one whitespace character present - the
# SEPARATED marker form.
_GATE_DELIM_WS = r"(?:[ \t]+|[ \t]*\n[ \t]*)"

# Marker BEFORE the value: any 3-letter uppercase word or symbol when it is
# separated from the value by whitespace; only a real currency code or symbol
# when it is attached.
_CURRENCY_MARKER = rf"(?:(?:{_ANY_CODE}|{_SYMBOL}){_GATE_DELIM_WS}|(?:\b{_ISO_CODE}|{_SYMBOL}){_GATE_DELIM})"
# Marker AFTER the value, same rule mirrored. The `\b` sits on the side the
# published grammar had it - leading for a marker that precedes the value,
# trailing for one that follows - so an attached "JPY9999" / "9999JPY" still snaps.
_CURRENCY_MARKER_POST = rf"(?:{_GATE_DELIM_WS}(?:{_ANY_CODE}|{_SYMBOL})|{_GATE_DELIM}(?:{_ISO_CODE}\b|{_SYMBOL}))"

# Every value alternative absorbs its decimal part into the SAME token. Two
# separate things go wrong without that:
#   - the fraction of "9999.99999" is a standalone 5+-digit run in its own
#     right, so the form-based branch latches onto it and rewrites a
#     cycle-count-accuracy KPI into "9999.100,000%" - a number the answer
#     never contained;
#   - in currency context the integer part snaps while the fraction dangles
#     ("JPY 1234.56" -> "JPY 1,000.56"), producing a value that is neither the
#     true figure nor on the grid.
# The fix is NOT to exempt decimals from the grid. An off-grid amount in
# explicit currency context still snaps - as ONE number ("JPY 1234.56" ->
# "JPY 1,000") - while a shrink rate, a percentage or a version reference,
# which was never a monetary token, is left byte-identical.
#
# The `(?!\.\d)` arm is what makes the absorption stick. A plain `(?:\.\d+)?`
# is a choice point: when the character after the fraction fails the trailing
# guard the engine backtracks out of the fraction and settles for the integer
# part alone, and "JPY 1234.56m" is right back to "JPY 1,000.56m". Either the
# fraction is taken whole, or there is none there.
_VAL_FRACTION = r"(?:\.\d+|(?!\.\d))"

# IDENTIFIER GUARDS, widened to THIS template's real render alphabet rather
# than a generic [A-Za-z0-9-]. They ADD to the `\b` assertions in the grammar
# above; they do not replace them (see _ANY_CODE - an ASCII class cannot see a
# CJK word boundary). The string this gate sees is assembled by
# src/nodes/output_format_node.py out of the fixed render frame plus the KB
# `title`, `source` and `excerpt` fields, so the alphabet is whatever those
# put next to a digit:
#
#   -   hyphenated compounds and record references. Present in the bundled KB
#       today ("30-day", "A-tier") and the shape of every identifier this
#       domain names: SKU-48210, PO-2026-004821 (kb-006 "PO number"),
#       LP-2026-00318 (kb-002 case file), and this repo's own citation ids
#       (kb-001).
#   _   the inert identifier alphabet this template declares for caller and
#       config identifiers, `^[a-z0-9_]{1,32}$` (src/nodes/input_validate_node.py,
#       src/nodes/pre_process_node.py), which is also the shape of its own KB
#       category keys ("pos_exceptions", "inventory_reconciliation"). A seeded
#       KB that names a POS/inventory export column renders "sku_48210"
#       against a digit run.
#   .   LEADING guard only, so no match can begin part-way through a number
#       and treat the tail of a fraction as a value of its own ("0.123456" can
#       be entered neither at "123456" nor at "23456"). It must NOT join the
#       trailing guard: an amount that ends a sentence ("...JPY 9999.") would
#       then be followed by a `.` and escape the grid entirely.
#
# Deliberately NOT in the class: `[`/`]` (the citation markers the frame
# renders - bracketing an amount would let "[12345]" escape) and `%` (no
# 5+-digit percentage exists in this domain; long-fraction KPIs are already
# covered by _VAL_FRACTION). Every character added here is a place a real leak
# could hide, so the class is evidence-driven, not defensive.
_LEAD_GUARD = r"(?<![A-Za-z0-9_.\-])"
_TRAIL_GUARD = r"(?![A-Za-z0-9_\-])"

_NUM_TOKEN_RE = re.compile(
    _LEAD_GUARD
    # marker THEN value: "JPY 9999", "JPY  -9999", "JPY\t9999", "¥9999", "USD\n+9999".
    # The value alternatives accept a comma-grouped form FIRST: the regex is
    # leftmost-first, so without it "JPY 1,234" would match as marker + "1"
    # (mangling the number on the snap) instead of as the whole grouped value
    # - an on-grid "JPY 1,000" must stay byte-identical, and an off-grid
    # "JPY 1,234" must snap as 1234, not as 1.
    + rf"(?:(?P<pre>{_CURRENCY_MARKER})"
    rf"(?P<val_after>[+-]?\d{{1,3}}(?:,\d{{3}})+{_VAL_FRACTION}|[+-]?\d{{1,4}}{_VAL_FRACTION})\b"
    # value THEN marker: "9999 JPY", "-9999\tJPY", "9999円", "+9999  $"
    rf"|(?P<val_before>[+-]?\d{{1,4}}{_VAL_FRACTION})(?P<post>{_CURRENCY_MARKER_POST})"
    # form-based, standalone at any magnitude: comma-grouped or 5+-digit runs
    rf"|(?P<val_form>[+-]?\d{{1,3}}(?:,\d{{3}})+{_VAL_FRACTION}|[+-]?\d{{5,}}{_VAL_FRACTION}))" + _TRAIL_GUARD
)


def _enforce_precision(result: str) -> tuple[str, int]:
    """Snap every monetary-form token onto the approved external grid.

    Returns (sanitised_result, redaction_count). A redaction means a
    full-precision monetary figure reached the external surface - the gate
    rounds it onto the approved grid. The currency marker, the original
    delimiter whitespace, and the explicit sign of the original token are all
    preserved on the snapped replacement.
    """
    redactions = 0

    def _snap(match: re.Match[str]) -> str:
        nonlocal redactions
        pre = match.group("pre") or ""
        post = match.group("post") or ""
        token = match.group("val_after") or match.group("val_before") or match.group("val_form")
        # float(), not int(): the token may carry a decimal part, and the WHOLE
        # amount - fraction included - is what has to land on the grid.
        value = float(token.replace(",", ""))  # float() understands leading +/-
        if value % _EXTERNAL_ROUND_UNIT == 0:
            return match.group(0)
        redactions += 1
        snapped = round(value / _EXTERNAL_ROUND_UNIT) * _EXTERNAL_ROUND_UNIT
        plus = "+" if token.startswith("+") and snapped >= 0 else ""
        return f"{pre}{plus}{snapped:,d}{post}"

    return _NUM_TOKEN_RE.sub(_snap, result), redactions


def _security_gate_output(content: str) -> Optional[str]:
    """Scan output for disallowed credential/secret patterns.

    Returns the first violation name, or None if the output is clean.
    Module-level function (not a node instance method) - the framework
    auto-wraps node instance methods on the real invoke path, so the gate
    must live at module level.
    """
    for pattern, name in _CREDENTIAL_PATTERNS:
        if re.search(pattern, content, re.IGNORECASE):
            return name
    return None


def _redact_blocked_fields(result: str, state: AgentState) -> tuple[str, List[str]]:
    """Replace verbatim embeddings of caller-derived state text with [REDACTED].

    Returns (sanitised_result, redacted_field_names). Only substantial values
    (len > 10) are matched so short incidental overlaps are not redacted.
    """
    redacted: List[str] = []
    sanitised = result
    for field in sorted(_BLOCKED_FIELDS):
        value = state.get(field)
        if isinstance(value, str) and len(value) > 10 and value in sanitised:
            sanitised = sanitised.replace(value, "[REDACTED]")
            redacted.append(field)
    return sanitised, redacted


class PostProcessNode(FunctionNode):
    """Apply the output gate and expose the final shrinkage Q&A answer.

    Outer backbone post_process slot.  Declared ANONYMOUS - trust was
    already enforced at PreProcessNode (VERIFIED_EXTERNAL).

    Input state keys:
        result:            str - rendered answer from the inner OutputFormatNode
        shrinkage_answer:  str - same rendered answer (fallback source)

    Output state keys (partial dict):
        formatted_output: str
        result:           str
        status:           str
        error_log:        list[str]  (only on ERROR)
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> dict[str, Any]:
        answer: str = state.get("result") or state.get("shrinkage_answer") or ""

        # -- Fallback for empty answer ----------------------------------------
        if not answer.strip():
            logger.warning("PostProcessNode: result is empty - using fallback message")
            answer = "[Shrinkage Q&A] No answer content generated. Check error_log for upstream failures."

        # -- Layer 1: credential scan (withhold entirely) ----------------------
        violation = _security_gate_output(answer)
        if violation:
            logger.error("PostProcessNode: credential pattern detected in output - %s", violation)
            emit_trace_event(
                "post_process_credential_violation",
                {"violation": violation},
                state,
            )
            sanitised = (
                f"[ANSWER REDACTED: output contained a disallowed pattern "
                f"({violation}). Review the generated output and retry.]"
            )
            return {
                "formatted_output": sanitised,
                "result": sanitised,
                "status": AgentStatus.ERROR.value,
                "error_log": [f"PostProcessNode: credential pattern detected - {violation}"],
            }

        # -- Layer 2: verbatim caller-text redaction ---------------------------
        sanitised_output, redacted_fields = _redact_blocked_fields(answer, state)
        if redacted_fields:
            logger.error(
                "PostProcessNode: caller-derived text embedded verbatim in output - %s",
                ", ".join(redacted_fields),
            )
            emit_trace_event(
                "post_process_blocked_field_redaction",
                {"fields": redacted_fields},
                state,
            )

        # -- Layer 3: monetary precision grid ----------------------------------
        sanitised_output, precision_redactions = _enforce_precision(sanitised_output)
        if precision_redactions:
            logger.warning(
                "PostProcessNode: %d off-grid monetary token(s) snapped to the external grid",
                precision_redactions,
            )
            emit_trace_event(
                "post_process_precision_redaction",
                {"redaction_count": precision_redactions},
                state,
            )

        logger.info("PostProcessNode: output gate passed - length=%d", len(sanitised_output))
        emit_trace_event(
            "post_process_complete",
            {"output_chars": len(sanitised_output)},
            state,
        )

        return {
            "formatted_output": sanitised_output,
            "result": sanitised_output,
            "status": AgentStatus.SUCCESS.value,
        }
