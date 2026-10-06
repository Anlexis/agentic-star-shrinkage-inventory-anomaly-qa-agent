"""AgentCore Platform v1.0"""

# RET-C2-005 - InputValidateNode
# Domain node 1: normalise the incoming shrinkage/inventory-anomaly question
# and resolve the caller's structured parameters.
#
# The outer GraphNode passes validated_input (from PreProcessNode) into the
# inner graph as its user_input; the caller's structured parameters (category
# filter, top_k retrieval-depth override) travel separately via input_context,
# seeded into inner state by DomainWorkflowGraph._extra_initial_state() (see
# src/graph/context_bridge.py). This node re-validates each parameter with the
# same strict rules as the ingest boundary (PreProcessNode), so a direct
# inner-graph invocation gets the same fail-closed contract, and publishes the
# result as query_filters for the downstream retrieval nodes.
#
# Wired by the inner graph (DomainWorkflowGraph).
# Node contract: execute(self, state) -> dict ONLY - no config parameter.
# Returns only changed state keys (partial dict).

import re
from typing import Any, ClassVar, Dict, List, Optional

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.schemas.state import to_json

# Hard cap on the normalised query length (defence-in-depth on input size).
_MAX_QUERY_CHARS = 2000

# Bounds for the caller-supplied parameters - must match the contract
# validated at the ingest boundary (PreProcessNode). Re-checked here so a
# direct inner-graph invocation gets the same fail-closed rule.
_TOP_K_MIN = 1
_TOP_K_MAX = 20
_IDENTIFIER_RE = re.compile(r"^[a-z0-9_]{1,32}$")

_WHITESPACE_RE = re.compile(r"\s+")


def _validate_top_k(value: Any) -> tuple[Optional[int], Optional[str]]:
    """Strict validation of the untrusted caller top_k override.

    Returns (top_k, error). The strict integer check rejects bools, floats,
    strings, and non-finite JSON values in one rule; the range bound covers
    the rest. Invalid values fail CLOSED with a field-naming error - never a
    silent fall-back.
    """
    if value is None:
        return None, None
    if not isinstance(value, int) or isinstance(value, bool) or not _TOP_K_MIN <= value <= _TOP_K_MAX:
        return None, f"input_context.top_k must be an integer between {_TOP_K_MIN} and {_TOP_K_MAX}"
    return value, None


def _validate_category(value: Any) -> tuple[Optional[str], Optional[str]]:
    """Strict validation of the untrusted caller category filter.

    Returns (category, error). The value selects a knowledge-base category,
    so it is locked to an inert identifier alphabet; free text fails CLOSED
    with a field-naming error.
    """
    if value is None:
        return None, None
    if not isinstance(value, str) or not _IDENTIFIER_RE.match(value):
        return None, "input_context.category must be a lowercase identifier (a-z, 0-9, _; 1-32 chars)"
    return value, None


class InputValidateNode(FunctionNode):
    """Normalise the question and resolve validated caller parameters.

    Input state keys:
        validated_input | user_input: the question text
        input_context:                caller parameters (category, top_k) -
                                      seeded by the inner graph's
                                      _extra_initial_state() context bridge

    Output state keys (partial dict):
        search_query:  normalised free-text search query
        query_filters: JSON dict {"category": str|None, "top_k": int|None}
        intake_notes:  (when anomalies were seen) JSON list[str]
        status/error_log: only on a failed-closed parameter validation
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> dict[str, Any]:
        raw = state.get("validated_input") or state.get("user_input", "")
        input_context = state.get("input_context")
        context: Dict[str, Any] = input_context if isinstance(input_context, dict) else {}
        notes: List[str] = []

        # -- Caller parameters (fail CLOSED; name the field, never the value) --
        category, category_error = _validate_category(context.get("category"))
        if category_error:
            emit_trace_event(
                "input_validate_failed",
                {"reason": "invalid_category"},
                state,
            )
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": [f"InputValidateNode: {category_error}"],
            }

        top_k, top_k_error = _validate_top_k(context.get("top_k"))
        if top_k_error:
            emit_trace_event(
                "input_validate_failed",
                {"reason": "invalid_top_k"},
                state,
            )
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": [f"InputValidateNode: {top_k_error}"],
            }

        # -- Normalise the question text ---------------------------------------
        query = raw.strip() if isinstance(raw, str) else ""
        if not query:
            notes.append("InputValidateNode: empty request - no query to search.")

        query = _WHITESPACE_RE.sub(" ", query).strip()
        if len(query) > _MAX_QUERY_CHARS:
            query = query[:_MAX_QUERY_CHARS]
            notes.append(f"InputValidateNode: query truncated to {_MAX_QUERY_CHARS} chars.")

        filters = {"category": category, "top_k": top_k}

        # Audit: request parsed and normalised.
        emit_trace_event(
            "input_validate_complete",
            {
                "query_chars": len(query),
                "has_category_filter": category is not None,
                "has_top_k_override": top_k is not None,
            },
            state,
        )

        out: Dict[str, Any] = {
            "search_query": query,
            "query_filters": to_json(filters),
        }
        if notes:
            out["intake_notes"] = to_json(notes)
        return out
