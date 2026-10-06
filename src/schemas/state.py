"""AgentCore Platform v1.0"""

# State must be a flat TypedDict - never Pydantic BaseModel.
# LangGraph checkpoints use msgpack serialization; Pydantic objects
# cause silent corruption.  Extend AgentState with agent-specific
# fields only.  Do NOT add credentials, secrets, or Pydantic models.
#
# Msgpack safety: structured fields (dict / list[dict]) are stored as JSON
# STRINGS, not bare Python containers - a bare dict/list in a checkpointed
# State field breaks checkpoint serialization. Producers serialize with
# to_json() on write; consumers deserialize with from_json() on read.
# Retrieval TUNING knobs (top_k / score_threshold / kb_path) are plain
# scalars, not JSON - see the note on the retrieval_* fields below.
#
# RET-C2-005 - Shrinkage & Inventory Anomaly Q&A Agent (Cat 2 RAG).
# Two-layer nested Cat 2 graph: outer backbone (AgentBaseGraph) + inner
# domain workflow (BaseGraph).  Fields below cover both layers.
#
# PII / confidentiality note: this template never names or accuses a specific
# associate; the KB and generated answers describe patterns and procedures
# only. No employee-identifying detail is written to State.
#
# Node contract: FunctionNode.execute() takes ONLY (self, state) - no config
# parameter. Retrieval tuning knobs flow config/config.yaml ->
# ShrinkageQAGraphNode._parent_config() -> inner graph config ->
# DomainWorkflowGraph._extra_initial_state() seeds the SCALAR state fields
# below -> RetrieveNode / RerankFilterNode read them directly from state,
# falling back to module defaults (mirroring config/config.yaml) when
# unseeded, e.g. a node instantiated directly in a unit test.

import json
import math
from typing import Any, Optional

from framework.schemas.agent_state import AgentState


def to_json(value: Any) -> Optional[str]:
    """Serialize a dict/list State field to a JSON string (msgpack safety).

    None passes through unchanged so an 'unset' field stays distinguishable
    from an empty container.
    """
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False)


def from_json(value: Optional[str], default: Any = None) -> Any:
    """Deserialize a JSON-string State field back to its dict/list.

    None / empty / malformed input -> the supplied ``default`` so a missing or
    corrupt field is non-fatal for the consuming node.
    """
    if not value:
        return default
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


def _finite_in_range(value: Any, lo: float, hi: float) -> Optional[float]:
    """Parse an untrusted numeric: FINITE float within [lo, hi], else None.

    Rejects bools, non-numerics, and - critically - non-finite values: float()
    happily parses "NaN"/"Infinity" (and Python's json accepts bare NaN in
    request bodies), and IEEE NaN comparisons are always False, which turns a
    threshold check into a silent no-op. Every number read from an untrusted
    or serialized source must come through here (or an equivalent explicit
    finite check).
    """
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed) or not lo <= parsed <= hi:
        return None
    return parsed


class State(AgentState):
    """Flat TypedDict for RET-C2-005.

    All shared fields (user_input, status, session_id, node_history,
    error_log, hitl_*, etc.) are inherited from AgentState.

    dict/list fields use JSON-serialized Optional[str].
    formatted_output is NOT re-declared here — it is inherited from AgentState
    (re-declaring it with a bare type breaks the state contract).
    """

    # ------------------------------------------------------------------
    # Outer layer - set by PreProcessNode / ShrinkageQAGraphNode.merge_output
    # ------------------------------------------------------------------

    # Validated request payload produced by PreProcessNode.
    validated_input: Optional[str]

    # Final shrinkage/inventory-anomaly Q&A answer, mapped from the inner
    # graph's formatted_answer output via merge_output.
    shrinkage_answer: Optional[str]

    # ------------------------------------------------------------------
    # Inner layer - domain nodes (DomainWorkflowGraph)
    # ------------------------------------------------------------------

    # InputValidateNode outputs
    # Normalised free-text search query (whitespace-collapsed, length-capped).
    search_query: Optional[str]

    # JSON STRING (to_json) of the validated caller parameters. Deserialised
    # dict shape: {"category": str | None, "top_k": int | None}. Sourced from
    # the caller's input_context (bridged into inner state - see
    # src/graph/context_bridge.py) and re-validated by InputValidateNode.
    # Consumers (RetrieveNode, RerankFilterNode) read it back via from_json().
    query_filters: Optional[str]

    # Declared `retrieval` settings forwarded by
    # ShrinkageQAGraphNode._parent_config() ->
    # DomainWorkflowGraph._extra_initial_state(). Plain SCALAR fields (the
    # node contract is execute(self, state) - no config param, so these
    # travel through State, not a JSON blob). Consumers (RetrieveNode,
    # RerankFilterNode) read them directly via state.get(...), falling back
    # to module defaults when unseeded.
    retrieval_top_k: int
    retrieval_score_threshold: float
    retrieval_kb_path: str

    # RetrieveNode output
    # JSON STRING (to_json) of scored KB candidates. Deserialised shape:
    # list[dict], each entry {"id": str, "title": str, "category": str,
    # "source": str, "score": float, "excerpt": str}.
    # Consumers (RerankFilterNode) read it back via from_json().
    retrieved_documents: Optional[str]

    # RerankFilterNode output
    # JSON STRING (to_json) of reranked + threshold-filtered passages, capped
    # at top_k. Same entry shape as retrieved_documents.
    # Consumers (GenerateAnswerNode) read it back via from_json().
    ranked_documents: Optional[str]

    # GenerateAnswerNode outputs
    # Rule-assembled grounded answer body with numbered citation markers.
    grounded_answer: Optional[str]

    # JSON STRING (to_json) of citations. Deserialised shape: list[dict],
    # each entry {"ref": int, "id": str, "title": str, "source": str}.
    # Consumers (OutputFormatNode) read it back via from_json().
    citations: Optional[str]

    # OutputFormatNode output
    # Final formatted answer (body + sources + scope disclaimer). Written by
    # OutputFormatNode; surfaced to the outer graph via get_output() ->
    # merge_output().
    formatted_answer: Optional[str]

    # Validation / parse notes accumulated during intake (no PII, no
    # employee-identifying detail).
    # JSON STRING (to_json) of list[str].
    intake_notes: Optional[str]
