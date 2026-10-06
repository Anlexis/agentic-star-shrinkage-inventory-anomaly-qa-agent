"""AgentCore Platform v1.0"""

# RET-C2-005 - RerankFilterNode
# Domain node 3: rerank the retrieval candidates and enforce the relevance
# floor. Deterministic: a small category-match boost on top of the retrieval
# score, drop everything below `score_threshold`, cap the survivors at
# `top_k`.
#
# Node contract: execute(self, state) -> dict ONLY - no config parameter.
# Retrieval tuning (top_k / score_threshold) is forwarded by
# ShrinkageQAGraphNode._parent_config() -> DomainWorkflowGraph.
# _extra_initial_state() as the SCALAR state fields retrieval_top_k /
# retrieval_score_threshold; this node reads them directly from state,
# falling back to module defaults (mirroring config/config.yaml) when
# unseeded. A caller-supplied top_k override (query_filters) wins when
# stricter. Serialized state scalars and candidate scores are re-parsed
# through a finite+bounded check - a non-finite value never reaches a
# comparison (a NaN score or threshold would otherwise turn the relevance
# floor into a silent no-op).
#
# Wired by the inner graph (DomainWorkflowGraph).
# Returns only changed state keys (partial dict).

from typing import Any, ClassVar, Dict, List

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.schemas.state import _finite_in_range, from_json, to_json

# Defaults mirror the declared `retrieval` block in config/config.yaml.
_DEFAULT_TOP_K = 4
_DEFAULT_SCORE_THRESHOLD = 0.25

# Boost applied when a candidate's category matches the caller's filter.
_CATEGORY_BOOST = 0.1


class RerankFilterNode(FunctionNode):
    """Rerank candidates, apply the score threshold, cap at top_k.

    Input state keys:
        retrieved_documents:        JSON list of scored candidates (from RetrieveNode)
        query_filters:               JSON dict with optional category / top_k override
        retrieval_top_k:             forwarded manifest top_k (scalar)
        retrieval_score_threshold:   forwarded manifest score_threshold (scalar)

    Output state keys (partial dict):
        ranked_documents: JSON list of surviving passages (score desc, <= top_k)
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> dict[str, Any]:
        candidates: List[Dict[str, Any]] = from_json(state.get("retrieved_documents"), []) or []
        filters = from_json(state.get("query_filters"), {}) or {}

        # Strict typing on seeded scalars: the config path forwards only
        # validated numerics, so a string here is corruption - fall back.
        raw_top_k = state.get("retrieval_top_k", _DEFAULT_TOP_K)
        parsed_top_k = _finite_in_range(raw_top_k, 1, 20) if isinstance(raw_top_k, (int, float)) else None
        top_k = int(parsed_top_k) if parsed_top_k is not None and parsed_top_k == int(parsed_top_k) else _DEFAULT_TOP_K
        # A stricter caller override (validated by InputValidateNode) wins.
        caller_top_k = filters.get("top_k")
        if isinstance(caller_top_k, int) and not isinstance(caller_top_k, bool) and 1 <= caller_top_k < top_k:
            top_k = caller_top_k

        raw_threshold = state.get("retrieval_score_threshold", _DEFAULT_SCORE_THRESHOLD)
        parsed_threshold = (
            _finite_in_range(raw_threshold, 0.0, 1.0) if isinstance(raw_threshold, (int, float)) else None
        )
        score_threshold = parsed_threshold if parsed_threshold is not None else _DEFAULT_SCORE_THRESHOLD

        category = filters.get("category")

        reranked: List[Dict[str, Any]] = []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            entry = dict(candidate)  # local copy - inputs stay immutable
            # Finite check: a non-finite score in a (possibly hand-edited or
            # replaced) KB serialization must never clear the relevance floor.
            raw_score = entry.get("score", 0.0)
            parsed_score = _finite_in_range(raw_score, 0.0, 1.0) if isinstance(raw_score, (int, float)) else None
            score = parsed_score if parsed_score is not None else 0.0
            if category and str(entry.get("category", "")).lower() == str(category).lower():
                score = min(1.0, score + _CATEGORY_BOOST)
            entry["score"] = round(score, 4)
            reranked.append(entry)

        # Deterministic ordering: score desc, then id asc for stable ties.
        reranked.sort(key=lambda c: (-c.get("score", 0.0), str(c.get("id", ""))))

        kept = [c for c in reranked if c.get("score", 0.0) >= score_threshold][:top_k]
        dropped = len(reranked) - len(kept)

        # Audit: rerank + relevance floor applied.
        emit_trace_event(
            "rerank_filter_complete",
            {
                "kept": len(kept),
                "dropped": dropped,
                "score_threshold": score_threshold,
                "top_k": top_k,
            },
            state,
        )

        return {"ranked_documents": to_json(kept)}
