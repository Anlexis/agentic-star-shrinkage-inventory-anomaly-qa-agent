# RET-C2-005 — Unit Tests: RerankFilterNode (inner domain node 3)
#
# Invocation canon: node(state) via BaseNode.__call__ with an ANONYMOUS caller
# — NO carve-out. Node contract: execute(self, state) is the
# ONLY signature; score_threshold / top_k overrides are read directly off
# SCALAR state keys (retrieval_score_threshold / retrieval_top_k, forwarded by
# DomainWorkflowGraph._extra_initial_state()) — every override test below
# seeds those keys in state and invokes exclusively through node(state).
#
# Mirrors docs/03_test_spec.md §2.4 (RRF-01..RRF-08).
# Deterministic — no LLM, no network. framework.* / src.* imports only.

from framework.schemas.trust_level import TrustLevel

from src.nodes.rerank_filter_node import RerankFilterNode
from src.schemas.state import from_json, to_json


def _doc(doc_id, score, category="anomaly_patterns"):
    return {
        "id": doc_id,
        "title": f"entry {doc_id}",
        "category": category,
        "source": "seeded kb",
        "score": score,
        "excerpt": "excerpt text",
    }


def _make_state(candidates, **extra) -> dict:
    state = {
        "retrieved_documents": to_json(candidates),
        "caller_trust_level": TrustLevel.ANONYMOUS.value,
        "node_history": [],
        "error_log": [],
        "session_id": "unit-session",
        "execution_time": {},
    }
    state.update(extra)
    return state


class TestThresholdAndCap:
    def test_rrf_01_default_threshold_drops_weak_candidates(self):
        result = RerankFilterNode()(_make_state([_doc("kb-a", 0.9), _doc("kb-b", 0.1)]))
        kept = from_json(result["ranked_documents"])
        assert [d["id"] for d in kept] == ["kb-a"]  # 0.1 < default 0.25 floor

    def test_rrf_02_state_score_threshold_override(self):
        state = _make_state(
            [_doc("kb-a", 0.9), _doc("kb-b", 0.3)],
            retrieval_score_threshold=0.5,
        )
        kept = from_json(RerankFilterNode()(state)["ranked_documents"])
        assert [d["id"] for d in kept] == ["kb-a"]

    def test_rrf_03_state_top_k_override(self):
        state = _make_state(
            [_doc("kb-a", 0.9), _doc("kb-b", 0.8), _doc("kb-c", 0.7)],
            retrieval_top_k=1,
        )
        kept = from_json(RerankFilterNode()(state)["ranked_documents"])
        assert [d["id"] for d in kept] == ["kb-a"]

    def test_ranked_documents_is_json_string(self):
        # List-shaped State fields travel as JSON strings (checkpoint safety).
        result = RerankFilterNode()(_make_state([_doc("kb-a", 0.9)]))
        assert isinstance(result["ranked_documents"], str)


class TestCategoryBoost:
    def test_rrf_04_matching_category_is_boosted_and_reranked(self):
        state = _make_state(
            [_doc("kb-a", 0.30, category="pos_exceptions"), _doc("kb-b", 0.25, category="employee_theft")],
            query_filters=to_json({"category": "employee_theft", "top_k": None}),
        )
        kept = from_json(RerankFilterNode()(state)["ranked_documents"])
        assert [d["id"] for d in kept] == ["kb-b", "kb-a"]
        assert kept[0]["score"] == 0.35  # 0.25 + 0.1 category boost

    def test_rrf_05_boost_is_capped_at_one(self):
        state = _make_state(
            [_doc("kb-a", 0.95, category="employee_theft")],
            query_filters=to_json({"category": "employee_theft", "top_k": None}),
        )
        kept = from_json(RerankFilterNode()(state)["ranked_documents"])
        assert kept[0]["score"] == 1.0


class TestCallerTopK:
    def test_rrf_06_stricter_caller_top_k_wins(self):
        state = _make_state(
            [_doc("kb-a", 0.9), _doc("kb-b", 0.8), _doc("kb-c", 0.7)],
            query_filters=to_json({"category": None, "top_k": 1}),
        )
        kept = from_json(RerankFilterNode()(state)["ranked_documents"])
        assert [d["id"] for d in kept] == ["kb-a"]

    def test_rrf_06_looser_caller_top_k_does_not_widen(self):
        state = _make_state(
            [_doc("kb-a", 0.9), _doc("kb-b", 0.8), _doc("kb-c", 0.7)],
            query_filters=to_json({"category": None, "top_k": 10}),
            retrieval_top_k=2,
        )
        kept = from_json(RerankFilterNode()(state)["ranked_documents"])
        assert [d["id"] for d in kept] == ["kb-a", "kb-b"]


class TestRobustness:
    def test_rrf_07_garbage_candidates_are_skipped_or_dropped(self):
        candidates = [
            "not-a-dict",
            {"id": "kb-bad", "title": "b", "category": "x", "source": "s", "score": "NaN?", "excerpt": "e"},
            _doc("kb-a", 0.9),
        ]
        kept = from_json(RerankFilterNode()(_make_state(candidates))["ranked_documents"])
        # The string entry is skipped; the uncoercible score becomes 0.0 and
        # falls below the relevance floor.
        assert [d["id"] for d in kept] == ["kb-a"]

    def test_rrf_08_deterministic_tie_break_by_id(self):
        kept = from_json(RerankFilterNode()(_make_state([_doc("kb-b", 0.5), _doc("kb-a", 0.5)]))["ranked_documents"])
        assert [d["id"] for d in kept] == ["kb-a", "kb-b"]
