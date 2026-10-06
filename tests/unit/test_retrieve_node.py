# RET-C2-005 — Unit Tests: RetrieveNode (inner domain node 2)
#
# Invocation canon: node(state) via BaseNode.__call__ with an ANONYMOUS caller
# — NO carve-out. Node contract: execute(self, state) is the
# ONLY signature; RetrieveNode takes no config parameter at all. Retrieval
# tuning (top_k / kb_path) is read directly off SCALAR state keys
# (retrieval_top_k / retrieval_kb_path, forwarded by
# DomainWorkflowGraph._extra_initial_state()) — every test below seeds those
# keys in state and invokes exclusively through node(state).
#
# Mirrors docs/03_test_spec.md §2.3 (RET-01..RET-08).
# Deterministic — keyword scoring over the seeded config/kb/shrinkage_kb.json;
# no LLM, no network. framework.* / src.* imports only.

from framework.schemas.trust_level import TrustLevel

from src.nodes.retrieve_node import RetrieveNode
from src.schemas.state import from_json, to_json

_ANOMALY_QUERY = (
    "what common shrinkage anomaly patterns should a store watch for, and "
    "what is the standard procedure for investigating them?"
)


def _make_state(query=_ANOMALY_QUERY, **extra) -> dict:
    state = {
        "search_query": query,
        "caller_trust_level": TrustLevel.ANONYMOUS.value,
        "node_history": [],
        "error_log": [],
        "session_id": "unit-session",
        "execution_time": {},
    }
    state.update(extra)
    return state


class TestRetrieveHappyPath:
    def test_ret_01_top_hit_is_anomaly_patterns_entry(self):
        result = RetrieveNode()(_make_state())
        docs = from_json(result["retrieved_documents"])
        assert docs, "expected candidates for the anomaly-patterns query"
        assert docs[0]["id"] == "kb-001"

    def test_ret_02_scores_sorted_descending(self):
        docs = from_json(RetrieveNode()(_make_state())["retrieved_documents"])
        scores = [d["score"] for d in docs]
        assert scores == sorted(scores, reverse=True)
        assert all(s > 0.0 for s in scores)

    def test_ret_03_entry_shape_and_excerpt_cap(self):
        docs = from_json(RetrieveNode()(_make_state())["retrieved_documents"])
        for doc in docs:
            assert set(doc.keys()) == {"id", "title", "category", "source", "score", "excerpt"}
            assert len(doc["excerpt"]) <= 400

    def test_retrieved_documents_is_json_string(self):
        # List-shaped State fields travel as JSON strings (checkpoint safety).
        result = RetrieveNode()(_make_state())
        assert isinstance(result["retrieved_documents"], str)


class TestRetrieveFilters:
    def test_ret_04_category_filter_restricts_pool(self):
        state = _make_state(
            query="escalation procedure",
            query_filters=to_json({"category": "investigation_procedures", "top_k": None}),
        )
        docs = from_json(RetrieveNode()(state)["retrieved_documents"])
        assert docs, "investigation_procedures category has a seeded entry"
        assert {d["category"] for d in docs} == {"investigation_procedures"}
        assert docs[0]["id"] == "kb-002"

    def test_ret_05_empty_query_yields_no_candidates(self):
        docs = from_json(RetrieveNode()(_make_state(query=""))["retrieved_documents"])
        assert docs == []


class TestRetrieveStateSeededConfig:
    """Node contract: tuning knobs travel ONLY through State (no config parameter)."""

    def test_ret_06_state_kb_path_override_via_call(self):
        state = _make_state(retrieval_kb_path="config/kb/does_not_exist.json")
        result = RetrieveNode()(state)
        assert from_json(result["retrieved_documents"]) == []
        notes = from_json(result.get("intake_notes"), [])
        assert any("not readable" in n for n in notes)

    def test_ret_07_non_numeric_top_k_seed_falls_back_to_default(self):
        # int(state["retrieval_top_k"]) raising must degrade to the module
        # default, never crash the node.
        state = _make_state(retrieval_top_k="not-a-number")
        docs = from_json(RetrieveNode()(state)["retrieved_documents"])
        assert docs and docs[0]["id"] == "kb-001"


class TestRetrieveNotesAccumulation:
    def test_ret_08_notes_append_never_clobber(self):
        state = _make_state(
            intake_notes=to_json(["earlier note from input validation"]),
            retrieval_kb_path="config/kb/bogus.json",
        )
        result = RetrieveNode()(state)
        notes = from_json(result["intake_notes"])
        assert notes[0] == "earlier note from input validation"
        assert len(notes) == 2
