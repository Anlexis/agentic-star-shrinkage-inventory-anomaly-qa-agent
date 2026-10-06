# RET-C2-005 — Unit Tests: DomainWorkflowGraph (inner BaseGraph)
#
# Inner-graph composition + a full inner invoke() over the seeded KB. The
# inner graph runs the 5 domain nodes (all ANONYMOUS) — the outer trust boundary
# is the AgentBaseGraph backbone's concern and is covered in
# test_graph_composition.py / the PoB suite.
#
# Mirrors docs/03_test_spec.md §3 (INT-01..INT-04).
# Deterministic — no LLM, no network. framework.* / src.* imports only.

from langgraph.graph import END

from framework.graph.base_graph import BaseGraph
from framework.schemas.agent_status import AgentStatus

from src.graph.context_bridge import set_caller_input_context
from src.graph.domain_workflow_graph import DomainWorkflowGraph
from src.graph.graph import ShrinkageQAGraphNode
from src.nodes.generate_answer_node import GenerateAnswerNode
from src.nodes.input_validate_node import InputValidateNode
from src.nodes.output_format_node import OutputFormatNode
from src.nodes.rerank_filter_node import RerankFilterNode
from src.nodes.retrieve_node import RetrieveNode
from src.schemas.state import State, from_json

_ANOMALY_QUERY = (
    "What common shrinkage anomaly patterns should a store watch for, and "
    "what is the standard procedure for investigating them?"
)


class TestInnerGraphConstruction:
    def test_int_01_inherits_base_graph(self):
        assert issubclass(DomainWorkflowGraph, BaseGraph)

    def test_int_01_registers_the_five_domain_nodes(self):
        inner = DomainWorkflowGraph()
        inner.register_nodes()
        assert set(inner._nodes.keys()) == {
            "input_validate",
            "retrieve",
            "rerank_filter",
            "generate_answer",
            "output_format",
        }
        assert isinstance(inner._nodes["input_validate"], InputValidateNode)
        assert isinstance(inner._nodes["retrieve"], RetrieveNode)
        assert isinstance(inner._nodes["rerank_filter"], RerankFilterNode)
        assert isinstance(inner._nodes["generate_answer"], GenerateAnswerNode)
        assert isinstance(inner._nodes["output_format"], OutputFormatNode)

    def test_inner_graph_name_and_schema(self):
        inner = DomainWorkflowGraph()
        assert inner.name == "ret_c2_005_shrinkage_qa_workflow"
        assert inner.state_schema is State

    def test_initialize_finalize_are_not_registered(self):
        # Outer backbone concerns must not leak into the inner topology.
        inner = DomainWorkflowGraph()
        inner.register_nodes()
        assert "initialize" not in inner._nodes
        assert "finalize" not in inner._nodes


class TestConfigForwarding:
    """_extra_initial_state() republishes the declared retrieval block as
    SCALAR state keys — never a JSON config blob — and seeds the caller's
    bridged input_context."""

    def test_int_02_extra_initial_state_republishes_retrieval_as_scalars(self):
        set_caller_input_context(None)
        inner = DomainWorkflowGraph(
            config={"configurable": {"retrieval": {"top_k": 2, "score_threshold": 0.4, "kb_path": "x.json"}}}
        )
        extra = inner._extra_initial_state()
        assert extra == {
            "input_context": {},
            "retrieval_top_k": 2,
            "retrieval_score_threshold": 0.4,
            "retrieval_kb_path": "x.json",
        }

    def test_int_02_partial_retrieval_block_forwards_only_present_keys(self):
        set_caller_input_context(None)
        inner = DomainWorkflowGraph(config={"configurable": {"retrieval": {"top_k": 7}}})
        assert inner._extra_initial_state() == {"input_context": {}, "retrieval_top_k": 7}

    def test_extra_initial_state_with_no_config_seeds_empty_context(self):
        set_caller_input_context(None)
        assert DomainWorkflowGraph()._extra_initial_state() == {"input_context": {}}

    def test_stashed_caller_context_is_seeded_into_inner_state(self):
        # The outer GraphNode stashes input_context before subgraph.invoke();
        # the inner graph must read it back (the framework GraphNode does not
        # forward it — see src/graph/context_bridge.py).
        set_caller_input_context({"category": "anomaly_patterns", "top_k": 3})
        try:
            extra = DomainWorkflowGraph()._extra_initial_state()
            assert extra["input_context"] == {"category": "anomaly_patterns", "top_k": 3}
        finally:
            set_caller_input_context(None)


class TestOutputShape:
    def test_int_03_get_output_shapes_the_merge_contract(self):
        inner = DomainWorkflowGraph()
        out = inner.get_output(
            {
                "formatted_answer": "ANSWER",
                "citations": "[]",
                "status": AgentStatus.SUCCESS.value,
                "node_history": ["InputValidateNode"],
            }
        )
        assert out["formatted_answer"] == "ANSWER"
        assert out["citations"] == "[]"
        assert out["status"] == AgentStatus.SUCCESS.value
        assert out["node_history"] == ["InputValidateNode"]

    def test_route_returns_end_on_error(self):
        inner = DomainWorkflowGraph()
        assert inner.route({"status": AgentStatus.ERROR.value}) == END
        assert inner.route({"status": AgentStatus.SUCCESS.value}) == "output_format"


class TestInnerEndToEnd:
    def _invoke(self, payload: str) -> dict:
        # Same construction path the outer GraphNode uses: manifest-derived
        # config via _parent_config(); domain nodes take NO ctor args.
        inner = DomainWorkflowGraph(config=ShrinkageQAGraphNode()._parent_config())
        return inner.invoke(payload, session_id="inner-e2e")

    def test_int_04_full_inner_run_produces_the_formatted_answer(self):
        result = self._invoke(_ANOMALY_QUERY)
        assert result["status"] == AgentStatus.SUCCESS.value
        answer = result["formatted_answer"]
        assert answer.startswith("# Shrinkage & Inventory Anomaly Q&A Result")
        assert "[1]" in answer
        assert "does not identify or accuse any individual" in answer
        citations = from_json(result["citations"])
        assert citations and citations[0]["id"] == "kb-001"

    def test_int_04_inner_node_history_is_the_linear_topology(self):
        history = self._invoke(_ANOMALY_QUERY)["node_history"]
        assert history == [
            "InputValidateNode",
            "RetrieveNode",
            "RerankFilterNode",
            "GenerateAnswerNode",
            "OutputFormatNode",
        ]

    def test_no_coverage_query_still_terminates_success(self):
        result = self._invoke("quantum telepathy sandwich recipes")
        assert result["status"] == AgentStatus.SUCCESS.value
        assert "does not contain sufficient coverage" in result["formatted_answer"]
