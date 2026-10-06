# RET-C2-005 — Unit Tests: nested Cat-2 graph composition (outer + end-to-end)
#
# Drives the REAL outer agent (ShrinkageInventoryQAAgent / Graph) end-to-end
# via AgentBaseGraph.invoke(). The e2e context is
# InvocationContext(caller_trust_level=TrustLevel.VERIFIED_EXTERNAL) — the
# manifest's declared caller level; for_internal() is NEVER used (it would
# over-privilege the run and hide trust-gate regressions).
#
# Mirrors docs/03_test_spec.md §3 (INT-05..INT-12).
# Deterministic — no LLM, no network. framework.* / src.* imports only.

import pathlib

from framework.graph.agent_base_graph import AgentBaseGraph
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import InvocationContext
from framework.schemas.trust_level import TrustLevel

import src.graph.graph
from src.graph.domain_workflow_graph import DomainWorkflowGraph
from src.graph.graph import (
    Graph,
    ShrinkageInventoryQAAgent,
    ShrinkageQAGraphNode,
)
from src.nodes.post_process_node import PostProcessNode
from src.nodes.pre_process_node import PreProcessNode
from src.schemas.state import State, from_json, to_json

_ANOMALY_QUERY = (
    "What common shrinkage anomaly patterns should a store watch for, and "
    "what is the standard procedure for investigating them?"
)


def _run(user_input: str, trust: TrustLevel = TrustLevel.VERIFIED_EXTERNAL) -> dict:
    ctx = InvocationContext(caller_trust_level=trust, caller_id="unit-suite")
    return Graph().invoke(user_input, ctx=ctx)


class TestOuterGraphConstruction:
    def test_int_05_inherits_agent_base_graph_directly(self):
        assert issubclass(ShrinkageInventoryQAAgent, AgentBaseGraph)

    def test_int_05_graph_alias(self):
        assert Graph is ShrinkageInventoryQAAgent

    def test_state_schema_is_state(self):
        assert ShrinkageInventoryQAAgent().state_schema is State

    def test_int_06_compile_fills_all_backbone_slots(self):
        agent = ShrinkageInventoryQAAgent()
        agent.compile()
        for slot in ("initialize", "pre_process", "main", "post_process", "finalize"):
            assert agent._nodes.get(slot) is not None, f"backbone slot not filled: {slot}"
        assert isinstance(agent._nodes["pre_process"], PreProcessNode)
        assert isinstance(agent._nodes["main"], ShrinkageQAGraphNode)
        assert isinstance(agent._nodes["post_process"], PostProcessNode)

    def test_add_edges_is_not_overridden(self):
        # Backbone wiring belongs to the framework — the template must not
        # redefine it.
        assert "add_edges" not in ShrinkageInventoryQAAgent.__dict__


class TestMainSlotGraphNode:
    def test_int_07_get_subgraph_returns_the_inner_graph(self):
        subgraph = ShrinkageQAGraphNode().get_subgraph()
        assert isinstance(subgraph, DomainWorkflowGraph)
        assert subgraph.config["configurable"]["retrieval"], "inner config must carry the retrieval block"

    def test_int_08_extract_input_prefers_validated_input(self):
        node = ShrinkageQAGraphNode()
        assert node.extract_input({"validated_input": "VI", "user_input": "UI"}) == "VI"
        assert node.extract_input({"user_input": "UI"}) == "UI"

    def test_int_09_merge_output_maps_the_inner_contract(self):
        node = ShrinkageQAGraphNode()
        citations = to_json([{"ref": 1, "id": "kb-001", "title": "t", "source": "s"}])
        delta = node.merge_output(
            {},
            {"formatted_answer": "ANSWER", "citations": citations, "status": AgentStatus.SUCCESS.value},
        )
        # The inner formatted_answer surfaces as BOTH shrinkage_answer and
        # result (PostProcessNode's output gate reads state["result"]).
        assert delta == {
            "shrinkage_answer": "ANSWER",
            "result": "ANSWER",
            "citations": citations,
            "status": AgentStatus.SUCCESS.value,
        }

    def test_error_strategy_is_propagate_and_hitl_is_contained(self):
        assert ShrinkageQAGraphNode.error_strategy == "propagate"
        assert ShrinkageQAGraphNode.propagate_hitl is False

    def test_int_10_parent_config_reads_the_runtime_config_file(self):
        # The declared runtime settings in config/config.yaml reach the inner
        # graph through _parent_config() — a dead declaration would silently
        # run the nodes on module defaults.
        cfg = ShrinkageQAGraphNode()._parent_config()
        assert cfg["configurable"]["retrieval"]["kb_path"] == "config/kb/shrinkage_kb.json"
        assert cfg["configurable"]["retrieval"]["top_k"] == 4
        assert cfg["configurable"]["retrieval"]["score_threshold"] == 0.25
        assert cfg["configurable"]["llm"] == {"temperature": 0.0, "max_tokens": 1500}

    def test_int_10b_missing_config_file_degrades_to_node_defaults(self, monkeypatch):
        # An unreadable runtime-config file must not crash graph construction;
        # the forwarded blocks come back empty and each node falls back to its
        # module default.
        monkeypatch.setattr(src.graph.graph, "_RUNTIME_CONFIG_PATH", pathlib.Path("/nonexistent/config.yaml"))
        cfg = ShrinkageQAGraphNode()._parent_config()
        assert cfg == {"configurable": {"retrieval": {}, "llm": {}}}

    def test_int_10c_malformed_config_values_are_not_forwarded(self, monkeypatch, tmp_path):
        # Non-finite / out-of-range / mistyped declared values never reach the
        # inner graph: a NaN score_threshold would silently drop every
        # candidate instead of applying the declared relevance floor.
        bad = tmp_path / "config.yaml"
        bad.write_text(
            "retrieval:\n"
            "  top_k: .nan\n"
            "  score_threshold: 7.5\n"
            "  kb_path: 42\n"
            "llm:\n"
            "  temperature: -1\n"
            "  max_tokens: true\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(src.graph.graph, "_RUNTIME_CONFIG_PATH", bad)
        cfg = ShrinkageQAGraphNode()._parent_config()
        assert cfg == {"configurable": {"retrieval": {}, "llm": {}}}


class TestEndToEndInvoke:
    """Full agent run: outer backbone + inner domain workflow, no LLM."""

    def test_int_11_invoke_returns_success(self):
        result = _run(_ANOMALY_QUERY)
        assert (
            result.get("status") == AgentStatus.SUCCESS.value
        ), f"Expected success, got {result.get('status')}. result={result!r}"

    def test_int_11_output_is_the_gated_formatted_answer(self):
        output = _run(_ANOMALY_QUERY).get("output")
        assert isinstance(output, str) and output.strip()
        assert output.startswith("# Shrinkage & Inventory Anomaly Q&A Result")
        assert "[1]" in output
        assert "does not identify or accuse any individual" in output

    def test_int_11_e2e_traverses_the_post_process_gate(self):
        history = _run(_ANOMALY_QUERY).get("node_history", [])
        for cls_name in ("PreProcessNode", "ShrinkageQAGraphNode", "PostProcessNode"):
            assert cls_name in history, f"node_history missing {cls_name}: {history}"

    def test_no_coverage_query_still_terminates_success(self):
        result = _run("quantum telepathy sandwich recipes")
        assert result.get("status") == AgentStatus.SUCCESS.value
        assert "does not contain sufficient coverage" in result.get("output", "")

    def test_int_12_anonymous_caller_is_denied_at_the_outer_boundary(self):
        """Trust gate at graph level: an ANONYMOUS invoke is refused by the
        VERIFIED_EXTERNAL pre_process slot. The error state short-circuits the
        main slot (its input gate sees status=error and skips the inner graph)
        and routes past post_process to finalize — no domain answer is ever
        produced."""
        result = _run(_ANOMALY_QUERY, trust=TrustLevel.ANONYMOUS)
        assert result.get("status") == AgentStatus.ERROR.value
        assert not result.get("output")
        history = result.get("node_history", [])
        assert "PostProcessNode" not in history
        assert history[:2] == ["InitializeNode", "PreProcessNode"]


class TestStateRoundTrip:
    """State JSON helpers: producers to_json() on write, consumers from_json() on read."""

    def test_to_from_json_list_round_trip(self):
        original = [{"id": "kb-001", "score": 0.55, "title": "shrinkage anomaly patterns"}]
        assert from_json(to_json(original)) == original

    def test_to_from_json_dict_round_trip(self):
        original = {"category": "anomaly_patterns", "top_k": 3}
        assert from_json(to_json(original)) == original

    def test_to_json_none_passes_through(self):
        assert to_json(None) is None

    def test_from_json_malformed_returns_default(self):
        assert from_json("{not valid json", default=[]) == []
        assert from_json(None, default={}) == {}
        assert from_json("", default=[]) == []
