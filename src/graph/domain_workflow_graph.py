"""AgentCore Platform v1.0"""

# RET-C2-005 - DomainWorkflowGraph (inner BaseGraph)
#
# This is the INNER graph for the Cat 2 two-layer nested architecture.
# It encapsulates the full shrinkage/inventory-anomaly Q&A domain workflow:
#
#   START -> input_validate -> retrieve -> rerank_filter
#         -> generate_answer -> output_format -> END
#
# Called by ShrinkageQAGraphNode.get_subgraph() (graph.py).
# get_output() shapes the sub_result dict consumed by merge_output() there.
#
# Rules enforced:
#   - Inherits BaseGraph (fully custom topology - no forced backbone)
#   - Implements all 7 BaseGraph ABC methods
#   - register_nodes() does NOT call super() (abstract in BaseGraph)
#   - register_nodes() instantiates every domain node with NO ctor args
#   - Does NOT register initialize / finalize (outer backbone concerns)
#   - _extra_initial_state() seeds the caller's input_context (context bridge)
#   - get_output() designed together with ShrinkageQAGraphNode.merge_output()
#   - No platform-SDK imports (framework/ and shared/ only)
#   - Not placed under src/subagents/

from typing import Any

from langgraph.graph import END, START

from framework.graph.base_graph import BaseGraph
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from src.graph.context_bridge import get_caller_input_context
from src.nodes.generate_answer_node import GenerateAnswerNode
from src.nodes.input_validate_node import InputValidateNode
from src.nodes.output_format_node import OutputFormatNode
from src.nodes.rerank_filter_node import RerankFilterNode
from src.nodes.retrieve_node import RetrieveNode
from src.schemas.state import State


class DomainWorkflowGraph(BaseGraph):
    """Inner domain workflow graph for RET-C2-005.

    Inherits BaseGraph directly for a fully custom node topology.
    Called by ShrinkageQAGraphNode.get_subgraph() in graph.py, which passes
    the validated runtime settings (_parent_config()) into the ctor.

    Pipeline (linear):
        START
          -> input_validate  (InputValidateNode)  - normalise query + caller filters
          -> retrieve        (RetrieveNode)       - keyword-score the seeded KB
          -> rerank_filter   (RerankFilterNode)   - boost / threshold / top_k cut
          -> generate_answer (GenerateAnswerNode) - grounded answer + citations
          -> output_format   (OutputFormatNode)   - final format + scope disclaimer
          -> END

    All nodes are FunctionNode subclasses returning partial-dict state updates
    with the canonical execute(self, state) -> dict signature (no config
    parameter). initialize / finalize are outer backbone concerns - not
    registered here.
    """

    # -- Identity --------------------------------------------------------------

    @property
    def name(self) -> str:
        """Unique identifier for this inner graph."""
        return "ret_c2_005_shrinkage_qa_workflow"

    @property
    def state_schema(self) -> type:
        """TypedDict subclass shared across inner and outer graph."""
        return State

    # -- Config validation -----------------------------------------------------

    def _validate_config(self) -> None:
        """Validate inner graph config before compilation.

        The forwarded `retrieval` block (top_k / score_threshold / kb_path) is
        already type/range-validated by ShrinkageQAGraphNode._parent_config()
        and is read per-call by the domain nodes with safe defaults, so
        absence is non-fatal. Validation is permissive here rather than
        raising ConfigError.
        """
        pass

    # -- Config + caller-context seeding into inner state ----------------------

    def _extra_initial_state(self) -> dict[str, Any]:
        """Seed inner state with the runtime settings and the caller's context.

        Two independent hand-offs happen here:

        1. Runtime settings: ShrinkageQAGraphNode._parent_config() forwards
           the validated `retrieval` block under config["configurable"]; this
           hook republishes it as SCALAR state fields (the node contract is
           execute(self, state) - no config parameter, so tuning knobs travel
           through State). RetrieveNode / RerankFilterNode read
           retrieval_top_k / retrieval_score_threshold / retrieval_kb_path
           from state, falling back to module defaults when unseeded.

        2. Caller context: GraphNode.execute() (framework, SDK 1.0.1) does not
           forward the outer state's input_context into subgraph.invoke(), so
           the outer graph stashes it in a ContextVar
           (ShrinkageQAGraphNode.extract_input) and this hook reads it back -
           see src/graph/context_bridge.py. Without this, inner-node reads of
           state["input_context"] (the per-invocation category / top_k
           overrides) would always see {}.
        """
        retrieval_raw = (self.config or {}).get("configurable", {}).get("retrieval") or {}
        retrieval: dict[str, Any] = retrieval_raw if isinstance(retrieval_raw, dict) else {}
        extra: dict[str, Any] = {"input_context": get_caller_input_context()}
        if "top_k" in retrieval:
            extra["retrieval_top_k"] = retrieval["top_k"]
        if "score_threshold" in retrieval:
            extra["retrieval_score_threshold"] = retrieval["score_threshold"]
        if "kb_path" in retrieval:
            extra["retrieval_kb_path"] = retrieval["kb_path"]
        return extra

    # -- Node registration -----------------------------------------------------

    def register_nodes(self) -> None:
        """Register all 5 domain nodes.

        No super() call - BaseGraph.register_nodes() is abstract.
        Do NOT register initialize or finalize; those are outer backbone
        concerns handled by AgentBaseGraph in graph.py.

        Every node is instantiated with NO constructor arguments - FunctionNode
        subclasses take no __init__; tuning config flows in via State (see
        _extra_initial_state() above), and execute(self, state) takes no
        config parameter. Every key registered here is referenced in
        add_edges().
        """
        self._nodes["input_validate"] = InputValidateNode()
        self._nodes["retrieve"] = RetrieveNode()
        self._nodes["rerank_filter"] = RerankFilterNode()
        self._nodes["generate_answer"] = GenerateAnswerNode()
        self._nodes["output_format"] = OutputFormatNode()

    # -- Edge wiring -----------------------------------------------------------

    def add_edges(self) -> None:
        """Wire the linear shrinkage-QA domain topology.

        Each step passes its partial-dict output into the shared State.
        For this template the topology is intentionally linear - no conditional
        branching between domain nodes. route() is implemented as required by
        the ABC but add_conditional_edges() is not used.
        """
        self._sg.add_edge(START, "input_validate")
        self._sg.add_edge("input_validate", "retrieve")
        self._sg.add_edge("retrieve", "rerank_filter")
        self._sg.add_edge("rerank_filter", "generate_answer")
        self._sg.add_edge("generate_answer", "output_format")
        self._sg.add_edge("output_format", END)

    # -- Routing ---------------------------------------------------------------

    def route(self, state: AgentState) -> str:
        """Conditional routing - required by BaseGraph ABC.

        For this linear topology add_conditional_edges() is not used, so this
        method is never called at runtime. It is implemented to satisfy the ABC
        contract. Returns END on error so an unexpected call does not re-enter a
        processing node.
        """
        if state.get("status") == AgentStatus.ERROR.value:
            return str(END)
        return "output_format"

    # -- Output shape ----------------------------------------------------------

    def get_output(self, state: AgentState) -> dict[str, Any]:
        """Shape the output dict returned to the outer graph as sub_result.

        This dict is received by ShrinkageQAGraphNode.merge_output() in
        graph.py as the `sub_result` argument. Both methods are designed
        together to guarantee field-name consistency:

            Inner get_output()  emits: "formatted_answer", "citations", "status", ...
            Outer merge_output() reads: sub_result.get("formatted_answer"),
                                        sub_result.get("citations"),
                                        sub_result.get("status")

        Additional fields (intake_notes, trace_id, correlation_id,
        node_history) are surfaced for observability / downstream extension.

        THE ANSWER IS CONDITIONED ON THIS GRAPH'S OWN TERMINAL STATUS.
        On any status other than SUCCESS the answer-bearing fields
        (formatted_answer, citations) are omitted entirely - not emptied, not
        truncated, not replaced with a partial draft. The reason is the shape
        of the egress one layer up: merge_output() copies the inner answer
        into the OUTER `result`, and AgentBaseGraph.get_output() publishes
        `formatted_output or result` with no status check. A non-SUCCESS outer
        status routes main -> finalize (AgentBaseGraph.route), so PostProcessNode
        - the output gate that would have written formatted_output - never
        runs, and whatever sits in `result` becomes the caller's answer by the
        fallback road. A draft this graph produced but did not stand behind
        must therefore never leave this method.

        The check lives here rather than only in merge_output() because this
        graph is a callable unit in its own right: it is instantiated and
        invoked directly by tests, by src/examples/, and by any future second
        caller, none of which pass through ShrinkageQAGraphNode. The outer
        merge keeps a matching presence check so it cannot resurrect what was
        withheld here (defence in depth, not a second gate).

        intake_notes / trace_id / correlation_id / node_history are always
        emitted: they are operational metadata authored by this template
        (validation notes, ids, node class names) and carry no retrieved
        knowledge-base content.
        """
        output: dict[str, Any] = {
            "status": state.get("status"),
            "intake_notes": state.get("intake_notes"),
            "trace_id": state.get("trace_id"),
            "correlation_id": state.get("correlation_id"),
            "node_history": state.get("node_history", []),
        }
        if state.get("status") == AgentStatus.SUCCESS.value:
            output["formatted_answer"] = state.get("formatted_answer")
            output["citations"] = state.get("citations")
        return output
