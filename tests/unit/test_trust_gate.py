# RET-C2-005 — Unit Tests: caller trust gate
#
# There is no test_main_node.py (a Cat-1-style MainNode stub is unused by
# this Cat-2 nested graph; the main slot is ShrinkageQAGraphNode in
# src/graph/graph.py). Covers the outer trust
# boundary and the template's full trust matrix (outer VERIFIED_EXTERNAL
# ingest gate + ANONYMOUS post_process + five inner ANONYMOUS domain nodes).
#
# Trust-gate contract: tests must invoke nodes via node(state) — through
# BaseNode.__call__, which runs trust gate -> input mask -> execute() ->
# output gate — never via node.execute(state) directly, which bypasses the
# gate. A denial RETURNS an error dict (never raises) with status ERROR and
# "trust gate denied" in error_log; execute() never runs, so execute-only
# output keys are ABSENT from the returned dict.

from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

from src.nodes.generate_answer_node import GenerateAnswerNode
from src.nodes.input_validate_node import InputValidateNode
from src.nodes.output_format_node import OutputFormatNode
from src.nodes.post_process_node import PostProcessNode
from src.nodes.pre_process_node import PreProcessNode
from src.nodes.rerank_filter_node import RerankFilterNode
from src.nodes.retrieve_node import RetrieveNode


def _make_state(trust_value: str, user_input: str = "what is our current shrinkage rate?", **extra) -> dict:
    state = {
        "user_input": user_input,
        "caller_trust_level": trust_value,
        "node_history": [],
        "error_log": [],
        "session_id": "test-session",
        "execution_time": {},
    }
    state.update(extra)
    return state


class TestTrustGate:
    """Trust gate tests — all invocations go through node(state) / __call__."""

    def test_anonymous_caller_allowed_on_inner_node(self):
        """An ANONYMOUS caller passes an ANONYMOUS inner domain node."""
        node = InputValidateNode()  # required_trust_level = ANONYMOUS
        result = node(_make_state(TrustLevel.ANONYMOUS.value, validated_input="test query"))
        assert "trust gate denied" not in str(result.get("error_log", []))
        assert result.get("search_query"), "inner node should produce a normalised query"

    def test_anonymous_caller_denied_on_pre_process(self):
        """Trust rejection: ANONYMOUS caller on the VERIFIED_EXTERNAL PreProcessNode.

        __call__ must RETURN an error dict (never raise) with status ERROR and
        'trust gate denied' in the error_log. execute() never ran, so the
        execute-only output key (validated_input) must be ABSENT.
        """
        node = PreProcessNode()  # required_trust_level = VERIFIED_EXTERNAL
        result = node(_make_state(TrustLevel.ANONYMOUS.value))
        assert result.get("status") == AgentStatus.ERROR.value
        error_log = result.get("error_log", [])
        assert any(
            "trust gate denied" in str(e) for e in error_log
        ), f"Expected 'trust gate denied' in error_log, got: {error_log}"
        assert "validated_input" not in result, "execute() must not run on a trust denial — validated_input leaked"

    def test_verified_external_caller_passes_pre_process(self):
        """A VERIFIED_EXTERNAL caller clears the pre_process gate and the node
        writes validated_input."""
        node = PreProcessNode()
        result = node(_make_state(TrustLevel.VERIFIED_EXTERNAL.value))
        assert result.get("status") == AgentStatus.SUCCESS.value
        assert result.get("validated_input")

    def test_pre_process_empty_input_rejected_after_gate(self):
        """The trust gate passes, then the node's own input validation rejects
        empty input."""
        node = PreProcessNode()
        result = node(_make_state(TrustLevel.VERIFIED_EXTERNAL.value, user_input=""))
        assert result.get("status") == AgentStatus.ERROR.value
        assert any("empty" in str(e) for e in result.get("error_log", []))

    def test_anonymous_caller_passes_post_process(self):
        """post_process is declared ANONYMOUS: trust is enforced once, at the
        pre_process ingest boundary, and the output gate must still run for
        callers the backbone routes through on any path."""
        node = PostProcessNode()
        result = node(
            _make_state(
                TrustLevel.ANONYMOUS.value,
                result="a clean shrinkage knowledge base answer",
            )
        )
        assert result.get("status") == AgentStatus.SUCCESS.value
        assert result.get("formatted_output")

    def test_verified_external_caller_passes_post_process(self):
        """A VERIFIED_EXTERNAL caller clears the post_process slot."""
        node = PostProcessNode()
        result = node(
            _make_state(
                TrustLevel.VERIFIED_EXTERNAL.value,
                result="a clean shrinkage knowledge base answer",
            )
        )
        assert result.get("status") == AgentStatus.SUCCESS.value
        assert result.get("formatted_output")


class TestTrustLevelMatrix:
    """The template's declared trust matrix (docs/02_design.md security section).

    The pre_process ingest slot requires VERIFIED_EXTERNAL (the manifest's
    required_trust_level); the post_process output gate and the five inner
    domain nodes run behind that boundary and are declared ANONYMOUS per the
    Cat-2 nested convention.
    """

    def test_ingest_gate_requires_verified_external(self):
        assert PreProcessNode.required_trust_level is TrustLevel.VERIFIED_EXTERNAL

    def test_post_process_is_anonymous(self):
        assert PostProcessNode.required_trust_level is TrustLevel.ANONYMOUS

    def test_inner_domain_nodes_admit_anonymous(self):
        for node_cls in (
            InputValidateNode,
            RetrieveNode,
            RerankFilterNode,
            GenerateAnswerNode,
            OutputFormatNode,
        ):
            assert node_cls.required_trust_level is TrustLevel.ANONYMOUS, (
                f"{node_cls.__name__} must declare TrustLevel.ANONYMOUS " "(inner Cat-2 domain node)"
            )
