# PB: the inner domain graph must not hand out its answer on a non-success run.
#
# The egress this pins is the one AgentBaseGraph inherits, not one this template
# wrote:
#
#     AgentBaseGraph.get_output() -> {"output": formatted_output or result, ...}
#
# There is no status check on that selection. AgentBaseGraph.route() sends any
# non-SUCCESS status from the `main` slot straight to `finalize`, so
# PostProcessNode - the output gate, and the only writer of formatted_output -
# never runs on that path. Whatever sits in state["result"] then becomes the
# caller's answer by the fallback road.
#
# state["result"] is written by ShrinkageQAGraphNode.merge_output() from the
# inner graph's sub_result. So a draft the inner pipeline composed but did not
# stand behind is one unconditional return away from the caller. The contract
# under test:
#
#   - DomainWorkflowGraph.get_output() emits the answer-bearing fields ONLY on
#     a SUCCESS inner status;
#   - ShrinkageQAGraphNode.merge_output() copies the answer through only when
#     the sub_result actually carries one - it never substitutes a fallback and
#     never reads the answer back out of outer state.
#
# How the failure is driven: a seeded knowledge-base fixture puts a recognisable
# sentinel into the retrieved passage, so the REAL inner pipeline (all five
# domain nodes) composes a REAL draft answer carrying it. A node wired
# downstream of output_format then fails, which is the only ordering in which a
# completed draft is in inner state at the moment the run ends non-success -
# once status is ERROR the framework's own BaseNode.__call__ short-circuits
# every later node, so an upstream failure never reaches the answer composer.
# Nothing here hand-builds a state mapping: every assertion is made against the
# output of a compiled graph, and twice over against the real ASGI /invoke.
#
# Deterministic - no LLM, no network.

import asyncio
import json
from typing import Any, ClassVar, Iterator

import pytest
from langgraph.graph import END, START

import src.graph.domain_workflow_graph as inner_module
import src.graph.graph as outer_module
from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from src.api.server import app
from src.graph.domain_workflow_graph import DomainWorkflowGraph
from src.graph.graph import ShrinkageQAGraphNode

# The draft marker. Deliberately lowercase and digit-free: the S-2 gate masks
# title-case proper nouns and the output gate rewrites monetary-form tokens, so
# either shape would make "the sentinel is absent" ambiguous between
# containment and incidental rewriting.
_SENTINEL = "zzq-inner-draft-sentinel-marker"

# Lowercase, and NOT byte-equal to the fixture title: PostProcessNode redacts a
# verbatim embedding of the caller's question, which would otherwise mask a
# real leak on the success control below.
_QUERY = "shrinkage anomaly marker patterns store audit"

_TOKEN = "pb-inner-egress-token"

# Every query token hits the fixture entry (five in the title, "marker" in the
# tags), so the passage clears the relevance floor and the composed answer
# carries the sentinel.
_KB_FIXTURE = [
    {
        "id": "kb-fixture-001",
        "title": "Store audit checks for shrinkage anomaly patterns",
        "category": "anomaly_patterns",
        "source": "fixture loss-prevention playbook",
        "tags": ["marker", "shrinkage", "audit"],
        "content": (
            "Draft passage reserved for this boundary test: "
            + _SENTINEL
            + " is the recognisable body text the inner pipeline renders into its answer."
        ),
    }
]


# ---------------------------------------------------------------------------
# Fault injection: a stage downstream of the answer composer
# ---------------------------------------------------------------------------


class _StallAfterAnswerNode(FunctionNode):
    """Terminate the inner run non-success AFTER the draft answer exists.

    Models a post-composition verification stage that times out. TIMEOUT is a
    declared AgentStatus and AgentBaseGraph.route() names it as a terminal
    status that skips post_process - so this is a fault on the data path, not
    on any gate.
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> dict[str, Any]:
        return {
            "status": AgentStatus.TIMEOUT.value,
            "error_log": ["inner verification stage timed out"],
        }


class _RaiseAfterAnswerNode(FunctionNode):
    """Raise AFTER the draft answer exists.

    The framework's BaseNode.__call__ catches it and writes the ERROR status
    itself, so the resulting terminal state is framework-authored rather than
    test-authored.
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> dict[str, Any]:
        raise RuntimeError("inner verification stage failed")


def _inner_graph_failing_with(node_class: type) -> type:
    """The production inner graph with one extra stage after output_format.

    register_nodes(), get_output() and all five domain nodes are the production
    implementations; only the topology gains a terminal stage.
    """

    class _FailingInnerGraph(DomainWorkflowGraph):  # type: ignore[misc]
        def register_nodes(self) -> None:
            super().register_nodes()
            self._nodes["verify"] = node_class()

        def add_edges(self) -> None:
            self._sg.add_edge(START, "input_validate")
            self._sg.add_edge("input_validate", "retrieve")
            self._sg.add_edge("retrieve", "rerank_filter")
            self._sg.add_edge("rerank_filter", "generate_answer")
            self._sg.add_edge("generate_answer", "output_format")
            self._sg.add_edge("output_format", "verify")
            self._sg.add_edge("verify", END)

    return _FailingInnerGraph


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


@pytest.fixture
def seeded_kb(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> str:
    """Point the declared retrieval config at the sentinel-bearing fixture KB."""
    kb_path = tmp_path / "inner_egress_kb.json"
    kb_path.write_text(json.dumps(_KB_FIXTURE), encoding="utf-8")
    monkeypatch.setattr(
        outer_module,
        "_runtime_config",
        lambda: {
            "max_retry": 3,
            "retrieval": {"top_k": 4, "score_threshold": 0.25, "kb_path": str(kb_path)},
        },
    )
    monkeypatch.setenv("INVOKE_AUTH_TOKEN", _TOKEN)
    return str(kb_path)


def _use_failing_inner(monkeypatch: pytest.MonkeyPatch, node_class: type) -> None:
    """Swap the inner graph the outer GraphNode instantiates.

    ShrinkageQAGraphNode.get_subgraph() imports DomainWorkflowGraph lazily, on
    every call, so patching the module attribute reaches the live invoke path.
    """
    monkeypatch.setattr(inner_module, "DomainWorkflowGraph", _inner_graph_failing_with(node_class))


def _post_invoke(payload: dict) -> tuple[int, dict]:
    """POST /invoke with a Bearer token through the real ASGI app."""
    body = json.dumps(payload).encode()
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/invoke",
        "raw_path": b"/invoke",
        "root_path": "",
        "query_string": b"",
        "headers": [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
            (b"authorization", f"Bearer {_TOKEN}".encode()),
        ],
        "client": ("127.0.0.1", 12345),
        "server": ("127.0.0.1", 8000),
    }

    messages: list[dict] = []
    sent = {"body": b""}

    async def receive() -> dict:
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message: dict) -> None:
        messages.append(message)
        if message["type"] == "http.response.body":
            sent["body"] += message.get("body", b"")

    asyncio.run(app(scope, receive, send))
    start = next(m for m in messages if m["type"] == "http.response.start")
    return start["status"], json.loads(sent["body"].decode() or "{}")


def _invoke(text: str) -> dict:
    status_code, body = _post_invoke({"input": text, "session_id": "pb-inner-egress"})
    assert status_code == 200, f"expected 200, got {status_code}: {body}"
    return body


def _walk(value: Any) -> Iterator[Any]:
    """Yield every key and every leaf of a nested mapping / sequence."""
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _walk(item)
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            yield from _walk(item)
    else:
        yield value


def _assert_draft_absent(payload: Any, where: str) -> None:
    for leaf in _walk(payload):
        assert _SENTINEL not in str(leaf), f"inner draft leaked through {where}: {leaf!r}"


def _inner_run(node_class: type) -> dict:
    """Invoke the compiled inner graph directly, exactly as the outer node does."""
    graph_class = _inner_graph_failing_with(node_class)
    inner = graph_class(config=ShrinkageQAGraphNode()._parent_config())
    return inner.invoke(_QUERY, session_id="pb-inner-egress")


# ---------------------------------------------------------------------------
# The fixture has to be able to leak before "no leak" means anything
# ---------------------------------------------------------------------------


class TestFixtureIsLoadBearing:
    def test_success_run_really_publishes_the_sentinel(self, seeded_kb: str) -> None:
        """Known-good probe: on a SUCCESS run the draft IS the published answer.

        Without this, every absence assertion below would also pass against a
        fixture that never reached the answer at all.
        """
        body = _invoke(_QUERY)

        assert body["status"] == AgentStatus.SUCCESS.value
        assert _SENTINEL in body["output"]

    def test_success_run_still_carries_the_answer_through_the_inner_contract(self, seeded_kb: str) -> None:
        """Withholding is conditional, not a blanket suppression."""
        inner = DomainWorkflowGraph(config=ShrinkageQAGraphNode()._parent_config())
        result = inner.invoke(_QUERY, session_id="pb-inner-egress")

        assert result["status"] == AgentStatus.SUCCESS.value
        assert _SENTINEL in result["formatted_answer"]
        assert result["citations"]


# ---------------------------------------------------------------------------
# The inner graph, compiled and really failed
# ---------------------------------------------------------------------------


class TestCompiledInnerGraphWithholdsItsDraft:
    def test_non_success_terminal_status_withholds_the_answer(self, seeded_kb: str) -> None:
        result = _inner_run(_StallAfterAnswerNode)

        assert result["status"] == AgentStatus.TIMEOUT.value
        assert "formatted_answer" not in result
        assert "citations" not in result
        _assert_draft_absent(result, "DomainWorkflowGraph.get_output() on TIMEOUT")

    def test_raised_failure_withholds_the_answer(self, seeded_kb: str) -> None:
        result = _inner_run(_RaiseAfterAnswerNode)

        assert result["status"] == AgentStatus.ERROR.value
        assert "formatted_answer" not in result
        _assert_draft_absent(result, "DomainWorkflowGraph.get_output() on ERROR")


# ---------------------------------------------------------------------------
# The whole stack, through the real ASGI /invoke
# ---------------------------------------------------------------------------


class TestInvokeDoesNotPublishTheDraft:
    def test_timed_out_inner_run_publishes_no_answer(self, seeded_kb: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """The leak path: non-SUCCESS skips the output gate, so `result` egresses."""
        _use_failing_inner(monkeypatch, _StallAfterAnswerNode)

        body = _invoke(_QUERY)

        assert body["status"] != AgentStatus.SUCCESS.value
        assert body["output"] is None
        _assert_draft_absent(body, "POST /invoke after a timed-out inner run")

    def test_raised_inner_failure_publishes_no_answer(self, seeded_kb: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """The propagate path: SubgraphError, so merge_output() is never reached."""
        _use_failing_inner(monkeypatch, _RaiseAfterAnswerNode)

        body = _invoke(_QUERY)

        assert body["status"] == AgentStatus.ERROR.value
        assert body["output"] is None
        _assert_draft_absent(body, "POST /invoke after a raised inner failure")


# ---------------------------------------------------------------------------
# The outer half of the contract
# ---------------------------------------------------------------------------


class TestMergeOutputDoesNotResurrectTheDraft:
    def test_absent_answer_is_not_backfilled_from_outer_state(self) -> None:
        """merge_output() must not read the answer out of a prior outer state."""
        delta = ShrinkageQAGraphNode().merge_output(
            {"shrinkage_answer": "STALE DRAFT", "result": "STALE DRAFT"},
            {"status": AgentStatus.TIMEOUT.value},
        )

        assert "result" not in delta
        assert "shrinkage_answer" not in delta
        assert delta["status"] == AgentStatus.TIMEOUT.value
        _assert_draft_absent(delta, "merge_output() with an answerless sub_result")

    def test_present_answer_still_maps_to_both_fields(self) -> None:
        delta = ShrinkageQAGraphNode().merge_output(
            {},
            {"formatted_answer": "ANSWER", "citations": "[]", "status": AgentStatus.SUCCESS.value},
        )

        assert delta == {
            "status": AgentStatus.SUCCESS.value,
            "shrinkage_answer": "ANSWER",
            "result": "ANSWER",
            "citations": "[]",
        }
