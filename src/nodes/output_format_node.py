"""AgentCore Platform v1.0"""

# RET-C2-005 - OutputFormatNode
# Domain node 5 (terminal): compose the final formatted answer - the grounded
# answer body, the Sources list, and the standing scope disclaimer. The
# disclaimer is part of THIS node's domain output contract, not of the outer
# post_process slot (post_process only gates, it does not compose).
#
# Output schema note: the external answer renders monetary figures in units
# of 1,000 (the outer output gate independently enforces that grid - see
# post_process_node.py). The bundled knowledge base contains no monetary
# figures, so the note line is appended only when the assembled answer
# actually carries a monetary-form token; the detection reuses the gate's own
# grammar so renderer and gate can never drift apart.
#
# Node contract: execute(self, state) -> dict ONLY - no config parameter.
# Wired by the inner graph (DomainWorkflowGraph). get_output() of the inner
# graph surfaces formatted_answer + status to the outer merge_output().
# Returns only changed state keys (partial dict).

from typing import Any, ClassVar, Dict, List

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.nodes.post_process_node import _NUM_TOKEN_RE
from src.schemas.state import from_json

# Standing scope disclaimer - appended to EVERY answer this template emits.
_SCOPE_DISCLAIMER = (
    "This answer is generated from the seeded loss-prevention knowledge base "
    "for informational purposes only. It describes general anomaly patterns "
    "and procedures, does not identify or accuse any individual, and is not "
    "a substitute for a formal investigation. Verify against your store's "
    "current SOPs and involve your regional LP lead before taking any "
    "personnel action."
)

# Printed when the answer renders monetary figures (approved external
# schema: amounts are expressed in units of 1,000).
_SCHEMA_NOTE = "Note: monetary figures in this answer are expressed in units of 1,000."


class OutputFormatNode(FunctionNode):
    """Compose the final answer: body + sources + scope disclaimer.

    Input state keys:
        grounded_answer: answer body with [n] citation markers
        citations:       JSON list [{ref, id, title, source}]

    Output state keys (partial dict):
        formatted_answer: final rendered answer string
        status:           AgentStatus.SUCCESS.value (plain string - never
                          write the bare enum to State)
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> dict[str, Any]:
        grounded_answer = state.get("grounded_answer") or ("No answer is available for this request.")
        citations: List[Dict[str, Any]] = from_json(state.get("citations"), []) or []

        lines: List[str] = []
        lines.append("# Shrinkage & Inventory Anomaly Q&A Result")
        lines.append("")
        lines.append(grounded_answer)
        lines.append("")
        lines.append("## Sources")
        if citations:
            for citation in citations:
                if not isinstance(citation, dict):
                    continue
                ref = citation.get("ref", "?")
                title = str(citation.get("title", "")).strip()
                source = str(citation.get("source", "")).strip()
                suffix = f" ({source})" if source else ""
                lines.append(f"- [{ref}] {title}{suffix}")
        else:
            lines.append("- none (no knowledge-base passage cleared the relevance threshold)")
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append(f"*{_SCOPE_DISCLAIMER}*")

        formatted_answer = "\n".join(lines)
        if _NUM_TOKEN_RE.search(formatted_answer):
            formatted_answer = "\n".join([*lines, "", _SCHEMA_NOTE])

        # Audit: final answer composed (disclaimer attached).
        emit_trace_event(
            "output_format_complete",
            {
                "answer_chars": len(formatted_answer),
                "citation_count": len(citations),
            },
            state,
        )

        return {
            "formatted_answer": formatted_answer,
            "status": AgentStatus.SUCCESS.value,
        }
