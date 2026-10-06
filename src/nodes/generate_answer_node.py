"""AgentCore Platform v1.0"""

# RET-C2-005 - GenerateAnswerNode
# Domain node 4: assemble the grounded answer from the ranked KB passages.
#
# The current build is DETERMINISTIC (no live LLM call): the answer is
# rule-assembled from the ranked passages only - a lead sentence plus one
# cited point per passage, each carrying a numbered citation marker [n].
# Nothing outside the ranked_documents input reaches the answer body, so the
# output is grounded by construction.
#
# The caller's question is deliberately NOT embedded in the answer: the
# response is built from retrieved knowledge-base content only. Echoing
# caller text would let a request smuggle arbitrary content - including fake
# [n] markers that masquerade as citations - into a document whose whole
# value is that every statement traces to a retrieved source.
#
# The LLM synthesis upgrade seam is documented in docs/02_design.md
# ("Implementation Note - LLM synthesis") and
# config/prompts/answer_synthesis_prompt.md: an upgraded node swaps the
# assembly for an LLM call over the same input and emits the same state
# contract.
#
# Node contract: execute(self, state) -> dict ONLY - no config parameter.
# Wired by the inner graph (DomainWorkflowGraph).
# Returns only changed state keys (partial dict).

from typing import Any, ClassVar, Dict, List

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.schemas.state import from_json, to_json

# Answer body used when no KB passage cleared the relevance threshold.
_NO_COVERAGE_ANSWER = (
    "The loss-prevention knowledge base does not contain sufficient coverage "
    "to answer this question. Rephrase the query with more specific "
    "anomaly, category, or procedure terms, or escalate to the regional LP "
    "lead for a manual review."
)

# Cited excerpt length per passage inside the answer body.
_POINT_EXCERPT_CHARS = 240


def _first_sentences(text: str, limit: int) -> str:
    """Trim an excerpt at a sentence boundary where possible, else hard-cap."""
    text = text.strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    period = cut.rfind(". ")
    if period > limit // 2:
        return cut[: period + 1]
    return cut.rstrip() + "..."


class GenerateAnswerNode(FunctionNode):
    """Rule-based grounded answer assembly with numbered citations.

    Input state keys:
        ranked_documents: JSON list of surviving passages (from RerankFilterNode)

    Output state keys (partial dict):
        grounded_answer: answer body with [n] citation markers
        citations:       JSON list [{ref, id, title, source}]
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> dict[str, Any]:
        ranked: List[Dict[str, Any]] = from_json(state.get("ranked_documents"), []) or []

        citations: List[Dict[str, Any]] = []

        if not ranked:
            grounded_answer = _NO_COVERAGE_ANSWER
        else:
            # The lead line never embeds the caller's question - the answer
            # body renders retrieved knowledge-base content only (see the
            # module docstring).
            lines: List[str] = []
            lines.append(
                "Based on the seeded loss-prevention knowledge base, the "
                "most relevant passages for this question are:"
            )
            lines.append("")
            for ref, doc in enumerate(ranked, start=1):
                if not isinstance(doc, dict):
                    continue
                title = str(doc.get("title", "")).strip()
                excerpt = _first_sentences(str(doc.get("excerpt", "")), _POINT_EXCERPT_CHARS)
                lines.append(f"[{ref}] {title}: {excerpt}")
                citations.append(
                    {
                        "ref": ref,
                        "id": str(doc.get("id", "")),
                        "title": title,
                        "source": str(doc.get("source", "")),
                    }
                )
            grounded_answer = "\n".join(lines)

        # Audit: grounded answer assembled.
        emit_trace_event(
            "generate_answer_complete",
            {
                "citation_count": len(citations),
                "answer_chars": len(grounded_answer),
                "no_coverage": not ranked,
            },
            state,
        )

        return {
            "grounded_answer": grounded_answer,
            "citations": to_json(citations),
        }
