# Answer Synthesis Prompt — RET-C2-005 (LLM upgrade seam)

> **The current build does NOT use this prompt at runtime.** The shipped
> `GenerateAnswerNode` is deterministic (rule-based grounded assembly over
> `ranked_documents`); no node reads this file. It documents the synthesis
> contract for the LLM upgrade described in `docs/02_design.md`
> ("Implementation Note — LLM synthesis"), so the swap changes only the
> inside of `GenerateAnswerNode.execute()`.

## Contract (upgraded GenerateAnswerNode)

- **Input:** the same `ranked_documents` JSON (id / title / category / source /
  score / excerpt) and `search_query` the current node reads.
- **Output:** the same state contract — `grounded_answer` (str, with numbered
  `[n]` citation markers) and `citations` (JSON list of
  `{ref, id, title, source}`).
- **Grounding rule:** every factual statement in the answer must be traceable
  to one of the supplied passages via a `[n]` marker; content not present in
  the passages must not be asserted.
- **No-coverage rule:** when no passage supports the question, say so and
  recommend refining the query or escalating to the regional LP lead — never
  answer from parametric knowledge.
- **Tone:** neutral, operations-appropriate, procedural (not accusatory toward
  any individual associate); the standing scope disclaimer is appended
  downstream by `OutputFormatNode`.

## Prompt template

```
You answer retail loss-prevention questions about shrinkage and inventory
anomalies strictly from the knowledge-base passages provided below.

Question:
{search_query}

Passages (each with a reference number):
{ranked_documents}

Rules:
1. Use ONLY the passages above. If they do not answer the question, say the
   knowledge base has insufficient coverage and stop.
2. Mark every factual statement with the [n] reference of its passage.
3. Do not name or accuse a specific individual; describe patterns and
   procedures only.
4. Keep the answer under 300 words.
```

## Configuration coupling

The `llm` block in `config/config.yaml` (`temperature`, `max_tokens`) is
already forwarded to the inner graph via
`ShrinkageQAGraphNode._parent_config()` under `config["configurable"]["llm"]`;
the upgraded node reads it from there.
