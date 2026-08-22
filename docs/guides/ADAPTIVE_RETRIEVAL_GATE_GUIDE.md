# Adaptive Retrieval Gate Guide

![Adaptive retrieval gate demo](../assets/adaptive_retrieval_gate.gif)

`AdaptiveRetrievalGate` decides whether corpus retrieval should run at all.
Inspired by Adaptive RAG and Self-RAG retrieval tokens, it returns `RETRIEVE`
or `SKIP` from deterministic lexical cues before any index lookup. This is
distinct from `SelfRagReflectionGate`, which evaluates evidence quality
*after* retrieval. When retrieval proceeds, downstream synthesis can use
GPT-5.5, Claude Sonnet 4.6, Gemini 3.x, or Kimi K2.

## Decision policy

1. Empty queries `SKIP`.
2. Exact chitchat / greeting matches such as `hello`, `thanks`, or
   `how are you` `SKIP`.
3. Meta/system phrases such as `what can you do` or `who are you` `SKIP`
   unless scholarly knowledge cues are also present.
4. Queries with cues like `paper`, `study`, `literature`, `evidence`,
   `method`, `compare`, or `pubmed` `RETRIEVE`.
5. Other queries with non-stopword content terms still `RETRIEVE`.
6. Stopword-only queries `SKIP`.

Each decision includes a stable reason string and the matched cue phrases or
terms used for the choice.

## Example

```python
from retrieval.adaptive_retrieval_gate import AdaptiveRetrievalGate

gate = AdaptiveRetrievalGate()
decision = gate.decide(user_query)

if decision.action == "SKIP":
    answer_without_retrieval(decision.reason)
else:
    results = await retriever.retrieve(user_query)
    synthesize(results)
```

Use this gate at the front of the Observe -> Decide -> Act loop to avoid
wasting retrieval budget on greetings and capability questions, while still
fetching evidence for knowledge-seeking research prompts.
