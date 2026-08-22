# Claim Verification Gate Guide

![Claim verification gate demo](../assets/claim_verification_gate.gif)

`ClaimVerificationGate` audits a draft answer against retrieved evidence before
the response is shown to a user. Inspired by RAGAS faithfulness and TruLens
groundedness checks, it splits the answer into claim sentences and scores each
claim with deterministic lexical overlap. No LLM or network call is required.
Verified answers can then be refined or explained with GPT-5.5, Claude Sonnet
4.6, Gemini 3.x, or Kimi K2.

## How verification works

1. Split the draft answer on sentence boundaries (`.`, `!`, `?`, or newlines).
2. Extract distinct non-stopword terms from each claim.
3. Measure what fraction of those terms appear across retrieved chunk titles
   and text.
4. Mark a claim `supported` when its support score meets `support_threshold`
   and at least one chunk contributes overlap.
5. Report overall `groundedness` as supported claims divided by all claims.

```text
support_score = covered claim terms / all claim terms
groundedness  = supported claims / all claims
```

The default `support_threshold` is `0.5`. Thresholds are inclusive, finite
values in `[0.0, 1.0]`.

## Example

```python
from retrieval.claim_verification_gate import ClaimVerificationGate

gate = ClaimVerificationGate(support_threshold=0.5)
report = gate.verify(draft_answer, retrieved_results)

if report.groundedness < 1.0:
    for claim in report.claims:
        if not claim.supported:
            flag_ungrounded(claim.claim)
```

Each verdict includes the claim text, boolean support flag, support score, and
supporting chunk ids. Empty answers produce an empty report with groundedness
`0.0`. An optional LLM adapter may be passed for future semantic checks and is
unused by the lexical path.
