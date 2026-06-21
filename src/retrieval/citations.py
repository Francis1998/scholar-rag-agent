"""Citation grounding and hallucination guard."""

from agent.models import AgentAnswer, Citation, Claim
from retrieval.models import Chunk
from retrieval.sparse import meaningful_terms


class CitationGrounder:
    """Validate claims against retrieved source chunks."""

    def ground(
        self, answer_text: str, claims: list[Claim], retrieved_chunks: list[Chunk]
    ) -> AgentAnswer:
        """Return an answer with citations and ungrounded claims flagged."""
        chunk_by_id = {chunk.chunk_id: chunk for chunk in retrieved_chunks}
        grounded_claims: list[Claim] = []
        citations: dict[str, Citation] = {}
        for claim in claims:
            supporting_ids = [chunk_id for chunk_id in claim.chunk_ids if chunk_id in chunk_by_id]
            claim_terms = meaningful_terms(claim.text)
            grounded_ids = []
            for chunk_id in supporting_ids:
                chunk = chunk_by_id[chunk_id]
                chunk_terms = meaningful_terms(chunk.text)
                if claim_terms and claim_terms & chunk_terms:
                    grounded_ids.append(chunk_id)
                    citations[chunk_id] = Citation(
                        chunk_id=chunk.chunk_id,
                        document_id=chunk.document_id,
                        title=chunk.title,
                        snippet=chunk.text[:240],
                    )
            grounded_claims.append(
                Claim(text=claim.text, chunk_ids=grounded_ids, grounded=bool(grounded_ids))
            )
        ungrounded = any(not claim.grounded for claim in grounded_claims)
        final_answer = answer_text if not ungrounded else f"[UNGROUNDED] {answer_text}"
        warnings = ["One or more claims lacked retrieved chunk support."] if ungrounded else []
        return AgentAnswer(
            answer=final_answer,
            citations=list(citations.values()),
            claims=grounded_claims,
            ungrounded=ungrounded,
            warnings=warnings,
        )
