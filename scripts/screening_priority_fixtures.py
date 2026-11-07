"""Synthetic labelled papers for the offline screening-prioritization simulation and demo.

Every title and abstract below is an original, generated fixture: no real publication,
author, dataset or finding is described. Labels exist only so a simulated reviewer can
replay screening; the application itself never sees or stores them as decisions.
"""

import random

from storage.screening_priority import SimulatedPaper

DATASET_SEED = 20261107
TOTAL_PAPERS = 100
RELEVANT_PAPERS = 20
REVIEW_QUESTION = (
    "Which papers ground generated answers to scientific questions in cited, retrieved evidence?"
)
_GENERIC = (
    "We evaluate on a public benchmark and report improvements over strong baselines.",
    "Code and synthetic data are released to support reproducibility.",
    "Limitations and directions for future work are discussed.",
    "Results are averaged over five random seeds.",
    "An ablation study isolates the contribution of each component.",
    "A small user study complements the automatic evaluation.",
)
# Field-wide background shared by every language-technology paper, relevant or not.
_FIELD = (
    "Large language models are increasingly deployed in real applications.",
    "Retrieval components are a common way to add external knowledge to models.",
    "Hallucinations remain a known failure mode of text generators.",
    "Human evaluation complements automatic metrics.",
    "Recent work on question answering motivates our design.",
)
# Related-work sentences that put relevant vocabulary into irrelevant near misses.
_CROSSOVER = (
    "Related work on retrieval-augmented question answering over scientific papers is reviewed.",
    "Unlike citation-grounded answer generators, our method does not cite sources.",
    "We discuss how retrieved evidence passages could be added in future work.",
)
_OFF_TOPIC = frozenset({"image-generation", "battery"})
# (topic, relevant, title patterns, title slots, abstract sentences). Most irrelevant topics
# are near misses that share query words, as in a real keyword-search export.
_TOPICS: tuple[tuple[str, bool, tuple[str, ...], tuple[str, ...], tuple[str, ...]], ...] = (
    (
        "grounded-rag",
        True,
        (
            "{} for scientific question answering",
            "{} for literature review questions",
            "{} with citation checks for biomedical questions",
        ),
        (
            "Retrieval-augmented generation",
            "Citation-grounded answer generation",
            "Evidence-linked passage retrieval",
            "Hybrid sparse and dense retrieval",
        ),
        (
            "We retrieve candidate passages from a corpus of scientific articles and condition "
            "a language model on the cited evidence.",
            "Each generated claim is checked for citation faithfulness against the retrieved "
            "passages.",
            "Grounding answers in source passages reduces unsupported statements and "
            "hallucinated references.",
            "We compare sparse lexical retrieval with dense embeddings and reciprocal rank "
            "fusion for evidence selection.",
            "Reviewers inspect the evidence trail behind every answer before trusting it.",
            "Answers cite the exact passages that support them so readers can verify claims.",
        ),
    ),
    (
        "source-attribution",
        True,
        ("Attributing {} to source documents", "Linking {} to supporting papers"),
        (
            "summary sentences in automated evidence synthesis",
            "claims in machine-written literature overviews",
        ),
        (
            "Each summary sentence links to the document that supports it.",
            "Sentences without a supporting source are flagged for human review.",
            "Attribution accuracy is measured against references chosen by expert readers.",
            "The system cites papers rather than inventing references.",
        ),
    ),
    (
        "claim-verification",
        True,
        ("Verifying {} against retrieved abstracts", "Checking {} with retrieved evidence"),
        ("scientific claims", "health claims from news articles"),
        (
            "Relevant abstracts are retrieved for each claim before a verdict is generated.",
            "The model labels each claim as supported or refuted and cites rationale sentences.",
            "Evidence retrieval errors are the main source of wrong verdicts.",
            "Rationales let readers audit why a claim was judged supported.",
        ),
    ),
    (
        "rag-code",
        False,
        ("{} for code completion", "{} from code repositories"),
        ("Retrieval-augmented generation", "Snippet retrieval"),
        (
            "Relevant code snippets are retrieved from the repository and added to the prompt.",
            "Completion accuracy improves for rarely used library functions.",
            "We measure exact-match accuracy on held-out repositories.",
            "Retrieved functions are reranked by syntactic similarity.",
        ),
    ),
    (
        "rag-support",
        False,
        ("{} for customer support", "{} for help-desk tickets"),
        ("Retrieval-augmented chatbots", "Retrieval-augmented answer generation"),
        (
            "Help-desk articles are retrieved and summarised into answers for customers.",
            "Agents rate generated replies for politeness and resolution.",
            "Hallucinated product details are reduced by retrieving manuals.",
            "Ticket resolution time falls after deployment.",
        ),
    ),
    (
        "citation-recommendation",
        False,
        ("Recommending {} for manuscript drafts", "Context-aware {}"),
        ("citations", "references to scientific papers"),
        (
            "Given a draft paragraph, the model ranks candidate papers to cite.",
            "Citation graphs and text embeddings are combined for recommendation.",
            "Recall at ten measures whether the cited paper appears in the list.",
            "No answers are generated; the output is a ranked reading list.",
        ),
    ),
    (
        "closed-book-qa",
        False,
        ("Closed-book {} on science exams", "Scaling language models for {}"),
        ("question answering", "scientific question answering"),
        (
            "A large language model answers science exam questions without retrieval.",
            "Accuracy grows with model size and chain-of-thought prompting.",
            "Multiple-choice questions cover physics, chemistry and biology.",
            "Answers are not linked to any source documents.",
        ),
    ),
    (
        "mt-hallucination",
        False,
        ("Detecting {} in neural machine translation", "Reducing {} in translation"),
        ("hallucinations", "unfaithful outputs"),
        (
            "Translations that ignore the source sentence are flagged as hallucinations.",
            "Faithfulness is estimated from attention patterns.",
            "Low-resource language pairs show the most hallucinated outputs.",
            "Human annotators label fluent but unfaithful translations.",
        ),
    ),
    (
        "music-retrieval",
        False,
        ("{} for playlist recommendation", "{} for music discovery"),
        ("Embedding retrieval of similar songs", "Fast nearest-neighbour song retrieval"),
        (
            "Listening sessions are embedded and similar tracks are retrieved for each user.",
            "Ranking quality is measured with listener skip rates.",
            "Audio features and play counts are combined in a two-tower model.",
            "Cold-start artists receive recommendations from content features alone.",
        ),
    ),
    (
        "table-qa",
        False,
        ("{} over spreadsheets", "{} over relational tables"),
        ("Question answering with SQL programs", "Numerical question answering"),
        (
            "Questions are translated into SQL queries executed over a single table.",
            "The parser handles aggregation, sorting and comparison operators.",
            "No external documents are retrieved; answers come from table cells.",
            "Execution accuracy improves with schema-aware decoding.",
        ),
    ),
    (
        "image-generation",
        False,
        ("{} for product photography", "{} for illustration"),
        ("Diffusion-based image generation", "Prompt-guided image synthesis"),
        (
            "A diffusion model generates images from short text prompts.",
            "Generated pictures are scored by human raters for realism.",
            "Guidance strength trades prompt fidelity against image diversity.",
            "We fine-tune the generator on a small style collection.",
        ),
    ),
    (
        "battery",
        False,
        ("{} in lithium-ion cells", "{} for fast-charging batteries"),
        ("Electrolyte additives", "Cathode coating strategies"),
        (
            "Capacity fade is measured over hundreds of charge cycles.",
            "Additives suppress dendrite growth at the anode interface.",
            "Impedance spectroscopy tracks degradation of the cathode.",
            "Thermal stability improves at high charging rates.",
        ),
    ),
)
# Twenty relevant papers in three differently worded subtopics; eighty mostly near misses.
_COUNTS = {
    "grounded-rag": 12,
    "source-attribution": 4,
    "claim-verification": 4,
    "rag-code": 10,
    "rag-support": 10,
    "citation-recommendation": 10,
    "closed-book-qa": 10,
    "mt-hallucination": 10,
    "music-retrieval": 8,
    "table-qa": 8,
    "image-generation": 7,
    "battery": 7,
}


def synthetic_papers() -> list[SimulatedPaper]:
    """Return the fixed 100-paper synthetic set in its canonical (topic-grouped) order."""
    generator = random.Random(DATASET_SEED)  # noqa: S311 - reproducible fixture, not security
    papers = []
    for topic, relevant, patterns, slots, sentences in _TOPICS:
        for index in range(_COUNTS[topic]):
            title = generator.choice(patterns).format(generator.choice(slots))
            parts = generator.sample(sentences, k=generator.choice((1, 2, 3)))
            if topic not in _OFF_TOPIC:
                parts.append(generator.choice(_FIELD))
                if not relevant and generator.random() < 0.3:
                    parts.append(generator.choice(_CROSSOVER))
            parts.append(generator.choice(_GENERIC))
            generator.shuffle(parts)
            abstract = " ".join(parts)
            papers.append(
                SimulatedPaper(
                    document_id=f"synthetic-{topic}-{index + 1:02d}",
                    title=title,
                    text=abstract,
                    relevant=relevant,
                )
            )
    if len(papers) != TOTAL_PAPERS or sum(paper.relevant for paper in papers) != RELEVANT_PAPERS:
        raise RuntimeError("The synthetic set must contain 100 papers, 20 of them relevant.")
    return papers


def collection_order(papers: list[SimulatedPaper], seed: int) -> list[SimulatedPaper]:
    """A deterministic shuffle standing in for an unranked search-export order."""
    shuffled = list(papers)
    random.Random(seed).shuffle(shuffled)  # noqa: S311 - reproducible fixture, not security
    return shuffled
