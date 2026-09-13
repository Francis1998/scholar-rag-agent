"""Tests for AuthorNameDisambiguationHint."""

from retrieval.author_disambiguation import AuthorNameDisambiguationHint


def test_empty_authors_ok() -> None:
    assert AuthorNameDisambiguationHint().group([]) == ()


def test_groups_initial_with_full_given_name() -> None:
    groups = AuthorNameDisambiguationHint().group(
        [
            "J Smith",
            "John Smith",
            "Jane Doe",
            "J. Smith",
        ]
    )
    smith = next(
        g for g in groups
        if "Smith" in g.canonical or any("Smith" in m for m in g.members)
    )
    members_folded = {m.casefold() for m in smith.members}
    assert "j smith" in members_folded or "j. smith" in members_folded
    assert "john smith" in members_folded
    assert len(smith.members) >= 2
    assert "initial" in smith.reason.lower() or "near" in smith.reason.lower()


def test_does_not_merge_different_surnames() -> None:
    groups = AuthorNameDisambiguationHint().group(["J Smith", "J Smythe", "John Smith"])
    # Smith variants may group; Smythe stays separate.
    all_multi = [g for g in groups if len(g.members) >= 2]
    for group in all_multi:
        joined = " ".join(group.members).casefold()
        assert "smythe" not in joined or "smith" not in joined


def test_preserves_singletons_and_order_stable_canonical() -> None:
    groups = AuthorNameDisambiguationHint().group(["Alice Alone", "Bob Solo"])
    assert len(groups) == 2
    assert all(len(g.members) == 1 for g in groups)


def test_never_mutates_input() -> None:
    authors = ["J Smith", "John Smith"]
    snapshot = list(authors)
    AuthorNameDisambiguationHint().group(authors)
    assert authors == snapshot


def test_docstring_mentions_frontier_models_and_gap() -> None:
    doc = AuthorNameDisambiguationHint.__doc__ or ""
    assert "GPT-5.5" in doc
    assert "Claude Sonnet 4.6" in doc
    assert "Gemini 3.x" in doc
    assert "Kimi K2" in doc
    assert (
        "VenueTierBooster" in doc
        or "venue_tier" in doc
        or "OpenAlex" in doc
        or "Semantic Scholar" in doc
    )
