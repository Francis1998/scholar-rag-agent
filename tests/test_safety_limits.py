"""Unit tests for :class:`agent.safety.SafetyLimits` clamping behavior."""

from __future__ import annotations

from agent.safety import SafetyLimits


def test_clamp_hops_honors_configured_limit_above_default() -> None:
    """A configured ``max_hops`` above the historical default must be honored.

    ``clamp_hops`` previously applied a hardcoded literal ceiling of ``5``
    alongside the configurable ``max_hops`` field, so a deployment (or caller)
    that raised ``max_hops`` above ``5`` had its requests silently capped at
    ``5`` despite the docstring promising to clamp to the *configured* range.
    The configured field must be the sole upper bound.
    """
    limits = SafetyLimits(max_hops=8)

    assert limits.clamp_hops(7) == 7
    assert limits.clamp_hops(9) == 8


def test_clamp_sources_honors_configured_limit_above_default() -> None:
    """A configured ``max_source_docs`` above the default must be honored.

    ``clamp_sources`` previously hardcoded a literal ceiling of ``50`` next to
    the configurable ``max_source_docs`` field, silently capping any configured
    value above ``50``.
    """
    limits = SafetyLimits(max_source_docs=100)

    assert limits.clamp_sources(80) == 80
    assert limits.clamp_sources(120) == 100


def test_clamp_preserves_default_bounds_and_floors() -> None:
    """Default limits and lower floors must be unchanged by the fix."""
    limits = SafetyLimits()

    assert limits.clamp_hops(99) == 5
    assert limits.clamp_hops(-3) == 0
    assert limits.clamp_sources(999) == 50
    assert limits.clamp_sources(0) == 1
