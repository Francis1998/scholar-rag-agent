"""Tests for the explicit agent state machine."""

from itertools import pairwise

import pytest

from agent.models import AgentState
from agent.state_machine import AgentStateMachine, InvalidTransitionError


def test_state_machine_accepts_required_path() -> None:
    """The state machine accepts the required successful run path."""
    machine = AgentStateMachine()
    path = [
        AgentState.IDLE,
        AgentState.PLANNING,
        AgentState.RETRIEVING,
        AgentState.REASONING,
        AgentState.ANSWERING,
        AgentState.DONE,
    ]
    for from_state, to_state in pairwise(path):
        machine.validate(from_state, to_state)


def test_state_machine_rejects_skipped_states() -> None:
    """The state machine rejects invalid skipped transitions."""
    machine = AgentStateMachine()
    with pytest.raises(InvalidTransitionError):
        machine.validate(AgentState.IDLE, AgentState.REASONING)
