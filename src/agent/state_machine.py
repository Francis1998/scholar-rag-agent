"""Explicit state transition validation for agent runs."""

from typing import ClassVar

from agent.models import AgentState


class InvalidTransitionError(ValueError):
    """Raised when an agent run attempts an invalid state transition."""


class AgentStateMachine:
    """Validate the required IDLE-to-DONE or ERROR state machine."""

    _allowed: ClassVar[dict[AgentState, set[AgentState]]] = {
        AgentState.IDLE: {AgentState.PLANNING, AgentState.ERROR},
        AgentState.PLANNING: {AgentState.RETRIEVING, AgentState.ERROR},
        AgentState.RETRIEVING: {AgentState.REASONING, AgentState.ERROR},
        AgentState.REASONING: {AgentState.ANSWERING, AgentState.ERROR},
        AgentState.ANSWERING: {AgentState.DONE, AgentState.ERROR},
        AgentState.DONE: set(),
        AgentState.ERROR: set(),
    }

    def can_transition(self, from_state: AgentState, to_state: AgentState) -> bool:
        """Return whether a transition is valid."""
        return to_state in self._allowed[from_state]

    def validate(self, from_state: AgentState, to_state: AgentState) -> None:
        """Raise when a transition is not part of the explicit state machine."""
        if not self.can_transition(from_state, to_state):
            message = f"invalid state transition: {from_state} -> {to_state}"
            raise InvalidTransitionError(message)
