"""Unit tests for md2short.state (docs/architecture.md section 3)."""

import pytest

from md2short.state import _SEQUENCE, validate_transition


def test_every_allowed_sequential_transition_passes() -> None:
    for current_state, next_state in zip(_SEQUENCE, _SEQUENCE[1:]):
        validate_transition(current_state, next_state)  # should not raise


@pytest.mark.parametrize("terminal_state", ["qc_passed", "qc_failed"])
def test_final_rendered_to_terminal_states_passes(terminal_state: str) -> None:
    validate_transition("final_rendered", terminal_state)  # should not raise


def test_disallowed_transition_raises_value_error() -> None:
    with pytest.raises(ValueError):
        validate_transition("created", "final_rendered")


@pytest.mark.parametrize(
    "non_terminal_state",
    ["created", "script_generated", "voice_generated", "transcribed", "subtitled", "final_rendered"],
)
def test_non_terminal_to_failed_always_succeeds(non_terminal_state: str) -> None:
    validate_transition(non_terminal_state, "failed")  # should not raise


def test_terminal_states_have_no_outgoing_transitions() -> None:
    for terminal_state in ["qc_passed", "qc_failed", "failed"]:
        with pytest.raises(ValueError):
            validate_transition(terminal_state, "created")
