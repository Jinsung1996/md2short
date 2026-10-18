"""Job status state machine (docs/architecture.md section 3).

Status is persisted to jobs/<job_id>/status.json. There is no database;
validate_transition() is the sole gate for allowed status transitions.

Progress events (photo_generating, photos_generated, photo_fallback; see
architecture.md section 3.1) are NOT part of this state machine — they are
surfaced via a separate on_stage callback mechanism implemented later in
pipeline.py, never written to status.json, and never passed through
validate_transition().
"""

from __future__ import annotations

# Linear happy-path sequence (architecture.md section 3). This is also exactly
# the set of non-terminal states, so NON_TERMINAL_STATES is derived from it
# rather than spelled out a second time that could drift out of step.
_SEQUENCE: list[str] = [
    "created",
    "script_generated",
    "voice_generated",
    "transcribed",
    "subtitled",
    "final_rendered",
]

NON_TERMINAL_STATES: list[str] = list(_SEQUENCE)

TERMINAL_STATES: list[str] = [
    "qc_passed",
    "qc_failed",
    "failed",
]

# ALLOWED_TRANSITIONS[current_state] = set of next_states that are valid from it.
ALLOWED_TRANSITIONS: dict[str, set[str]] = {}

for _index, _state in enumerate(_SEQUENCE):
    _next_states: set[str] = set()
    if _index + 1 < len(_SEQUENCE):
        _next_states.add(_SEQUENCE[_index + 1])
    if _state == "final_rendered":
        _next_states.update({"qc_passed", "qc_failed"})
    # Any non-terminal state may always transition to "failed".
    _next_states.add("failed")
    ALLOWED_TRANSITIONS[_state] = _next_states

# Terminal states have no outgoing transitions.
for _state in TERMINAL_STATES:
    ALLOWED_TRANSITIONS.setdefault(_state, set())


def validate_transition(current_state: str, next_state: str) -> None:
    """Raise ValueError if current_state -> next_state is not an allowed transition.

    Returns None (does nothing) when the transition is valid.
    """
    allowed = ALLOWED_TRANSITIONS.get(current_state)
    if allowed is None or next_state not in allowed:
        raise ValueError(
            f"Invalid state transition: {current_state!r} -> {next_state!r} is not allowed"
        )
