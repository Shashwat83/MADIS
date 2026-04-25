from __future__ import annotations

from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from .models import DisasterObservation, DisasterState, DroneActions


class DisasterSurveillanceEnvClient(
    EnvClient[DroneActions, DisasterObservation, DisasterState]
):
    """Typed client for the disaster surveillance environment."""

    def _step_payload(self, action: DroneActions) -> Dict[str, Any]:
        return action.model_dump()

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[DisasterObservation]:
        obs_data = dict(payload.get("observation", {}))
        obs_data.setdefault("reward", payload.get("reward"))
        obs_data.setdefault("done", payload.get("done", False))
        observation = DisasterObservation.model_validate(obs_data)
        return StepResult(
            observation=observation,
            reward=payload.get("reward", observation.reward),
            done=payload.get("done", observation.done),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> DisasterState:
        return DisasterState.model_validate(payload)
