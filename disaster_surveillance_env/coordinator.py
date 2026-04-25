from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple

from .models import Coord, manhattan_distance

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment]


DEFAULT_ROUTER_BASE_URL = "https://router.huggingface.co/v1"
DEFAULT_REMOTE_LLM = "Qwen/Qwen2.5-72B-Instruct:fastest"


class CoordinatorAgent(ABC):
    """Interface for high-level coordination policies."""

    @abstractmethod
    def act(self, observation: Mapping[str, Any]) -> Dict[str, Coord]:
        """Return one target coordinate per drone."""


class TextGenerationBackend(Protocol):
    def generate(self, prompt: str) -> str:
        """Return generated text for the prompt."""


@dataclass
class CoordinatorDecision:
    targets: Dict[str, Coord]
    metadata: Dict[str, Any]


class HeuristicCoordinator(CoordinatorAgent):
    """Simple non-learning coordinator used as a fallback and baseline."""

    REGION_ANCHORS: Tuple[Coord, ...] = ((1, 1), (8, 1), (5, 8))
    SEVERITY_PRIORITY = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}

    def act(self, observation: Mapping[str, Any]) -> Dict[str, Coord]:
        return self.decide(observation).targets

    def decide(self, observation: Mapping[str, Any]) -> CoordinatorDecision:
        drone_positions: Mapping[str, Coord] = observation["drone_positions"]
        visible_events: Sequence[Mapping[str, Any]] = observation.get("visible_active_events", [])
        frontier_cells: Sequence[Coord] = observation.get("team_frontier_cells", [])
        grid_size = int(observation.get("grid_size", 10))

        remaining_targets: Dict[str, Coord] = {}
        assigned_cells: set[Coord] = set()
        prioritized_events = sorted(
            visible_events,
            key=lambda event: (
                self.SEVERITY_PRIORITY.get(str(event["severity"]), 99),
                int(event.get("deadline_remaining", 999)),
            ),
        )

        for drone_id, position in drone_positions.items():
            event_target = self._select_event_target(position, prioritized_events, assigned_cells)
            if event_target is not None:
                remaining_targets[drone_id] = event_target
                assigned_cells.add(event_target)
                continue

            frontier_target = self._select_frontier_target(position, frontier_cells, assigned_cells)
            if frontier_target is not None:
                remaining_targets[drone_id] = frontier_target
                assigned_cells.add(frontier_target)
                continue

            region_target = self._select_region_anchor(drone_id, position, grid_size, assigned_cells)
            remaining_targets[drone_id] = region_target
            assigned_cells.add(region_target)

        return CoordinatorDecision(
            targets=remaining_targets,
            metadata={
                "decision_source": "heuristic",
                "reason": "priority_frontier_region_assignment",
            },
        )

    def _select_event_target(
        self,
        position: Coord,
        prioritized_events: Sequence[Mapping[str, Any]],
        assigned_cells: set[Coord],
    ) -> Coord | None:
        available = [
            tuple(event["location"])
            for event in prioritized_events
            if tuple(event["location"]) not in assigned_cells
        ]
        if not available:
            return None
        return min(available, key=lambda location: manhattan_distance(position, location))

    def _select_frontier_target(
        self,
        position: Coord,
        frontier_cells: Sequence[Coord],
        assigned_cells: set[Coord],
    ) -> Coord | None:
        available = [tuple(cell) for cell in frontier_cells if tuple(cell) not in assigned_cells]
        if not available:
            return None
        return min(available, key=lambda location: manhattan_distance(position, location))

    def _select_region_anchor(
        self,
        drone_id: str,
        position: Coord,
        grid_size: int,
        assigned_cells: set[Coord],
    ) -> Coord:
        index = max(0, int(drone_id.split("_")[-1]) - 1)
        preferred = self.REGION_ANCHORS[index % len(self.REGION_ANCHORS)]
        preferred = (
            min(grid_size - 1, preferred[0]),
            min(grid_size - 1, preferred[1]),
        )
        if preferred not in assigned_cells:
            return preferred

        remaining: List[Coord] = [
            anchor
            for anchor in self.REGION_ANCHORS
            if anchor not in assigned_cells
        ]
        if remaining:
            return min(remaining, key=lambda location: manhattan_distance(position, location))
        return preferred


class HFRouterOpenAIBackend:
    """Text-generation backend using Hugging Face's OpenAI-compatible router."""

    def __init__(
        self,
        model_name: str = DEFAULT_REMOTE_LLM,
        api_token: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 20.0,
    ) -> None:
        if OpenAI is None:
            raise RuntimeError("openai is not available.")

        self.model_name = os.environ.get("MODEL_NAME", model_name)
        self.api_token = api_token or os.environ.get("HF_TOKEN")
        if not self.api_token:
            raise RuntimeError("HF_TOKEN is not set in the current shell environment.")
        self.base_url = (base_url or os.environ.get("API_BASE_URL") or DEFAULT_ROUTER_BASE_URL).rstrip("/")
        self.timeout = timeout
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_token,
            timeout=timeout,
        )

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a drone coordination planner that returns only JSON.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            max_tokens=220,
            temperature=0.2,
        )
        content = response.choices[0].message.content
        if content is None:
            raise RuntimeError(f"Router returned an empty completion for model '{self.model_name}'.")
        return str(content)


class LLMCoordinator(CoordinatorAgent):
    """Inference-only LLM coordinator with heuristic fallback."""

    def __init__(
        self,
        backend: Optional[TextGenerationBackend] = None,
        model_name: Optional[str] = None,
        fallback: Optional[HeuristicCoordinator] = None,
    ) -> None:
        self.model_name = model_name or os.environ.get("HF_COORDINATOR_MODEL") or os.environ.get("MODEL_NAME") or DEFAULT_REMOTE_LLM
        self.fallback = fallback or HeuristicCoordinator()
        self.backend = backend
        self.last_metadata: Dict[str, Any] = {}
        self._backend_error: Optional[str] = None

    def act(self, observation: Mapping[str, Any]) -> Dict[str, Coord]:
        decision = self.decide(observation)
        self.last_metadata = dict(decision.metadata)
        return decision.targets

    def decide(self, observation: Mapping[str, Any]) -> CoordinatorDecision:
        prompt = self._build_prompt(observation)
        started_at = time.perf_counter()
        backend = self._ensure_backend()
        if backend is None:
            fallback_decision = self.fallback.decide(observation)
            fallback_decision.metadata.update(
                {
                    "decision_source": "heuristic_fallback",
                    "fallback_reason": self._backend_error or "no_llm_backend_available",
                    "model_name": self.model_name,
                    "llm_enabled": False,
                }
            )
            return fallback_decision

        try:
            raw_text = backend.generate(prompt)
            latency_ms = 1000.0 * (time.perf_counter() - started_at)
            targets = self._parse_targets(raw_text, observation)
            return CoordinatorDecision(
                targets=targets,
                metadata={
                    "decision_source": "llm",
                    "model_name": self.model_name,
                    "llm_enabled": True,
                    "llm_latency_ms": latency_ms,
                    "llm_raw_response": raw_text,
                },
            )
        except Exception as exc:
            latency_ms = 1000.0 * (time.perf_counter() - started_at)
            fallback_decision = self.fallback.decide(observation)
            fallback_decision.metadata.update(
                {
                    "decision_source": "heuristic_fallback",
                    "fallback_reason": f"{type(exc).__name__}: {exc}",
                    "model_name": self.model_name,
                    "llm_enabled": True,
                    "llm_latency_ms": latency_ms,
                }
            )
            return fallback_decision

    def _ensure_backend(self) -> Optional[TextGenerationBackend]:
        if self.backend is not None:
            return self.backend
        self.backend = self._build_default_backend(self.model_name)
        return self.backend

    def _build_default_backend(self, model_name: str) -> Optional[TextGenerationBackend]:
        try:
            self._backend_error = None
            return HFRouterOpenAIBackend(model_name=model_name)
        except Exception as exc:
            self._backend_error = f"{type(exc).__name__}: {exc}"
            return None

    def _build_prompt(self, observation: Mapping[str, Any]) -> str:
        compact_observation = {
            "timestep": observation["timestep"],
            "grid_size": observation["grid_size"],
            "drone_positions": observation["drone_positions"],
            "visible_active_events": observation.get("visible_active_events", []),
            "team_frontier_cells": observation.get("team_frontier_cells", [])[:20],
            "recent_team_coverage_ratio": observation.get("recent_team_coverage_ratio", 0.0),
            "known_detected_events": observation.get("known_detected_events", [])[-10:],
        }
        return (
            "You are a disaster-response drone coordinator.\n"
            "Assign one target grid cell to each drone.\n"
            "Prioritize HIGH severity events, then MEDIUM, then LOW.\n"
            "Avoid assigning the same target to multiple drones unless necessary.\n"
            "Return JSON only with the shape:\n"
            '{"drone_1": [x, y], "drone_2": [x, y], "drone_3": [x, y]}\n'
            f"Observation:\n{json.dumps(compact_observation, separators=(',', ':'))}\n"
        )

    def _parse_targets(
        self,
        raw_text: str,
        observation: Mapping[str, Any],
    ) -> Dict[str, Coord]:
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("LLM response did not contain a JSON object.")

        parsed = json.loads(raw_text[start : end + 1])
        drone_positions = observation["drone_positions"]
        grid_size = int(observation.get("grid_size", 10))
        targets: Dict[str, Coord] = {}
        for drone_id in drone_positions:
            if drone_id not in parsed:
                raise ValueError(f"LLM response missing target for {drone_id}.")
            value = parsed[drone_id]
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError(f"Invalid target format for {drone_id}: {value}.")
            x = max(0, min(grid_size - 1, int(value[0])))
            y = max(0, min(grid_size - 1, int(value[1])))
            targets[drone_id] = (x, y)
        return targets
