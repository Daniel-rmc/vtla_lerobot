"""Feature mapping utilities for LeRobot-to-VTLA batches."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class FeatureMap:
    """Names of LeRobot columns consumed by the VTLA training batch."""

    state: str = "observation.state"
    action: str = "action"
    cameras: dict[str, str] = field(default_factory=dict)
    tactile: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> "FeatureMap":
        if raw is None:
            return cls()
        cameras = dict(raw.get("cameras", {}) or {})
        tactile = dict(raw.get("tactile", {}) or {})
        return cls(
            state=str(raw.get("state", cls.state)),
            action=str(raw.get("action", cls.action)),
            cameras={str(k): str(v) for k, v in cameras.items()},
            tactile={str(k): str(v) for k, v in tactile.items()},
        )

    def merged(self, override: Mapping[str, Any] | None) -> "FeatureMap":
        if not override:
            return self
        raw = dict(override)
        cameras = dict(self.cameras)
        cameras.update({str(k): str(v) for k, v in (raw.get("cameras", {}) or {}).items()})
        tactile = dict(self.tactile)
        tactile.update({str(k): str(v) for k, v in (raw.get("tactile", {}) or {}).items()})
        return FeatureMap(
            state=str(raw.get("state", self.state)),
            action=str(raw.get("action", self.action)),
            cameras=cameras,
            tactile=tactile,
        )

    @property
    def image_columns(self) -> list[str]:
        return list(self.cameras.values()) + list(self.tactile.values())

    def require_training_features(self) -> None:
        if not self.state:
            raise ValueError("feature_map.state must be set")
        if not self.action:
            raise ValueError("feature_map.action must be set")
        if not self.cameras:
            raise ValueError("feature_map.cameras must contain at least one camera")
        if not self.tactile:
            raise ValueError("feature_map.tactile must contain at least one tactile image")
