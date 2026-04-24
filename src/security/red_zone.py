"""Red Zone System — prevents interaction with protected system areas."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class RedZoneAction(StrEnum):
    BLOCK = "BLOCK"
    GATE = "GATE"
    ALERT = "ALERT"


@dataclass
class RedZone:
    name: str
    action: RedZoneAction


_DEFAULT_RED_ZONES: list[RedZone] = [
    RedZone("System Settings", RedZoneAction.BLOCK),
    RedZone("Package Manager", RedZoneAction.BLOCK),
    RedZone("Network Configuration", RedZoneAction.BLOCK),
    RedZone("Authentication prompts", RedZoneAction.BLOCK),
]


class RedZoneSystem:
    """Prevents actions on protected system zones."""

    def __init__(self, zones: list[dict] | None = None) -> None:
        self._zones: list[RedZone] = list(_DEFAULT_RED_ZONES)
        if zones:
            for z in zones:
                action_val = z.get("action", "BLOCK").upper()
                try:
                    action = RedZoneAction(action_val)
                except ValueError:
                    action = RedZoneAction.BLOCK
                self._zones.append(RedZone(name=z["name"], action=action))

    def check(self, element_label: str, app: str = "") -> RedZoneAction | None:
        """Check if an element is in a red zone. Returns action or None."""
        label_lower = element_label.lower()
        app_lower = app.lower()

        for zone in self._zones:
            zone_lower = zone.name.lower()
            if zone_lower in label_lower or zone_lower in app_lower:
                return zone.action
        return None

    def filter_clickable_zones(self, zones: list[dict]) -> list[dict]:
        """Filter out red-zone elements from a list of clickable UI zones."""
        safe = []
        for zone in zones:
            label = zone.get("label", "")
            app = zone.get("app", "")
            action = self.check(label, app)
            if action is None:
                safe.append(zone)
        return safe
