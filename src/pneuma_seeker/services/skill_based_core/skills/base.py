from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pneuma_seeker.services.skill_based_core.agent import SkillsAgent


@dataclass
class SkillResult:
    """Returned by every skill's execute() method."""
    content: str          # sent back to the LLM as the skill's output
    terminates: bool = False  # True only for skills that end the turn (i.e. respond)


class SkillBase:
    """Base class for all skills.

    Each concrete subclass must set `name` as a class attribute and implement `execute`.
    The agent discovers subclasses at startup by scanning the skills/ directory.
    """
    name: str = ""
    config_flag: str | None = None  # Config attribute that must be True to enable this skill

    def execute(self, args: dict, agent: SkillsAgent) -> SkillResult:
        raise NotImplementedError(f"Skill '{self.name}' must implement execute()")
