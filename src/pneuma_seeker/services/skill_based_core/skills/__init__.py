"""Auto-discovery of skill groups.

A skill group is a sub-package that contains:
  - SKILLS.md  — LLM-facing documentation for the skills in this group
  - impl.py    — one or more SkillBase subclasses

Adding a new skill group is just creating a new directory with those two files.
"""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import TYPE_CHECKING

from pneuma_seeker.services.skill_based_core.skills.base import SkillBase, SkillResult

if TYPE_CHECKING:
    from pneuma_seeker.shared.config import Config

_PACKAGE_PREFIX = "pneuma_seeker.services.skill_based_core.skills"


def discover_skills(config: Config) -> tuple[dict[str, SkillBase], str]:
    """Scan skills/ sub-packages, collect enabled skills and assemble the prompt.

    Returns:
        registry   — name → SkillBase instance for all enabled skills
        skills_doc — concatenated SKILLS.md text for the LLM system prompt
    """
    skills_dir = Path(__file__).parent
    registry: dict[str, SkillBase] = {}
    prompt_sections: list[str] = []

    for group_dir in sorted(skills_dir.iterdir()):
        if not group_dir.is_dir() or group_dir.name.startswith("_"):
            continue
        skills_md_path = group_dir / "SKILLS.md"
        impl_path = group_dir / "impl.py"
        if not skills_md_path.exists() or not impl_path.exists():
            continue

        prompt_sections.append(skills_md_path.read_text().strip())

        module = importlib.import_module(f"{_PACKAGE_PREFIX}.{group_dir.name}.impl")

        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (
                isinstance(attr, type)
                and issubclass(attr, SkillBase)
                and attr is not SkillBase
                and getattr(attr, "name", "")
            ):
                instance = attr()
                flag = instance.config_flag
                if flag is None or getattr(config, flag, False):
                    registry[instance.name] = instance

    return registry, "\n\n---\n\n".join(prompt_sections)


__all__ = ["SkillBase", "SkillResult", "discover_skills"]
