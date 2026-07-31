"""Skill registry."""

from __future__ import annotations

from .config import SKILLS_DIR
from .context import ctx
from .utils import parse_frontmatter

# Module-level alias so ``from mcodecore.skills import SKILL_REGISTRY`` works.
SKILL_REGISTRY = ctx.skill_registry


def _scan_skills() -> None:
    """Scan the ``skills/`` directory for all ``SKILL.md`` files and populate the registry."""
    if not SKILLS_DIR.exists():
        return
    for d in sorted(SKILLS_DIR.iterdir()):
        if not d.is_dir():
            continue
        manifest = d / "SKILL.md"
        if manifest.exists():
            raw = manifest.read_text(encoding="utf-8")
            meta, body = parse_frontmatter(raw)
            name = meta.get("name", d.name)
            desc = meta.get("description", raw.split("\n")[0].lstrip("#").strip())
            ctx.skill_registry[name] = {"name": name, "description": desc, "content": raw}


def list_skills() -> str:
    """Return a Markdown list of available skills."""
    if not ctx.skill_registry:
        return "(no skills found)"
    return "\n".join(f"- **{s['name']}**: {s['description']}"
                     for s in ctx.skill_registry.values())


def load_skill(name: str) -> str:
    """Load and return the full text of a skill."""
    skill = ctx.skill_registry.get(name)
    if not skill:
        return f"Skill not found: {name}"
    return skill["content"]


# Scan once at import time.
_scan_skills()
