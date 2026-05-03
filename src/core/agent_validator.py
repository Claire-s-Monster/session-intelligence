"""Validate agent names against real agent files under ~/.claude/agents and extract metadata."""

import difflib
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_AGENTS_ROOT_ENV = "SESSION_INTELLIGENCE_AGENTS_DIR"
_MODE_ENV = "SESSION_INTELLIGENCE_AGENT_VALIDATION"
_VALID_MODES = {"strict", "lenient", "off"}
_DEFAULT_AGENTS_ROOT = Path("~/.claude/agents").expanduser()
_DESCRIPTION_MAX_LEN = 500


class AgentNotFoundError(ValueError):
    """Raised when an agent name has no matching file and mode is strict."""


class AmbiguousAgentNameError(ValueError):
    """Raised when an agent name matches multiple files under different subdirs."""


@dataclass(frozen=True)
class ValidatedAgent:
    name: str
    agent_type: str
    description: str
    file_path: Path
    raw_frontmatter: dict[str, Any]


def _parse_frontmatter(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return {}

    if not text.startswith("---"):
        return {}

    end = text.find("---", 3)
    if end == -1:
        return {}

    try:
        data = yaml.safe_load(text[3:end])
        return data if isinstance(data, dict) else {}
    except yaml.YAMLError:
        return {}


def _normalize_description(raw: str) -> str:
    stripped = raw.strip()
    collapsed = re.sub(r"\s+", " ", stripped)
    if len(collapsed) > _DESCRIPTION_MAX_LEN:
        return collapsed[:_DESCRIPTION_MAX_LEN] + "…"
    return collapsed


class AgentValidator:
    def __init__(
        self,
        agents_root: Path | None = None,
        mode: str | None = None,
    ) -> None:
        self._agents_root = self._resolve_root(agents_root)
        self._mode = self._resolve_mode(mode)
        self._cache: dict[str, list[Path]] = {}

        if self._mode != "off":
            if self._agents_root is None or not self._agents_root.exists():
                logger.warning(
                    "agents_root %r does not exist — agent validation disabled",
                    self._agents_root,
                )
                self._mode = "off"
            else:
                self._cache = self._build_cache()

    @staticmethod
    def _resolve_root(agents_root: Path | None) -> Path:
        if agents_root is not None:
            return agents_root
        env_val = os.environ.get(_AGENTS_ROOT_ENV)
        if env_val:
            return Path(env_val)
        return _DEFAULT_AGENTS_ROOT

    @staticmethod
    def _resolve_mode(mode: str | None) -> str:
        if mode is not None:
            if mode not in _VALID_MODES:
                raise ValueError(f"mode must be one of {_VALID_MODES}, got {mode!r}")
            return mode
        env_val = os.environ.get(_MODE_ENV, "").lower()
        if env_val in _VALID_MODES:
            return env_val
        return "strict"

    def _build_cache(self) -> dict[str, list[Path]]:
        cache: dict[str, list[Path]] = {}
        for md_file in self._agents_root.glob("**/*.md"):
            name = md_file.stem
            cache.setdefault(name, []).append(md_file)
        return cache

    def validate(self, agent_name: str) -> ValidatedAgent | None:
        if self._mode == "off":
            return None

        matches = self._cache.get(agent_name, [])

        if len(matches) > 1:
            paths_str = ", ".join(str(p) for p in matches)
            raise AmbiguousAgentNameError(
                f"Multiple agent files named {agent_name!r}: {paths_str}"
            )

        if len(matches) == 1:
            path = matches[0]
            frontmatter = _parse_frontmatter(path)
            raw_desc = frontmatter.get("description", "") or ""
            description = _normalize_description(str(raw_desc))
            agent_type = path.parent.name
            return ValidatedAgent(
                name=agent_name,
                agent_type=agent_type,
                description=description,
                file_path=path,
                raw_frontmatter=frontmatter,
            )

        # No match
        closest = self.find_closest(agent_name)
        if self._mode == "strict":
            raise AgentNotFoundError(
                f"Agent {agent_name!r} not found in {self._agents_root}. "
                f"Closest matches: {closest}"
            )
        logger.warning(
            "Agent %r not found in %s (closest: %s)",
            agent_name,
            self._agents_root,
            closest,
        )
        return None

    def find_closest(self, agent_name: str, n: int = 5) -> list[str]:
        return difflib.get_close_matches(agent_name, self.list_known_agents(), n=n, cutoff=0.5)

    def list_known_agents(self) -> list[str]:
        return list(self._cache.keys())
