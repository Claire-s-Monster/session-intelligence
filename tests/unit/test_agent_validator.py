"""Unit tests for core/agent_validator.py.

All tests use tmp_path to create isolated fake agent directories and are
completely isolated from the real ~/.claude/agents/ directory tree.
"""

import logging
from pathlib import Path

import pytest

from core.agent_validator import (
    AgentNotFoundError,
    AgentValidator,
    AmbiguousAgentNameError,
    ValidatedAgent,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_agent(base: Path, subdir: str, stem: str, frontmatter: str = "") -> Path:
    """Write a fake agent .md file and return its path."""
    target_dir = base / subdir
    target_dir.mkdir(parents=True, exist_ok=True)
    if frontmatter:
        content = f"---\n{frontmatter}\n---\n# Agent\nBody text.\n"
    else:
        content = "# Agent\nBody text.\n"
    path = target_dir / f"{stem}.md"
    path.write_text(content, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# 1. Basic resolution
# ---------------------------------------------------------------------------


class TestValidatorFindsAgentInSubdir:
    def test_finds_domain_agent(self, tmp_path: Path) -> None:
        _write_agent(
            tmp_path,
            "domain",
            "foo-agent",
            "name: foo-agent\ndescription: Foo bar\nmodel: sonnet",
        )
        validator = AgentValidator(agents_root=tmp_path, mode="strict")
        result = validator.validate("foo-agent")

        assert isinstance(result, ValidatedAgent)
        assert result.name == "foo-agent"
        assert result.agent_type == "domain"
        assert result.description == "Foo bar"


# ---------------------------------------------------------------------------
# 2. Strict mode — unknown agent
# ---------------------------------------------------------------------------


class TestValidatorStrictRaisesOnUnknown:
    def test_raises_agent_not_found(self, tmp_path: Path) -> None:
        _write_agent(tmp_path, "domain", "real-agent", "name: real-agent\ndescription: Real")
        validator = AgentValidator(agents_root=tmp_path, mode="strict")

        with pytest.raises(AgentNotFoundError) as exc_info:
            validator.validate("nonexistent")

        msg = str(exc_info.value)
        assert "Closest matches" in msg
        # At least one known agent name should appear
        assert "real-agent" in msg or len(validator.list_known_agents()) > 0


# ---------------------------------------------------------------------------
# 3. Lenient mode — unknown agent
# ---------------------------------------------------------------------------


class TestValidatorLenientReturnsNone:
    def test_returns_none_and_logs_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        _write_agent(tmp_path, "domain", "some-agent", "name: some-agent")
        validator = AgentValidator(agents_root=tmp_path, mode="lenient")

        with caplog.at_level(logging.WARNING, logger="core.agent_validator"):
            result = validator.validate("nonexistent-agent")

        assert result is None
        assert any("not found" in record.message.lower() for record in caplog.records)


# ---------------------------------------------------------------------------
# 4. Off mode
# ---------------------------------------------------------------------------


class TestValidatorOffReturnsNoneAlways:
    def test_returns_none_for_known_agent(self, tmp_path: Path) -> None:
        _write_agent(tmp_path, "domain", "known-agent", "name: known-agent")
        validator = AgentValidator(agents_root=tmp_path, mode="off")
        assert validator.validate("known-agent") is None

    def test_returns_none_for_unknown_agent(self, tmp_path: Path) -> None:
        _write_agent(tmp_path, "domain", "known-agent", "name: known-agent")
        validator = AgentValidator(agents_root=tmp_path, mode="off")
        assert validator.validate("totally-unknown") is None


# ---------------------------------------------------------------------------
# 5. Ambiguous name
# ---------------------------------------------------------------------------


class TestValidatorAmbiguousNameRaises:
    def test_raises_ambiguous_error(self, tmp_path: Path) -> None:
        _write_agent(tmp_path, "domain", "dup", "name: dup")
        _write_agent(tmp_path, "meta", "dup", "name: dup")
        validator = AgentValidator(agents_root=tmp_path, mode="strict")

        with pytest.raises(AmbiguousAgentNameError) as exc_info:
            validator.validate("dup")

        msg = str(exc_info.value)
        # Both paths should appear in the error
        assert "domain" in msg
        assert "meta" in msg


# ---------------------------------------------------------------------------
# 6. Missing root — auto-degrades to off
# ---------------------------------------------------------------------------


class TestValidatorHandlesMissingRoot:
    def test_degrades_to_off_when_root_missing(self, tmp_path: Path) -> None:
        nonexistent = tmp_path / "does_not_exist"
        validator = AgentValidator(agents_root=nonexistent, mode="strict")
        # After degradation, validate returns None instead of raising
        result = validator.validate("any-agent")
        assert result is None


# ---------------------------------------------------------------------------
# 7. No frontmatter
# ---------------------------------------------------------------------------


class TestValidatorHandlesNoFrontmatter:
    def test_empty_frontmatter(self, tmp_path: Path) -> None:
        target = tmp_path / "domain"
        target.mkdir()
        (target / "bare-agent.md").write_text("# Agent\nNo frontmatter here.\n", encoding="utf-8")

        validator = AgentValidator(agents_root=tmp_path, mode="strict")
        result = validator.validate("bare-agent")

        assert isinstance(result, ValidatedAgent)
        assert result.raw_frontmatter == {}
        assert result.description == ""


# ---------------------------------------------------------------------------
# 8. Malformed YAML
# ---------------------------------------------------------------------------


class TestValidatorHandlesMalformedYaml:
    def test_malformed_yaml_gives_empty_frontmatter(self, tmp_path: Path) -> None:
        target = tmp_path / "domain"
        target.mkdir()
        (target / "broken-agent.md").write_text(
            "---\nname: [unclosed\n---\n# Body\n", encoding="utf-8"
        )

        validator = AgentValidator(agents_root=tmp_path, mode="strict")
        result = validator.validate("broken-agent")

        assert isinstance(result, ValidatedAgent)
        assert result.raw_frontmatter == {}
        assert result.description == ""


# ---------------------------------------------------------------------------
# 9. Long description truncation
# ---------------------------------------------------------------------------


class TestValidatorTruncatesLongDescription:
    def test_description_truncated_at_500(self, tmp_path: Path) -> None:
        long_desc = "x" * 600
        frontmatter = f"name: verbose-agent\ndescription: {long_desc}"
        _write_agent(tmp_path, "domain", "verbose-agent", frontmatter)
        validator = AgentValidator(agents_root=tmp_path, mode="strict")
        result = validator.validate("verbose-agent")

        assert isinstance(result, ValidatedAgent)
        assert len(result.description) <= 503  # 500 chars + up to 3-char ellipsis
        assert result.description.endswith("…") or result.description.endswith("...")


# ---------------------------------------------------------------------------
# 10. Multi-line description collapsed
# ---------------------------------------------------------------------------


class TestValidatorCollapsesMultilineDescription:
    def test_block_scalar_description_collapsed(self, tmp_path: Path) -> None:
        frontmatter = (
            "name: multi-agent\ndescription: |\n"
            "  First line.\n  Second line.\n  Third line.\n"
        )
        _write_agent(tmp_path, "domain", "multi-agent", frontmatter)
        validator = AgentValidator(agents_root=tmp_path, mode="strict")
        result = validator.validate("multi-agent")

        assert isinstance(result, ValidatedAgent)
        assert "\n" not in result.description
        # Consecutive whitespace should be collapsed
        assert "  " not in result.description


# ---------------------------------------------------------------------------
# 11. find_closest top-N
# ---------------------------------------------------------------------------


class TestFindClosestReturnsTopN:
    def test_returns_at_most_n_matches(self, tmp_path: Path) -> None:
        for i in range(5):
            _write_agent(tmp_path, "domain", f"foobar-agent-{i}", f"name: foobar-agent-{i}")
        validator = AgentValidator(agents_root=tmp_path, mode="strict")
        closest = validator.find_closest("foobar", n=3)
        assert len(closest) <= 3


# ---------------------------------------------------------------------------
# 12. Env var controls mode
# ---------------------------------------------------------------------------


class TestEnvVarControlsMode:
    def test_env_off_prevents_raise(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _write_agent(tmp_path, "domain", "real-agent", "name: real-agent")
        monkeypatch.setenv("SESSION_INTELLIGENCE_AGENT_VALIDATION", "off")
        monkeypatch.setenv("SESSION_INTELLIGENCE_AGENTS_DIR", str(tmp_path))
        # Instantiate with no explicit args — should pick up env vars
        validator = AgentValidator()
        result = validator.validate("nonexistent-agent")
        assert result is None


# ---------------------------------------------------------------------------
# 13. Env var controls agents dir
# ---------------------------------------------------------------------------


class TestEnvVarControlsAgentsDir:
    def test_env_dir_picked_up(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _write_agent(tmp_path, "domain", "env-agent", "name: env-agent")
        monkeypatch.setenv("SESSION_INTELLIGENCE_AGENTS_DIR", str(tmp_path))
        monkeypatch.setenv("SESSION_INTELLIGENCE_AGENT_VALIDATION", "strict")
        validator = AgentValidator()
        known = validator.list_known_agents()
        assert "env-agent" in known
