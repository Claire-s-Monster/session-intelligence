"""
Regression tests for the strict type gate: `quality` must depend on the
real `type-check` task (mypy src/), not a `|| echo`-swallowed lenient
variant that would let mypy regressions pass CI silently.
"""

import tomllib
from pathlib import Path

import pytest

pytestmark = pytest.mark.regression

PYPROJECT_PATH = Path(__file__).resolve().parents[2] / "pyproject.toml"


def _load_pyproject() -> dict:
    with PYPROJECT_PATH.open("rb") as fh:
        return tomllib.load(fh)


def _iter_pixi_tasks(config: dict):
    """Yield (task_name, task_value) from [tool.pixi.tasks] and any
    [tool.pixi.feature.*.tasks] sections."""
    pixi = config.get("tool", {}).get("pixi", {})
    yield from pixi.get("tasks", {}).items()
    for feature in pixi.get("feature", {}).values():
        yield from feature.get("tasks", {}).items()


def _task_cmd(task_value) -> str:
    """Normalize a pixi task definition to its command string, if any."""
    if isinstance(task_value, str):
        return task_value
    if isinstance(task_value, dict):
        return str(task_value.get("cmd", ""))
    return ""


class TestQualityDependsOnStrictTypeCheck:
    def test_quality_depends_on_type_check(self):
        config = _load_pyproject()
        quality = config["tool"]["pixi"]["tasks"]["quality"]
        depends_on = quality["depends-on"]
        assert "type-check" in depends_on, (
            "quality must depend on the strict 'type-check' task (mypy src/) "
            "so a mypy failure fails CI instead of being silently accepted."
        )

    def test_quality_does_not_depend_on_type_check_lenient(self):
        config = _load_pyproject()
        quality = config["tool"]["pixi"]["tasks"]["quality"]
        depends_on = quality["depends-on"]
        assert "type-check-lenient" not in depends_on, (
            "type-check-lenient wrapped mypy in `|| echo`, turning a real "
            "mypy failure into an exit-0 warning that CI never saw. It must "
            "not be reintroduced as a dependency of quality."
        )

    def test_type_check_task_is_exactly_mypy_src(self):
        config = _load_pyproject()
        type_check = config["tool"]["pixi"]["tasks"]["type-check"]
        cmd = _task_cmd(type_check)
        assert cmd == "mypy src/", (
            "type-check must run bare 'mypy src/' with no fallback/suppression "
            f"suffix (e.g. '|| echo ...'); got: {cmd!r}"
        )


class TestNoSilentlySwallowedTaskFailures:
    def test_no_pixi_task_swallows_its_own_failure(self):
        """A `cmd || echo ...` pattern makes a task exit 0 even when the
        underlying tool fails, which is exactly how type-check-lenient made
        mypy regressions invisible to CI. No pixi task should do this."""
        config = _load_pyproject()
        offenders = [
            name for name, value in _iter_pixi_tasks(config) if "|| echo" in _task_cmd(value)
        ]
        assert not offenders, (
            "These pixi tasks swallow failures via '|| echo', which lets a "
            f"real failure exit 0 and pass CI silently: {offenders}"
        )
