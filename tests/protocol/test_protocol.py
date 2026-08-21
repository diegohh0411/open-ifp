from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

EXPECTED_RUNTIME = {
    "mlx": "0.31.2",
    "mlx-lm": "0.31.3",
    "huggingface_hub": "1.27.0",
}
EXPECTED_DEV = {
    "PyYAML": "6.0.3",
    "jsonschema": "4.26.0",
    "pytest": "9.1.1",
}
FORBIDDEN = {"torch", "torchvision", "torchtune"}


def read_pins(path: Path) -> dict[str, str]:
    pins: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        assert "==" in line, f"direct dependency is not equality-pinned: {line}"
        name, version = line.split("==", maxsplit=1)
        pins[name] = version
    return pins


def test_runtime_dependencies_are_exact() -> None:
    assert read_pins(ROOT / "requirements.txt") == EXPECTED_RUNTIME


def test_dev_dependencies_are_exact() -> None:
    assert read_pins(ROOT / "requirements-dev.txt") == EXPECTED_DEV


def test_forbidden_runtime_dependencies_are_absent() -> None:
    names = {name.lower() for name in read_pins(ROOT / "requirements.txt")}
    assert names.isdisjoint(FORBIDDEN)


def test_setup_installs_runtime_and_dev_requirements() -> None:
    setup = (ROOT / "setup.sh").read_text(encoding="utf-8")
    assert "python -m pip install -r requirements.txt -r requirements-dev.txt" in setup


def test_generation_resolves_the_immutable_cached_snapshot() -> None:
    generate = (ROOT / "scripts/generate.py").read_text(encoding="utf-8")
    assert 'DEFAULT_MODEL_REVISION = "c540970f9e29518b1d8f06ab8b24cba66ad77b6d"' in generate
    assert "revision=args.revision" in generate
    assert "local_files_only=True" in generate
    assert "load(snapshot_path)" in generate
