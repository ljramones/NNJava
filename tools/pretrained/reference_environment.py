"""Validate reference-tool versions against the installable pins rather than a second version list."""
from importlib.metadata import version
from pathlib import Path


def validate_versions():
    expected = {}
    for line in Path(__file__).with_name("reference-requirements.txt").read_text().splitlines():
        line = line.partition("#")[0].strip()
        if not line:
            continue
        parts = line.split("==")
        if len(parts) != 2 or not all(parts):
            raise ValueError(f"Reference requirements must have exact version pins: {line}")
        expected[parts[0]] = parts[1]
    versions = {name: version(name) for name in expected}
    for name, pinned in expected.items():
        actual = versions[name]
        # Official Linux CPU wheels add a local build identifier; record it in provenance.
        if actual != pinned and not (name == "torch" and actual == pinned + "+cpu"):
            raise ValueError(f"Reference version differs from requirements: {name}=={actual}; expected {pinned}")
    return versions
