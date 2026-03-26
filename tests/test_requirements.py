"""
Validate that every package version spec in requirements.txt is satisfiable
by versions actually published on PyPI.

Catches the "Oh no" crash caused by pinning non-existent versions
(e.g. streamlit==1.43.0) before the code ever reaches Streamlit Cloud.
"""

import json
import os
import re
import urllib.request
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.version import Version


REQUIREMENTS_PATH = Path(__file__).parent.parent / "requirements.txt"


def _parse_requirements(path: Path) -> list[Requirement]:
    reqs = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        reqs.append(Requirement(line))
    return reqs


def _pypi_versions(package_name: str) -> list[Version]:
    """Fetch all published versions for a package from PyPI JSON API."""
    url = f"https://pypi.org/pypi/{package_name}/json"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read())
        return [Version(v) for v in data["releases"].keys()]
    except Exception as exc:
        pytest.skip(f"PyPI unreachable for {package_name}: {exc}")
        return []


def _any_version_satisfies(req: Requirement, versions: list[Version]) -> bool:
    return any(v in req.specifier for v in versions)


@pytest.mark.parametrize("req", _parse_requirements(REQUIREMENTS_PATH), ids=lambda r: r.name)
def test_package_version_exists_on_pypi(req):
    """Each package spec must be satisfiable by at least one version on PyPI."""
    versions = _pypi_versions(req.name)
    if not versions:
        pytest.skip(f"No versions found for {req.name}")
    assert _any_version_satisfies(req, versions), (
        f"No PyPI release of '{req.name}' satisfies '{req.specifier}'. "
        f"Available: {sorted(versions)[-5:]}"
    )
