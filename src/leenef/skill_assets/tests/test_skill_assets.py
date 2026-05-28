"""Guards for the bundled method-selection skill assets.

The skill (``SKILL.md`` plus reference docs, source, benchmarks, tests, colab
notebooks, and base-config defaults) is shipped inside the package as
``leenef/skill_assets`` so ``pip install leenef`` users get it.  These tests
ensure the packaged copies stay byte-identical to the canonical repository
sources, that the manifest covers the expected ground, that the installer
materializes a usable skill directory, and that the dropped plain-language
reference is not reintroduced.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from leenef.install_skill import (
    SKILL_NAME,
    _packaged_assets_dir,
    _repo_root,
    install,
    iter_canonical_paths,
)

REPO_ROOT = _repo_root()
requires_repo = pytest.mark.skipif(REPO_ROOT is None, reason="requires a repository checkout")


def test_packaged_assets_exist():
    assets = _packaged_assets_dir()
    assert (assets / "SKILL.md").is_file()
    # Sanity: representative files from each tree are present.
    for rel in (
        "docs/USAGE.md",
        "docs/SSUP-I.md",
        "docs/SSUP-II.md",
        "docs/SSUP-III.md",
        "README.md",
        "AGENTS.md",
        "LICENSE",
        "pyproject.toml",
        "src/leenef/__init__.py",
        "src/leenef/rl.py",
        "benchmarks/run_rl_paper_dense.py",
        "tests/test_rl.py",
        "results/rl_clean/base_config.json",
    ):
        assert (assets / rel).is_file(), f"missing bundled asset {rel}"


@requires_repo
def test_bundled_assets_match_canonical():
    assets = _packaged_assets_dir()
    stale: list[str] = []
    missing: list[str] = []
    for rel in iter_canonical_paths(REPO_ROOT):
        bundled = assets / rel
        if not bundled.is_file():
            missing.append(rel)
            continue
        if (REPO_ROOT / rel).read_bytes() != bundled.read_bytes():
            stale.append(rel)
    assert not missing, (
        "Missing bundled assets (run `leenef-install-skill --sync-assets`): " + ", ".join(missing)
    )
    assert not stale, (
        "Stale bundled assets (run `leenef-install-skill --sync-assets`): " + ", ".join(stale)
    )


@requires_repo
def test_skill_assets_dir_has_no_extras():
    """Bundled tree must contain only files produced by ``iter_canonical_paths``."""
    from leenef.install_skill import _SKIP_DIRS

    expected = set(iter_canonical_paths(REPO_ROOT))
    assets = _packaged_assets_dir()
    actual: set[str] = set()
    for path in assets.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(assets)
        # Ignore caches that tools may drop into the bundled tree at runtime
        # (e.g. ``.ruff_cache`` next to the mirrored ``pyproject.toml``).
        if any(part in _SKIP_DIRS for part in rel.parts):
            continue
        rel_str = rel.as_posix()
        if rel_str == ".asset-count":
            continue
        actual.add(rel_str)
    extras = sorted(actual - expected)
    assert not extras, f"unexpected bundled files (regenerate with --sync-assets): {extras}"


def test_skill_references_bundled_docs():
    skill = (_packaged_assets_dir() / "SKILL.md").read_text()
    for doc in ("docs/USAGE.md", "docs/SSUP-I.md", "docs/SSUP-II.md", "docs/SSUP-III.md"):
        assert doc in skill, f"SKILL.md should reference {doc}"


def test_plain_language_reference_dropped():
    skill = (_packaged_assets_dir() / "SKILL.md").read_text()
    assert "SSUP-plain-language" not in skill
    assert not (_packaged_assets_dir() / "docs" / "SSUP-plain-language.md").exists()


def test_install_copies_skill(tmp_path: Path):
    target = install(tmp_path, force=True)
    assert target == tmp_path / SKILL_NAME
    assert (target / "SKILL.md").is_file()
    assert (target / "docs" / "USAGE.md").is_file()
    assert (target / "docs" / "SSUP-III.md").is_file()
    assert (target / "src" / "leenef" / "rl.py").is_file()
    assert (target / "benchmarks" / "run_rl_paper_dense.py").is_file()
    assert (target / "results" / "rl_clean" / "base_config.json").is_file()


def test_install_requires_force_when_exists(tmp_path: Path):
    install(tmp_path)
    with pytest.raises(FileExistsError):
        install(tmp_path)
    install(tmp_path, force=True)
