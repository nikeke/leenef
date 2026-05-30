"""Guards for the manifest-driven leenef Agent Skill."""

from __future__ import annotations

from pathlib import Path

import pytest

from leenef.install_skill import (
    SKILL_NAME,
    _repo_root,
    install,
    installed_relative_path,
    iter_manifest_paths,
    materialize_assets,
)

REPO_ROOT = _repo_root()
requires_repo = pytest.mark.skipif(REPO_ROOT is None, reason="requires a repository checkout")


@requires_repo
def test_skill_manifest_selects_useful_assets():
    assets = set(iter_manifest_paths(REPO_ROOT))
    expected = {
        "skills/leenef/SKILL.md",
        "skills/leenef/assets.txt",
        "docs/usage.md",
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
        "results/rl_clean/base_config.json",
    }
    assert expected <= assets


@requires_repo
def test_skill_manifest_excludes_unneeded_assets():
    assets = set(iter_manifest_paths(REPO_ROOT))
    assert "docs/SSUP-plain-language.md" not in assets
    assert not any(path.startswith("tests/") for path in assets)
    assert not any(path.startswith("colab/") for path in assets)
    assert not any(path.startswith("src/leenef/skill_assets/") for path in assets)


@requires_repo
def test_skill_references_manifest_assets():
    skill = (REPO_ROOT / "skills" / "leenef" / "SKILL.md").read_text()
    for doc in ("docs/usage.md", "docs/SSUP-I.md", "docs/SSUP-II.md", "docs/SSUP-III.md"):
        assert doc in skill, f"SKILL.md should reference {doc}"
    assert "SSUP-plain-language" not in skill
    assert "tests/" not in skill
    assert "colab/" not in skill


@requires_repo
def test_materialize_assets_can_preserve_repo_layout(tmp_path: Path):
    count = materialize_assets(REPO_ROOT, tmp_path, install_layout=False)
    assert count == len(iter_manifest_paths(REPO_ROOT))
    assert (tmp_path / "skills" / "leenef" / "SKILL.md").is_file()
    assert (tmp_path / "skills" / "leenef" / "assets.txt").is_file()
    assert (tmp_path / "docs" / "usage.md").is_file()
    assert not (tmp_path / "tests").exists()
    assert not (tmp_path / "colab").exists()


@requires_repo
def test_installed_relative_path_strips_source_skill_prefix():
    assert installed_relative_path("skills/leenef/SKILL.md") == Path("SKILL.md")
    assert installed_relative_path("skills/leenef/assets.txt") == Path("assets.txt")
    assert installed_relative_path("docs/usage.md") == Path("docs/usage.md")


def test_install_copies_skill(tmp_path: Path):
    target = install(tmp_path, force=True)
    assert target == tmp_path / SKILL_NAME
    assert (target / "SKILL.md").is_file()
    assert (target / "assets.txt").is_file()
    assert (target / "docs" / "usage.md").is_file()
    assert (target / "docs" / "SSUP-III.md").is_file()
    assert (target / "src" / "leenef" / "rl.py").is_file()
    assert (target / "benchmarks" / "run_rl_paper_dense.py").is_file()
    assert (target / "results" / "rl_clean" / "base_config.json").is_file()
    assert (target / "LICENSE").is_file()
    assert not (target / "skills").exists()
    assert not (target / "tests").exists()
    assert not (target / "colab").exists()


def test_install_requires_force_when_exists(tmp_path: Path):
    install(tmp_path)
    with pytest.raises(FileExistsError):
        install(tmp_path)
    install(tmp_path, force=True)
