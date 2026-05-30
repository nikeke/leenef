"""Guards for the manifest-driven leenef Agent Skill."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from leenef._skill_cli import load_installer

REPO_ROOT = Path(__file__).resolve().parents[1]
INSTALLER_PATH = REPO_ROOT / "skills" / "leenef" / "install_skill.py"


def _load_installer():
    spec = importlib.util.spec_from_file_location("_test_leenef_skill_installer", INSTALLER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


install_skill = _load_installer()


def test_skill_manifest_selects_useful_assets():
    assets = set(install_skill.iter_manifest_paths(REPO_ROOT))
    expected = {
        "skills/leenef/SKILL.md",
        "skills/leenef/assets.txt",
        "skills/leenef/install_skill.py",
        "docs/usage.md",
        "docs/SSUP-I.md",
        "docs/SSUP-II.md",
        "docs/SSUP-III.md",
        "README.md",
        "AGENTS.md",
        "LICENSE",
        "pyproject.toml",
        ".github/workflows/publish-pypi.yml",
        "scripts/build_dist.sh",
        "src/leenef/__init__.py",
        "src/leenef/rl.py",
        "benchmarks/run_rl_paper_dense.py",
        "results/rl_clean/base_config.json",
        "results/rl_sparse/baseline_mc_fqi.json",
        "results/rl_sparse/logs/baseline_mc_fqi.log",
        "results/rl_sparse/p2_supervisor.sh",
    }
    assert expected <= assets


def test_skill_manifest_excludes_unneeded_assets():
    assets = set(install_skill.iter_manifest_paths(REPO_ROOT))
    assert "docs/SSUP-plain-language.md" not in assets
    assert not any(path.startswith("tests/") for path in assets)
    assert not any(path.startswith("colab/") for path in assets)
    assert not any(path.startswith("src/leenef/skill_assets/") for path in assets)


def test_skill_references_manifest_assets():
    skill = (REPO_ROOT / "skills" / "leenef" / "SKILL.md").read_text()
    for doc in ("docs/usage.md", "docs/SSUP-I.md", "docs/SSUP-II.md", "docs/SSUP-III.md"):
        assert doc in skill, f"SKILL.md should reference {doc}"
    assert "SSUP-plain-language" not in skill
    assert "tests/" not in skill
    assert "colab/" not in skill


def test_materialize_assets_can_preserve_repo_layout(tmp_path: Path):
    count = install_skill.materialize_assets(REPO_ROOT, tmp_path, install_layout=False)
    assert count == len(install_skill.iter_manifest_paths(REPO_ROOT))
    assert (tmp_path / "skills" / "leenef" / "SKILL.md").is_file()
    assert (tmp_path / "skills" / "leenef" / "assets.txt").is_file()
    assert (tmp_path / "skills" / "leenef" / "install_skill.py").is_file()
    assert (tmp_path / "docs" / "usage.md").is_file()
    assert (tmp_path / "results" / "rl_sparse" / "baseline_mc_fqi.json").is_file()
    assert not (tmp_path / "tests").exists()
    assert not (tmp_path / "colab").exists()


def test_installed_relative_path_strips_source_skill_prefix():
    assert install_skill.installed_relative_path("skills/leenef/SKILL.md") == Path("SKILL.md")
    assert install_skill.installed_relative_path("skills/leenef/assets.txt") == Path("assets.txt")
    assert install_skill.installed_relative_path("skills/leenef/install_skill.py") == Path(
        "install_skill.py"
    )
    assert install_skill.installed_relative_path("docs/usage.md") == Path("docs/usage.md")


def test_package_cli_loads_canonical_installer():
    assert load_installer().SKILL_NAME == install_skill.SKILL_NAME


def test_install_copies_skill(tmp_path: Path):
    target = install_skill.install(tmp_path, force=True)
    assert target == tmp_path / install_skill.SKILL_NAME
    assert (target / "SKILL.md").is_file()
    assert (target / "assets.txt").is_file()
    assert (target / "install_skill.py").is_file()
    assert (target / "docs" / "usage.md").is_file()
    assert (target / "docs" / "SSUP-III.md").is_file()
    assert (target / "src" / "leenef" / "rl.py").is_file()
    assert (target / "benchmarks" / "run_rl_paper_dense.py").is_file()
    assert (target / ".github" / "workflows" / "publish-pypi.yml").is_file()
    assert (target / "scripts" / "build_dist.sh").is_file()
    assert (target / "results" / "rl_clean" / "base_config.json").is_file()
    assert (target / "results" / "rl_sparse" / "baseline_mc_fqi.json").is_file()
    assert (target / "results" / "rl_sparse" / "logs" / "baseline_mc_fqi.log").is_file()
    assert (target / "LICENSE").is_file()
    assert not (target / "skills").exists()
    assert not (target / "tests").exists()
    assert not (target / "colab").exists()


def test_install_requires_force_when_exists(tmp_path: Path):
    install_skill.install(tmp_path)
    with pytest.raises(FileExistsError):
        install_skill.install(tmp_path)
    install_skill.install(tmp_path, force=True)
