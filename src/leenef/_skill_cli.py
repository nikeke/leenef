"""Console-script shim for the manifest-driven leenef Agent Skill installer."""

from __future__ import annotations

import importlib.util
from importlib import resources
from pathlib import Path
from types import ModuleType

_IMPL_NAME = "_leenef_skill_installer_impl"
_SKILL_INSTALLER = Path("skills/leenef/install_skill.py")


def _load_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(_IMPL_NAME, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load leenef skill installer from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _repo_installer() -> Path | None:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / _SKILL_INSTALLER
        if candidate.is_file():
            return candidate
    return None


def _packaged_installer() -> Path:
    return Path(
        str(
            resources.files("leenef").joinpath(
                "skill_assets", "skills", "leenef", "install_skill.py"
            )
        )
    )


def load_installer() -> ModuleType:
    """Load the canonical installer from a checkout or generated package assets."""
    repo = _repo_installer()
    if repo is not None:
        return _load_module(repo)

    packaged = _packaged_installer()
    if packaged.is_file():
        return _load_module(packaged)

    raise FileNotFoundError(
        "leenef skill installer was not found. Reinstall leenef from a wheel that "
        "includes generated skill assets, or run from a repository checkout."
    )


def main(argv: list[str] | None = None) -> int:
    return load_installer().main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
