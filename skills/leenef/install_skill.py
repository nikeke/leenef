"""Install the leenef Agent Skill into an agent skills directory.

The canonical skill lives in ``skills/leenef/`` at the repository root.  Its
``assets.txt`` manifest lists the docs, source files, benchmark drivers, and
result artifacts the agent should have available.  Repository installs read
those files directly.  Wheel builds materialize the same manifest under
``leenef/skill_assets`` at build time so ``pip install leenef`` users can still
run ``leenef-install-skill`` without a source checkout.
"""

from __future__ import annotations

import argparse
import shutil
from collections.abc import Iterator
from pathlib import Path

SKILL_NAME = "leenef"
SKILL_DIR = Path("skills/leenef")
MANIFEST_PATH = SKILL_DIR / "assets.txt"
SKILL_FILE = SKILL_DIR / "SKILL.md"
_SKIP_DIRS: frozenset[str] = frozenset(
    {"__pycache__", ".git", ".mypy_cache", ".pytest_cache", ".ruff_cache", "skill_assets"}
)


def _has_glob_magic(pattern: str) -> bool:
    return any(char in pattern for char in "*?[")


def _validate_manifest_pattern(pattern: str) -> None:
    path = Path(pattern)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"asset manifest patterns must be repo-relative: {pattern!r}")


def _manifest_lines(manifest: Path) -> Iterator[tuple[bool, str]]:
    for raw in manifest.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        excluded = line.startswith("!")
        pattern = line[1:].strip() if excluded else line
        if not pattern:
            continue
        _validate_manifest_pattern(pattern)
        yield excluded, pattern


def _matching_files(root: Path, pattern: str) -> list[str]:
    if _has_glob_magic(pattern):
        paths = root.glob(pattern)
    else:
        paths = [root / pattern]

    matches: list[str] = []
    for path in sorted(paths):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if any(part in _SKIP_DIRS for part in rel.parts):
            continue
        matches.append(rel.as_posix())
    return matches


def iter_manifest_paths(root: Path, *, strict: bool = True) -> list[str]:
    """Return repo-relative asset paths selected by ``skills/leenef/assets.txt``."""
    root = Path(root)
    manifest = root / MANIFEST_PATH
    if not manifest.is_file():
        raise FileNotFoundError(f"skill asset manifest not found at {manifest}")

    selected: set[str] = set()
    excluded_paths: set[str] = set()
    for excluded, pattern in _manifest_lines(manifest):
        matches = _matching_files(root, pattern)
        if strict and not matches and not excluded:
            raise FileNotFoundError(f"skill asset pattern matched no files: {pattern}")
        if excluded:
            selected.difference_update(matches)
            excluded_paths.update(matches)
        else:
            selected.update(match for match in matches if match not in excluded_paths)
    return sorted(selected)


def iter_canonical_paths(root: Path) -> Iterator[str]:
    """Backward-compatible iterator for callers that used the old manifest API."""
    yield from iter_manifest_paths(root)


def installed_relative_path(rel: str) -> Path:
    """Map a repo-relative asset path to its path inside an installed skill."""
    path = Path(rel)
    try:
        return path.relative_to(SKILL_DIR)
    except ValueError:
        return path


def materialize_assets(
    source_root: Path,
    target: Path,
    *,
    link: bool = False,
    install_layout: bool = True,
) -> int:
    """Copy or symlink manifest-selected assets from ``source_root`` to ``target``.

    ``install_layout=True`` strips the source-only ``skills/leenef/`` prefix so
    the installed skill has ``SKILL.md`` at its root.  Wheel builds pass
    ``install_layout=False`` to preserve repo-relative paths under the package's
    generated ``skill_assets`` tree.
    """
    source_root = Path(source_root)
    target = Path(target)
    count = 0
    for rel in iter_manifest_paths(source_root):
        dst_rel = installed_relative_path(rel) if install_layout else Path(rel)
        src = source_root / rel
        dst = target / dst_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if link:
            dst.symlink_to(src.resolve())
        else:
            shutil.copy2(src, dst)
        count += 1
    return count


def _repo_root() -> Path | None:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file() and (parent / MANIFEST_PATH).is_file():
            return parent
    return None


def _source_root() -> Path:
    root = _repo_root()
    if root is not None:
        return root
    raise FileNotFoundError(
        "leenef skill assets were not found. Reinstall leenef from a wheel that "
        "includes generated skill assets, or run this command from a repository checkout."
    )


def install(dest: Path, *, link: bool = False, force: bool = False) -> Path:
    """Copy (or symlink) the skill into ``dest/leenef``."""
    dest = Path(dest).expanduser()
    target = dest / SKILL_NAME
    source_root = _source_root()

    if target.exists() or target.is_symlink():
        if not force:
            raise FileExistsError(f"{target} already exists; pass --force to overwrite.")
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        else:
            target.unlink()

    target.mkdir(parents=True, exist_ok=True)
    materialize_assets(source_root, target, link=link, install_layout=True)
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="leenef-install-skill",
        description="Install the leenef Agent Skill into a skills directory.",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        required=True,
        help="Skills directory to install into (creates <dest>/leenef/).",
    )
    parser.add_argument(
        "--link",
        action="store_true",
        help="Symlink manifest assets instead of copying (best for development checkouts).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing leenef skill directory.",
    )
    args = parser.parse_args(argv)

    target = install(args.dest, link=args.link, force=args.force)
    verb = "Linked" if args.link else "Installed"
    print(f"{verb} skill at {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
