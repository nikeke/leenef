"""Install the leenef method-selection Agent Skill into a skills directory.

The skill is ``SKILL.md`` plus the reference material it tells the agent to
read (the usage manual, the three SSUP papers, the project notes, the package
source, the benchmark drivers, the test suite, the Colab notebooks, and the
``base_config*.json`` defaults).  Those assets ship *inside* the installed
package as ``leenef/skill_assets`` so they are available even after a plain
``pip install leenef`` where the repository tree is absent.

``leenef-install-skill --dest <dir>`` copies (or, with ``--link``, symlinks)
the bundled assets into ``<dir>/leenef-method-selection/`` so the skill's
relative reference links resolve wherever it lands.

``--sync-assets`` (development only) regenerates ``leenef/skill_assets`` from
the repository's canonical files; the packaged copy is kept byte-identical
to those sources and guarded by ``tests/test_skill_assets.py``.
"""

from __future__ import annotations

import argparse
import shutil
from collections.abc import Callable, Iterator
from importlib import resources
from pathlib import Path

SKILL_NAME = "leenef-method-selection"

# Files always bundled (exact repo-root paths).
_ROOT_FILES: tuple[str, ...] = (
    "SKILL.md",
    "README.md",
    "AGENTS.md",
    "LICENSE",
    "pyproject.toml",
)

# Per-tree inclusion rules.  Each entry: (subdir, recursive, predicate).
# ``predicate(path)`` decides whether to bundle the file.
_TREE_RULES: tuple[tuple[str, bool, Callable[[Path], bool]], ...] = (
    # Conceptual + execution-log markdown (drop plain-language overview per design).
    ("docs", False, lambda p: p.suffix == ".md" and p.name != "SSUP-plain-language.md"),
    # Library source: Python + py.typed.  ``skill_assets/`` itself is filtered
    # out via ``_SKIP_DIRS`` so sync doesn't recurse into its own output.
    ("src/leenef", True, lambda p: p.suffix == ".py" or p.name == "py.typed"),
    # Benchmark drivers, helpers, base configs, READMEs.
    ("benchmarks", True, lambda p: p.suffix in {".py", ".sh", ".md", ".json"}),
    # Test suite as executable usage examples and reference patterns.
    ("tests", True, lambda p: p.suffix == ".py"),
    # Colab notebooks and their READMEs.
    ("colab", True, lambda p: p.suffix in {".md", ".ipynb"}),
    # Only the default-config JSONs from results/ (the rest is heavy result data).
    ("results", True, lambda p: p.name.startswith("base_config") and p.suffix == ".json"),
)

# Directory names skipped during tree recursion (caches, vendored builds, etc).
_SKIP_DIRS: frozenset[str] = frozenset(
    {"__pycache__", ".pytest_cache", ".ruff_cache", "skill_assets", "attic"}
)


def iter_canonical_paths(root: Path) -> Iterator[str]:
    """Yield repo-relative paths that should be bundled into ``skill_assets``."""
    for name in _ROOT_FILES:
        if (root / name).is_file():
            yield name
    for subdir, recursive, predicate in _TREE_RULES:
        base = root / subdir
        if not base.is_dir():
            continue
        iterator = base.rglob("*") if recursive else base.glob("*")
        for path in sorted(iterator):
            if not path.is_file():
                continue
            rel = path.relative_to(root)
            if any(part in _SKIP_DIRS for part in rel.parts):
                continue
            if not predicate(path):
                continue
            yield str(rel)


def _packaged_assets_dir() -> Path:
    """Real on-disk path of the bundled ``skill_assets`` directory."""
    return Path(str(resources.files("leenef").joinpath("skill_assets")))


def _src_assets_dir() -> Path:
    """Source-tree ``skill_assets`` directory (target of ``--sync-assets``)."""
    return Path(__file__).resolve().parent / "skill_assets"


def _repo_root() -> Path | None:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file() and (parent / "docs").is_dir():
            return parent
    return None


def sync_assets() -> Path:
    """Regenerate the packaged skill assets from the canonical repo tree."""
    root = _repo_root()
    if root is None:
        raise RuntimeError(
            "--sync-assets requires a repository checkout (pyproject.toml + docs/)."
        )
    assets = _src_assets_dir()
    if assets.exists():
        shutil.rmtree(assets)
    assets.mkdir(parents=True)
    count = 0
    for rel in iter_canonical_paths(root):
        dst = assets / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / rel, dst)
        count += 1
    (assets / ".asset-count").write_text(f"{count}\n")
    return assets


def install(dest: Path, *, link: bool = False, force: bool = False) -> Path:
    """Copy (or symlink) the bundled skill into ``dest/leenef-method-selection``."""
    dest = Path(dest).expanduser()
    target = dest / SKILL_NAME
    assets = _packaged_assets_dir()
    if not assets.is_dir():
        raise FileNotFoundError(f"bundled skill assets not found at {assets}")

    if target.exists() or target.is_symlink():
        if not force:
            raise FileExistsError(f"{target} already exists; pass --force to overwrite.")
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        else:
            target.unlink()

    dest.mkdir(parents=True, exist_ok=True)
    if link:
        target.symlink_to(assets.resolve(), target_is_directory=True)
    else:
        shutil.copytree(assets, target)
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="leenef-install-skill",
        description="Install the leenef method-selection Agent Skill into a skills directory.",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        help="Skills directory to install into (creates <dest>/leenef-method-selection/).",
    )
    parser.add_argument(
        "--link",
        action="store_true",
        help="Symlink the bundled assets instead of copying (development checkouts only).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing leenef-method-selection skill directory.",
    )
    parser.add_argument(
        "--sync-assets",
        action="store_true",
        help="Development only: regenerate packaged skill assets from canonical repo files.",
    )
    args = parser.parse_args(argv)

    if args.sync_assets:
        assets = sync_assets()
        print(f"Synced skill assets at {assets}")
        return 0

    if args.dest is None:
        parser.error("--dest is required (or use --sync-assets).")

    target = install(args.dest, link=args.link, force=args.force)
    verb = "Linked" if args.link else "Installed"
    print(f"{verb} skill at {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
