from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py

ROOT = Path(__file__).resolve().parent


def _load_install_skill_module():
    spec = importlib.util.spec_from_file_location(
        "_leenef_install_skill", ROOT / "skills" / "leenef" / "install_skill.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load leenef install_skill build helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class build_py(_build_py):
    def run(self):
        super().run()
        install_skill = _load_install_skill_module()
        target = Path(self.build_lib) / "leenef" / "skill_assets"
        if target.exists():
            shutil.rmtree(target)
        install_skill.materialize_assets(ROOT, target, install_layout=False)


setup(cmdclass={"build_py": build_py})
