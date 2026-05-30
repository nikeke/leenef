# PyPI release workflow

Use PyPI Trusted Publishing from GitHub Actions.  That is the preferred
workflow: PyPI and GitHub exchange an OpenID Connect token at release time, so
no PyPI API token is stored in the repository or on a developer machine.

## First-time setup

Do this once for PyPI and once for TestPyPI:

1. Create or claim the `leenef` project on the target index, or create a
   pending trusted publisher for a project that does not exist yet.
2. Add a trusted publisher:
   - owner: `nikeke`
   - repository: `leenef`
   - workflow name:
     - PyPI: `publish-pypi.yml`
     - TestPyPI: `publish-testpypi.yml`
   - environment:
     - PyPI: `pypi`
     - TestPyPI: `testpypi`
3. In GitHub, create matching environments named `pypi` and `testpypi`.
   Add required reviewers if releases should require manual approval.

Keep account recovery codes, API tokens, or other credentials out of the repo.
With Trusted Publishing, normal releases do not need an API token.

## Local preflight

Before tagging a release:

```bash
source venv/bin/activate
ruff check src/leenef/ tests/ benchmarks/ setup.py
ruff format --check src/leenef/ tests/ benchmarks/ setup.py
PYTHONPATH=$PWD pytest -q
bash scripts/build_dist.sh
```

The build script removes old `dist/` artifacts, builds the wheel and sdist, and
runs `twine check dist/*` so README rendering problems are caught before
upload.

## TestPyPI dry run

1. Bump `version` in `pyproject.toml` to a version that has never been uploaded
   to TestPyPI or PyPI.  PyPI indexes reject replacing an existing file.
   Prefer a disposable prerelease for TestPyPI, for example:

   - TestPyPI: `0.1.1rc1`, `0.1.1rc2`, ...
   - Final PyPI release: `0.1.1`

   That keeps the real release version clean while still exercising the full
   publish workflow.
2. Commit the version bump.
3. Run the **Publish to TestPyPI** workflow manually from the GitHub Actions UI.
4. Smoke-test the TestPyPI package in a fresh environment:

```bash
python3 -m venv /tmp/leenef-testpypi
/tmp/leenef-testpypi/bin/pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  leenef==0.1.1rc1
/tmp/leenef-testpypi/bin/leenef-install-skill --dest /tmp/leenef-skills
```

## PyPI release

1. After the TestPyPI dry run, bump `pyproject.toml` from the disposable
   prerelease (`0.1.1rc1`) to the final release (`0.1.1`) and commit that
   change.
2. Create a GitHub release with a matching tag, for example `v0.1.1`.
3. Publishing the release triggers `.github/workflows/publish-pypi.yml`.
4. After the workflow finishes, verify:

```bash
python3 -m venv /tmp/leenef-pypi
/tmp/leenef-pypi/bin/pip install leenef==0.1.1
/tmp/leenef-pypi/bin/leenef-install-skill --dest /tmp/leenef-skills
```

## Why not upload manually?

Manual `twine upload dist/*` works, but it requires a PyPI API token and makes
release provenance depend on a local machine.  Trusted Publishing ties the
upload to a specific repository, workflow, and environment, which is easier to
audit and rotate.  Use manual upload only as an emergency fallback.
