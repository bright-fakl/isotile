# Releasing isotile

Maintainer procedure. Nothing is published on a merge to `main`; a release is a tag `vX.Y.Z` (or a GitHub release that
creates it) on a commit whose `pyproject.toml` has the version `X.Y.Z`.

## What the automation does

| Workflow | Trigger | Effect |
| --- | --- | --- |
| `main.yml` | push to `main`, pull request | lint, types, tests (Python 3.10 to 3.13, macOS, Windows), tactile equivalence, data check, minimum dependency versions, wheel smoke test, docs build |
| `pypi-publish.yml` | tag `v*` pushed, GitHub release published | builds, tests and checks the distributions, then uploads to PyPI (environment `pypi`) |
| `pypi-publish.yml` | manual run | the same, but uploads to TestPyPI (environment `testpypi`); never to PyPI |
| `deploy-docs.yml` | push to `main`, manual | builds the documentation (strict) and publishes it to GitHub Pages |

`pypi-publish.yml` stops before the upload if the tag is not `v` plus the version in `pyproject.toml` (PEP 440
normalization applies: `v0.2.0-rc1` equals `0.2.0rc1`), if the tests fail, if `scripts/check_dist.py` finds a missing
data file, license text or `py.typed`, if `twine check --strict` rejects the metadata, or if the smoke test of the wheel
in a clean environment fails. A pre-release or development version (`rc`, `a`, `b`, `.dev`) goes to TestPyPI instead of
PyPI. If the version is already on the target index, the upload is skipped. That is the normal case for the second run
when a release is created together with its tag: both events start the workflow, the runs are serialized, and the
second one uploads nothing.

Authentication is trusted publishing (OIDC). The repository holds no PyPI token.

### Environments and refs

- The job `publish-pypi` runs only when `scripts/release_plan.py` returns the target `pypi`, which requires a tag
  event whose tag is `v` plus the package version of a final version. For any other ref the script exits with an error
  before the job exists, so the job, and with it the `pypi` environment, is never used on a branch ref. A manual run
  always targets `testpypi`, also when started on a tag.
- A `release: published` event runs on the tag ref (`refs/tags/vX.Y.Z`), so the restriction to the tag pattern `v*`
  admits it, as it admits a pushed tag.
- `gh release create vX.Y.Z --target <sha>` creates the tag `vX.Y.Z` on that commit (the tag does not need to exist) and
  publishes the release. The workflow starts on the tag and checks out `<sha>`; `pyproject.toml` at `<sha>` must have
  the version `X.Y.Z`. Both the tag push and the release event start a run: the first waits for the approval of the
  `pypi` deployment, the second waits in the concurrency queue, then finds the version on PyPI and uploads nothing, so
  there is a single approval. A draft release creates the tag only when it is published.
- The manual dry run from `main` uses `testpypi`, which has no restriction.

## One-time setup

Done once, before the first release.

1. **PyPI.** Log in at <https://pypi.org>, open Account settings, Publishing, and add a *pending publisher* with the
   values below. The project `isotile` is created by the first upload.

   | Field | Value |
   | --- | --- |
   | PyPI project name | `isotile` |
   | Owner | `bright-fakl` |
   | Repository name | `isotile` |
   | Workflow name | `pypi-publish.yml` |
   | Environment name | `pypi` |

2. **TestPyPI** (optional, for dry runs). The same at <https://test.pypi.org> with the environment name `testpypi`.
3. **GitHub.** Repository Settings, Environments:

   | Environment | Protection |
   | --- | --- |
   | `pypi` | Required reviewers: the maintainer, so each upload waits for an approval. Deployment branches and tags: *Selected branches and tags* with the tag pattern `v*` only. |
   | `testpypi` | None (used by the manual dry run from `main`). |

The environment names are part of the trusted publisher on the index; a mismatch gives the error
`invalid-publisher` in the upload step.

## Version and changelog

`pyproject.toml` is the only place of the version; `isotile.__version__` reads the installed metadata, and the
documentation takes the changelog from `CHANGELOG.md`. Between releases `main` carries a development version
(`0.1.0.dev0`), which the publish workflow never sends to PyPI. The release changes it in one pull request.

## Release checklist

1. **Prepare the release pull request.**

   - Set `version = "X.Y.Z"` in `pyproject.toml`.
   - Run `uv lock` (the lock file holds the version of the project; `uv lock --check` in CI fails otherwise).
   - In `CHANGELOG.md`, rename the heading `## [X.Y.Z] - Unreleased` to `## [X.Y.Z] - YYYY-MM-DD` (for a later release, move the
     entries of `## [Unreleased]` into a new `## [X.Y.Z] - YYYY-MM-DD` section) and read it against the merged changes.
   - Read the built documentation (`uv run --group docs mkdocs serve`); the changelog and the migration notes are the
     pages most likely to be wrong, because the tests do not check their content.

2. **Check the pull request.** CI must be green, including `wheel-smoke-test`, `minimum-versions` and `check-docs`.
   Locally:

   ```bash
   uv build -o dist
   uv run python scripts/check_dist.py dist --version X.Y.Z
   uvx twine check --strict dist/*
   ```

3. **Merge** and wait for `main.yml` and `deploy-docs.yml` to pass on the merge commit.

4. **Optional dry run on TestPyPI** (needs the one-time setup of TestPyPI). Run it before the version bump of step 1, or
   with a release candidate version: TestPyPI refuses to replace a file, so each version can be uploaded there once.

   ```bash
   gh workflow run pypi-publish.yml --ref main
   ```

   Then install from TestPyPI into a clean environment, outside the repository:

   ```bash
   cd /tmp && uv venv /tmp/tpypi && VIRTUAL_ENV=/tmp/tpypi uv pip install \
     --index-strategy unsafe-best-match --index-url https://test.pypi.org/simple/ \
     --extra-index-url https://pypi.org/simple/ "isotile==X.Y.Z"
   VIRTUAL_ENV=/tmp/tpypi uv run --no-project python <path-to-checkout>/scripts/smoke_test.py
   ```

   Pin the exact version (older versions on TestPyPI would otherwise be installed instead), use
   `--index-strategy unsafe-best-match` (numpy and scipy come from PyPI) and run from outside the checkout.

5. **Create the release.** The tag must be on the merge commit of step 3. Either command fires `pypi-publish.yml`:

   ```bash
   gh release create vX.Y.Z --target <merge commit sha> --title "isotile X.Y.Z" --notes-file <(awk '/^## \[/ {f = ($0 ~ /^## \[X.Y.Z\]/); next} f' CHANGELOG.md)
   ```

   or `git tag -a vX.Y.Z <sha> -m "isotile X.Y.Z" && git push origin vX.Y.Z`. With a required reviewer on the
   `pypi` environment, approve the pending deployment in the Actions run. A version on PyPI can never be replaced or
   re-uploaded (it can be yanked); if the workflow fails before the upload, fix the cause and re-run the failed run,
   or delete the tag and create it again on a fixed commit.

6. **Verify the published package** in a clean environment, outside the repository:

   ```bash
   cd /tmp && uv venv /tmp/pypi && VIRTUAL_ENV=/tmp/pypi uv pip install isotile
   VIRTUAL_ENV=/tmp/pypi uv run --no-project python <path-to-checkout>/scripts/smoke_test.py --version X.Y.Z
   ```

   Also open <https://pypi.org/project/isotile/> (description, links, license), and the documentation site.

7. **Start the next development version.** In a pull request, set `version = "X.Y.(Z+1).dev0"` (or the next minor),
   run `uv lock` and add an empty `## [Unreleased]` section to `CHANGELOG.md`.

8. **Switch the consumers** (carto-flow and isohedral-tiling-ui) from their git or archive pin to `isotile>=X.Y.Z`.
