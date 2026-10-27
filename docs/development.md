# Development

The repository is [bright-fakl/isotile](https://github.com/bright-fakl/isotile). The project uses
[uv](https://docs.astral.sh/uv/); CI pins uv 0.8.17.

## Set up the environment

```bash
git clone https://github.com/bright-fakl/isotile.git
cd isotile
uv sync                    # runtime and development dependencies, including tactile
uv sync --no-group tactile # without tactile; its equivalence tests are skipped
uv sync --group docs       # adds the documentation dependencies
```

`uv sync` installs the groups `dev` and `tactile` by default. tactile is only used by the tests that compare isotile
with it. Install the pre-commit hooks with `uv run pre-commit install`.

## Run the tests

```bash
uv run pytest                    # all tests
uv run pytest -m tactile         # only the comparison with tactile
uv run pytest -m "not tactile"   # everything else
uv run pytest -m "not slow"      # without the tests that draw many random curved tilings
uv run pytest --cov              # with coverage
```

The tests also run every Python code block of the documentation pages (`tests/test_docs_examples.py`) and check that
every public name is in the API reference.

## Lint and type-check

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy                      # mypy --strict on src
```

`uv lock --check` verifies that `uv.lock` matches `pyproject.toml`.

## Regenerate the data file

The arrays of the 81 types are stored in `src/isotile/data/types.npz`, built from tactile by `isotile._build_data`. The
build is deterministic. This command rebuilds the data in memory and fails if it differs from the stored file:

```bash
uv run python -m isotile._build_data --check
```

Run `uv run python -m isotile._build_data` without `--check` to write the file. Both need the `tactile` group.

## Build the documentation

```bash
uv sync --group docs
uv run mkdocs build -s -d site   # strict: warnings and unresolved references fail the build
uv run mkdocs serve              # live preview at http://127.0.0.1:8000
```

The API reference is generated from the docstrings (numpy style) by mkdocstrings. A name that is added to
`isotile.__all__` needs an entry in `docs/api/`. Code in the pages `index.md`, `quickstart.md`, `concepts.md`, `curves.md` and
`adjacency.md` fenced as `python` is executed by the tests; use `text` or `bash` for code that must not run.

## Regenerate the figures

The figures of the pages are SVG files in `docs/assets/figures/`, drawn by `scripts/make_doc_figures.py` with matplotlib
(docs group). Each figure has a light and a dark variant; the pages show the variant of the active color scheme with the
`#only-light` and `#only-dark` URL fragments. Where a figure illustrates a code block of a page, the script runs the
blocks of the page up to that block, so the figure shows what the code produces.

```bash
uv run --group docs python scripts/make_doc_figures.py           # write the files
uv run --group docs python scripts/make_doc_figures.py --check   # fail if a stored file is out of date
```

The output is deterministic for a given matplotlib version. A figure is a function in the script; add it to `FIGURES`
and reference both variants in the page.

## Benchmarks

```bash
uv run python benchmarks/bench.py
```
