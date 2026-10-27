# isotile (prototype)

Prototype of the core of `isotile`, a NumPy-based library for isohedral tilings that replaces `tactile`.
It was written as evidence for the design in the dev notes (`proposal-isohedral-tiling-library`, issue
bright-fakl/carto-flow#159) and is not yet a package: `pyproject.toml` configures only ruff and pytest.

The type data in `isotile_proto/data/` is derived from Craig S. Kaplan's Tactile (BSD 3-Clause, see
`isotile_proto/data/LICENSE-tactile`).

    uv run --project /home/fakl/carto-flow python -m pytest
