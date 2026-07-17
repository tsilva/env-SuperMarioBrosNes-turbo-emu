# Contributing

Thank you for helping improve SuperMarioBros-Nes-turbo. Contributions should
stay within the documented mapper 0/NROM scope and preserve the public
Gymnasium, determinism, state, and performance contracts in `SPECS.md`.

Participation is governed by [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## Development setup

Install [uv](https://docs.astral.sh/uv/) and a Rust toolchain, then run:

```bash
git clone https://github.com/tsilva/SuperMarioBros-Nes-turbo.git
cd SuperMarioBros-Nes-turbo
uv sync --frozen --extra dev --group dev
uv run maturin develop --release
```

The default test suite does not require a ROM:

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
make test
```

Changes affecting emulation, states, rewards, termination, actions, or
preprocessing must also pass the local ROM-backed oracle suite:

```bash
make test-retro-oracle
```

Never commit or attach ROMs. Report only the canonical ROM SHA-256 documented
in the README.

## Pull requests

- Open an issue first for public API, scope, saved-state, determinism, or
  performance-contract changes.
- Keep changes focused and add regression tests for observable behavior.
- Update `CHANGES.md` and public documentation when behavior changes.
- Describe validation, compatibility effects, and any benchmark evidence in
  the pull request template.
- Do not add ROMs, extracted game assets, proprietary firmware, secrets,
  generated run directories, or benchmark artifacts containing private host
  data.

Performance changes must use the repository benchmark workflow. Do not publish
speed claims from unmatched workloads, busy-host runs, or results that skipped
the required correctness checks.

## Third-party content

Only contribute code and assets you are authorized to redistribute. Gameplay
captures, logos, packaged states, names, and marks require separate rights
review and are not granted rights by the MIT license. See [NOTICE.md](NOTICE.md).
