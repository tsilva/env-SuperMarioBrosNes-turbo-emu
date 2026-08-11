.PHONY: develop develop-release play release test test-python test-rust test-retro-oracle

PYTHON ?= .venv/bin/python
UV_CACHE_DIR ?= .uv-cache
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
RUSTFLAGS_EXT ?= -C link-arg=-undefined -C link-arg=dynamic_lookup
else
RUSTFLAGS_EXT ?=
endif
PLAY_ARGS ?= Level1-1
PYTEST_ARGS ?=

develop:
	UV_CACHE_DIR=$(UV_CACHE_DIR) $(PYTHON) -m maturin develop

develop-release:
	UV_CACHE_DIR=$(UV_CACHE_DIR) $(PYTHON) -m maturin develop --release

play: develop-release
	$(PYTHON) play.py $(PLAY_ARGS)

release:
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv sync --frozen --extra dev --group dev
	scripts/release.py

test-rust:
	RUSTFLAGS="$(RUSTFLAGS_EXT)" cargo test --workspace

test-python:
	$(PYTHON) -m pytest -m "not retro_oracle" $(PYTEST_ARGS)

test-retro-oracle:
	$(PYTHON) -m pytest -m retro_oracle $(PYTEST_ARGS)

test: test-rust test-python
