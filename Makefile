.PHONY: develop develop-release play release test test-python test-rust test-retro-oracle verify-retro-oracle

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
TURBOBENCH ?= $(abspath ../turbobench/.venv/bin/turbobench)
ORACLE_OUTPUT ?=
ORACLE_RECEIPT ?=

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
	@output="$(ORACLE_OUTPUT)"; \
	if [ -z "$$output" ]; then output="$$(mktemp -d)/supermario-semantic-oracle"; fi; \
	$(TURBOBENCH) oracle supermario/canonical-v2 \
		--left stable-retro@1.0.1 \
		--right env-supermariobrosnes-turbo-emu@checkout:$(CURDIR) \
		--output "$$output" \
		--allow-dirty; \
	echo "Semantic-oracle receipt: $$output"

verify-retro-oracle:
	@test -n "$(ORACLE_RECEIPT)" || \
		(echo "Set ORACLE_RECEIPT to an external TurboBench receipt" >&2; exit 2)
	$(TURBOBENCH) verify-oracle "$(ORACLE_RECEIPT)" \
		--require-canonical \
		--require-provider env-supermariobrosnes-turbo-emu

test: test-rust test-python
