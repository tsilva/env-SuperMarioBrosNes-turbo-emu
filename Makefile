.PHONY: develop develop-release parity parity-release play release test test-python test-rust verify-parity

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
PARITY_OUTPUT ?=
PARITY_RECEIPT ?=
PARITY_WHEEL ?=

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
	$(PYTHON) -m pytest $(PYTEST_ARGS)

parity:
	@output="$(PARITY_OUTPUT)"; \
	if [ -z "$$output" ]; then output="$$(mktemp -d)/supermario-parity"; fi; \
	$(TURBOBENCH) parity supermario/canonical-v2 \
		--candidate env-supermariobrosnes-turbo-emu@checkout:$(CURDIR) \
		--output "$$output" \
		--allow-dirty --quick; \
	echo "Diagnostic parity receipt: $$output"

parity-release:
	@test -f "$(PARITY_WHEEL)" || (echo "Set PARITY_WHEEL to the exact final wheel" >&2; exit 2)
	@test -n "$(PARITY_OUTPUT)" || (echo "Set PARITY_OUTPUT to an external receipt path" >&2; exit 2)
	$(TURBOBENCH) parity supermario/canonical-v2 \
		--candidate env-supermariobrosnes-turbo-emu@artifact:$(abspath $(PARITY_WHEEL)) \
		--output "$(PARITY_OUTPUT)"
	$(TURBOBENCH) verify-parity "$(PARITY_OUTPUT)" --require-canonical \
		--require-provider env-supermariobrosnes-turbo-emu

verify-parity:
	@test -n "$(PARITY_RECEIPT)" || \
		(echo "Set PARITY_RECEIPT to an external TurboBench receipt" >&2; exit 2)
	$(TURBOBENCH) verify-parity "$(PARITY_RECEIPT)" \
		--require-canonical \
		--require-provider env-supermariobrosnes-turbo-emu

test: test-rust test-python
