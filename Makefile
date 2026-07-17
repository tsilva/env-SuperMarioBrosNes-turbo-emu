.PHONY: autoresearch-accept autoresearch-accept-full autoresearch-calibrate autoresearch-checks autoresearch-diagnose autoresearch-profile autoresearch-screen benchmark benchmark-local benchmark-report develop develop-release play release test test-python test-rust test-retro-oracle

PYTHON ?= .venv/bin/python
UV_CACHE_DIR ?= .uv-cache
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
RUSTFLAGS_EXT ?= -C link-arg=-undefined -C link-arg=dynamic_lookup
else
RUSTFLAGS_EXT ?=
endif
BENCHMARK_STEPS ?= 5000
BENCHMARK_REPEATS ?= 3
BENCHMARK_WARMUP ?= 500
BENCHMARK_ENV_ARGS ?= $(shell $(PYTHON) scripts/benchmark_workload.py)
BENCHMARK_LOAD_ARGS ?= --skip-load-preflight
BENCHMARK_ARGS ?=
BENCHMARK_REPORT_ARGS ?=
PLAY_ARGS ?= Level1-1
BASELINE_REF ?=
CANDIDATE_REF ?=
CALIBRATE_REF ?= HEAD
PYTEST_ARGS ?=

develop:
	UV_CACHE_DIR=$(UV_CACHE_DIR) $(PYTHON) -m maturin develop

develop-release:
	UV_CACHE_DIR=$(UV_CACHE_DIR) $(PYTHON) -m maturin develop --release

benchmark: develop-release benchmark-local

benchmark-local:
	$(PYTHON) scripts/benchmark_sps.py $(BENCHMARK_ENV_ARGS) --steps $(BENCHMARK_STEPS) --repeats $(BENCHMARK_REPEATS) --warmup $(BENCHMARK_WARMUP) $(BENCHMARK_LOAD_ARGS) $(BENCHMARK_ARGS)

benchmark-report: develop-release
	$(PYTHON) scripts/benchmark_report.py $(BENCHMARK_REPORT_ARGS)

play: develop-release
	$(PYTHON) play.py $(PLAY_ARGS)

autoresearch-diagnose:
	$(PYTHON) scripts/autoresearch.py diagnose

autoresearch-profile:
	$(PYTHON) scripts/autoresearch.py diagnose --profile

autoresearch-screen:
	test -n "$(BASELINE_REF)" && test -n "$(CANDIDATE_REF)"
	$(PYTHON) scripts/autoresearch.py screen $(BASELINE_REF) $(CANDIDATE_REF)

autoresearch-accept:
	test -n "$(BASELINE_REF)" && test -n "$(CANDIDATE_REF)"
	$(PYTHON) scripts/autoresearch.py accept $(BASELINE_REF) $(CANDIDATE_REF)

autoresearch-accept-full:
	test -n "$(BASELINE_REF)" && test -n "$(CANDIDATE_REF)"
	$(PYTHON) scripts/autoresearch.py accept $(BASELINE_REF) $(CANDIDATE_REF) --full

autoresearch-calibrate:
	$(PYTHON) scripts/autoresearch.py calibrate $(CALIBRATE_REF)

autoresearch-checks:
	$(PYTHON) scripts/autoresearch.py checks

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
