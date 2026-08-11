# Benchmarks

This page contains the official, host-specific TurboBench package results for
SuperMarioBros-Nes-turbo. Each provider comparison remains bound to its own
paired evidence bundle.

## Official TurboBench package results

On 2026-08-11,
[`supermariobrosnes-turbo==0.6.2`](https://pypi.org/project/supermariobrosnes-turbo/0.6.2/)
was compared using the [documented machine profile](#machine-profile) against
both
[`stable-retro-turbo==1.0.1.post37`](https://pypi.org/project/stable-retro-turbo/1.0.1.post37/)
and the original
[`stable-retro==1.0.1`](https://pypi.org/project/stable-retro/1.0.1/). Both
comparisons passed every TurboBench validity gate and produced official,
verified result bundles.

| Envs | Matched provider | Turbo median SPS | Provider median SPS | Paired speedup | 95% paired bootstrap CI |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | `stable-retro-turbo==1.0.1.post37` | 10,566.6 | 1,416.0 | 7.4635x | 7.3962x–7.4735x |
| 1 | `stable-retro==1.0.1` | 10,543.8 | 626.8 | 16.8314x | 16.7373x–16.9237x |
| 16 | `stable-retro-turbo==1.0.1.post37` | 52,864.5 | 7,085.0 | 7.4614x | 7.4391x–7.5529x |
| 16 | `stable-retro==1.0.1` | 52,368.7 | 3,346.5 | 15.6262x | 15.5387x–15.8200x |
| 32 | `stable-retro-turbo==1.0.1.post37` | 66,439.3 | 8,269.6 | 8.0526x | 7.9997x–8.0687x |
| 32 | `stable-retro==1.0.1` | 65,930.0 | 3,689.6 | 17.9426x | 17.7317x–18.0522x |

The two baselines were measured in separate paired runs. Their rows must not be
treated as a direct paired comparison between Stable Retro Turbo and original
Stable Retro. SPS means environment steps per second.

### Evidence

| Comparison | Recorded bundle ID |
| --- | --- |
| `supermariobrosnes-turbo==0.6.2` versus `stable-retro-turbo==1.0.1.post37` | `1d3508ab40a81377d7f8d0fc5270c55a85e9f935893a41518761597f75e63760` |
| `supermariobrosnes-turbo==0.6.2` versus `stable-retro==1.0.1` | `ca1853067f0a28c276dc97343981e3c806c468ea5b1cde094b40fc263cfa3e19` |

Each bundle contains 119 hash-bound artifacts and passed `turbobench verify`
without errors. The IDs above identify the recorded local evidence; they are
not download links, and this repository does not publish the bundles. The runs
used [TurboBench 1.0.0](https://pypi.org/project/turbobench-cli/1.0.0/) from
[source commit `d986efa72c81a7d0b5ea689ac37898d8fc38732f`](https://github.com/tsilva/turbobench/commit/d986efa72c81a7d0b5ea689ac37898d8fc38732f),
harness source SHA-256
`2c64aefe52d5db7f2887b0f9d9d32c23c49f6590319a02eee5b6e2398b710319`,
and immutable profile `supermario/canonical-v1` with profile SHA-256
`326c6d47c4cc0bc2bbafdf003a430ea80cc27877f8a4144dfbb65dbea6bb2cd7`.

### TurboBench protocol

For shapes 1, 16, and 32, TurboBench first required exact matched policy
observations, normalized raw RGB frames, rewards, termination and truncation,
selective reset points, completion, and semantic infos. Each timed shape then
used one unmeasured warmup pair followed by seven alternating AB/BA measured
pairs. Each invocation contained three repetitions of 250 vector steps, and a
deterministic 20,000-resample paired bootstrap produced the 95% interval.

The matched workload used `Level1-1` through `Level1-4`, frame skip 4, no
max-pooling, four grayscale frames, a zeroed 32-row HUD, area resize to `84x84`,
CHW output, and deterministic precomputed actions. Timed SPS included
preprocessing, IPC, infos, terminal detection, and selective resets. It excluded
construction, initial reset, action generation, warmup, correctness replay,
rendering, and encoding.

Install the published CLI for new comparisons with:

```bash
uv tool install \
  --exclude-newer-package turbobench-cli=2026-08-12T00:00:00Z \
  turbobench-cli==1.0.0
```

The recorded bundles retain the exact source identity shown above. Reproduce
them from that source revision, setting `TURBOBENCH_ASSET_ROOT` to a Stable
Retro-compatible data directory that contains `SuperMarioBros-Nes-v0/rom.nes`
and the canonical `Level1-1` through `Level1-4` state files:

```bash
git clone https://github.com/tsilva/turbobench.git
cd turbobench
git checkout --detach d986efa72c81a7d0b5ea689ac37898d8fc38732f
uv sync --frozen --group dev

export TURBOBENCH_ASSET_ROOT=/absolute/path/to/stable-retro/data/stable
uv run turbobench doctor supermario/canonical-v1

uv run turbobench compare supermario/canonical-v1 \
  --left supermariobrosnes-turbo@0.6.2 \
  --right stable-retro-turbo@1.0.1.post37

uv run turbobench compare supermario/canonical-v1 \
  --left supermariobrosnes-turbo@0.6.2 \
  --right stable-retro@1.0.1
```

### Machine profile

| Component | Specification |
| --- | --- |
| CPU | AMD Ryzen 5 7600X (Zen 4), 6 physical cores / 12 threads, boost enabled, 5.457 GHz reported maximum |
| CPU cache | 384 KiB L1, 6 MiB L2, 32 MiB L3 |
| Memory | 65,396,760,576 bytes reported (60.9 GiB) |
| OS | Ubuntu 26.04 LTS, Linux 7.0.0-29-generic, x86_64 |
| Runtime | CPython 3.14.6; `numpy==2.4.2`; `gymnasium==1.2.2` |

The one-minute system load was below TurboBench's 6.0 threshold for both runs;
no diagnostic overrides were used.
