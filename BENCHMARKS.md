# Benchmarks

This page contains the detailed, host-specific benchmark results for
SuperMarioBros-Nes-turbo. The summary chart in the [README](README.md#benchmark)
links here for exact values, confidence intervals, protocol details, and machine
specifications.

![SuperMarioBros-Nes-turbo versus Stable Retro median environment throughput](media/benchmark-throughput.svg)

The chart above summarizes the earlier published `0.3.0` checkout results
documented later on this page. The current package results below use TurboBench
and keep each provider comparison bound to its own paired evidence bundle.

## Official TurboBench package results

On 2026-08-11, `supermariobrosnes-turbo==0.6.2` was compared on `beast-3`
against both `stable-retro-turbo==1.0.1.post37` and the original
`stable-retro==1.0.1`. Both comparisons passed every TurboBench validity gate
and produced official, independently verified result bundles.

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

| Comparison | Verified bundle ID |
| --- | --- |
| `supermariobrosnes-turbo==0.6.2` versus `stable-retro-turbo==1.0.1.post37` | `1d3508ab40a81377d7f8d0fc5270c55a85e9f935893a41518761597f75e63760` |
| `supermariobrosnes-turbo==0.6.2` versus `stable-retro==1.0.1` | `ca1853067f0a28c276dc97343981e3c806c468ea5b1cde094b40fc263cfa3e19` |

Each bundle contains 119 hash-bound artifacts and passed `turbobench verify`
without errors. The runs used TurboBench `1.0.0` from source commit
`d986efa72c81a7d0b5ea689ac37898d8fc38732f`, harness source SHA-256
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

Reproduce the package comparisons with the canonical ROM available to
TurboBench:

```bash
turbobench compare supermario/canonical-v1 \
  --left supermariobrosnes-turbo@0.6.2 \
  --right stable-retro-turbo@1.0.1.post37

turbobench compare supermario/canonical-v1 \
  --left supermariobrosnes-turbo@0.6.2 \
  --right stable-retro@1.0.1
```

### `beast-3`

| Component | Specification |
| --- | --- |
| CPU | AMD Ryzen 5 7600X (Zen 4), 6 physical cores / 12 threads, boost enabled, 5.457 GHz reported maximum |
| CPU cache | 384 KiB L1, 6 MiB L2, 32 MiB L3 |
| Memory | 65,396,760,576 bytes reported (60.9 GiB) |
| OS | Ubuntu 26.04 LTS, Linux 7.0.0-29-generic, x86_64 |
| Runtime | CPython 3.14.6; `numpy==2.4.2`; `gymnasium==1.2.2` |

The one-minute system load was below TurboBench's 6.0 threshold for both runs;
no diagnostic overrides were used.

## Earlier published results

SuperMarioBros-Nes-turbo was compared with upstream `stable-retro==1.0.1`
using seven alternating paired runs per environment count. Turbo versions are
listed with each machine's runtime specifications.

| Machine ID | Commit | Envs | Median SPS | Baseline median SPS | Median speedup | 95% bootstrap CI | Measured pairs |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `amd-ryzen-5-3600-6c` | `3131fc0c` | 1 | 5,403.6 | 406.0 | 13.10x | 13.07x–13.35x | 7 |
| `amd-ryzen-5-3600-6c` | `3131fc0c` | 16 | 28,234.4 | 1,841.4 | 15.36x | 15.22x–15.64x | 7 |
| `amd-ryzen-5-3600-6c` | `3131fc0c` | 32 | 35,104.4 | 1,956.3 | 17.95x | 17.80x–18.08x | 7 |
| `apple-m1-pro-8c` | `ae1171e` | 1 | 8,574.5 | 584.3 | 14.68x | 14.61x–14.78x | 7 |
| `apple-m1-pro-8c` | `ae1171e` | 16 | 36,675.3 | 2,608.5 | 13.79x | 13.45x–14.55x | 7 |
| `apple-m1-pro-8c` | `ae1171e` | 32 | 43,443.0 | 2,555.0 | 17.23x | 16.38x–17.86x | 7 |

SPS means environment steps per second. Each confidence interval is the 95%
bootstrap interval for the paired speedup ratio.

## Protocol

Both backends use the canonical public `step()` workload: canonical round-robin
`Level1-1` through `Level1-4` lane states, frame skip 4, no max-pooling, four
grayscale frames, a zeroed 32-row HUD, integer area resize to `84x84`, CHW
output, deterministic sampled actions, and manual terminal-lane resets. Runs
alternate backend order within each environment-count shape to reduce ordering
bias.

Reproduce the paired report from a clean checkout with:

```bash
make benchmark-report
```

Results are host-specific. Publishable comparisons require the canonical ROM,
the ROM-backed correctness checks, a clean commit, and the report's load
preflight to pass.

## Machine specifications

### `amd-ryzen-5-3600-6c`

| Component | Specification |
| --- | --- |
| System | ASUS desktop; ROG STRIX B550-F GAMING (WI-FI) motherboard, BIOS 2803 |
| CPU | AMD Ryzen 5 3600 (Zen 2), 6 physical cores / 12 threads, boost enabled, 4.208 GHz reported maximum |
| CPU cache | 384 KiB L1 (192 KiB data + 192 KiB instruction), 3 MiB L2, 32 MiB L3 |
| Memory | 32 GiB system RAM |
| Storage | 1 TB nominal WDC WDS100T2B0C-00PXH0 NVMe SSD (931.5 GiB reported) |
| OS | Ubuntu 26.04, Linux 7.0.0-27-generic, glibc 2.43, x86_64 |
| CPU frequency policy | `amd_pstate` active, `powersave` scaling governor |
| Runtime | CPython 3.13.14; `supermariobrosnes-turbo==0.3.2`, `stable-retro==1.0.1`, `numpy==2.5.0`, `gymnasium==1.3.0` |

The Ryzen results were measured from clean commit `3131fc0c` after the
ROM-backed parity checks passed. The session-start one-minute load was 0.36,
below the protocol limit of 4.0.

### `apple-m1-pro-8c`

| Component | Specification |
| --- | --- |
| System | 14-inch MacBook Pro (2021), `MacBookPro18,3`, model `MKGP3PO/A` |
| CPU | Apple M1 Pro, 8 physical/logical cores (6 performance + 2 efficiency), 8 threads |
| CPU cache | Performance cores: 192 KiB L1 instruction + 128 KiB L1 data per core, 12 MiB L2 per 3-core cluster; efficiency cores: 128 KiB L1 instruction + 64 KiB L1 data per core, 4 MiB shared L2 |
| GPU | Integrated 14-core Apple M1 Pro GPU |
| Memory | 16 GiB unified memory |
| Storage | 500.3 GB nominal internal APPLE SSD AP0512R (494.4 GB APFS capacity) |
| OS | macOS 26.5.2 build 25F84, Darwin 25.5.0, arm64 |
| CPU frequency policy | Apple-managed heterogeneous performance/efficiency scheduling; no user-selectable macOS scaling governor |
| Runtime | CPython 3.14.4; `supermariobrosnes-turbo==0.3.0`, `stable-retro==1.0.1`, `numpy==2.5.0`, `gymnasium==1.3.0` |

The Apple results were measured from clean commit `ae1171e`.
