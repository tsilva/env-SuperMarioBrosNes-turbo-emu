# Benchmarks

This page contains the detailed, host-specific benchmark results for
SuperMarioBros-Nes-turbo. The summary chart in the [README](README.md#benchmark)
links here for exact values, confidence intervals, protocol details, and machine
specifications.

![SuperMarioBros-Nes-turbo versus Stable Retro median environment throughput](media/benchmark-throughput.svg)

## Results

SuperMarioBros-Nes-turbo `0.3.0` was compared with upstream
`stable-retro==1.0.1` using seven alternating paired runs per environment count.

| Machine ID | Commit | Envs | Median SPS | Baseline median SPS | Median speedup | 95% bootstrap CI | Measured pairs |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `amd-ryzen-5-3600-6c` | `545131bf` | 1 | 5,458.4 | 411.4 | 13.27x | 12.85x–13.33x | 7 |
| `amd-ryzen-5-3600-6c` | `545131bf` | 16 | 28,591.6 | 1,847.7 | 15.49x | 15.30x–15.83x | 7 |
| `amd-ryzen-5-3600-6c` | `545131bf` | 32 | 35,778.4 | 1,958.0 | 18.27x | 18.20x–18.30x | 7 |
| `apple-m1-pro-8c` | `ae1171e` | 1 | 8,574.5 | 584.3 | 14.68x | 14.61x–14.78x | 7 |
| `apple-m1-pro-8c` | `ae1171e` | 16 | 36,675.3 | 2,608.5 | 13.79x | 13.45x–14.55x | 7 |
| `apple-m1-pro-8c` | `ae1171e` | 32 | 43,443.0 | 2,555.0 | 17.23x | 16.38x–17.86x | 7 |

SPS means environment steps per second. Each confidence interval is the 95%
bootstrap interval for the paired speedup ratio.

## Protocol

Both backends use the canonical public `step()` workload: frame skip 4, four
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
| Runtime | CPython 3.13.14; `supermariobrosnes-turbo==0.3.0`, `stable-retro==1.0.1`, `numpy==2.5.0`, `gymnasium==1.3.0` |

The Ryzen results were measured from clean commit `545131bf` after the
ROM-backed parity checks passed. The session-start one-minute load was 0.23,
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
