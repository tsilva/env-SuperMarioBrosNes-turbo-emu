# Benchmarks

This page contains the official, host-specific TurboBench package results for
env-SuperMarioBrosNes-turbo-emu. The comparison remains bound to its paired evidence
bundle.

## Official TurboBench package results

On 2026-08-13,
[`env-supermariobrosnes-turbo-emu==0.6.4`](https://pypi.org/project/env-supermariobrosnes-turbo-emu/0.6.4/)
was compared using the [documented machine profile](#machine-profile) against
the original
[`stable-retro==1.0.1`](https://pypi.org/project/stable-retro/1.0.1/). The
comparison passed every TurboBench validity gate and produced an official,
verified result bundle published under an immutable Hugging Face tag.

| Envs | Matched provider | Turbo median SPS | Provider median SPS | Paired speedup | 95% paired bootstrap CI |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | `stable-retro==1.0.1` | 10,399.6 | 629.6 | 16.6064x | 16.4006x–16.6895x |
| 16 | `stable-retro==1.0.1` | 51,109.7 | 3,327.6 | 15.4219x | 15.1743x–15.4474x |
| 32 | `stable-retro==1.0.1` | 63,606.3 | 3,676.8 | 17.2701x | 17.1991x–17.3980x |

SPS means environment steps per second.

### Evidence

| Comparison | Immutable evidence | Bundle ID |
| --- | --- | --- |
| `env-supermariobrosnes-turbo-emu==0.6.4` versus `stable-retro==1.0.1` | [Hugging Face bundle](https://huggingface.co/datasets/tsilva/env-supermariobrosnes-turbo-emu-benchmarks/tree/v0.6.4/bundles/v0.6.4/vs-stable-retro-1.0.1/65eb59b9c84d0420483a051f09df08b57d334d817671cbac685a5cd1dd11fc21) | `65eb59b9c84d0420483a051f09df08b57d334d817671cbac685a5cd1dd11fc21` |

The 119 hash-bound artifacts are published in the immutable
[`v0.6.4` dataset tag](https://huggingface.co/datasets/tsilva/env-supermariobrosnes-turbo-emu-benchmarks/tree/v0.6.4)
at [commit `adaeb62c3c4c45aa6fa439d874945b786c97bc3f`](https://huggingface.co/datasets/tsilva/env-supermariobrosnes-turbo-emu-benchmarks/commit/adaeb62c3c4c45aa6fa439d874945b786c97bc3f).
A fresh download of that tag passed `turbobench verify` without errors using
[TurboBench 1.0.2](https://pypi.org/project/turbobench-cli/1.0.2/). This is
first-party, self-verified evidence, not an independent reproduction or author
authentication.

The benchmark itself used
[TurboBench 1.0.1](https://pypi.org/project/turbobench-cli/1.0.1/) from
[source commit `917c3d70b04b54779a05f94055a748ceda524b20`](https://github.com/tsilva/turbobench/commit/917c3d70b04b54779a05f94055a748ceda524b20),
harness source SHA-256
`78dce9803b7b3413668bb6d8168f661e34f974089a131c0dd34223c993535fc9`,
and immutable profile `supermario/canonical-v1` with profile SHA-256
`cbe40b5f203cd6fc5c397a216cc353e984685a52b04d01b7d550cd784b3935a1`.
The recorded provider artifact identities are
`7ce7e04110f4c993adbc1b15cc6d0ccaa7ddc12861aa31c6dcda4430746f8795`
for Turbo and
`1310a4c0f9a5d6c0dc99c4412318ffd34529083abad262bad20146d1bff2366a`
for Stable Retro. Their selected Linux wheel SHA-256 digests are respectively
`a37962b48504fbf509ed8785564229cc1b441e5f2a591b122a2adc1b59744e67`
and `4451c5f8209dbdbf343d29ab028ec7d40d28b9c7459ac4e632be990bc2f1eac3`.

Download and verify the exact publication with:

```bash
uv tool install --refresh --force \
  --exclude-newer-package turbobench-cli=2026-08-13T15:41:56Z \
  turbobench-cli==1.0.2
hf download tsilva/env-supermariobrosnes-turbo-emu-benchmarks \
  --type dataset --revision v0.6.4 --local-dir ./benchmark-evidence
turbobench verify \
  ./benchmark-evidence/bundles/v0.6.4/vs-stable-retro-1.0.1/65eb59b9c84d0420483a051f09df08b57d334d817671cbac685a5cd1dd11fc21
```

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

Install the benchmark's recorded CLI for reproduction with:

```bash
uv tool install \
  --exclude-newer-package turbobench-cli=2026-08-13T14:50:17Z \
  turbobench-cli==1.0.1
```

The recorded bundles retain the exact source identity shown above. Reproduce
them from that source revision, setting `TURBOBENCH_ASSET_ROOT` to a Stable
Retro-compatible data directory that contains `SuperMarioBros-Nes-v0/rom.nes`
and the canonical `Level1-1` through `Level1-4` state files:

```bash
git clone https://github.com/tsilva/turbobench.git
cd turbobench
git checkout --detach 917c3d70b04b54779a05f94055a748ceda524b20
uv sync --frozen --group dev

export TURBOBENCH_ASSET_ROOT=/absolute/path/to/stable-retro/data/stable
uv run turbobench doctor supermario/canonical-v1

uv run turbobench compare supermario/canonical-v1 \
  --left env-supermariobrosnes-turbo-emu@0.6.4 \
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

The recorded one-minute system load was 2.476, below TurboBench's 6.0 threshold;
no diagnostic overrides were used.
