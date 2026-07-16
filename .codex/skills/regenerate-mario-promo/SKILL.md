---
name: regenerate-mario-promo
description: Regenerate the verified Stable Retro versus SuperMarioBros-Nes-turbo Level 1-1 promotional video and README GIF preview. Use when asked to rebuild, refresh, rerun, or update the Mario throughput comparison, its exact shared GymRec trajectory, local benchmark numbers, verification evidence, final tracked MP4, or animated GIF preview.
---

# Regenerate Mario Promo

Produce one Git-eligible MP4 from an isolated GymRec recording, exact two-backend replay verification, and the repository's canonical matched benchmark.

## Run

1. Read the root `SPECS.md` with `$specs-author` and read `.codex/skills/autoresearch-speed/SKILL.md` plus its `SPECS.md` before benchmarking.
2. Confirm that `../gymrec` is the intended GymRec checkout and that the explicit `--rom-path` or Stable Retro-compatible `RETRO_DATA_PATH` tree resolves to the canonical ROM. Do not use or replace the user's normal `~/.gymrec` dataset.
3. Run from the repository root:

```bash
.venv/bin/python .codex/skills/regenerate-mario-promo/scripts/regenerate.py
```

4. Keep the command under observation. The five-pair Stable Retro baseline is intentionally much slower than Turbo and can take several minutes.
5. Inspect `media/mario-promo/work/verification-report.json`, `media/mario-promo/work/canonical-benchmark/report.md`, and preview frames. Report any failed parity, trajectory, load, sample-count, or speedup gate instead of presenting the video as verified.
6. Generate the README GIF preview from the final MP4.
7. Return `media/mario-promo/mario-throughput-comparison.mp4` and `media/mario-promo/mario-throughput-comparison.gif`. Do not stage or publish unless the user asks.

## GIF Preview

Generate the tracked GIF after the final MP4 is verified. Keep temporary palette files in the ignored work directory:

```bash
ffmpeg -y -loglevel error -i media/mario-promo/mario-throughput-comparison.mp4 \
  -vf 'crop=1280:548:0:80,fps=12,scale=640:-2:flags=lanczos,palettegen=max_colors=96:stats_mode=diff' \
  -frames:v 1 media/mario-promo/work/mario-throughput-comparison-palette.png

ffmpeg -y -loglevel error -i media/mario-promo/mario-throughput-comparison.mp4 \
  -i media/mario-promo/work/mario-throughput-comparison-palette.png \
  -filter_complex '[0:v]crop=1280:548:0:80,fps=12,scale=640:-2:flags=lanczos[video];[video][1:v]paletteuse=dither=none' \
  media/mario-promo/mario-throughput-comparison.gif
```

The crop intentionally removes the MP4's headline and verification footnotes so
the README preview shows only the two labeled gameplay panels. Confirm the GIF
is 640×274 at 12 FPS and inspect an early, mid-run, and final frame. Keep the
GIF Git-eligible because `README.md` embeds it; keep all other generated assets
ignored.

## Guarantees

- Record the public `hf://tsilva/SuperMarioBros-Nes-v0_Level1-1` deterministic policy through GymRec's `supermariobrosnes-turbo` backend.
- Isolate recordings under the ignored work directory; never mutate `~/.gymrec`.
- Cut the shared action stream at the first Level 1-1 to Level 1-2 transition.
- Require both backends to use the same ROM, seed, one-frame skip, zero sticky-action probability, action array, and `Level1-1` state.
- Require equal completion steps, raw observation frames, rewards, terminal flags, and common semantic info fields.
- Derive the displayed speed ratio from the repository's matched canonical shape-1 benchmark, excluding video encoding.
- Time-compress Turbo gameplay by the measured paired SPS ratio and hold its final frame while Stable Retro remains at 1x recorded gameplay speed.
- Abort before replacing the final MP4 when verification or benchmark correctness fails.

## Options

Use `--gymrec-root`, `--rom-path`, or `--final-output` only when the corresponding location differs. Use `--reuse-recording` or `--reuse-benchmark` only to resume a known-good interrupted run whose work artifacts were produced by this script.
