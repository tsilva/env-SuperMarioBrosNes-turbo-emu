#!/usr/bin/env python3
"""Regenerate the verified Stable Retro versus Turbo Mario promo video."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Sequence

from supermariobrosnes_turbo import resolve_required_rom_path


ENV_ID = "SuperMarioBros-Nes-v0"
HF_POLICY = "hf://tsilva/SuperMarioBros-Nes-v0_Level1-1"
ROM_SHA256 = "f61548fdf1670cffefcc4f0b7bdcdd9eaba0c226e3b74f8666071496988248de"
TURBO = "supermariobrosnes-turbo"
STABLE = "stable-retro"
VIDEO_FPS = 60
COMMON_INFO_KEYS = (
    "levelHi",
    "levelLo",
    "lives",
    "score",
    "time",
    "scrolling",
    "xscrollHi",
    "xscrollLo",
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: Sequence[str | os.PathLike[str]], *, cwd: Path, check: bool = True) -> subprocess.CompletedProcess[str]:
    rendered = [str(part) for part in command]
    print("+", " ".join(rendered), flush=True)
    return subprocess.run(rendered, cwd=cwd, text=True, check=check)


def load_gymrec(gymrec_root: Path, storage_root: Path, rom_path: Path):
    sys.path.insert(0, str(gymrec_root))
    try:
        import main as gymrec
    finally:
        sys.path.pop(0)
    gymrec._lazy_init()
    gymrec.CONFIG["storage"]["local_dir"] = str(storage_root)
    os.environ["ROMS_PATH"] = str(rom_path)
    return gymrec


def record_phase(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gymrec-root", type=Path, required=True)
    parser.add_argument("--storage-root", type=Path, required=True)
    parser.add_argument("--rom-path", type=Path, required=True)
    args = parser.parse_args(argv)
    args.storage_root.mkdir(parents=True, exist_ok=True)
    gymrec = load_gymrec(args.gymrec_root, args.storage_root, args.rom_path)
    old_argv = sys.argv
    try:
        sys.argv = [
            "gymrec",
            "record",
            HF_POLICY,
            "--backend",
            TURBO,
            "--roms-path",
            str(args.rom_path),
            "--headless",
            "--episodes",
            "1",
            "--deterministic",
            "--storage",
            "lossless-video",
            "--dry-run",
        ]
        asyncio.run(gymrec.main())
    finally:
        sys.argv = old_argv
    dataset_path = Path(gymrec.get_local_dataset_path(ENV_ID))
    if not dataset_path.is_dir():
        raise SystemExit(f"GymRec did not create {dataset_path}")
    return 0


def replay_phase(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gymrec-root", type=Path, required=True)
    parser.add_argument("--storage-root", type=Path, required=True)
    parser.add_argument("--rom-path", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    gymrec = load_gymrec(args.gymrec_root, args.storage_root, args.rom_path)
    np = gymrec.np
    dataset = gymrec.load_local_dataset(ENV_ID)
    if dataset is None:
        raise SystemExit("isolated GymRec dataset is missing")

    initial_level: tuple[int, int] | None = None
    transition_row: int | None = None
    for index, row in enumerate(dataset):
        if gymrec._is_terminal_action(row.get("actions")):
            break
        info = json.loads(row["infos"])
        level = (int(info["levelHi"]), int(info["levelLo"]))
        if initial_level is None:
            initial_level = level
        elif level != initial_level:
            transition_row = index
            break
    if initial_level is None or transition_row is None:
        raise SystemExit("recording did not contain a level transition")

    action_count = transition_row + 1
    rows = [dataset[index] for index in range(action_count + 1)]
    actions = [np.asarray(rows[index]["actions"], dtype=np.int8) for index in range(action_count)]
    seed = int(rows[0]["seed"])
    action_hash = hashlib.sha256(b"".join(action.tobytes() for action in actions)).hexdigest()
    expected_frame_hashes = [str(row["frame_sha256"]) for row in rows]
    args.work_dir.mkdir(parents=True, exist_ok=True)

    def make_env(backend: str):
        return gymrec.create_env(
            ENV_ID,
            backend=backend,
            stable_retro_state="Level1-1",
            human_recording=True,
        )

    def rgb(observation):
        frame = np.asarray(observation)
        if frame.shape != (224, 240, 3):
            raise RuntimeError(f"unexpected observation shape {frame.shape}")
        return np.ascontiguousarray(frame, dtype=np.uint8)

    def encode(backend: str, turbo_hashes: list[str] | None = None) -> dict[str, Any]:
        output = args.work_dir / ("turbo-replay.mp4" if backend == TURBO else "stable-retro-replay.mp4")
        env = make_env(backend)
        process: subprocess.Popen[bytes] | None = None
        try:
            observation, reset_info = env.reset(seed=seed)
            command = [
                shutil.which("ffmpeg") or "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                "240x224",
                "-r",
                str(VIDEO_FPS),
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "15",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(output),
            ]
            process = subprocess.Popen(command, stdin=subprocess.PIPE)
            frame_hashes: list[str] = []
            infos: list[dict[str, Any]] = []
            rewards: list[float] = []
            terminals: list[tuple[bool, bool]] = []

            def emit(frame) -> None:
                raw = frame.tobytes()
                frame_hashes.append(hashlib.sha256(raw).hexdigest())
                assert process is not None and process.stdin is not None
                process.stdin.write(raw)

            emit(rgb(observation))
            for step, action in enumerate(actions, start=1):
                observation, reward, terminated, truncated, info = env.step(action)
                emit(rgb(observation))
                rewards.append(float(reward))
                terminals.append((bool(terminated), bool(truncated)))
                infos.append(dict(info))
                if terminated or truncated:
                    raise RuntimeError(f"{backend} terminated unexpectedly at action {step}")
            assert process.stdin is not None
            process.stdin.close()
            process.stdin = None
            return_code = process.wait()
            if return_code:
                raise RuntimeError(f"ffmpeg failed for {backend} with exit {return_code}")
            transitions = [
                index + 1
                for index, info in enumerate(infos)
                if (int(info.get("levelHi", -1)), int(info.get("levelLo", -1))) != initial_level
            ]
            return {
                "video": str(output),
                "frame_hashes": frame_hashes,
                "infos": infos,
                "rewards": rewards,
                "terminals": terminals,
                "reset_info": dict(reset_info),
                "final_info": infos[-1],
                "first_level_transition_step": transitions[0] if transitions else None,
                "recorded_frame_match_count": sum(
                    actual == expected for actual, expected in zip(frame_hashes, expected_frame_hashes)
                ),
                "turbo_frame_match_count": (
                    len(frame_hashes)
                    if turbo_hashes is None
                    else sum(actual == expected for actual, expected in zip(frame_hashes, turbo_hashes))
                ),
            }
        finally:
            if process is not None and process.poll() is None:
                if process.stdin is not None:
                    process.stdin.close()
                process.terminate()
                process.wait()
            env.close()

    turbo = encode(TURBO)
    stable = encode(STABLE, turbo["frame_hashes"])
    semantic_mismatches = []
    for step, (turbo_info, stable_info) in enumerate(zip(turbo["infos"], stable["infos"]), start=1):
        fields = {
            key: [turbo_info.get(key), stable_info.get(key)]
            for key in COMMON_INFO_KEYS
            if turbo_info.get(key) != stable_info.get(key)
        }
        if fields:
            semantic_mismatches.append({"step": step, "fields": fields})
    reward_mismatches = [
        index + 1
        for index, values in enumerate(zip(turbo["rewards"], stable["rewards"]))
        if values[0] != values[1]
    ]
    terminal_mismatches = [
        index + 1
        for index, values in enumerate(zip(turbo["terminals"], stable["terminals"]))
        if values[0] != values[1]
    ]
    frame_count = action_count + 1
    checks = {
        "same_actions_by_construction": True,
        "both_reached_target_level": all(
            replay["first_level_transition_step"] == action_count for replay in (turbo, stable)
        ),
        "level_transition_step_equal": turbo["first_level_transition_step"] == stable["first_level_transition_step"],
        "pixel_equal_frames": stable["turbo_frame_match_count"],
        "recorded_frame_match_count": turbo["recorded_frame_match_count"],
        "total_frames": frame_count,
        "reward_mismatch_count": len(reward_mismatches),
        "terminal_mismatch_count": len(terminal_mismatches),
        "common_semantic_mismatch_count": len(semantic_mismatches),
    }
    valid = (
        checks["both_reached_target_level"]
        and checks["level_transition_step_equal"]
        and checks["pixel_equal_frames"] == frame_count
        and checks["recorded_frame_match_count"] == frame_count
        and not reward_mismatches
        and not terminal_mismatches
        and not semantic_mismatches
    )
    report = {
        "valid": valid,
        "logical_env_id": ENV_ID,
        "state": "Level1-1",
        "rom_path": str(args.rom_path),
        "rom_sha256": rows[0]["rom_sha256"],
        "seed": seed,
        "action_encoding": rows[0]["action_encoding"],
        "button_order": json.loads(rows[0]["nes_button_order"]),
        "frame_skip": int(rows[0]["frame_skip"]),
        "sticky_action_prob": float(rows[0]["sticky_action_prob"]),
        "action_count": action_count,
        "action_prefix_sha256": action_hash,
        "initial_level": list(initial_level),
        "video_fps": VIDEO_FPS,
        "video_frame_count": frame_count,
        "checks": checks,
        "reward_mismatch_steps": reward_mismatches,
        "terminal_mismatch_steps": terminal_mismatches,
        "first_semantic_mismatches": semantic_mismatches[:20],
        "backends": {
            TURBO: {
                "video": turbo["video"],
                "first_level_transition_step": turbo["first_level_transition_step"],
                "final_info": turbo["final_info"],
            },
            STABLE: {
                "video": stable["video"],
                "first_level_transition_step": stable["first_level_transition_step"],
                "final_info": stable["final_info"],
            },
        },
    }
    report_path = args.work_dir / "verification-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(report_path)
    if not valid:
        raise SystemExit("backend replay verification failed")
    return 0


def find_font(bold: bool) -> str:
    candidates = (
        [
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ]
        if bold
        else [
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
    )
    for candidate in candidates:
        if Path(candidate).is_file():
            return candidate
    raise SystemExit("could not find a supported Arial or DejaVu font")


def compose_phase(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--final-output", type=Path, required=True)
    args = parser.parse_args(argv)
    from PIL import Image, ImageDraw, ImageFont

    verification = json.loads((args.work_dir / "verification-report.json").read_text())
    aggregate = json.loads((args.benchmark_dir / "aggregate.json").read_text())
    if not verification.get("valid"):
        raise SystemExit("trajectory verification is not valid")
    if not aggregate.get("correctness", {}).get("passed"):
        raise SystemExit("ROM parity checks did not pass")
    result = next((item for item in aggregate.get("results", []) if item.get("num_envs") == 1), None)
    if not result or not result.get("claim_passed"):
        raise SystemExit("canonical shape-1 benchmark gates did not pass")
    speedup = float(result["pair_speedup"]["median"])
    turbo_sps = float(result["turbo_invocation_median_sps"]["median"])
    stable_sps = float(result["stable_retro_invocation_median_sps"]["median"])

    card = args.work_dir / "promo-frame.png"
    image = Image.new("RGB", (1280, 720), "#0b1020")
    draw = ImageDraw.Draw(image)
    bold_path = find_font(True)
    regular_path = find_font(False)

    def font(path: str, size: int):
        return ImageFont.truetype(path, size)

    def centered(text: str, y: int, selected_font, color: str, x0: int = 0, x1: int = 1280) -> None:
        box = draw.textbbox((0, 0), text, font=selected_font)
        x = x0 + ((x1 - x0) - (box[2] - box[0])) / 2
        draw.text((x, y), text, font=selected_font, fill=color)

    draw.rectangle((0, 0, 1280, 8), fill="#ffcf33")
    centered(
        f"SAME MARIO. SAME ACTIONS. {speedup:.2f}× MORE THROUGHPUT.",
        25,
        font(bold_path, 34),
        "#f8fafc",
    )
    centered("STABLE RETRO", 88, font(bold_path, 25), "#cbd5e1", 40, 600)
    centered("SUPERMARIOBROS-NES-TURBO", 88, font(bold_path, 25), "#ffcf33", 680, 1240)
    centered(f"{stable_sps:,.0f} SPS  •  1×", 119, font(bold_path, 23), "#94a3b8", 40, 600)
    centered(f"{turbo_sps:,.0f} SPS  •  {speedup:.2f}×", 119, font(bold_path, 23), "#fbbf24", 680, 1240)
    for x, color in ((80, "#64748b"), (720, "#f59e0b")):
        draw.rounded_rectangle((x - 8, 147, x + 488, 611), radius=9, fill="#020617", outline=color, width=4)
    draw.rectangle((638, 150, 642, 608), fill="#26324a")
    action_count = int(verification["action_count"])
    frame_count = int(verification["video_frame_count"])
    centered(
        f"One deterministic Level 1-1 controller trajectory  •  {action_count:,} actions",
        630,
        font(bold_path, 22),
        "#e2e8f0",
    )
    centered(
        f"Both reach Level 1-2 on action {action_count:,}  •  {frame_count:,} / {frame_count:,} raw frames identical",
        660,
        font(regular_path, 19),
        "#a7b3c7",
    )
    centered(
        f"Local matched run, {result['pair_count']} alternating pairs  •  gameplay time scaled by measured SPS ratio",
        689,
        font(regular_path, 14),
        "#64748b",
    )
    image.save(card)

    duration = frame_count / VIDEO_FPS
    temporary_output = args.work_dir / "mario-throughput-comparison.mp4"
    filter_graph = (
        f"[0:v]trim=duration={duration:.9f},setpts=PTS-STARTPTS[bg];"
        "[1:v]scale=480:448:flags=neighbor,setpts=PTS-STARTPTS[stable];"
        f"[2:v]scale=480:448:flags=neighbor,setpts=(PTS-STARTPTS)/{speedup:.12f},"
        "fps=60,tpad=stop_mode=clone:stop_duration=60[fast];"
        "[bg][stable]overlay=80:155:shortest=1[tmp];"
        "[tmp][fast]overlay=720:155:shortest=1[out]"
    )
    run(
        [
            shutil.which("ffmpeg") or "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-loop",
            "1",
            "-framerate",
            str(VIDEO_FPS),
            "-i",
            card,
            "-i",
            args.work_dir / "stable-retro-replay.mp4",
            "-i",
            args.work_dir / "turbo-replay.mp4",
            "-filter_complex",
            filter_graph,
            "-map",
            "[out]",
            "-an",
            "-r",
            str(VIDEO_FPS),
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "17",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            temporary_output,
        ],
        cwd=repo_root(),
    )
    probe = subprocess.run(
        [
            shutil.which("ffprobe") or "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,nb_frames,duration",
            "-of",
            "json",
            str(temporary_output),
        ],
        text=True,
        stdout=subprocess.PIPE,
        check=True,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    expected = {
        "width": 1280,
        "height": 720,
        "r_frame_rate": "60/1",
        "nb_frames": str(frame_count),
    }
    for key, value in expected.items():
        if stream.get(key) != value:
            raise SystemExit(f"final video {key}={stream.get(key)!r}, expected {value!r}")
    preview_frames = (
        min(30, frame_count - 1),
        min(180, frame_count - 1),
        max(frame_count - VIDEO_FPS, 0),
    )
    for index, preview_frame in enumerate(preview_frames, start=1):
        run(
            [
                shutil.which("ffmpeg") or "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                temporary_output,
                "-vf",
                f"select=eq(n\\,{preview_frame})",
                "-fps_mode",
                "vfr",
                "-frames:v",
                "1",
                args.work_dir / f"preview-{index}.png",
            ],
            cwd=repo_root(),
        )
    args.final_output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(temporary_output, args.final_output)
    manifest = {
        "final_output": str(args.final_output),
        "final_sha256": sha256_file(args.final_output),
        "final_bytes": args.final_output.stat().st_size,
        "video_stream": stream,
        "median_paired_speedup": speedup,
        "turbo_median_sps": turbo_sps,
        "stable_retro_median_sps": stable_sps,
        "benchmark_claim_passed": aggregate.get("claim_passed"),
        "benchmark_shape_claim_passed": result.get("claim_passed"),
        "trajectory_verification_valid": verification.get("valid"),
    }
    (args.work_dir / "final-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(args.final_output)
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gymrec-root", type=Path, default=root.parent / "gymrec")
    parser.add_argument("--rom-path", type=Path)
    parser.add_argument("--work-dir", type=Path, default=root / "media" / "mario-promo" / "work")
    parser.add_argument(
        "--final-output",
        type=Path,
        default=root / "media" / "mario-promo" / "mario-throughput-comparison.mp4",
    )
    parser.add_argument("--benchmark-pairs", type=int, default=5)
    parser.add_argument("--benchmark-steps", type=int, default=5000)
    parser.add_argument("--reuse-recording", action="store_true")
    parser.add_argument("--reuse-benchmark", action="store_true")
    return parser.parse_args(argv)


def orchestrate(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = repo_root()
    try:
        rom_path = resolve_required_rom_path(args.rom_path).resolve()
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if not rom_path.is_file():
        raise SystemExit(f"ROM does not exist: {rom_path}")
    actual_rom_hash = sha256_file(rom_path)
    if actual_rom_hash != ROM_SHA256:
        raise SystemExit(f"ROM SHA-256 {actual_rom_hash} does not match canonical {ROM_SHA256}")
    gymrec_root = args.gymrec_root.expanduser().resolve()
    gymrec_python = gymrec_root / ".venv" / "bin" / "python"
    if not (gymrec_root / "main.py").is_file() or not gymrec_python.is_file():
        raise SystemExit(f"GymRec checkout or virtualenv missing under {gymrec_root}")
    for executable in ("ffmpeg", "ffprobe", "make"):
        if shutil.which(executable) is None:
            raise SystemExit(f"required executable not found: {executable}")
    if args.benchmark_pairs < 5:
        raise SystemExit("--benchmark-pairs must be at least 5 for the displayed claim")
    if args.benchmark_steps <= 0:
        raise SystemExit("--benchmark-steps must be positive")

    work_dir = args.work_dir.expanduser().resolve()
    storage_root = work_dir / "gymrec-datasets"
    benchmark_dir = work_dir / "canonical-benchmark"
    if not args.reuse_recording and not args.reuse_benchmark and work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve()
    if not args.reuse_recording:
        if storage_root.exists():
            shutil.rmtree(storage_root)
        run(
            [
                gymrec_python,
                script,
                "_record",
                "--gymrec-root",
                gymrec_root,
                "--storage-root",
                storage_root,
                "--rom-path",
                rom_path,
            ],
            cwd=root,
        )
    elif not storage_root.is_dir():
        raise SystemExit(f"--reuse-recording requested but {storage_root} is missing")

    run(
        [
            gymrec_python,
            script,
            "_replay",
            "--gymrec-root",
            gymrec_root,
            "--storage-root",
            storage_root,
            "--rom-path",
            rom_path,
            "--work-dir",
            work_dir,
        ],
        cwd=root,
    )

    if not args.reuse_benchmark:
        if benchmark_dir.exists():
            shutil.rmtree(benchmark_dir)
        run(["make", "develop-release"], cwd=root)
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain=v1"],
                cwd=root,
                text=True,
                stdout=subprocess.PIPE,
                check=True,
            ).stdout.strip()
        )
        command: list[str | os.PathLike[str]] = [
            root / ".venv" / "bin" / "python",
            root / "scripts" / "benchmark_report.py",
            "--shapes",
            "1",
            "--pairs",
            str(args.benchmark_pairs),
            "--warmup-pairs",
            "1",
            "--steps",
            str(args.benchmark_steps),
            "--repeats",
            "3",
            "--warmup",
            "500",
            "--rom-path",
            rom_path,
            "--output-dir",
            benchmark_dir,
        ]
        if dirty:
            command.append("--allow-dirty")
        benchmark_process = run(command, cwd=root, check=False)
        if benchmark_process.returncode not in (0, 2) or not (benchmark_dir / "aggregate.json").is_file():
            raise SystemExit(f"canonical benchmark failed with exit {benchmark_process.returncode}")
    elif not (benchmark_dir / "aggregate.json").is_file():
        raise SystemExit(f"--reuse-benchmark requested but {benchmark_dir / 'aggregate.json'} is missing")

    run(
        [
            gymrec_python,
            script,
            "_compose",
            "--work-dir",
            work_dir,
            "--benchmark-dir",
            benchmark_dir,
            "--final-output",
            args.final_output.expanduser().resolve(),
        ],
        cwd=root,
    )
    return 0


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] in {"_record", "_replay", "_compose"}:
        phase = sys.argv[1]
        phase_args = sys.argv[2:]
        if phase == "_record":
            return record_phase(phase_args)
        if phase == "_replay":
            return replay_phase(phase_args)
        return compose_phase(phase_args)
    return orchestrate()


if __name__ == "__main__":
    raise SystemExit(main())
