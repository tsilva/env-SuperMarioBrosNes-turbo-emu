use nes_turbo_nrom_core::{Cartridge, Profiler, VISIBLE_FRAME_HEIGHT, VISIBLE_FRAME_WIDTH};
use smb_turbo_driver::{NesEmulator, EXPECTED_SMB_ROM_SHA256};
use std::path::{Path, PathBuf};
use std::process::Command;

fn local_rom_path() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("ROM_PATH").map(PathBuf::from) {
        return path.is_file().then_some(path);
    }
    let env = std::fs::read_to_string(".env").ok()?;
    env.lines().find_map(|line| {
        let path = line.strip_prefix("ROM_PATH=")?.trim();
        let path = PathBuf::from(path);
        path.is_file().then_some(path)
    })
}

fn load_state(path: &Path) -> Vec<u8> {
    let output = Command::new("gzip")
        .args(["-dc"])
        .arg(path)
        .output()
        .expect("run gzip for canonical state");
    assert!(output.status.success(), "decompress {}", path.display());
    output.stdout
}

#[test]
fn enabled_and_disabled_smb_fast_paths_match_canonical_state_traces() {
    let Some(rom_path) = local_rom_path() else {
        eprintln!("skipping ROM-backed fast-path trace: ROM_PATH is unavailable");
        return;
    };
    let state_root = Path::new("python/supermariobrosnes_turbo/data/SuperMarioBros-Nes-v0");
    let cart = Cartridge::load_ines_for(rom_path, EXPECTED_SMB_ROM_SHA256).expect("load SMB ROM");

    for state_name in ["Level1-1", "Level1-2", "Level1-3", "Level1-4"] {
        let state_path = state_root.join(format!("{state_name}.state"));
        let state = load_state(&state_path);
        let mut fast = NesEmulator::new_with_options(cart.clone(), true);
        let mut interpreted = NesEmulator::new_with_options(cart.clone(), true);
        interpreted.disable_fast_paths();
        fast.load_fceu_state(&state).expect("load fast state");
        interpreted
            .load_fceu_state(&state)
            .expect("load interpreted state");

        let mut random = 0x9e37_79b9u32;
        let mut fast_profile = Profiler::new();
        let mut interpreted_profile = Profiler::new();
        for step in 0..256 {
            random ^= random << 13;
            random ^= random >> 17;
            random ^= random << 5;
            let action = random as u8;
            let (fast_reward, interpreted_reward) = if step % 7 == 0 {
                (
                    fast.step_frame_profiled(action, &mut fast_profile),
                    interpreted.step_frame_profiled(action, &mut interpreted_profile),
                )
            } else {
                (fast.step_frame(action), interpreted.step_frame(action))
            };
            assert_eq!(fast_reward, interpreted_reward, "{state_name} step {step}");
            assert_eq!(
                fast.is_done(),
                interpreted.is_done(),
                "{state_name} step {step}"
            );
            assert_eq!(
                fast.signals(),
                interpreted.signals(),
                "{state_name} step {step}"
            );
            assert_eq!(
                fast.snapshot(),
                interpreted.snapshot(),
                "{state_name} step {step}"
            );

            let mut fast_frame = vec![0; VISIBLE_FRAME_WIDTH * VISIBLE_FRAME_HEIGHT];
            let mut interpreted_frame = vec![0; fast_frame.len()];
            fast.write_gray_visible_frame_cropped(&mut fast_frame, 0, VISIBLE_FRAME_HEIGHT);
            interpreted.write_gray_visible_frame_cropped(
                &mut interpreted_frame,
                0,
                VISIBLE_FRAME_HEIGHT,
            );
            assert_eq!(fast_frame, interpreted_frame, "{state_name} step {step}");

            if fast.is_done() {
                break;
            }
        }

        fast.reset();
        interpreted.reset();
        assert_eq!(fast.signals(), interpreted.signals(), "{state_name} reset");
        assert_eq!(
            fast.snapshot(),
            interpreted.snapshot(),
            "{state_name} reset"
        );
    }
}
