use nes_turbo_nrom_core::{
    Cartridge, FastPathOutcome, FrameBudget, NromCore, NromGame, NromMachine, PpuTiming,
    ResumePolicy, StepResult, PPU_VBLANK_DOT,
};

#[derive(Clone)]
struct SyntheticGame;

#[derive(Clone)]
struct SyntheticTiming;

impl PpuTiming for SyntheticTiming {
    const SPRITE0_HIT_DOT: Option<usize> = None;
}

#[derive(Clone, Default)]
struct SyntheticFastPaths {
    increment_signal: bool,
}

impl NromGame for SyntheticGame {
    type Options = ();
    type Signals = u8;
    type StepContext = u8;
    type FastPaths = SyntheticFastPaths;
    type PpuTiming = SyntheticTiming;

    const EXPECTED_ROM_SHA256: &'static str = "synthetic-test-rom";

    fn detect_fast_paths(prg_rom: &[u8], _prg_addr_mask: usize) -> Self::FastPaths {
        SyntheticFastPaths {
            increment_signal: prg_rom.starts_with(&[0xea, 0xea]),
        }
    }

    fn synchronize(core: &mut NromCore<SyntheticTiming>, signals: &mut u8) {
        *signals = core.ram[0x10];
    }

    fn pre_step(signals: &u8) -> u8 {
        *signals
    }

    fn post_step(_options: &(), signals: &u8, before: u8) -> StepResult {
        StepResult {
            reward: signals.wrapping_sub(before) as f32,
            done: *signals == u8::MAX,
        }
    }

    fn dispatch_fast_path(
        core: &mut NromCore<SyntheticTiming>,
        fast_paths: &SyntheticFastPaths,
        budget: &mut FrameBudget,
    ) -> FastPathOutcome {
        if core.cpu.pc != 0x8000 || !fast_paths.increment_signal {
            return FastPathOutcome::Miss;
        }
        core.ram[0x10] = core.ram[0x10].wrapping_add(1);
        core.cpu.pc = core.cpu.pc.wrapping_add(1);
        budget.cpu_cycle_guard += 2;
        budget.pending_ppu_cycles += 6;
        FastPathOutcome::Applied(ResumePolicy::Continue)
    }
}

fn synthetic_cart(first_bytes: &[u8]) -> Cartridge {
    let mut prg_rom = vec![0xea; 32 * 1024];
    prg_rom[..first_bytes.len()].copy_from_slice(first_bytes);
    prg_rom[0x7ffc..0x7ffe].copy_from_slice(&0x8000u16.to_le_bytes());
    Cartridge {
        prg_rom,
        chr_rom: vec![0; 8 * 1024],
        vertical_mirroring: false,
    }
}

#[test]
fn external_driver_owns_signals_timing_and_signature_guarded_handler() {
    let cart = synthetic_cart(&[0xea, 0xea]);
    let fast_paths = SyntheticGame::detect_fast_paths(&cart.prg_rom, cart.prg_rom.len() - 1);
    let mut core = NromCore::<SyntheticTiming>::new(cart);
    core.reset();
    let mut budget = FrameBudget::default();

    assert_eq!(
        SyntheticGame::dispatch_fast_path(&mut core, &fast_paths, &mut budget),
        FastPathOutcome::Applied(ResumePolicy::Continue)
    );
    assert_eq!(core.ram[0x10], 0);
    assert_eq!(core.cpu.pc, 0x8001);
    assert_eq!(budget.pending_ppu_cycles, 6);

    assert_eq!(SyntheticTiming::SPRITE0_HIT_DOT, None);
    assert_eq!(core.ppu.cycles_until_next_event(), PPU_VBLANK_DOT);
}

#[test]
fn signature_miss_is_side_effect_free_and_interpreter_fallback_remains_available() {
    let cart = synthetic_cart(&[0xa9, 0x2a]);
    let fast_paths = SyntheticGame::detect_fast_paths(&cart.prg_rom, cart.prg_rom.len() - 1);
    let mut machine = NromMachine::<SyntheticGame>::new_with_options(cart, ());
    let before_pc = machine.core.cpu.pc;
    let before_ram = machine.core.ram;
    let mut budget = FrameBudget::default();

    assert_eq!(
        SyntheticGame::dispatch_fast_path(&mut machine.core, &fast_paths, &mut budget),
        FastPathOutcome::Miss
    );
    assert_eq!(machine.core.cpu.pc, before_pc);
    assert_eq!(machine.core.ram, before_ram);
    assert_eq!(budget, FrameBudget::default());

    machine.step_frame(0);
    assert_eq!(machine.core.cpu.a, 0x2a);
    assert!(machine.core.cpu.pc > 0x8002);
}
