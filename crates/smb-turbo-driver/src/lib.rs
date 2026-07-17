use nes_turbo_nrom_core::{
    FastPathOutcome, FrameBudget, NromCore, NromGame, NromMachine, PpuTiming, ResumePolicy,
    StepResult, CPU_CYCLES_PER_FRAME_GUARD, FLAG_C, FLAG_N, FLAG_U, FLAG_Z,
};
#[cfg(test)]
use nes_turbo_nrom_core::{FLAG_D, FLAG_I, FLAG_V, PPU_PRERENDER_DOT, PPU_VBLANK_DOT};

pub const EXPECTED_SMB_ROM_SHA256: &str =
    "f61548fdf1670cffefcc4f0b7bdcdd9eaba0c226e3b74f8666071496988248de";
const PPU_SPRITE0_DOT: usize = (22 + 30) * 341 + 1;

pub type SmbOptions = bool;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SmbSignals {
    pub x_pos: u16,
    pub coins: u8,
    pub level_hi: i16,
    pub level_lo: i16,
    pub lives: i16,
    pub score: u32,
    pub scrolling: i16,
    pub time: u16,
    pub xscroll_hi: u8,
    pub xscroll_lo: u8,
}

#[derive(Clone)]
pub struct SmbPpuTiming;

impl PpuTiming for SmbPpuTiming {
    const SPRITE0_HIT_DOT: Option<usize> = Some(PPU_SPRITE0_DOT);
}

#[derive(Clone)]
pub struct SuperMarioBros;

pub type NesEmulator = NromMachine<SuperMarioBros>;

const SMB_IDLE_JMP_PC: u16 = 0x8057;
const SMB_IDLE_JMP_PPU_CYCLES: usize = 9;
const SMB_SPRITE0_POLL_PC: u16 = 0x8150;
const SMB_SPRITE0_POLL_PPU_CYCLES: usize = 27;
const SMB_SPRITE0_POLL_EXIT_CPU_CYCLES: usize = 109;
const SMB_SPRITE0_POLL_EXIT_PPU_CYCLES: usize = SMB_SPRITE0_POLL_EXIT_CPU_CYCLES * 3;
const SMB_TIMER_CONTROL_LOOP_PC: u16 = 0x810e;
const SMB_OAM_CLEAR_PC: u16 = 0x8223;
const SMB_OAM_CLEAR_CPU_CYCLES: usize = 1017;
const SMB_OAM_CLEAR_PPU_CYCLES: usize = SMB_OAM_CLEAR_CPU_CYCLES * 3;
const SMB_SCROLL_SLOT_LOOP_PC: u16 = 0x81cf;
const SMB_CONTROLLER_READ_PC: u16 = 0x8e6a;
const SMB_CONTROLLER_READ_TAKEN_CPU_CYCLES: usize = 257;
const SMB_CONTROLLER_READ_NOT_TAKEN_CPU_CYCLES: usize = 258;
const SMB_DIGIT_MATH_LOOP_PC: u16 = 0x8fa1;
const SMB_BOUNDING_BOX_NIBBLE_PC: u16 = 0x9be1;
const SMB_BOUNDING_BOX_NIBBLE_CPU_CYCLES: usize = 41;
const SMB_BOUNDING_BOX_HELPER_PC: u16 = 0xe3f0;
const SMB_BOUNDING_BOX_HELPER_MAX_CPU_CYCLES: usize = 160;
const SMB_OFFSCREEN_BITS_SUBS_PC: u16 = 0xf1c0;
const SMB_OFFSCREEN_BITS_SUBS_MAX_CPU_CYCLES: usize = 660;
const SMB_RELATIVE_POSITION_HELPER_PC: u16 = 0xf26d;
const SMB_DRAW_SPRITE_OBJECT_PC: u16 = 0xf282;
fn prg_rom_supports_smb_idle_jmp(prg_rom: &[u8], mask: usize) -> bool {
    let offset = (SMB_IDLE_JMP_PC as usize).wrapping_sub(0x8000) & mask;
    prg_rom.get(offset) == Some(&0x4c)
        && prg_rom.get((offset + 1) & mask) == Some(&((SMB_IDLE_JMP_PC & 0xff) as u8))
        && prg_rom.get((offset + 2) & mask) == Some(&((SMB_IDLE_JMP_PC >> 8) as u8))
}

fn prg_rom_supports_smb_sprite0_poll(prg_rom: &[u8], mask: usize) -> bool {
    let offset = (SMB_SPRITE0_POLL_PC as usize).wrapping_sub(0x8000) & mask;
    let expected = [0xad, 0x02, 0x20, 0x29, 0x40, 0xf0, 0xf9];
    expected
        .iter()
        .enumerate()
        .all(|(index, byte)| prg_rom.get((offset + index) & mask) == Some(byte))
}

fn prg_rom_supports_smb_sprite0_poll_exit(prg_rom: &[u8], mask: usize) -> bool {
    let offset = (SMB_SPRITE0_POLL_PC as usize).wrapping_sub(0x8000) & mask;
    let expected = [
        0xad, 0x02, 0x20, 0x29, 0x40, 0xf0, 0xf9, 0xa0, 0x14, 0x88, 0xd0, 0xfd,
    ];
    expected
        .iter()
        .enumerate()
        .all(|(index, byte)| prg_rom.get((offset + index) & mask) == Some(byte))
}

fn prg_rom_supports_smb_timer_control_loop(prg_rom: &[u8], mask: usize) -> bool {
    let offset = (SMB_TIMER_CONTROL_LOOP_PC as usize).wrapping_sub(0x8000) & mask;
    let expected = [
        0xbd, 0x80, 0x07, 0xf0, 0x03, 0xde, 0x80, 0x07, 0xca, 0x10, 0xf5,
    ];
    expected
        .iter()
        .enumerate()
        .all(|(index, byte)| prg_rom.get((offset + index) & mask) == Some(byte))
}

fn prg_rom_supports_smb_oam_clear(prg_rom: &[u8], mask: usize) -> bool {
    let offset = (SMB_OAM_CLEAR_PC as usize).wrapping_sub(0x8000) & mask;
    let expected = [
        0xa0, 0x04, 0xa9, 0xf8, 0x99, 0x00, 0x02, 0xc8, 0xc8, 0xc8, 0xc8, 0xd0, 0xf7, 0x60,
    ];
    expected
        .iter()
        .enumerate()
        .all(|(index, byte)| prg_rom.get((offset + index) & mask) == Some(byte))
}

fn prg_rom_supports_smb_scroll_slot_loop(prg_rom: &[u8], mask: usize) -> bool {
    let offset = (SMB_SCROLL_SLOT_LOOP_PC as usize).wrapping_sub(0x8000) & mask;
    let expected = [
        0xbd, 0xe4, 0x06, 0xc5, 0x00, 0x90, 0x0f, 0xac, 0xe0, 0x06, 0x18, 0x79, 0xe1, 0x06, 0x90,
        0x03, 0x18, 0x65, 0x00, 0x9d, 0xe4, 0x06, 0xca, 0x10, 0xe7,
    ];
    expected
        .iter()
        .enumerate()
        .all(|(index, byte)| prg_rom.get((offset + index) & mask) == Some(byte))
}

fn prg_rom_supports_smb_controller_read(prg_rom: &[u8], mask: usize) -> bool {
    let offset = (SMB_CONTROLLER_READ_PC as usize).wrapping_sub(0x8000) & mask;
    let expected = [
        0xa0, 0x08, 0x48, 0xbd, 0x16, 0x40, 0x85, 0x00, 0x4a, 0x05, 0x00, 0x4a, 0x68, 0x2a, 0x88,
        0xd0, 0xf1, 0x9d, 0xfc, 0x06, 0x48, 0x29, 0x30, 0x3d, 0x4a, 0x07, 0xf0, 0x07, 0x68, 0x29,
        0xcf, 0x9d, 0xfc, 0x06, 0x60, 0x68, 0x9d, 0x4a, 0x07, 0x60,
    ];
    expected
        .iter()
        .enumerate()
        .all(|(index, byte)| prg_rom.get((offset + index) & mask) == Some(byte))
}

fn prg_rom_supports_smb_digit_math_loop(prg_rom: &[u8], mask: usize) -> bool {
    let offset = (SMB_DIGIT_MATH_LOOP_PC as usize).wrapping_sub(0x8000) & mask;
    let expected = [
        0xbd, 0xdd, 0x07, 0xf9, 0xd7, 0x07, 0xca, 0x88, 0x10, 0xf6, 0x90, 0x0e, 0xe8, 0xc8, 0xbd,
        0xdd, 0x07, 0x99, 0xd7, 0x07, 0xe8, 0xc8, 0xc0, 0x06, 0x90, 0xf4, 0x60,
    ];
    expected
        .iter()
        .enumerate()
        .all(|(index, byte)| prg_rom.get((offset + index) & mask) == Some(byte))
}

fn prg_rom_supports_smb_bounding_box_nibble(prg_rom: &[u8], mask: usize) -> bool {
    let offset = (SMB_BOUNDING_BOX_NIBBLE_PC as usize).wrapping_sub(0x8000) & mask;
    let expected = [
        0x48, 0x4a, 0x4a, 0x4a, 0x4a, 0xa8, 0xb9, 0xdf, 0x9b, 0x85, 0x07, 0x68, 0x29, 0x0f, 0x18,
        0x79, 0xdd, 0x9b, 0x85, 0x06, 0x60,
    ];
    expected
        .iter()
        .enumerate()
        .all(|(index, byte)| prg_rom.get((offset + index) & mask) == Some(byte))
}

fn prg_rom_supports_smb_bounding_box_helper(prg_rom: &[u8], mask: usize) -> bool {
    let offset = (SMB_BOUNDING_BOX_HELPER_PC as usize).wrapping_sub(0x8000) & mask;
    let expected = [
        0x48, 0x84, 0x04, 0xb9, 0xb0, 0xe3, 0x18, 0x75, 0x86, 0x85, 0x05, 0xb5, 0x6d, 0x69, 0x00,
        0x29, 0x01, 0x4a, 0x05, 0x05, 0x6a, 0x4a, 0x4a, 0x4a, 0x20, 0xe1, 0x9b, 0xa4, 0x04, 0xb5,
        0xce, 0x18, 0x79, 0xcc, 0xe3, 0x29, 0xf0, 0x38, 0xe9, 0x20, 0x85, 0x02, 0xa8, 0xb1, 0x06,
        0x85, 0x03, 0xa4, 0x04, 0x68, 0xd0, 0x05, 0xb5, 0xce, 0x4c, 0x2b, 0xe4, 0xb5, 0x86, 0x29,
        0x0f, 0x85, 0x04, 0xa5, 0x03, 0x60,
    ];
    expected
        .iter()
        .enumerate()
        .all(|(index, byte)| prg_rom.get((offset + index) & mask) == Some(byte))
        && prg_rom_supports_smb_bounding_box_nibble(prg_rom, mask)
}

fn prg_rom_supports_smb_relative_position_helper(prg_rom: &[u8], mask: usize) -> bool {
    let offset = (SMB_RELATIVE_POSITION_HELPER_PC as usize).wrapping_sub(0x8000) & mask;
    let expected = [
        0x85, 0x05, 0xa5, 0x07, 0xc5, 0x06, 0xb0, 0x0c, 0x4a, 0x4a, 0x4a, 0x29, 0x07, 0xc0, 0x01,
        0xb0, 0x02, 0x65, 0x05, 0xaa, 0x60,
    ];
    expected
        .iter()
        .enumerate()
        .all(|(index, byte)| prg_rom.get((offset + index) & mask) == Some(byte))
}

const SMB_OFFSCREEN_BITS_SEGMENTS: &[(u16, &[u8])] = &[
    (
        0xf1c0,
        &[
            0x98, 0x48, 0x20, 0xd7, 0xf1, 0x0a, 0x0a, 0x0a, 0x0a, 0x05, 0x00, 0x85, 0x00, 0x68,
            0xa8, 0xa5, 0x00, 0x99, 0xd0, 0x03, 0xa6, 0x08, 0x60,
        ],
    ),
    (
        0xf1d7,
        &[
            0x20, 0xf6, 0xf1, 0x4a, 0x4a, 0x4a, 0x4a, 0x85, 0x00, 0x4c, 0x39, 0xf2,
        ],
    ),
    (
        0xf1e3,
        &[
            0x7f, 0x3f, 0x1f, 0x0f, 0x07, 0x03, 0x01, 0x00, 0x80, 0xc0, 0xe0, 0xf0, 0xf8, 0xfc,
            0xfe, 0xff,
        ],
    ),
    (0xf1f3, &[0x07, 0x0f, 0x07]),
    (
        0xf1f6,
        &[
            0x86, 0x04, 0xa0, 0x01, 0xb9, 0x1c, 0x07, 0x38, 0xf5, 0x86, 0x85, 0x07, 0xb9, 0x1a,
            0x07, 0xf5, 0x6d, 0xbe, 0xf3, 0xf1, 0xc9, 0x00, 0x30, 0x10, 0xbe, 0xf4, 0xf1, 0xc9,
            0x01, 0x10, 0x09, 0xa9, 0x38, 0x85, 0x06, 0xa9, 0x08, 0x20, 0x6d, 0xf2, 0xbd, 0xe3,
            0xf1, 0xa6, 0x04, 0xc9, 0x00, 0xd0, 0x03, 0x88, 0x10, 0xd0, 0x60,
        ],
    ),
    (
        0xf22b,
        &[0x00, 0x08, 0x0c, 0x0e, 0x0f, 0x07, 0x03, 0x01, 0x00],
    ),
    (0xf234, &[0x04, 0x00, 0x04]),
    (0xf237, &[0xff, 0x00]),
    (
        0xf239,
        &[
            0x86, 0x04, 0xa0, 0x01, 0xb9, 0x37, 0xf2, 0x38, 0xf5, 0xce, 0x85, 0x07, 0xa9, 0x01,
            0xf5, 0xb5, 0xbe, 0x34, 0xf2, 0xc9, 0x00, 0x30, 0x10, 0xbe, 0x35, 0xf2, 0xc9, 0x01,
            0x10, 0x09, 0xa9, 0x20, 0x85, 0x06, 0xa9, 0x04, 0x20, 0x6d, 0xf2, 0xbd, 0x2b, 0xf2,
            0xa6, 0x04, 0xc9, 0x00, 0xd0, 0x03, 0x88, 0x10, 0xd1, 0x60,
        ],
    ),
];

fn prg_rom_supports_smb_offscreen_bits_subs(prg_rom: &[u8], mask: usize) -> bool {
    SMB_OFFSCREEN_BITS_SEGMENTS
        .iter()
        .all(|(address, expected)| {
            let offset = (*address as usize).wrapping_sub(0x8000) & mask;
            expected
                .iter()
                .enumerate()
                .all(|(index, byte)| prg_rom.get((offset + index) & mask) == Some(byte))
        })
        && prg_rom_supports_smb_relative_position_helper(prg_rom, mask)
}

fn prg_rom_supports_smb_draw_sprite_object(prg_rom: &[u8], mask: usize) -> bool {
    let offset = (SMB_DRAW_SPRITE_OBJECT_PC as usize).wrapping_sub(0x8000) & mask;
    let expected = [
        0xa5, 0x03, 0x4a, 0x4a, 0xa5, 0x00, 0x90, 0x0c, 0x99, 0x05, 0x02, 0xa5, 0x01, 0x99, 0x01,
        0x02, 0xa9, 0x40, 0xd0, 0x0a, 0x99, 0x01, 0x02, 0xa5, 0x01, 0x99, 0x05, 0x02, 0xa9, 0x00,
        0x05, 0x04, 0x99, 0x02, 0x02, 0x99, 0x06, 0x02, 0xa5, 0x02, 0x99, 0x00, 0x02, 0x99, 0x04,
        0x02, 0xa5, 0x05, 0x99, 0x03, 0x02, 0x18, 0x69, 0x08, 0x99, 0x07, 0x02, 0xa5, 0x02, 0x18,
        0x69, 0x08, 0x85, 0x02, 0x98, 0x18, 0x69, 0x08, 0xa8, 0xe8, 0xe8, 0x60,
    ];
    expected
        .iter()
        .enumerate()
        .all(|(index, byte)| prg_rom.get((offset + index) & mask) == Some(byte))
}

macro_rules! smb_fast_paths {
    (
        capabilities { $( $capability:ident => $detector:ident, [ $( $dependency:ident ),* ]; )* }
        sites { $( $pc:ident => $site_capability:ident => $handler:ident => $policy:ident; )* }
    ) => {
        #[derive(Clone, Debug, Default, PartialEq, Eq)]
        pub struct SmbFastPaths { $( pub $capability: bool, )* }

        impl SmbFastPaths {
            fn detect(prg_rom: &[u8], mask: usize) -> Self {
                let mut capabilities = Self {
                    $( $capability: $detector(prg_rom, mask), )*
                };
                $( capabilities.$capability &= true $( && capabilities.$dependency )*; )*
                capabilities
            }

            fn dispatch(
                &self,
                core: &mut NromCore<SmbPpuTiming>,
                budget: &mut FrameBudget,
            ) -> FastPathOutcome {
                match core.cpu.pc {
                    $( $pc if self.$site_capability => {
                        if SuperMarioBros::$handler(
                            core,
                            self,
                            &mut budget.cpu_cycle_guard,
                            &mut budget.pending_ppu_cycles,
                        ) {
                            FastPathOutcome::Applied(ResumePolicy::$policy)
                        } else {
                            FastPathOutcome::Miss
                        }
                    } )*
                    _ => FastPathOutcome::Miss,
                }
            }
        }
    };
}

smb_fast_paths! {
    capabilities {
        idle_jmp => prg_rom_supports_smb_idle_jmp, [];
        sprite0_poll => prg_rom_supports_smb_sprite0_poll, [];
        sprite0_poll_exit => prg_rom_supports_smb_sprite0_poll_exit, [sprite0_poll];
        timer_control_loop => prg_rom_supports_smb_timer_control_loop, [];
        oam_clear => prg_rom_supports_smb_oam_clear, [];
        scroll_slot_loop => prg_rom_supports_smb_scroll_slot_loop, [];
        controller_read => prg_rom_supports_smb_controller_read, [];
        digit_math_loop => prg_rom_supports_smb_digit_math_loop, [];
        bounding_box_nibble => prg_rom_supports_smb_bounding_box_nibble, [];
        bounding_box_helper => prg_rom_supports_smb_bounding_box_helper, [bounding_box_nibble];
        offscreen_bits_subs => prg_rom_supports_smb_offscreen_bits_subs, [];
        relative_position_helper => prg_rom_supports_smb_relative_position_helper, [];
        draw_sprite_object => prg_rom_supports_smb_draw_sprite_object, [];
    }
    sites {
        SMB_IDLE_JMP_PC => idle_jmp => try_fast_forward_idle_jmp => FlushNow;
        SMB_SPRITE0_POLL_PC => sprite0_poll => try_fast_forward_sprite0_poll => FlushNow;
        SMB_TIMER_CONTROL_LOOP_PC => timer_control_loop => try_fast_forward_timer_control_loop => Continue;
        SMB_OAM_CLEAR_PC => oam_clear => try_fast_forward_oam_clear => FlushIfEventDue;
        SMB_SCROLL_SLOT_LOOP_PC => scroll_slot_loop => try_fast_forward_scroll_slot_loop => Continue;
        SMB_CONTROLLER_READ_PC => controller_read => try_fast_forward_controller_read => Continue;
        SMB_DIGIT_MATH_LOOP_PC => digit_math_loop => try_fast_forward_digit_math_loop => Continue;
        SMB_BOUNDING_BOX_HELPER_PC => bounding_box_helper => try_fast_forward_bounding_box_helper => Continue;
        SMB_BOUNDING_BOX_NIBBLE_PC => bounding_box_nibble => try_fast_forward_bounding_box_nibble => Continue;
        SMB_OFFSCREEN_BITS_SUBS_PC => offscreen_bits_subs => try_fast_forward_offscreen_bits_subs => Continue;
        SMB_RELATIVE_POSITION_HELPER_PC => relative_position_helper => try_fast_forward_relative_position_helper => Continue;
        SMB_DRAW_SPRITE_OBJECT_PC => draw_sprite_object => try_fast_forward_draw_sprite_object => Continue;
    }
}

impl SuperMarioBros {
    fn try_fast_forward_idle_jmp(
        core: &mut NromCore<SmbPpuTiming>,
        capabilities: &SmbFastPaths,
        cpu_cycle_guard: &mut usize,
        pending_ppu_cycles: &mut usize,
    ) -> bool {
        if core.cpu.pc != SMB_IDLE_JMP_PC || !capabilities.idle_jmp {
            return false;
        }

        let ppu_cycles_until_event = core.ppu.cycles_until_next_event();
        let remaining = ppu_cycles_until_event.saturating_sub(*pending_ppu_cycles);
        let jumps = remaining.div_ceil(SMB_IDLE_JMP_PPU_CYCLES).max(1);
        *cpu_cycle_guard += jumps * 3;
        *pending_ppu_cycles += jumps * SMB_IDLE_JMP_PPU_CYCLES;
        *pending_ppu_cycles >= ppu_cycles_until_event
            || *cpu_cycle_guard >= CPU_CYCLES_PER_FRAME_GUARD
    }

    #[inline(never)]
    fn try_fast_forward_sprite0_poll(
        core: &mut NromCore<SmbPpuTiming>,
        capabilities: &SmbFastPaths,
        cpu_cycle_guard: &mut usize,
        pending_ppu_cycles: &mut usize,
    ) -> bool {
        if core.cpu.pc != SMB_SPRITE0_POLL_PC || !capabilities.sprite0_poll {
            return false;
        }

        if core.ppu.sprite0_hit_set() {
            return Self::try_fast_forward_sprite0_poll_exit(
                core,
                capabilities,
                cpu_cycle_guard,
                pending_ppu_cycles,
            );
        }

        let ppu_cycles_until_event = core.ppu.cycles_until_next_event();
        let remaining = ppu_cycles_until_event.saturating_sub(*pending_ppu_cycles);
        let loops = remaining.div_ceil(SMB_SPRITE0_POLL_PPU_CYCLES).max(1);
        core.cpu.a = 0;
        core.set_zn(0);
        *cpu_cycle_guard += loops * 9;
        *pending_ppu_cycles += loops * SMB_SPRITE0_POLL_PPU_CYCLES;
        *pending_ppu_cycles >= ppu_cycles_until_event
            || *cpu_cycle_guard >= CPU_CYCLES_PER_FRAME_GUARD
    }

    #[inline(never)]
    fn try_fast_forward_sprite0_poll_exit(
        core: &mut NromCore<SmbPpuTiming>,
        capabilities: &SmbFastPaths,
        cpu_cycle_guard: &mut usize,
        pending_ppu_cycles: &mut usize,
    ) -> bool {
        if !capabilities.sprite0_poll_exit {
            return false;
        }

        let ppu_cycles_until_event = core.ppu.cycles_until_next_event();
        let remaining = ppu_cycles_until_event.saturating_sub(*pending_ppu_cycles);
        if SMB_SPRITE0_POLL_EXIT_PPU_CYCLES > remaining {
            return false;
        }

        let status = core.ppu.cpu_read_register(0x2002);
        core.cpu.a = status & 0x40;
        core.set_zn(core.cpu.a);
        core.cpu.y = 0;
        core.set_zn(0);
        core.cpu.pc = SMB_SPRITE0_POLL_PC + 12;
        *cpu_cycle_guard += SMB_SPRITE0_POLL_EXIT_CPU_CYCLES;
        *pending_ppu_cycles += SMB_SPRITE0_POLL_EXIT_PPU_CYCLES;
        true
    }

    #[inline(never)]
    fn try_fast_forward_timer_control_loop(
        core: &mut NromCore<SmbPpuTiming>,
        capabilities: &SmbFastPaths,
        cpu_cycle_guard: &mut usize,
        pending_ppu_cycles: &mut usize,
    ) -> bool {
        if core.cpu.pc != SMB_TIMER_CONTROL_LOOP_PC
            || !capabilities.timer_control_loop
            || core.cpu.x & 0x80 != 0
        {
            return false;
        }

        let (cycles, last_a) = Self::timer_control_loop_cycles_and_last_a(core, core.cpu.x);
        let ppu_cycles = cycles * 3;
        let remaining = core
            .ppu
            .cycles_until_next_event()
            .saturating_sub(*pending_ppu_cycles);
        if ppu_cycles >= remaining {
            return false;
        }

        let mut x = core.cpu.x;
        loop {
            let timer = &mut core.ram[0x0780 + x as usize];
            if *timer != 0 {
                *timer = timer.wrapping_sub(1);
            }
            x = x.wrapping_sub(1);
            if x & 0x80 != 0 {
                break;
            }
        }
        core.cpu.a = last_a;
        core.cpu.x = x;
        core.set_zn(x);
        core.cpu.pc = SMB_TIMER_CONTROL_LOOP_PC + 11;
        *cpu_cycle_guard += cycles;
        *pending_ppu_cycles += ppu_cycles;
        true
    }

    fn timer_control_loop_cycles_and_last_a(
        core: &mut NromCore<SmbPpuTiming>,
        start_x: u8,
    ) -> (usize, u8) {
        let mut cycles = 0usize;
        let mut x = start_x;
        let last_a = loop {
            let value = core.ram[0x0780 + x as usize];
            cycles += if value == 0 { 7 } else { 13 };
            x = x.wrapping_sub(1);
            cycles += if x & 0x80 == 0 { 5 } else { 4 };
            if x & 0x80 != 0 {
                break value;
            }
        };
        (cycles, last_a)
    }

    #[inline(never)]
    fn try_fast_forward_oam_clear(
        core: &mut NromCore<SmbPpuTiming>,
        capabilities: &SmbFastPaths,
        cpu_cycle_guard: &mut usize,
        pending_ppu_cycles: &mut usize,
    ) -> bool {
        if core.cpu.pc != SMB_OAM_CLEAR_PC || !capabilities.oam_clear {
            return false;
        }

        let ppu_cycles_until_event = core.ppu.cycles_until_next_event();
        let remaining = ppu_cycles_until_event.saturating_sub(*pending_ppu_cycles);
        if SMB_OAM_CLEAR_PPU_CYCLES > remaining {
            return false;
        }

        for offset in (4usize..=252).step_by(4) {
            core.ram[0x0200 + offset] = 0xf8;
        }
        core.cpu.a = 0xf8;
        core.cpu.y = 0;
        core.set_zn(0);
        core.cpu.pc = core.pop_u16().wrapping_add(1);
        *cpu_cycle_guard += SMB_OAM_CLEAR_CPU_CYCLES;
        *pending_ppu_cycles += SMB_OAM_CLEAR_PPU_CYCLES;
        true
    }

    #[inline(never)]
    fn try_fast_forward_scroll_slot_loop(
        core: &mut NromCore<SmbPpuTiming>,
        capabilities: &SmbFastPaths,
        cpu_cycle_guard: &mut usize,
        pending_ppu_cycles: &mut usize,
    ) -> bool {
        if core.cpu.pc != SMB_SCROLL_SLOT_LOOP_PC || !capabilities.scroll_slot_loop {
            return false;
        }

        let iterations = if core.cpu.x < 0x80 {
            core.cpu.x as usize + 1
        } else {
            1
        };
        let max_ppu_cycles = iterations * 40 * 3;
        let remaining = core
            .ppu
            .cycles_until_next_event()
            .saturating_sub(*pending_ppu_cycles);
        if max_ppu_cycles > remaining {
            return false;
        }

        let mut cycles = 0usize;
        loop {
            let x = core.cpu.x;
            let slot_addr = 0x06e4usize + x as usize;
            core.cpu.a = core.ram_read(slot_addr);
            cycles += 4;

            let threshold = core.ram_read(0);
            core.cmp(core.cpu.a, threshold);
            cycles += 3;
            if !core.flag(FLAG_C) {
                cycles += 3;
            } else {
                cycles += 2;
                core.cpu.y = core.ram_read(0x06e0);
                core.set_zn(core.cpu.y);
                cycles += 4;
                core.set_flag(FLAG_C, false);
                cycles += 2;
                let add_addr = 0x06e1u16.wrapping_add(core.cpu.y as u16);
                let addend = core.cpu_read(add_addr);
                core.adc(addend);
                cycles += 4 + page_crossed(0x06e1, add_addr) as usize;
                if !core.flag(FLAG_C) {
                    cycles += 3;
                } else {
                    cycles += 2;
                    core.set_flag(FLAG_C, false);
                    cycles += 2;
                    core.adc(threshold);
                    cycles += 3;
                }
                core.ram_write(slot_addr, core.cpu.a);
                cycles += 5;
            }

            core.cpu.x = x.wrapping_sub(1);
            core.set_zn(core.cpu.x);
            cycles += 2;
            if core.flag(FLAG_N) {
                cycles += 2;
                break;
            }
            cycles += 3;
        }

        core.cpu.pc = 0x81e8;
        *cpu_cycle_guard += cycles;
        *pending_ppu_cycles += cycles * 3;
        true
    }

    #[inline(never)]
    fn try_fast_forward_controller_read(
        core: &mut NromCore<SmbPpuTiming>,
        capabilities: &SmbFastPaths,
        cpu_cycle_guard: &mut usize,
        pending_ppu_cycles: &mut usize,
    ) -> bool {
        if core.cpu.pc != SMB_CONTROLLER_READ_PC || !capabilities.controller_read || core.cpu.x > 1
        {
            return false;
        }

        let x = core.cpu.x as usize;
        let mut a = core.cpu.a;
        let mut last_read = 0u8;
        let mut controller_shift = core.controller_shift;
        let mut carry = core.flag(FLAG_C);
        for _ in 0..8 {
            let value = if x == 0 {
                let bit = if core.controller_strobe {
                    core.controller_state & 1
                } else {
                    let bit = controller_shift & 1;
                    controller_shift = (controller_shift >> 1) | 0x80;
                    bit
                };
                0x40 | bit
            } else {
                0
            };
            last_read = value;
            let old_a = a;
            a = (old_a << 1) | (value & 1);
            carry = old_a & 0x80 != 0;
        }

        let prior_buttons = core.ram_read(0x074a + x);
        let duplicate_start_select = (a & 0x30) & prior_buttons;
        let branch_taken = duplicate_start_select == 0;
        let routine_cycles = if branch_taken {
            SMB_CONTROLLER_READ_TAKEN_CPU_CYCLES
        } else {
            SMB_CONTROLLER_READ_NOT_TAKEN_CPU_CYCLES
        };
        let extra_cycles = core.extra_cycles as usize;
        let total_cycles = routine_cycles + extra_cycles;
        let ppu_cycles = total_cycles * 3;
        let ppu_cycles_until_event = core.ppu.cycles_until_next_event();
        let remaining = ppu_cycles_until_event.saturating_sub(*pending_ppu_cycles);
        if ppu_cycles > remaining {
            return false;
        }

        core.controller_shift = controller_shift;
        core.extra_cycles = 0;
        core.ram_write(0x0000, last_read);
        core.ram_write(0x06fc + x, a);
        core.ram_write(0x0100 | core.cpu.sp as usize, a);
        if carry {
            core.cpu.p |= FLAG_C;
        } else {
            core.cpu.p &= !FLAG_C;
        }
        core.cpu.p |= FLAG_U;
        if branch_taken {
            core.cpu.a = a;
            core.set_zn(a);
            core.ram_write(0x074a + x, a);
        } else {
            core.cpu.a = a & 0xcf;
            core.set_zn(core.cpu.a);
            core.ram_write(0x06fc + x, core.cpu.a);
        }
        core.cpu.y = 0;
        core.cpu.pc = core.pop_u16().wrapping_add(1);
        *cpu_cycle_guard += total_cycles;
        *pending_ppu_cycles += ppu_cycles;
        true
    }

    fn digit_math_loop_cycles(core: &mut NromCore<SmbPpuTiming>) -> usize {
        let mut cycles = core.extra_cycles as usize;
        let mut x = core.cpu.x;
        let mut y = core.cpu.y;
        let mut carry = core.flag(FLAG_C);

        loop {
            let source_base = 0x07ddu16;
            let source_addr = source_base.wrapping_add(x as u16);
            let a = core.ram_read(source_addr as usize);
            cycles += 4 + page_crossed(source_base, source_addr) as usize;

            let sub_base = 0x07d7u16;
            let sub_addr = sub_base.wrapping_add(y as u16);
            let value = core.ram_read(sub_addr as usize);
            let sum = a as u16 + (!value) as u16 + carry as u16;
            carry = sum > 0xff;
            cycles += 4 + page_crossed(sub_base, sub_addr) as usize;

            x = x.wrapping_sub(1);
            y = y.wrapping_sub(1);
            cycles += 4;
            if y & 0x80 != 0 {
                cycles += 2;
                break;
            }
            cycles += 3;
        }

        if !carry {
            return cycles + 3 + 6;
        }
        cycles += 2;
        x = x.wrapping_add(1);
        y = y.wrapping_add(1);
        cycles += 4;

        loop {
            let source_base = 0x07ddu16;
            let source_addr = source_base.wrapping_add(x as u16);
            cycles += 4 + page_crossed(source_base, source_addr) as usize;
            cycles += 5;
            x = x.wrapping_add(1);
            y = y.wrapping_add(1);
            cycles += 6;
            if y >= 6 {
                cycles += 2;
                break;
            }
            cycles += 3;
        }
        cycles + 6
    }

    #[inline(never)]
    fn try_fast_forward_digit_math_loop(
        core: &mut NromCore<SmbPpuTiming>,
        capabilities: &SmbFastPaths,
        cpu_cycle_guard: &mut usize,
        pending_ppu_cycles: &mut usize,
    ) -> bool {
        if core.cpu.pc != SMB_DIGIT_MATH_LOOP_PC || !capabilities.digit_math_loop {
            return false;
        }

        let total_cycles = Self::digit_math_loop_cycles(core);
        let ppu_cycles = total_cycles * 3;
        let remaining = core
            .ppu
            .cycles_until_next_event()
            .saturating_sub(*pending_ppu_cycles);
        if ppu_cycles >= remaining || *cpu_cycle_guard + total_cycles >= CPU_CYCLES_PER_FRAME_GUARD
        {
            return false;
        }

        let mut cycles = core.extra_cycles as usize;
        loop {
            let source_base = 0x07ddu16;
            let source_addr = source_base.wrapping_add(core.cpu.x as u16);
            core.cpu.a = core.ram_read(source_addr as usize);
            core.set_zn(core.cpu.a);
            cycles += 4 + page_crossed(source_base, source_addr) as usize;

            let sub_base = 0x07d7u16;
            let sub_addr = sub_base.wrapping_add(core.cpu.y as u16);
            let value = core.ram_read(sub_addr as usize);
            core.sbc(value);
            cycles += 4 + page_crossed(sub_base, sub_addr) as usize;

            core.cpu.x = core.cpu.x.wrapping_sub(1);
            core.set_zn(core.cpu.x);
            core.cpu.y = core.cpu.y.wrapping_sub(1);
            core.set_zn(core.cpu.y);
            cycles += 4;
            if core.flag(FLAG_N) {
                cycles += 2;
                break;
            }
            cycles += 3;
        }

        if !core.flag(FLAG_C) {
            cycles += 3;
        } else {
            cycles += 2;
            core.cpu.x = core.cpu.x.wrapping_add(1);
            core.set_zn(core.cpu.x);
            core.cpu.y = core.cpu.y.wrapping_add(1);
            core.set_zn(core.cpu.y);
            cycles += 4;

            loop {
                let source_base = 0x07ddu16;
                let source_addr = source_base.wrapping_add(core.cpu.x as u16);
                core.cpu.a = core.ram_read(source_addr as usize);
                core.set_zn(core.cpu.a);
                cycles += 4 + page_crossed(source_base, source_addr) as usize;

                let destination = 0x07d7u16.wrapping_add(core.cpu.y as u16);
                core.ram_write(destination as usize, core.cpu.a);
                cycles += 5;

                core.cpu.x = core.cpu.x.wrapping_add(1);
                core.set_zn(core.cpu.x);
                core.cpu.y = core.cpu.y.wrapping_add(1);
                core.set_zn(core.cpu.y);
                core.cmp(core.cpu.y, 6);
                cycles += 6;
                if core.flag(FLAG_C) {
                    cycles += 2;
                    break;
                }
                cycles += 3;
            }
        }

        core.cpu.pc = core.pop_u16().wrapping_add(1);
        cycles += 6;
        core.extra_cycles = 0;
        debug_assert_eq!(cycles, total_cycles);
        *cpu_cycle_guard += total_cycles;
        *pending_ppu_cycles += ppu_cycles;
        true
    }

    #[inline(never)]
    fn try_fast_forward_bounding_box_nibble(
        core: &mut NromCore<SmbPpuTiming>,
        capabilities: &SmbFastPaths,
        cpu_cycle_guard: &mut usize,
        pending_ppu_cycles: &mut usize,
    ) -> bool {
        if core.cpu.pc != SMB_BOUNDING_BOX_NIBBLE_PC || !capabilities.bounding_box_nibble {
            return false;
        }

        let extra_cycles = core.extra_cycles as usize;
        let total_cycles = SMB_BOUNDING_BOX_NIBBLE_CPU_CYCLES + extra_cycles;
        let ppu_cycles = total_cycles * 3;
        let remaining = core
            .ppu
            .cycles_until_next_event()
            .saturating_sub(*pending_ppu_cycles);
        if ppu_cycles > remaining {
            return false;
        }

        core.push(core.cpu.a);
        core.cpu.a = core.lsr(core.cpu.a);
        core.cpu.a = core.lsr(core.cpu.a);
        core.cpu.a = core.lsr(core.cpu.a);
        core.cpu.a = core.lsr(core.cpu.a);
        core.cpu.y = core.cpu.a;
        core.set_zn(core.cpu.y);
        let high_addr = 0x9bdfu16.wrapping_add(core.cpu.y as u16);
        core.cpu.a = core.prg_read(high_addr);
        core.set_zn(core.cpu.a);
        core.ram_write(0x0007, core.cpu.a);
        core.cpu.a = core.pop();
        core.set_zn(core.cpu.a);
        core.cpu.a &= 0x0f;
        core.set_zn(core.cpu.a);
        core.set_flag(FLAG_C, false);
        let low_addr = 0x9bddu16.wrapping_add(core.cpu.y as u16);
        let addend = core.prg_read(low_addr);
        core.adc(addend);
        core.ram_write(0x0006, core.cpu.a);
        core.cpu.pc = core.pop_u16().wrapping_add(1);
        core.extra_cycles = 0;
        *cpu_cycle_guard += total_cycles;
        *pending_ppu_cycles += ppu_cycles;
        true
    }

    #[inline(never)]
    fn try_fast_forward_bounding_box_helper(
        core: &mut NromCore<SmbPpuTiming>,
        capabilities: &SmbFastPaths,
        cpu_cycle_guard: &mut usize,
        pending_ppu_cycles: &mut usize,
    ) -> bool {
        if core.cpu.pc != SMB_BOUNDING_BOX_HELPER_PC || !capabilities.bounding_box_helper {
            return false;
        }

        let extra_cycles = core.extra_cycles as usize;
        let max_ppu_cycles = (SMB_BOUNDING_BOX_HELPER_MAX_CPU_CYCLES + extra_cycles) * 3;
        let remaining = core
            .ppu
            .cycles_until_next_event()
            .saturating_sub(*pending_ppu_cycles);
        if max_ppu_cycles > remaining {
            return false;
        }

        let mut cycles = extra_cycles;
        let x = core.cpu.x;
        core.push(core.cpu.a);
        cycles += 3;
        core.ram_write(0x0004, core.cpu.y);
        cycles += 3;

        let table1_base = 0xe3b0u16;
        let table1_addr = table1_base.wrapping_add(core.cpu.y as u16);
        core.cpu.a = core.prg_read(table1_addr);
        core.set_zn(core.cpu.a);
        cycles += 4 + page_crossed(table1_base, table1_addr) as usize;
        core.set_flag(FLAG_C, false);
        cycles += 2;
        let x86 = 0x86u8.wrapping_add(x) as usize;
        let addend = core.ram_read(x86);
        core.adc(addend);
        cycles += 4;
        core.ram_write(0x0005, core.cpu.a);
        cycles += 3;
        let x6d = 0x6du8.wrapping_add(x) as usize;
        core.cpu.a = core.ram_read(x6d);
        core.set_zn(core.cpu.a);
        cycles += 4;
        core.adc(0);
        cycles += 2;
        core.cpu.a &= 0x01;
        core.set_zn(core.cpu.a);
        cycles += 2;
        core.cpu.a = core.lsr(core.cpu.a);
        cycles += 2;
        core.ora(core.ram_read(0x0005));
        cycles += 3;
        core.cpu.a = core.ror(core.cpu.a);
        cycles += 2;
        core.cpu.a = core.lsr(core.cpu.a);
        core.cpu.a = core.lsr(core.cpu.a);
        core.cpu.a = core.lsr(core.cpu.a);
        cycles += 6;

        core.push_u16(0xe40a);
        cycles += 6;
        core.push(core.cpu.a);
        core.cpu.a = core.lsr(core.cpu.a);
        core.cpu.a = core.lsr(core.cpu.a);
        core.cpu.a = core.lsr(core.cpu.a);
        core.cpu.a = core.lsr(core.cpu.a);
        core.cpu.y = core.cpu.a;
        core.set_zn(core.cpu.y);
        let high_addr = 0x9bdfu16.wrapping_add(core.cpu.y as u16);
        core.cpu.a = core.prg_read(high_addr);
        core.set_zn(core.cpu.a);
        core.ram_write(0x0007, core.cpu.a);
        core.cpu.a = core.pop();
        core.set_zn(core.cpu.a);
        core.cpu.a &= 0x0f;
        core.set_zn(core.cpu.a);
        core.set_flag(FLAG_C, false);
        let low_addr = 0x9bddu16.wrapping_add(core.cpu.y as u16);
        let addend = core.prg_read(low_addr);
        core.adc(addend);
        core.ram_write(0x0006, core.cpu.a);
        core.cpu.pc = core.pop_u16().wrapping_add(1);
        debug_assert_eq!(core.cpu.pc, 0xe40b);
        cycles += SMB_BOUNDING_BOX_NIBBLE_CPU_CYCLES;

        core.cpu.y = core.ram_read(0x0004);
        core.set_zn(core.cpu.y);
        cycles += 3;
        let xce = 0xceu8.wrapping_add(x) as usize;
        core.cpu.a = core.ram_read(xce);
        core.set_zn(core.cpu.a);
        cycles += 4;
        core.set_flag(FLAG_C, false);
        cycles += 2;
        let table2_base = 0xe3ccu16;
        let table2_addr = table2_base.wrapping_add(core.cpu.y as u16);
        let addend = core.prg_read(table2_addr);
        core.adc(addend);
        cycles += 4 + page_crossed(table2_base, table2_addr) as usize;
        core.cpu.a &= 0xf0;
        core.set_zn(core.cpu.a);
        cycles += 2;
        core.set_flag(FLAG_C, true);
        cycles += 2;
        core.sbc(0x20);
        cycles += 2;
        core.ram_write(0x0002, core.cpu.a);
        cycles += 3;
        core.cpu.y = core.cpu.a;
        core.set_zn(core.cpu.y);
        cycles += 2;
        let ptr = core.ram_read(0x0006) as u16 | ((core.ram_read(0x0007) as u16) << 8);
        let indirect_addr = ptr.wrapping_add(core.cpu.y as u16);
        core.cpu.a = core.cpu_read(indirect_addr);
        core.set_zn(core.cpu.a);
        cycles += 5 + page_crossed(ptr, indirect_addr) as usize;
        core.ram_write(0x0003, core.cpu.a);
        cycles += 3;
        core.cpu.y = core.ram_read(0x0004);
        core.set_zn(core.cpu.y);
        cycles += 3;
        core.cpu.a = core.pop();
        core.set_zn(core.cpu.a);
        cycles += 4;
        if core.cpu.a != 0 {
            cycles += 3;
            core.cpu.a = core.ram_read(x86);
            core.set_zn(core.cpu.a);
            cycles += 4;
        } else {
            cycles += 2;
            core.cpu.a = core.ram_read(xce);
            core.set_zn(core.cpu.a);
            cycles += 4;
            core.cpu.pc = 0xe42b;
            cycles += 3;
        }
        core.cpu.a &= 0x0f;
        core.set_zn(core.cpu.a);
        cycles += 2;
        core.ram_write(0x0004, core.cpu.a);
        cycles += 3;
        core.cpu.a = core.ram_read(0x0003);
        core.set_zn(core.cpu.a);
        cycles += 3;
        core.cpu.pc = core.pop_u16().wrapping_add(1);
        cycles += 6;
        core.extra_cycles = 0;
        *cpu_cycle_guard += cycles;
        *pending_ppu_cycles += cycles * 3;
        true
    }

    #[inline]
    fn fast_forward_divide_pdiff_body(core: &mut NromCore<SmbPpuTiming>) -> usize {
        let mut cycles = 0;
        core.ram_write(0x0005, core.cpu.a);
        cycles += 3;
        core.cpu.a = core.ram_read(0x0007);
        core.set_zn(core.cpu.a);
        cycles += 3;
        core.cmp(core.cpu.a, core.ram_read(0x0006));
        cycles += 3;
        if core.flag(FLAG_C) {
            cycles += 3;
            core.pop_u16();
            return cycles + 6;
        }
        cycles += 2;
        core.cpu.a = core.lsr(core.cpu.a);
        core.cpu.a = core.lsr(core.cpu.a);
        core.cpu.a = core.lsr(core.cpu.a);
        cycles += 6;
        core.cpu.a &= 0x07;
        core.set_zn(core.cpu.a);
        cycles += 2;
        core.cmp(core.cpu.y, 1);
        cycles += 2;
        if core.flag(FLAG_C) {
            cycles += 3;
        } else {
            cycles += 2;
            core.adc(core.ram_read(0x0005));
            cycles += 3;
        }
        core.cpu.x = core.cpu.a;
        core.set_zn(core.cpu.x);
        cycles += 2;
        core.pop_u16();
        cycles + 6
    }

    #[inline]
    fn fast_forward_x_offscreen_bits_body(core: &mut NromCore<SmbPpuTiming>) -> usize {
        let mut cycles = 0;
        core.ram_write(0x0004, core.cpu.x);
        cycles += 3;
        core.cpu.y = 1;
        core.set_zn(core.cpu.y);
        cycles += 2;
        loop {
            let y = core.cpu.y as usize;
            core.cpu.a = core.ram_read(0x071c + y);
            core.set_zn(core.cpu.a);
            cycles += 4;
            core.set_flag(FLAG_C, true);
            cycles += 2;
            let object_x = core.ram_read(0x86u8.wrapping_add(core.cpu.x) as usize);
            core.sbc(object_x);
            cycles += 4;
            core.ram_write(0x0007, core.cpu.a);
            cycles += 3;
            core.cpu.a = core.ram_read(0x071a + y);
            core.set_zn(core.cpu.a);
            cycles += 4;
            let object_page = core.ram_read(0x6du8.wrapping_add(core.cpu.x) as usize);
            core.sbc(object_page);
            cycles += 4;
            core.cpu.x = core.prg_read(0xf1f3u16.wrapping_add(core.cpu.y as u16));
            core.set_zn(core.cpu.x);
            cycles += 4;
            core.cmp(core.cpu.a, 0);
            cycles += 2;
            if core.flag(FLAG_N) {
                cycles += 3;
            } else {
                cycles += 2;
                core.cpu.x = core.prg_read(0xf1f4u16.wrapping_add(core.cpu.y as u16));
                core.set_zn(core.cpu.x);
                cycles += 4;
                core.cmp(core.cpu.a, 1);
                cycles += 2;
                if !core.flag(FLAG_N) {
                    cycles += 3;
                } else {
                    cycles += 2;
                    core.cpu.a = 0x38;
                    core.set_zn(core.cpu.a);
                    cycles += 2;
                    core.ram_write(0x0006, core.cpu.a);
                    cycles += 3;
                    core.cpu.a = 0x08;
                    core.set_zn(core.cpu.a);
                    cycles += 2;
                    core.push_u16(0xf21d);
                    cycles += 6;
                    cycles += Self::fast_forward_divide_pdiff_body(core);
                }
            }
            core.cpu.a = core.prg_read(0xf1e3u16.wrapping_add(core.cpu.x as u16));
            core.set_zn(core.cpu.a);
            cycles += 4;
            core.cpu.x = core.ram_read(0x0004);
            core.set_zn(core.cpu.x);
            cycles += 3;
            core.cmp(core.cpu.a, 0);
            cycles += 2;
            if !core.flag(FLAG_Z) {
                cycles += 3;
                core.pop_u16();
                return cycles + 6;
            }
            cycles += 2;
            core.cpu.y = core.cpu.y.wrapping_sub(1);
            core.set_zn(core.cpu.y);
            cycles += 2;
            if !core.flag(FLAG_N) {
                cycles += 4;
            } else {
                cycles += 2;
                core.pop_u16();
                return cycles + 6;
            }
        }
    }

    #[inline]
    fn fast_forward_y_offscreen_bits_body(core: &mut NromCore<SmbPpuTiming>) -> (usize, u16) {
        let mut cycles = 0;
        core.ram_write(0x0004, core.cpu.x);
        cycles += 3;
        core.cpu.y = 1;
        core.set_zn(core.cpu.y);
        cycles += 2;
        loop {
            core.cpu.a = core.prg_read(0xf237u16.wrapping_add(core.cpu.y as u16));
            core.set_zn(core.cpu.a);
            cycles += 4;
            core.set_flag(FLAG_C, true);
            cycles += 2;
            let object_y = core.ram_read(0xceu8.wrapping_add(core.cpu.x) as usize);
            core.sbc(object_y);
            cycles += 4;
            core.ram_write(0x0007, core.cpu.a);
            cycles += 3;
            core.cpu.a = 1;
            core.set_zn(core.cpu.a);
            cycles += 2;
            let object_high = core.ram_read(0xb5u8.wrapping_add(core.cpu.x) as usize);
            core.sbc(object_high);
            cycles += 4;
            core.cpu.x = core.prg_read(0xf234u16.wrapping_add(core.cpu.y as u16));
            core.set_zn(core.cpu.x);
            cycles += 4;
            core.cmp(core.cpu.a, 0);
            cycles += 2;
            if core.flag(FLAG_N) {
                cycles += 3;
            } else {
                cycles += 2;
                core.cpu.x = core.prg_read(0xf235u16.wrapping_add(core.cpu.y as u16));
                core.set_zn(core.cpu.x);
                cycles += 4;
                core.cmp(core.cpu.a, 1);
                cycles += 2;
                if !core.flag(FLAG_N) {
                    cycles += 3;
                } else {
                    cycles += 2;
                    core.cpu.a = 0x20;
                    core.set_zn(core.cpu.a);
                    cycles += 2;
                    core.ram_write(0x0006, core.cpu.a);
                    cycles += 3;
                    core.cpu.a = 0x04;
                    core.set_zn(core.cpu.a);
                    cycles += 2;
                    core.push_u16(0xf25f);
                    cycles += 6;
                    cycles += Self::fast_forward_divide_pdiff_body(core);
                }
            }
            core.cpu.a = core.prg_read(0xf22bu16.wrapping_add(core.cpu.x as u16));
            core.set_zn(core.cpu.a);
            cycles += 4;
            core.cpu.x = core.ram_read(0x0004);
            core.set_zn(core.cpu.x);
            cycles += 3;
            core.cmp(core.cpu.a, 0);
            cycles += 2;
            if !core.flag(FLAG_Z) {
                cycles += 3;
                let return_address = core.pop_u16();
                return (cycles + 6, return_address);
            }
            cycles += 2;
            core.cpu.y = core.cpu.y.wrapping_sub(1);
            core.set_zn(core.cpu.y);
            cycles += 2;
            if !core.flag(FLAG_N) {
                cycles += 3;
            } else {
                cycles += 2;
                let return_address = core.pop_u16();
                return (cycles + 6, return_address);
            }
        }
    }

    #[inline(never)]
    fn try_fast_forward_offscreen_bits_subs(
        core: &mut NromCore<SmbPpuTiming>,
        capabilities: &SmbFastPaths,
        cpu_cycle_guard: &mut usize,
        pending_ppu_cycles: &mut usize,
    ) -> bool {
        if core.cpu.pc != SMB_OFFSCREEN_BITS_SUBS_PC || !capabilities.offscreen_bits_subs {
            return false;
        }
        let max_cycles = SMB_OFFSCREEN_BITS_SUBS_MAX_CPU_CYCLES + core.extra_cycles as usize;
        let remaining = core
            .ppu
            .cycles_until_next_event()
            .saturating_sub(*pending_ppu_cycles);
        if max_cycles * 3 >= remaining
            || *cpu_cycle_guard + max_cycles >= CPU_CYCLES_PER_FRAME_GUARD
        {
            return false;
        }

        let mut total_cycles = core.extra_cycles as usize;
        core.extra_cycles = 0;
        core.cpu.a = core.cpu.y;
        core.set_zn(core.cpu.a);
        total_cycles += 2;
        core.push(core.cpu.a);
        total_cycles += 3;
        core.push_u16(0xf1c4);
        total_cycles += 6;
        core.push_u16(0xf1d9);
        total_cycles += 6;
        total_cycles += Self::fast_forward_x_offscreen_bits_body(core);
        core.cpu.a = core.lsr(core.cpu.a);
        core.cpu.a = core.lsr(core.cpu.a);
        core.cpu.a = core.lsr(core.cpu.a);
        core.cpu.a = core.lsr(core.cpu.a);
        total_cycles += 8;
        core.ram_write(0x0000, core.cpu.a);
        total_cycles += 3;
        total_cycles += 3;
        let (y_cycles, return_address) = Self::fast_forward_y_offscreen_bits_body(core);
        total_cycles += y_cycles;
        debug_assert_eq!(return_address, 0xf1c4);
        core.cpu.a = core.asl(core.cpu.a);
        core.cpu.a = core.asl(core.cpu.a);
        core.cpu.a = core.asl(core.cpu.a);
        core.cpu.a = core.asl(core.cpu.a);
        total_cycles += 8;
        core.ora(core.ram_read(0x0000));
        total_cycles += 3;
        core.ram_write(0x0000, core.cpu.a);
        total_cycles += 3;
        core.cpu.a = core.pop();
        total_cycles += 4;
        core.cpu.y = core.cpu.a;
        core.set_zn(core.cpu.y);
        total_cycles += 2;
        core.cpu.a = core.ram_read(0x0000);
        core.set_zn(core.cpu.a);
        total_cycles += 3;
        core.ram_write(0x03d0 + core.cpu.y as usize, core.cpu.a);
        total_cycles += 5;
        core.cpu.x = core.ram_read(0x0008);
        core.set_zn(core.cpu.x);
        total_cycles += 3;
        let return_address = core.pop_u16();
        total_cycles += 6;
        debug_assert!(total_cycles <= max_cycles);
        core.cpu.pc = return_address.wrapping_add(1);
        *cpu_cycle_guard += total_cycles;
        *pending_ppu_cycles += total_cycles * 3;
        true
    }

    #[inline(never)]
    fn try_fast_forward_relative_position_helper(
        core: &mut NromCore<SmbPpuTiming>,
        capabilities: &SmbFastPaths,
        cpu_cycle_guard: &mut usize,
        pending_ppu_cycles: &mut usize,
    ) -> bool {
        if core.cpu.pc != SMB_RELATIVE_POSITION_HELPER_PC || !capabilities.relative_position_helper
        {
            return false;
        }

        let routine_cycles = if core.ram_read(0x0007) >= core.ram_read(0x0006) {
            18
        } else if core.cpu.y >= 1 {
            32
        } else {
            34
        };
        let total_cycles = routine_cycles + core.extra_cycles as usize;
        let ppu_cycles = total_cycles * 3;
        let remaining = core
            .ppu
            .cycles_until_next_event()
            .saturating_sub(*pending_ppu_cycles);
        if ppu_cycles >= remaining || *cpu_cycle_guard + total_cycles >= CPU_CYCLES_PER_FRAME_GUARD
        {
            return false;
        }

        core.ram_write(0x0005, core.cpu.a);
        core.cpu.a = core.ram_read(0x0007);
        core.set_zn(core.cpu.a);
        core.cmp(core.cpu.a, core.ram_read(0x0006));
        if !core.flag(FLAG_C) {
            core.cpu.a = core.lsr(core.cpu.a);
            core.cpu.a = core.lsr(core.cpu.a);
            core.cpu.a = core.lsr(core.cpu.a);
            core.and(0x07);
            core.cmp(core.cpu.y, 1);
            if !core.flag(FLAG_C) {
                core.adc(core.ram_read(0x0005));
            }
            core.cpu.x = core.cpu.a;
            core.set_zn(core.cpu.x);
        }
        core.cpu.pc = core.pop_u16().wrapping_add(1);
        core.extra_cycles = 0;
        *cpu_cycle_guard += total_cycles;
        *pending_ppu_cycles += ppu_cycles;
        true
    }

    #[inline(never)]
    fn try_fast_forward_draw_sprite_object(
        core: &mut NromCore<SmbPpuTiming>,
        capabilities: &SmbFastPaths,
        cpu_cycle_guard: &mut usize,
        pending_ppu_cycles: &mut usize,
    ) -> bool {
        if core.cpu.pc != SMB_DRAW_SPRITE_OBJECT_PC || !capabilities.draw_sprite_object {
            return false;
        }

        let flip = core.ram_read(0x0003) & 0x02 != 0;
        let routine_cycles = if flip { 101 } else { 99 };
        let total_cycles = routine_cycles + core.extra_cycles as usize;
        let ppu_cycles = total_cycles * 3;
        let remaining = core
            .ppu
            .cycles_until_next_event()
            .saturating_sub(*pending_ppu_cycles);
        if ppu_cycles >= remaining || *cpu_cycle_guard + total_cycles >= CPU_CYCLES_PER_FRAME_GUARD
        {
            return false;
        }

        let y = core.cpu.y as usize;
        let tile_0 = core.ram_read(0x0000);
        let tile_1 = core.ram_read(0x0001);
        if flip {
            core.ram_write(0x0205 + y, tile_0);
            core.ram_write(0x0201 + y, tile_1);
        } else {
            core.ram_write(0x0201 + y, tile_0);
            core.ram_write(0x0205 + y, tile_1);
        }
        let attributes = core.ram_read(0x0004) | if flip { 0x40 } else { 0 };
        core.ram_write(0x0202 + y, attributes);
        core.ram_write(0x0206 + y, attributes);
        let sprite_y = core.ram_read(0x0002);
        core.ram_write(0x0200 + y, sprite_y);
        core.ram_write(0x0204 + y, sprite_y);
        let sprite_x = core.ram_read(0x0005);
        core.ram_write(0x0203 + y, sprite_x);
        core.ram_write(0x0207 + y, sprite_x.wrapping_add(8));
        core.ram_write(0x0002, sprite_y.wrapping_add(8));

        core.cpu.a = core.cpu.y;
        core.set_flag(FLAG_C, false);
        core.adc(8);
        core.cpu.y = core.cpu.a;
        core.set_zn(core.cpu.y);
        core.cpu.x = core.cpu.x.wrapping_add(1);
        core.set_zn(core.cpu.x);
        core.cpu.x = core.cpu.x.wrapping_add(1);
        core.set_zn(core.cpu.x);
        core.cpu.pc = core.pop_u16().wrapping_add(1);
        core.extra_cycles = 0;
        *cpu_cycle_guard += total_cycles;
        *pending_ppu_cycles += ppu_cycles;
        true
    }
}

impl NromGame for SuperMarioBros {
    type Options = SmbOptions;
    type Signals = SmbSignals;
    type StepContext = u8;
    type FastPaths = SmbFastPaths;
    type PpuTiming = SmbPpuTiming;

    const EXPECTED_ROM_SHA256: &'static str = EXPECTED_SMB_ROM_SHA256;

    fn detect_fast_paths(prg_rom: &[u8], prg_addr_mask: usize) -> Self::FastPaths {
        SmbFastPaths::detect(prg_rom, prg_addr_mask)
    }

    #[inline]
    fn synchronize(core: &mut NromCore<SmbPpuTiming>, signals: &mut SmbSignals) {
        signals.x_pos = ((core.ram[0x006d] as u16) << 8) | core.ram[0x0086] as u16;
        signals.coins = core.ram[0x075e];
        signals.level_hi = sign_extend_u8(core.ram[0x075f]);
        signals.level_lo = sign_extend_u8(core.ram[0x075c]);
        signals.lives = sign_extend_u8(core.ram[0x075a]);
        signals.score = u32::from(core.ram[0x07dd] & 0x0f) * 100_000
            + u32::from(core.ram[0x07de] & 0x0f) * 10_000
            + u32::from(core.ram[0x07df] & 0x0f) * 1_000
            + u32::from(core.ram[0x07e0] & 0x0f) * 100
            + u32::from(core.ram[0x07e1] & 0x0f) * 10
            + u32::from(core.ram[0x07e2] & 0x0f);
        signals.scrolling = sign_extend_u8(core.ram[0x0778]);
        signals.time = u16::from(core.ram[0x07f8] & 0x0f) * 100
            + u16::from(core.ram[0x07f9] & 0x0f) * 10
            + u16::from(core.ram[0x07fa] & 0x0f);
        signals.xscroll_hi = core.ram[0x071a];
        signals.xscroll_lo = core.ram[0x071c];
        core.ppu.set_scroll_override_x(None);
    }

    #[inline]
    fn pre_step(signals: &SmbSignals) -> u8 {
        signals.xscroll_lo
    }

    #[inline]
    fn post_step(options: &SmbOptions, signals: &SmbSignals, before: u8) -> StepResult {
        StepResult {
            reward: (signals.xscroll_lo as i16 - before as i16).max(0) as f32,
            done: signals.lives == -1 || (*options && signals.x_pos >= 3160),
        }
    }

    #[inline]
    fn dispatch_fast_path(
        core: &mut NromCore<SmbPpuTiming>,
        fast_paths: &SmbFastPaths,
        budget: &mut FrameBudget,
    ) -> FastPathOutcome {
        fast_paths.dispatch(core, budget)
    }
}

#[inline]
fn sign_extend_u8(value: u8) -> i16 {
    (value as i8) as i16
}

#[inline]
fn page_crossed(a: u16, b: u16) -> bool {
    (a & 0xff00) != (b & 0xff00)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nes_turbo_nrom_core::Cartridge;

    fn make_test_cart_with_prg(prg_rom: Vec<u8>) -> Cartridge {
        Cartridge {
            prg_rom,
            chr_rom: vec![0; 8192],
            vertical_mirroring: true,
        }
    }

    trait SmbFastPathTestSupport {
        fn try_fast_forward_sprite0_poll(&mut self, cpu: &mut usize, ppu: &mut usize) -> bool;
        fn try_fast_forward_timer_control_loop(&mut self, cpu: &mut usize, ppu: &mut usize)
            -> bool;
        fn try_fast_forward_oam_clear(&mut self, cpu: &mut usize, ppu: &mut usize) -> bool;
        fn try_fast_forward_scroll_slot_loop(&mut self, cpu: &mut usize, ppu: &mut usize) -> bool;
        fn try_fast_forward_controller_read(&mut self, cpu: &mut usize, ppu: &mut usize) -> bool;
        fn try_fast_forward_digit_math_loop(&mut self, cpu: &mut usize, ppu: &mut usize) -> bool;
        fn try_fast_forward_bounding_box_nibble(
            &mut self,
            cpu: &mut usize,
            ppu: &mut usize,
        ) -> bool;
        fn try_fast_forward_bounding_box_helper(
            &mut self,
            cpu: &mut usize,
            ppu: &mut usize,
        ) -> bool;
        fn try_fast_forward_offscreen_bits_subs(
            &mut self,
            cpu: &mut usize,
            ppu: &mut usize,
        ) -> bool;
        fn try_fast_forward_relative_position_helper(
            &mut self,
            cpu: &mut usize,
            ppu: &mut usize,
        ) -> bool;
        fn try_fast_forward_draw_sprite_object(&mut self, cpu: &mut usize, ppu: &mut usize)
            -> bool;
        fn timer_control_loop_cycles_and_last_a(&mut self, start_x: u8) -> (usize, u8);
    }

    macro_rules! impl_test_fast_paths {
        ($( $name:ident ),* $(,)?) => {
            impl SmbFastPathTestSupport for NesEmulator {
                fn timer_control_loop_cycles_and_last_a(&mut self, start_x: u8) -> (usize, u8) {
                    SuperMarioBros::timer_control_loop_cycles_and_last_a(&mut self.core, start_x)
                }
                $(
                    fn $name(&mut self, cpu: &mut usize, ppu: &mut usize) -> bool {
                        SuperMarioBros::$name(&mut self.core, &self.fast_paths, cpu, ppu)
                    }
                )*
            }
        };
    }

    impl_test_fast_paths!(
        try_fast_forward_sprite0_poll,
        try_fast_forward_timer_control_loop,
        try_fast_forward_oam_clear,
        try_fast_forward_scroll_slot_loop,
        try_fast_forward_controller_read,
        try_fast_forward_digit_math_loop,
        try_fast_forward_bounding_box_nibble,
        try_fast_forward_bounding_box_helper,
        try_fast_forward_offscreen_bits_subs,
        try_fast_forward_relative_position_helper,
        try_fast_forward_draw_sprite_object,
    );

    #[test]
    fn smb_idle_jmp_signature_is_exact() {
        let mut prg = vec![0xea; 32768];
        let offset = (SMB_IDLE_JMP_PC - 0x8000) as usize;
        prg[offset..offset + 3].copy_from_slice(&[0x4c, 0x57, 0x80]);
        assert!(prg_rom_supports_smb_idle_jmp(&prg, 0x7fff));
        prg[offset + 1] ^= 1;
        assert!(!prg_rom_supports_smb_idle_jmp(&prg, 0x7fff));
    }

    #[test]
    fn dispatch_sites_are_unique() {
        let mut sites = [
            SMB_IDLE_JMP_PC,
            SMB_SPRITE0_POLL_PC,
            SMB_TIMER_CONTROL_LOOP_PC,
            SMB_OAM_CLEAR_PC,
            SMB_SCROLL_SLOT_LOOP_PC,
            SMB_CONTROLLER_READ_PC,
            SMB_DIGIT_MATH_LOOP_PC,
            SMB_BOUNDING_BOX_HELPER_PC,
            SMB_BOUNDING_BOX_NIBBLE_PC,
            SMB_OFFSCREEN_BITS_SUBS_PC,
            SMB_RELATIVE_POSITION_HELPER_PC,
            SMB_DRAW_SPRITE_OBJECT_PC,
        ];
        sites.sort_unstable();
        assert!(sites.windows(2).all(|pair| pair[0] != pair[1]));
    }

    #[test]
    fn smb_sprite0_poll_signature_is_exact() {
        let mut prg = vec![0xea; 32768];
        let offset = (SMB_SPRITE0_POLL_PC - 0x8000) as usize;
        prg[offset..offset + 7].copy_from_slice(&[0xad, 0x02, 0x20, 0x29, 0x40, 0xf0, 0xf9]);

        assert!(prg_rom_supports_smb_sprite0_poll(&prg, prg.len() - 1));

        prg[offset + 6] = 0xf7;
        assert!(!prg_rom_supports_smb_sprite0_poll(&prg, prg.len() - 1));
    }

    #[test]
    fn smb_sprite0_poll_exit_signature_is_exact() {
        let mut prg = vec![0xea; 32768];
        let offset = (SMB_SPRITE0_POLL_PC - 0x8000) as usize;
        prg[offset..offset + 12].copy_from_slice(&[
            0xad, 0x02, 0x20, 0x29, 0x40, 0xf0, 0xf9, 0xa0, 0x14, 0x88, 0xd0, 0xfd,
        ]);

        assert!(prg_rom_supports_smb_sprite0_poll_exit(&prg, prg.len() - 1));

        prg[offset + 8] = 0x13;
        assert!(!prg_rom_supports_smb_sprite0_poll_exit(&prg, prg.len() - 1));
    }

    #[test]
    fn smb_timer_control_loop_signature_is_exact() {
        let mut prg = vec![0xea; 32768];
        let offset = (SMB_TIMER_CONTROL_LOOP_PC - 0x8000) as usize;
        prg[offset..offset + 11].copy_from_slice(&[
            0xbd, 0x80, 0x07, 0xf0, 0x03, 0xde, 0x80, 0x07, 0xca, 0x10, 0xf5,
        ]);

        assert!(prg_rom_supports_smb_timer_control_loop(&prg, prg.len() - 1));

        prg[offset + 10] = 0xf4;
        assert!(!prg_rom_supports_smb_timer_control_loop(
            &prg,
            prg.len() - 1
        ));
    }

    #[test]
    fn smb_oam_clear_signature_is_exact() {
        let mut prg = vec![0xea; 32768];
        let offset = (SMB_OAM_CLEAR_PC - 0x8000) as usize;
        prg[offset..offset + 14].copy_from_slice(&[
            0xa0, 0x04, 0xa9, 0xf8, 0x99, 0x00, 0x02, 0xc8, 0xc8, 0xc8, 0xc8, 0xd0, 0xf7, 0x60,
        ]);

        assert!(prg_rom_supports_smb_oam_clear(&prg, prg.len() - 1));

        prg[offset + 11] = 0xf0;
        assert!(!prg_rom_supports_smb_oam_clear(&prg, prg.len() - 1));
    }

    #[test]
    fn smb_controller_read_signature_is_exact() {
        let mut prg = vec![0xea; 32768];
        let offset = (SMB_CONTROLLER_READ_PC - 0x8000) as usize;
        prg[offset..offset + 40].copy_from_slice(&[
            0xa0, 0x08, 0x48, 0xbd, 0x16, 0x40, 0x85, 0x00, 0x4a, 0x05, 0x00, 0x4a, 0x68, 0x2a,
            0x88, 0xd0, 0xf1, 0x9d, 0xfc, 0x06, 0x48, 0x29, 0x30, 0x3d, 0x4a, 0x07, 0xf0, 0x07,
            0x68, 0x29, 0xcf, 0x9d, 0xfc, 0x06, 0x60, 0x68, 0x9d, 0x4a, 0x07, 0x60,
        ]);

        assert!(prg_rom_supports_smb_controller_read(&prg, prg.len() - 1));

        prg[offset + 27] = 0x06;
        assert!(!prg_rom_supports_smb_controller_read(&prg, prg.len() - 1));
    }

    #[test]
    fn smb_digit_math_loop_signature_is_exact() {
        let mut prg = vec![0xea; 32768];
        let offset = (SMB_DIGIT_MATH_LOOP_PC - 0x8000) as usize;
        prg[offset..offset + 27].copy_from_slice(&[
            0xbd, 0xdd, 0x07, 0xf9, 0xd7, 0x07, 0xca, 0x88, 0x10, 0xf6, 0x90, 0x0e, 0xe8, 0xc8,
            0xbd, 0xdd, 0x07, 0x99, 0xd7, 0x07, 0xe8, 0xc8, 0xc0, 0x06, 0x90, 0xf4, 0x60,
        ]);

        assert!(prg_rom_supports_smb_digit_math_loop(&prg, prg.len() - 1));

        prg[offset + 25] = 0xf2;
        assert!(!prg_rom_supports_smb_digit_math_loop(&prg, prg.len() - 1));
    }

    #[test]
    fn smb_draw_sprite_object_signature_is_exact() {
        let mut prg = vec![0xea; 32768];
        add_draw_sprite_object_routine(&mut prg);

        assert!(prg_rom_supports_smb_draw_sprite_object(&prg, prg.len() - 1));

        let offset = (SMB_DRAW_SPRITE_OBJECT_PC - 0x8000) as usize;
        prg[offset + 68] = 0xca;
        assert!(!prg_rom_supports_smb_draw_sprite_object(
            &prg,
            prg.len() - 1
        ));
    }

    #[test]
    fn smb_offscreen_bits_subs_signature_is_exact() {
        let mut prg = vec![0xea; 32768];
        add_offscreen_bits_subs(&mut prg);

        assert!(prg_rom_supports_smb_offscreen_bits_subs(
            &prg,
            prg.len() - 1
        ));

        let offset = (SMB_OFFSCREEN_BITS_SUBS_PC - 0x8000) as usize;
        prg[offset + 5] = 0x49;
        assert!(!prg_rom_supports_smb_offscreen_bits_subs(
            &prg,
            prg.len() - 1
        ));
    }

    #[test]
    fn smb_scroll_slot_loop_signature_is_exact() {
        let mut prg = vec![0xea; 32768];
        let offset = (SMB_SCROLL_SLOT_LOOP_PC - 0x8000) as usize;
        prg[offset..offset + 25].copy_from_slice(&[
            0xbd, 0xe4, 0x06, 0xc5, 0x00, 0x90, 0x0f, 0xac, 0xe0, 0x06, 0x18, 0x79, 0xe1, 0x06,
            0x90, 0x03, 0x18, 0x65, 0x00, 0x9d, 0xe4, 0x06, 0xca, 0x10, 0xe7,
        ]);

        assert!(prg_rom_supports_smb_scroll_slot_loop(&prg, prg.len() - 1));

        prg[offset + 24] = 0xe5;
        assert!(!prg_rom_supports_smb_scroll_slot_loop(&prg, prg.len() - 1));
    }

    #[test]
    fn smb_bounding_box_nibble_signature_is_exact() {
        let mut prg = vec![0xea; 32768];
        let offset = (SMB_BOUNDING_BOX_NIBBLE_PC - 0x8000) as usize;
        prg[offset..offset + 21].copy_from_slice(&[
            0x48, 0x4a, 0x4a, 0x4a, 0x4a, 0xa8, 0xb9, 0xdf, 0x9b, 0x85, 0x07, 0x68, 0x29, 0x0f,
            0x18, 0x79, 0xdd, 0x9b, 0x85, 0x06, 0x60,
        ]);

        assert!(prg_rom_supports_smb_bounding_box_nibble(
            &prg,
            prg.len() - 1
        ));

        prg[offset + 15] = 0x7d;
        assert!(!prg_rom_supports_smb_bounding_box_nibble(
            &prg,
            prg.len() - 1
        ));
    }

    #[test]
    fn smb_bounding_box_helper_signature_is_exact() {
        let mut prg = vec![0xea; 32768];
        let helper_offset = (SMB_BOUNDING_BOX_HELPER_PC - 0x8000) as usize;
        prg[helper_offset..helper_offset + 66].copy_from_slice(&[
            0x48, 0x84, 0x04, 0xb9, 0xb0, 0xe3, 0x18, 0x75, 0x86, 0x85, 0x05, 0xb5, 0x6d, 0x69,
            0x00, 0x29, 0x01, 0x4a, 0x05, 0x05, 0x6a, 0x4a, 0x4a, 0x4a, 0x20, 0xe1, 0x9b, 0xa4,
            0x04, 0xb5, 0xce, 0x18, 0x79, 0xcc, 0xe3, 0x29, 0xf0, 0x38, 0xe9, 0x20, 0x85, 0x02,
            0xa8, 0xb1, 0x06, 0x85, 0x03, 0xa4, 0x04, 0x68, 0xd0, 0x05, 0xb5, 0xce, 0x4c, 0x2b,
            0xe4, 0xb5, 0x86, 0x29, 0x0f, 0x85, 0x04, 0xa5, 0x03, 0x60,
        ]);
        let nibble_offset = (SMB_BOUNDING_BOX_NIBBLE_PC - 0x8000) as usize;
        prg[nibble_offset..nibble_offset + 21].copy_from_slice(&[
            0x48, 0x4a, 0x4a, 0x4a, 0x4a, 0xa8, 0xb9, 0xdf, 0x9b, 0x85, 0x07, 0x68, 0x29, 0x0f,
            0x18, 0x79, 0xdd, 0x9b, 0x85, 0x06, 0x60,
        ]);

        assert!(prg_rom_supports_smb_bounding_box_helper(
            &prg,
            prg.len() - 1
        ));

        prg[helper_offset + 50] = 0xf0;
        assert!(!prg_rom_supports_smb_bounding_box_helper(
            &prg,
            prg.len() - 1
        ));
    }

    #[test]
    fn smb_relative_position_helper_signature_is_exact() {
        let mut prg = vec![0xea; 32768];
        let offset = (SMB_RELATIVE_POSITION_HELPER_PC - 0x8000) as usize;
        prg[offset..offset + 21].copy_from_slice(&[
            0x85, 0x05, 0xa5, 0x07, 0xc5, 0x06, 0xb0, 0x0c, 0x4a, 0x4a, 0x4a, 0x29, 0x07, 0xc0,
            0x01, 0xb0, 0x02, 0x65, 0x05, 0xaa, 0x60,
        ]);

        assert!(prg_rom_supports_smb_relative_position_helper(
            &prg,
            prg.len() - 1
        ));

        prg[offset + 16] = 0x03;
        assert!(!prg_rom_supports_smb_relative_position_helper(
            &prg,
            prg.len() - 1
        ));
    }

    #[test]
    fn sprite0_poll_fast_forward_skips_failed_poll_iterations() {
        let mut prg = vec![0xea; 32768];
        let offset = (SMB_SPRITE0_POLL_PC - 0x8000) as usize;
        prg[offset..offset + 7].copy_from_slice(&[0xad, 0x02, 0x20, 0x29, 0x40, 0xf0, 0xf9]);
        let mut emu = NesEmulator::new_with_options(make_test_cart_with_prg(prg), true);
        emu.cpu.pc = SMB_SPRITE0_POLL_PC;
        emu.cpu.a = 0xff;
        emu.cpu.p = FLAG_U | FLAG_N;
        emu.ppu.status &= !0x40;
        emu.ppu.set_dot(PPU_PRERENDER_DOT);

        let mut cpu_cycle_guard = 0usize;
        let mut pending_ppu_cycles = 0usize;

        assert!(emu.try_fast_forward_sprite0_poll(&mut cpu_cycle_guard, &mut pending_ppu_cycles));
        assert_eq!(emu.cpu.pc, SMB_SPRITE0_POLL_PC);
        assert_eq!(emu.cpu.a, 0);
        assert!(emu.flag(FLAG_Z));
        assert!(!emu.flag(FLAG_N));
        assert!(pending_ppu_cycles >= PPU_SPRITE0_DOT - PPU_PRERENDER_DOT);
        assert_eq!(
            cpu_cycle_guard,
            (pending_ppu_cycles / SMB_SPRITE0_POLL_PPU_CYCLES) * 9
        );
    }

    #[test]
    fn sprite0_poll_fast_forward_stops_once_hit_is_set() {
        let mut prg = vec![0xea; 32768];
        let offset = (SMB_SPRITE0_POLL_PC - 0x8000) as usize;
        prg[offset..offset + 7].copy_from_slice(&[0xad, 0x02, 0x20, 0x29, 0x40, 0xf0, 0xf9]);
        let mut emu = NesEmulator::new_with_options(make_test_cart_with_prg(prg), true);
        emu.cpu.pc = SMB_SPRITE0_POLL_PC;
        emu.ppu.status |= 0x40;

        let mut cpu_cycle_guard = 0usize;
        let mut pending_ppu_cycles = 0usize;

        assert!(!emu.try_fast_forward_sprite0_poll(&mut cpu_cycle_guard, &mut pending_ppu_cycles));
        assert_eq!(cpu_cycle_guard, 0);
        assert_eq!(pending_ppu_cycles, 0);
    }

    #[test]
    fn sprite0_poll_exit_fast_forward_matches_interpreted_loop() {
        let mut prg = vec![0xea; 32768];
        let offset = (SMB_SPRITE0_POLL_PC - 0x8000) as usize;
        prg[offset..offset + 12].copy_from_slice(&[
            0xad, 0x02, 0x20, 0x29, 0x40, 0xf0, 0xf9, 0xa0, 0x14, 0x88, 0xd0, 0xfd,
        ]);
        let mut fast = NesEmulator::new_with_options(make_test_cart_with_prg(prg.clone()), true);
        let mut interpreted = NesEmulator::new_with_options(make_test_cart_with_prg(prg), true);
        for emu in [&mut fast, &mut interpreted] {
            emu.cpu.pc = SMB_SPRITE0_POLL_PC;
            emu.cpu.a = 0x7a;
            emu.cpu.x = 0x55;
            emu.cpu.y = 0xaa;
            emu.cpu.p = FLAG_U | FLAG_C | FLAG_N;
            emu.cpu.sp = 0xee;
            emu.ppu.status = 0xc0;
            emu.ppu.set_dot(PPU_SPRITE0_DOT);
        }

        let mut cpu_cycle_guard = 0usize;
        let mut pending_ppu_cycles = 0usize;
        assert!(fast.try_fast_forward_sprite0_poll(&mut cpu_cycle_guard, &mut pending_ppu_cycles));

        let mut interpreted_cycles = 0usize;
        while interpreted.cpu.pc != SMB_SPRITE0_POLL_PC + 12 {
            interpreted_cycles += interpreted.interpret_one_for_test() as usize;
        }

        assert_eq!(interpreted_cycles, SMB_SPRITE0_POLL_EXIT_CPU_CYCLES);
        assert_eq!(cpu_cycle_guard, SMB_SPRITE0_POLL_EXIT_CPU_CYCLES);
        assert_eq!(pending_ppu_cycles, SMB_SPRITE0_POLL_EXIT_PPU_CYCLES);
        assert_eq!(fast.cpu.a, interpreted.cpu.a);
        assert_eq!(fast.cpu.x, interpreted.cpu.x);
        assert_eq!(fast.cpu.y, interpreted.cpu.y);
        assert_eq!(fast.cpu.sp, interpreted.cpu.sp);
        assert_eq!(fast.cpu.pc, interpreted.cpu.pc);
        assert_eq!(fast.cpu.p, interpreted.cpu.p);
        assert_eq!(fast.ppu.status, interpreted.ppu.status);
    }

    #[test]
    fn timer_control_loop_fast_forward_matches_interpreted_loop() {
        let mut prg = vec![0xea; 32768];
        let offset = (SMB_TIMER_CONTROL_LOOP_PC - 0x8000) as usize;
        prg[offset..offset + 11].copy_from_slice(&[
            0xbd, 0x80, 0x07, 0xf0, 0x03, 0xde, 0x80, 0x07, 0xca, 0x10, 0xf5,
        ]);
        let mut fast = NesEmulator::new_with_options(make_test_cart_with_prg(prg.clone()), true);
        let mut interpreted = NesEmulator::new_with_options(make_test_cart_with_prg(prg), true);
        for emu in [&mut fast, &mut interpreted] {
            emu.cpu.pc = SMB_TIMER_CONTROL_LOOP_PC;
            emu.cpu.a = 0x7a;
            emu.cpu.x = 0x23;
            emu.cpu.y = 0xaa;
            emu.cpu.p = FLAG_U | FLAG_C | FLAG_Z;
            emu.cpu.sp = 0xee;
            emu.ppu.set_dot(PPU_PRERENDER_DOT);
            for index in 0..=0x23usize {
                emu.ram[0x0780 + index] = ((index * 13 + 5) % 4) as u8;
            }
        }

        let mut cpu_cycle_guard = 0usize;
        let mut pending_ppu_cycles = 0usize;
        assert!(
            fast.try_fast_forward_timer_control_loop(&mut cpu_cycle_guard, &mut pending_ppu_cycles)
        );

        let mut interpreted_cycles = 0usize;
        while interpreted.cpu.pc != SMB_TIMER_CONTROL_LOOP_PC + 11 {
            interpreted_cycles += interpreted.interpret_one_for_test() as usize;
        }

        assert_eq!(cpu_cycle_guard, interpreted_cycles);
        assert_eq!(pending_ppu_cycles, interpreted_cycles * 3);
        assert_eq!(fast.cpu.a, interpreted.cpu.a);
        assert_eq!(fast.cpu.x, interpreted.cpu.x);
        assert_eq!(fast.cpu.y, interpreted.cpu.y);
        assert_eq!(fast.cpu.sp, interpreted.cpu.sp);
        assert_eq!(fast.cpu.pc, interpreted.cpu.pc);
        assert_eq!(fast.cpu.p, interpreted.cpu.p);
        assert_eq!(fast.ram[0x0780..=0x07a3], interpreted.ram[0x0780..=0x07a3]);
    }

    #[test]
    fn timer_control_loop_fast_forward_does_not_cross_ppu_event() {
        let mut prg = vec![0xea; 32768];
        let offset = (SMB_TIMER_CONTROL_LOOP_PC - 0x8000) as usize;
        prg[offset..offset + 11].copy_from_slice(&[
            0xbd, 0x80, 0x07, 0xf0, 0x03, 0xde, 0x80, 0x07, 0xca, 0x10, 0xf5,
        ]);
        let mut emu = NesEmulator::new_with_options(make_test_cart_with_prg(prg), true);
        emu.cpu.pc = SMB_TIMER_CONTROL_LOOP_PC;
        emu.cpu.x = 0x23;
        for index in 0..=0x23usize {
            emu.ram[0x0780 + index] = 1;
        }
        let (cycles, _) = emu.timer_control_loop_cycles_and_last_a(emu.cpu.x);
        emu.ppu.set_dot(PPU_SPRITE0_DOT - cycles * 3);

        let mut cpu_cycle_guard = 0usize;
        let mut pending_ppu_cycles = 0usize;
        assert!(
            !emu.try_fast_forward_timer_control_loop(&mut cpu_cycle_guard, &mut pending_ppu_cycles)
        );
        assert_eq!(emu.cpu.pc, SMB_TIMER_CONTROL_LOOP_PC);
        assert_eq!(emu.cpu.x, 0x23);
        assert_eq!(cpu_cycle_guard, 0);
        assert_eq!(pending_ppu_cycles, 0);
    }

    #[test]
    fn sprite0_poll_exit_fast_forward_does_not_cross_ppu_event() {
        let mut prg = vec![0xea; 32768];
        let offset = (SMB_SPRITE0_POLL_PC - 0x8000) as usize;
        prg[offset..offset + 12].copy_from_slice(&[
            0xad, 0x02, 0x20, 0x29, 0x40, 0xf0, 0xf9, 0xa0, 0x14, 0x88, 0xd0, 0xfd,
        ]);
        let mut emu = NesEmulator::new_with_options(make_test_cart_with_prg(prg), true);
        emu.cpu.pc = SMB_SPRITE0_POLL_PC;
        emu.ppu.status |= 0x40;
        emu.ppu
            .set_dot(PPU_VBLANK_DOT - SMB_SPRITE0_POLL_EXIT_PPU_CYCLES + 1);

        let mut cpu_cycle_guard = 0usize;
        let mut pending_ppu_cycles = 0usize;
        assert!(!emu.try_fast_forward_sprite0_poll(&mut cpu_cycle_guard, &mut pending_ppu_cycles));
        assert_eq!(emu.cpu.pc, SMB_SPRITE0_POLL_PC);
        assert_eq!(cpu_cycle_guard, 0);
        assert_eq!(pending_ppu_cycles, 0);
    }

    #[test]
    fn oam_clear_fast_forward_matches_interpreted_routine() {
        let mut prg = vec![0xea; 32768];
        let offset = (SMB_OAM_CLEAR_PC - 0x8000) as usize;
        prg[offset..offset + 14].copy_from_slice(&[
            0xa0, 0x04, 0xa9, 0xf8, 0x99, 0x00, 0x02, 0xc8, 0xc8, 0xc8, 0xc8, 0xd0, 0xf7, 0x60,
        ]);
        let mut fast = NesEmulator::new_with_options(make_test_cart_with_prg(prg.clone()), true);
        let mut interpreted = NesEmulator::new_with_options(make_test_cart_with_prg(prg), true);
        for (index, value) in fast.ram.iter_mut().enumerate() {
            *value = ((index * 17 + 3) & 0xff) as u8;
        }
        interpreted.ram = fast.ram;
        for emu in [&mut fast, &mut interpreted] {
            emu.cpu.pc = SMB_OAM_CLEAR_PC;
            emu.cpu.a = 0x35;
            emu.cpu.x = 0x91;
            emu.cpu.y = 0xa7;
            emu.cpu.p = FLAG_U | FLAG_C | FLAG_N;
            emu.cpu.sp = 0xfd;
            emu.push_u16(0x9000);
            emu.ppu.set_dot(PPU_VBLANK_DOT);
        }

        let mut cpu_cycle_guard = 0usize;
        let mut pending_ppu_cycles = 0usize;
        assert!(fast.try_fast_forward_oam_clear(&mut cpu_cycle_guard, &mut pending_ppu_cycles));

        let mut interpreted_cycles = 0usize;
        while interpreted.cpu.pc != 0x9001 {
            interpreted_cycles += interpreted.interpret_one_for_test() as usize;
        }

        assert_eq!(interpreted_cycles, SMB_OAM_CLEAR_CPU_CYCLES);
        assert_eq!(cpu_cycle_guard, SMB_OAM_CLEAR_CPU_CYCLES);
        assert_eq!(pending_ppu_cycles, SMB_OAM_CLEAR_PPU_CYCLES);
        assert_eq!(fast.cpu.a, interpreted.cpu.a);
        assert_eq!(fast.cpu.x, interpreted.cpu.x);
        assert_eq!(fast.cpu.y, interpreted.cpu.y);
        assert_eq!(fast.cpu.sp, interpreted.cpu.sp);
        assert_eq!(fast.cpu.pc, interpreted.cpu.pc);
        assert_eq!(fast.cpu.p, interpreted.cpu.p);
        assert_eq!(fast.ram, interpreted.ram);
    }

    #[test]
    fn oam_clear_fast_forward_does_not_cross_ppu_event() {
        let mut prg = vec![0xea; 32768];
        let offset = (SMB_OAM_CLEAR_PC - 0x8000) as usize;
        prg[offset..offset + 14].copy_from_slice(&[
            0xa0, 0x04, 0xa9, 0xf8, 0x99, 0x00, 0x02, 0xc8, 0xc8, 0xc8, 0xc8, 0xd0, 0xf7, 0x60,
        ]);
        let mut emu = NesEmulator::new_with_options(make_test_cart_with_prg(prg), true);
        emu.cpu.pc = SMB_OAM_CLEAR_PC;
        emu.ppu.set_dot(PPU_VBLANK_DOT - 1);

        let mut cpu_cycle_guard = 0usize;
        let mut pending_ppu_cycles = 0usize;
        assert!(!emu.try_fast_forward_oam_clear(&mut cpu_cycle_guard, &mut pending_ppu_cycles));
        assert_eq!(emu.cpu.pc, SMB_OAM_CLEAR_PC);
        assert_eq!(cpu_cycle_guard, 0);
        assert_eq!(pending_ppu_cycles, 0);
    }

    fn assert_controller_read_fast_forward_matches_interpreted(
        x: u8,
        controller_state: u8,
        prior_buttons: u8,
        expected_cycles: usize,
    ) {
        let mut prg = vec![0xea; 32768];
        let offset = (SMB_CONTROLLER_READ_PC - 0x8000) as usize;
        prg[offset..offset + 40].copy_from_slice(&[
            0xa0, 0x08, 0x48, 0xbd, 0x16, 0x40, 0x85, 0x00, 0x4a, 0x05, 0x00, 0x4a, 0x68, 0x2a,
            0x88, 0xd0, 0xf1, 0x9d, 0xfc, 0x06, 0x48, 0x29, 0x30, 0x3d, 0x4a, 0x07, 0xf0, 0x07,
            0x68, 0x29, 0xcf, 0x9d, 0xfc, 0x06, 0x60, 0x68, 0x9d, 0x4a, 0x07, 0x60,
        ]);
        let mut fast = NesEmulator::new_with_options(make_test_cart_with_prg(prg.clone()), true);
        let mut interpreted = NesEmulator::new_with_options(make_test_cart_with_prg(prg), true);
        for (index, value) in fast.ram.iter_mut().enumerate() {
            *value = ((index * 19 + 11) & 0xff) as u8;
        }
        fast.ram[0x074a + x as usize] = prior_buttons;
        interpreted.ram = fast.ram;
        for emu in [&mut fast, &mut interpreted] {
            emu.cpu.pc = SMB_CONTROLLER_READ_PC;
            emu.cpu.a = 0x00;
            emu.cpu.x = x;
            emu.cpu.y = 0x55;
            emu.cpu.p = FLAG_U | FLAG_C | FLAG_N;
            emu.cpu.sp = 0xfd;
            emu.push_u16(0x9000);
            emu.controller_state = controller_state;
            emu.controller_shift = controller_state;
            emu.controller_strobe = false;
            emu.ppu.set_dot(PPU_VBLANK_DOT);
        }

        let mut cpu_cycle_guard = 0usize;
        let mut pending_ppu_cycles = 0usize;
        assert!(
            fast.try_fast_forward_controller_read(&mut cpu_cycle_guard, &mut pending_ppu_cycles)
        );

        let mut interpreted_cycles = 0usize;
        while interpreted.cpu.pc != 0x9001 {
            interpreted_cycles += interpreted.interpret_one_for_test() as usize;
        }

        assert_eq!(interpreted_cycles, expected_cycles);
        assert_eq!(cpu_cycle_guard, expected_cycles);
        assert_eq!(pending_ppu_cycles, expected_cycles * 3);
        assert_eq!(fast.cpu.a, interpreted.cpu.a);
        assert_eq!(fast.cpu.x, interpreted.cpu.x);
        assert_eq!(fast.cpu.y, interpreted.cpu.y);
        assert_eq!(fast.cpu.sp, interpreted.cpu.sp);
        assert_eq!(fast.cpu.pc, interpreted.cpu.pc);
        assert_eq!(fast.cpu.p, interpreted.cpu.p);
        assert_eq!(fast.ram, interpreted.ram);
        assert_eq!(fast.controller_shift, interpreted.controller_shift);
        assert_eq!(fast.controller_strobe, interpreted.controller_strobe);
    }

    #[test]
    fn controller_read_fast_forward_matches_interpreted_new_buttons() {
        assert_controller_read_fast_forward_matches_interpreted(
            0,
            0b0011_0101,
            0x00,
            SMB_CONTROLLER_READ_TAKEN_CPU_CYCLES,
        );
    }

    #[test]
    fn controller_read_fast_forward_matches_interpreted_duplicate_start_select() {
        assert_controller_read_fast_forward_matches_interpreted(
            0,
            0b0000_1100,
            0x30,
            SMB_CONTROLLER_READ_NOT_TAKEN_CPU_CYCLES,
        );
    }

    #[test]
    fn controller_read_fast_forward_matches_interpreted_second_controller() {
        assert_controller_read_fast_forward_matches_interpreted(
            1,
            0b1111_1111,
            0x00,
            SMB_CONTROLLER_READ_TAKEN_CPU_CYCLES,
        );
    }

    #[test]
    fn controller_read_fast_forward_does_not_cross_ppu_event() {
        let mut prg = vec![0xea; 32768];
        let offset = (SMB_CONTROLLER_READ_PC - 0x8000) as usize;
        prg[offset..offset + 40].copy_from_slice(&[
            0xa0, 0x08, 0x48, 0xbd, 0x16, 0x40, 0x85, 0x00, 0x4a, 0x05, 0x00, 0x4a, 0x68, 0x2a,
            0x88, 0xd0, 0xf1, 0x9d, 0xfc, 0x06, 0x48, 0x29, 0x30, 0x3d, 0x4a, 0x07, 0xf0, 0x07,
            0x68, 0x29, 0xcf, 0x9d, 0xfc, 0x06, 0x60, 0x68, 0x9d, 0x4a, 0x07, 0x60,
        ]);
        let mut emu = NesEmulator::new_with_options(make_test_cart_with_prg(prg), true);
        emu.cpu.pc = SMB_CONTROLLER_READ_PC;
        emu.cpu.x = 0;
        emu.ppu
            .set_dot(PPU_PRERENDER_DOT - SMB_CONTROLLER_READ_TAKEN_CPU_CYCLES * 3 + 1);

        let mut cpu_cycle_guard = 0usize;
        let mut pending_ppu_cycles = 0usize;
        assert!(
            !emu.try_fast_forward_controller_read(&mut cpu_cycle_guard, &mut pending_ppu_cycles)
        );
        assert_eq!(emu.cpu.pc, SMB_CONTROLLER_READ_PC);
        assert_eq!(cpu_cycle_guard, 0);
        assert_eq!(pending_ppu_cycles, 0);
    }

    #[test]
    fn scroll_slot_loop_fast_forward_matches_interpreted_loop() {
        let mut prg = vec![0xea; 32768];
        let offset = (SMB_SCROLL_SLOT_LOOP_PC - 0x8000) as usize;
        prg[offset..offset + 25].copy_from_slice(&[
            0xbd, 0xe4, 0x06, 0xc5, 0x00, 0x90, 0x0f, 0xac, 0xe0, 0x06, 0x18, 0x79, 0xe1, 0x06,
            0x90, 0x03, 0x18, 0x65, 0x00, 0x9d, 0xe4, 0x06, 0xca, 0x10, 0xe7,
        ]);
        let mut interpreted =
            NesEmulator::new_with_options(make_test_cart_with_prg(prg.clone()), true);
        let mut fast = NesEmulator::new_with_options(make_test_cart_with_prg(prg), true);
        for idx in 0..0x800usize {
            let value = ((idx * 17 + idx / 3 + 29) & 0xff) as u8;
            interpreted.ram[idx] = value;
            fast.ram[idx] = value;
        }
        for emu in [&mut interpreted, &mut fast] {
            emu.cpu.pc = SMB_SCROLL_SLOT_LOOP_PC;
            emu.cpu.x = 0x0e;
            emu.cpu.a = 0x55;
            emu.cpu.y = 0x03;
            emu.cpu.p = FLAG_U | FLAG_C;
            emu.ppu.set_dot(PPU_VBLANK_DOT);
            emu.ram[0] = 0x28;
            emu.ram[0x06e0] = 0x02;
            emu.ram[0x06e1] = 0xf0;
            emu.ram[0x06e2] = 0x18;
            emu.ram[0x06e3] = 0x44;
        }

        let mut interpreted_cycles = 0usize;
        while interpreted.cpu.pc != 0x81e8 {
            interpreted_cycles += interpreted.interpret_one_for_test() as usize;
        }
        let mut cpu_cycle_guard = 0usize;
        let mut pending_ppu_cycles = 0usize;
        assert!(
            fast.try_fast_forward_scroll_slot_loop(&mut cpu_cycle_guard, &mut pending_ppu_cycles)
        );

        assert_eq!(interpreted_cycles, cpu_cycle_guard);
        assert_eq!(pending_ppu_cycles, interpreted_cycles * 3);
        assert_eq!(fast.cpu.a, interpreted.cpu.a);
        assert_eq!(fast.cpu.x, interpreted.cpu.x);
        assert_eq!(fast.cpu.y, interpreted.cpu.y);
        assert_eq!(fast.cpu.p, interpreted.cpu.p);
        assert_eq!(fast.cpu.pc, interpreted.cpu.pc);
        assert_eq!(&fast.ram[..], &interpreted.ram[..]);
    }

    #[test]
    fn scroll_slot_loop_fast_forward_does_not_cross_ppu_event() {
        let mut prg = vec![0xea; 32768];
        let offset = (SMB_SCROLL_SLOT_LOOP_PC - 0x8000) as usize;
        prg[offset..offset + 25].copy_from_slice(&[
            0xbd, 0xe4, 0x06, 0xc5, 0x00, 0x90, 0x0f, 0xac, 0xe0, 0x06, 0x18, 0x79, 0xe1, 0x06,
            0x90, 0x03, 0x18, 0x65, 0x00, 0x9d, 0xe4, 0x06, 0xca, 0x10, 0xe7,
        ]);
        let mut emu = NesEmulator::new_with_options(make_test_cart_with_prg(prg), true);
        emu.cpu.pc = SMB_SCROLL_SLOT_LOOP_PC;
        emu.cpu.x = 0x0e;
        emu.ppu.set_dot(PPU_PRERENDER_DOT - 1);
        let mut cpu_cycle_guard = 0usize;
        let mut pending_ppu_cycles = 0usize;

        assert!(
            !emu.try_fast_forward_scroll_slot_loop(&mut cpu_cycle_guard, &mut pending_ppu_cycles)
        );
        assert_eq!(emu.cpu.pc, SMB_SCROLL_SLOT_LOOP_PC);
        assert_eq!(cpu_cycle_guard, 0);
        assert_eq!(pending_ppu_cycles, 0);
    }

    fn add_digit_math_routine(prg: &mut [u8]) {
        let offset = (SMB_DIGIT_MATH_LOOP_PC - 0x8000) as usize;
        prg[offset..offset + 27].copy_from_slice(&[
            0xbd, 0xdd, 0x07, 0xf9, 0xd7, 0x07, 0xca, 0x88, 0x10, 0xf6, 0x90, 0x0e, 0xe8, 0xc8,
            0xbd, 0xdd, 0x07, 0x99, 0xd7, 0x07, 0xe8, 0xc8, 0xc0, 0x06, 0x90, 0xf4, 0x60,
        ]);
    }

    fn assert_digit_math_fast_forward_matches_interpreted(
        carry_out: bool,
        start_x: u8,
        start_y: u8,
    ) {
        let mut prg = vec![0xea; 32768];
        add_digit_math_routine(&mut prg);
        let mut fast = NesEmulator::new_with_options(make_test_cart_with_prg(prg.clone()), true);
        let mut interpreted = NesEmulator::new_with_options(make_test_cart_with_prg(prg), true);
        for (index, value) in fast.ram.iter_mut().enumerate() {
            *value = ((index * 31 + index / 7 + 19) & 0xff) as u8;
        }
        let mut last_x = start_x;
        let mut last_y = start_y;
        loop {
            let next_y = last_y.wrapping_sub(1);
            if next_y & 0x80 != 0 {
                break;
            }
            last_x = last_x.wrapping_sub(1);
            last_y = next_y;
        }
        let source = (0x07ddusize + last_x as usize) & 0x07ff;
        let subtrahend = (0x07d7usize + last_y as usize) & 0x07ff;
        fast.ram[source] = if carry_out { 0xf0 } else { 0x10 };
        fast.ram[subtrahend] = if carry_out { 0x10 } else { 0xf0 };
        interpreted.ram = fast.ram;
        for emu in [&mut fast, &mut interpreted] {
            emu.cpu.pc = SMB_DIGIT_MATH_LOOP_PC;
            emu.cpu.a = 0x55;
            emu.cpu.x = start_x;
            emu.cpu.y = start_y;
            emu.cpu.p = FLAG_U | FLAG_C | FLAG_D | FLAG_I | FLAG_V;
            emu.cpu.sp = 0xf9;
            emu.push_u16(0x9122);
            emu.extra_cycles = 7;
            emu.ppu.set_dot(PPU_VBLANK_DOT);
        }

        let mut cpu_cycle_guard = 0usize;
        let mut pending_ppu_cycles = 0usize;
        assert!(
            fast.try_fast_forward_digit_math_loop(&mut cpu_cycle_guard, &mut pending_ppu_cycles)
        );

        let mut interpreted_cycles = 0usize;
        while interpreted.cpu.pc != 0x9123 {
            interpreted_cycles += interpreted.interpret_one_for_test() as usize;
            assert!(interpreted_cycles < 500);
        }

        assert_eq!(cpu_cycle_guard, interpreted_cycles);
        assert_eq!(pending_ppu_cycles, interpreted_cycles * 3);
        assert_eq!(fast.cpu.a, interpreted.cpu.a);
        assert_eq!(fast.cpu.x, interpreted.cpu.x);
        assert_eq!(fast.cpu.y, interpreted.cpu.y);
        assert_eq!(fast.cpu.sp, interpreted.cpu.sp);
        assert_eq!(fast.cpu.pc, interpreted.cpu.pc);
        assert_eq!(fast.cpu.p, interpreted.cpu.p);
        assert_eq!(fast.extra_cycles, interpreted.extra_cycles);
        assert_eq!(fast.ram, interpreted.ram);
    }

    #[test]
    fn digit_math_fast_forward_matches_interpreted_carry_clear() {
        assert_digit_math_fast_forward_matches_interpreted(false, 2, 1);
    }

    #[test]
    fn digit_math_fast_forward_matches_interpreted_carry_set() {
        assert_digit_math_fast_forward_matches_interpreted(true, 2, 1);
    }

    #[test]
    fn digit_math_fast_forward_matches_page_crossing_indices() {
        assert_digit_math_fast_forward_matches_interpreted(true, 0x40, 0x81);
    }

    #[test]
    fn digit_math_fast_forward_does_not_cross_ppu_event() {
        let mut prg = vec![0xea; 32768];
        add_digit_math_routine(&mut prg);
        let mut emu = NesEmulator::new_with_options(make_test_cart_with_prg(prg), true);
        emu.cpu.pc = SMB_DIGIT_MATH_LOOP_PC;
        emu.cpu.x = 2;
        emu.cpu.y = 1;
        emu.ppu.set_dot(PPU_PRERENDER_DOT - 1);
        let original_cpu = emu.cpu;
        let original_ram = emu.ram;
        let mut cpu_cycle_guard = 0usize;
        let mut pending_ppu_cycles = 0usize;

        assert!(
            !emu.try_fast_forward_digit_math_loop(&mut cpu_cycle_guard, &mut pending_ppu_cycles)
        );
        assert_eq!(emu.cpu.a, original_cpu.a);
        assert_eq!(emu.cpu.x, original_cpu.x);
        assert_eq!(emu.cpu.y, original_cpu.y);
        assert_eq!(emu.cpu.sp, original_cpu.sp);
        assert_eq!(emu.cpu.pc, original_cpu.pc);
        assert_eq!(emu.cpu.p, original_cpu.p);
        assert_eq!(emu.ram, original_ram);
        assert_eq!(cpu_cycle_guard, 0);
        assert_eq!(pending_ppu_cycles, 0);
    }

    fn add_relative_position_helper(prg: &mut [u8]) {
        let offset = (SMB_RELATIVE_POSITION_HELPER_PC - 0x8000) as usize;
        prg[offset..offset + 21].copy_from_slice(&[
            0x85, 0x05, 0xa5, 0x07, 0xc5, 0x06, 0xb0, 0x0c, 0x4a, 0x4a, 0x4a, 0x29, 0x07, 0xc0,
            0x01, 0xb0, 0x02, 0x65, 0x05, 0xaa, 0x60,
        ]);
    }

    fn add_draw_sprite_object_routine(prg: &mut [u8]) {
        let offset = (SMB_DRAW_SPRITE_OBJECT_PC - 0x8000) as usize;
        prg[offset..offset + 72].copy_from_slice(&[
            0xa5, 0x03, 0x4a, 0x4a, 0xa5, 0x00, 0x90, 0x0c, 0x99, 0x05, 0x02, 0xa5, 0x01, 0x99,
            0x01, 0x02, 0xa9, 0x40, 0xd0, 0x0a, 0x99, 0x01, 0x02, 0xa5, 0x01, 0x99, 0x05, 0x02,
            0xa9, 0x00, 0x05, 0x04, 0x99, 0x02, 0x02, 0x99, 0x06, 0x02, 0xa5, 0x02, 0x99, 0x00,
            0x02, 0x99, 0x04, 0x02, 0xa5, 0x05, 0x99, 0x03, 0x02, 0x18, 0x69, 0x08, 0x99, 0x07,
            0x02, 0xa5, 0x02, 0x18, 0x69, 0x08, 0x85, 0x02, 0x98, 0x18, 0x69, 0x08, 0xa8, 0xe8,
            0xe8, 0x60,
        ]);
    }

    fn add_offscreen_bits_subs(prg: &mut [u8]) {
        for (address, bytes) in SMB_OFFSCREEN_BITS_SEGMENTS {
            let offset = (*address - 0x8000) as usize;
            prg[offset..offset + bytes.len()].copy_from_slice(bytes);
        }
        add_relative_position_helper(prg);
    }

    #[test]
    fn offscreen_bits_subs_fast_forward_matches_interpreter_states() {
        let mut prg = vec![0xea; 32768];
        add_offscreen_bits_subs(&mut prg);
        let fast_template =
            NesEmulator::new_with_options(make_test_cart_with_prg(prg.clone()), true);
        let interpreted_template =
            NesEmulator::new_with_options(make_test_cart_with_prg(prg), true);

        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        for case in 0..128u8 {
            let mut fast = fast_template.clone();
            let mut interpreted = interpreted_template.clone();
            for value in &mut fast.ram {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                *value = seed as u8;
            }
            interpreted.ram = fast.ram;
            for emu in [&mut fast, &mut interpreted] {
                emu.cpu.pc = SMB_OFFSCREEN_BITS_SUBS_PC;
                emu.cpu.a = case.wrapping_mul(37);
                emu.cpu.x = case.wrapping_mul(11);
                emu.cpu.y = case.wrapping_mul(19);
                emu.cpu.p = FLAG_U | (case & (FLAG_C | FLAG_D | FLAG_I | FLAG_V | FLAG_N));
                emu.cpu.sp = 0xf7;
                emu.push_u16(0x9122);
                emu.extra_cycles = 7;
                emu.ppu.set_dot(PPU_VBLANK_DOT);
            }

            let mut cpu_cycle_guard = 0usize;
            let mut pending_ppu_cycles = 0usize;
            assert!(fast.try_fast_forward_offscreen_bits_subs(
                &mut cpu_cycle_guard,
                &mut pending_ppu_cycles
            ));

            let mut interpreted_cycles = 0usize;
            while interpreted.cpu.pc != 0x9123 {
                interpreted_cycles += interpreted.interpret_one_for_test() as usize;
                assert!(interpreted_cycles < SMB_OFFSCREEN_BITS_SUBS_MAX_CPU_CYCLES + 20);
            }

            assert_eq!(cpu_cycle_guard, interpreted_cycles, "cycle case {case}");
            assert_eq!(pending_ppu_cycles, interpreted_cycles * 3);
            assert_eq!(fast.cpu.a, interpreted.cpu.a, "a case {case}");
            assert_eq!(fast.cpu.x, interpreted.cpu.x, "x case {case}");
            assert_eq!(fast.cpu.y, interpreted.cpu.y, "y case {case}");
            assert_eq!(fast.cpu.sp, interpreted.cpu.sp, "sp case {case}");
            assert_eq!(fast.cpu.pc, interpreted.cpu.pc, "pc case {case}");
            assert_eq!(fast.cpu.p, interpreted.cpu.p, "p case {case}");
            assert_eq!(fast.extra_cycles, interpreted.extra_cycles);
            assert_eq!(fast.ram, interpreted.ram, "ram case {case}");
        }
    }

    #[test]
    fn offscreen_bits_subs_fast_forward_does_not_cross_ppu_event() {
        let mut prg = vec![0xea; 32768];
        add_offscreen_bits_subs(&mut prg);
        let mut emu = NesEmulator::new_with_options(make_test_cart_with_prg(prg), true);
        emu.cpu.pc = SMB_OFFSCREEN_BITS_SUBS_PC;
        emu.ppu.set_dot(PPU_PRERENDER_DOT - 1);
        let original_cpu = emu.cpu;
        let original_ram = emu.ram;
        let mut cpu_cycle_guard = 0usize;
        let mut pending_ppu_cycles = 0usize;

        assert!(!emu
            .try_fast_forward_offscreen_bits_subs(&mut cpu_cycle_guard, &mut pending_ppu_cycles));
        assert_eq!(emu.cpu.a, original_cpu.a);
        assert_eq!(emu.cpu.x, original_cpu.x);
        assert_eq!(emu.cpu.y, original_cpu.y);
        assert_eq!(emu.cpu.sp, original_cpu.sp);
        assert_eq!(emu.cpu.pc, original_cpu.pc);
        assert_eq!(emu.cpu.p, original_cpu.p);
        assert_eq!(emu.ram, original_ram);
        assert_eq!(cpu_cycle_guard, 0);
        assert_eq!(pending_ppu_cycles, 0);
    }

    fn assert_draw_sprite_object_fast_forward_matches_interpreted(
        flip: bool,
        start_x: u8,
        start_y: u8,
    ) {
        let mut prg = vec![0xea; 32768];
        add_draw_sprite_object_routine(&mut prg);
        let mut fast = NesEmulator::new_with_options(make_test_cart_with_prg(prg.clone()), true);
        let mut interpreted = NesEmulator::new_with_options(make_test_cart_with_prg(prg), true);
        for (index, value) in fast.ram.iter_mut().enumerate() {
            *value = ((index * 29 + index / 11 + 7) & 0xff) as u8;
        }
        fast.ram[0x0003] = if flip { 0x02 } else { 0x00 };
        interpreted.ram = fast.ram;
        for emu in [&mut fast, &mut interpreted] {
            emu.cpu.pc = SMB_DRAW_SPRITE_OBJECT_PC;
            emu.cpu.a = 0x55;
            emu.cpu.x = start_x;
            emu.cpu.y = start_y;
            emu.cpu.p = FLAG_U | FLAG_C | FLAG_D | FLAG_I | FLAG_V | FLAG_N;
            emu.cpu.sp = 0xf9;
            emu.push_u16(0x9122);
            emu.extra_cycles = 7;
            emu.ppu.set_dot(PPU_VBLANK_DOT);
        }

        let mut cpu_cycle_guard = 0usize;
        let mut pending_ppu_cycles = 0usize;
        assert!(
            fast.try_fast_forward_draw_sprite_object(&mut cpu_cycle_guard, &mut pending_ppu_cycles)
        );

        let mut interpreted_cycles = 0usize;
        while interpreted.cpu.pc != 0x9123 {
            interpreted_cycles += interpreted.interpret_one_for_test() as usize;
            assert!(interpreted_cycles < 200);
        }

        assert_eq!(cpu_cycle_guard, interpreted_cycles);
        assert_eq!(pending_ppu_cycles, interpreted_cycles * 3);
        assert_eq!(fast.cpu.a, interpreted.cpu.a);
        assert_eq!(fast.cpu.x, interpreted.cpu.x);
        assert_eq!(fast.cpu.y, interpreted.cpu.y);
        assert_eq!(fast.cpu.sp, interpreted.cpu.sp);
        assert_eq!(fast.cpu.pc, interpreted.cpu.pc);
        assert_eq!(fast.cpu.p, interpreted.cpu.p);
        assert_eq!(fast.extra_cycles, interpreted.extra_cycles);
        assert_eq!(fast.ram, interpreted.ram);
    }

    #[test]
    fn draw_sprite_object_fast_forward_matches_unflipped_path() {
        assert_draw_sprite_object_fast_forward_matches_interpreted(false, 0x20, 0x18);
    }

    #[test]
    fn draw_sprite_object_fast_forward_matches_flipped_wrapping_path() {
        assert_draw_sprite_object_fast_forward_matches_interpreted(true, 0xff, 0xfc);
    }

    #[test]
    fn draw_sprite_object_fast_forward_does_not_cross_ppu_event() {
        let mut prg = vec![0xea; 32768];
        add_draw_sprite_object_routine(&mut prg);
        let mut emu = NesEmulator::new_with_options(make_test_cart_with_prg(prg), true);
        emu.cpu.pc = SMB_DRAW_SPRITE_OBJECT_PC;
        emu.ppu.set_dot(PPU_PRERENDER_DOT - 1);
        let original_cpu = emu.cpu;
        let original_ram = emu.ram;
        let mut cpu_cycle_guard = 0usize;
        let mut pending_ppu_cycles = 0usize;

        assert!(
            !emu.try_fast_forward_draw_sprite_object(&mut cpu_cycle_guard, &mut pending_ppu_cycles)
        );
        assert_eq!(emu.cpu.a, original_cpu.a);
        assert_eq!(emu.cpu.x, original_cpu.x);
        assert_eq!(emu.cpu.y, original_cpu.y);
        assert_eq!(emu.cpu.sp, original_cpu.sp);
        assert_eq!(emu.cpu.pc, original_cpu.pc);
        assert_eq!(emu.cpu.p, original_cpu.p);
        assert_eq!(emu.ram, original_ram);
        assert_eq!(cpu_cycle_guard, 0);
        assert_eq!(pending_ppu_cycles, 0);
    }

    fn assert_relative_position_fast_forward_matches_interpreted(
        value: u8,
        threshold: u8,
        y: u8,
        expected_routine_cycles: usize,
    ) {
        let mut prg = vec![0xea; 32768];
        add_relative_position_helper(&mut prg);
        let mut fast = NesEmulator::new_with_options(make_test_cart_with_prg(prg.clone()), true);
        let mut interpreted = NesEmulator::new_with_options(make_test_cart_with_prg(prg), true);
        for emu in [&mut fast, &mut interpreted] {
            emu.cpu.pc = SMB_RELATIVE_POSITION_HELPER_PC;
            emu.cpu.a = 0x13;
            emu.cpu.x = 0xa5;
            emu.cpu.y = y;
            emu.cpu.p = FLAG_U | FLAG_C | FLAG_D | FLAG_I | FLAG_V | FLAG_N;
            emu.cpu.sp = 0xf9;
            emu.push_u16(0x9122);
            emu.ram[0x0005] = 0xcc;
            emu.ram[0x0006] = threshold;
            emu.ram[0x0007] = value;
            emu.extra_cycles = 5;
            emu.ppu.set_dot(PPU_VBLANK_DOT);
        }

        let mut cpu_cycle_guard = 0usize;
        let mut pending_ppu_cycles = 0usize;
        assert!(fast.try_fast_forward_relative_position_helper(
            &mut cpu_cycle_guard,
            &mut pending_ppu_cycles
        ));

        let mut interpreted_cycles = 0usize;
        while interpreted.cpu.pc != 0x9123 {
            interpreted_cycles += interpreted.interpret_one_for_test() as usize;
            assert!(interpreted_cycles < 100);
        }

        assert_eq!(interpreted_cycles, expected_routine_cycles + 5);
        assert_eq!(cpu_cycle_guard, interpreted_cycles);
        assert_eq!(pending_ppu_cycles, interpreted_cycles * 3);
        assert_eq!(fast.cpu.a, interpreted.cpu.a);
        assert_eq!(fast.cpu.x, interpreted.cpu.x);
        assert_eq!(fast.cpu.y, interpreted.cpu.y);
        assert_eq!(fast.cpu.sp, interpreted.cpu.sp);
        assert_eq!(fast.cpu.pc, interpreted.cpu.pc);
        assert_eq!(fast.cpu.p, interpreted.cpu.p);
        assert_eq!(fast.extra_cycles, interpreted.extra_cycles);
        assert_eq!(fast.ram, interpreted.ram);
    }

    #[test]
    fn relative_position_fast_forward_matches_first_carry_path() {
        assert_relative_position_fast_forward_matches_interpreted(0x40, 0x20, 0, 18);
    }

    #[test]
    fn relative_position_fast_forward_matches_y_carry_path() {
        assert_relative_position_fast_forward_matches_interpreted(0x10, 0x20, 1, 32);
    }

    #[test]
    fn relative_position_fast_forward_matches_adc_path() {
        assert_relative_position_fast_forward_matches_interpreted(0x10, 0x20, 0, 34);
    }

    #[test]
    fn relative_position_fast_forward_does_not_cross_ppu_event() {
        let mut prg = vec![0xea; 32768];
        add_relative_position_helper(&mut prg);
        let mut emu = NesEmulator::new_with_options(make_test_cart_with_prg(prg), true);
        emu.cpu.pc = SMB_RELATIVE_POSITION_HELPER_PC;
        emu.ppu.set_dot(PPU_PRERENDER_DOT - 1);
        let original_cpu = emu.cpu;
        let original_ram = emu.ram;
        let mut cpu_cycle_guard = 0usize;
        let mut pending_ppu_cycles = 0usize;

        assert!(!emu.try_fast_forward_relative_position_helper(
            &mut cpu_cycle_guard,
            &mut pending_ppu_cycles
        ));
        assert_eq!(emu.cpu.a, original_cpu.a);
        assert_eq!(emu.cpu.x, original_cpu.x);
        assert_eq!(emu.cpu.y, original_cpu.y);
        assert_eq!(emu.cpu.sp, original_cpu.sp);
        assert_eq!(emu.cpu.pc, original_cpu.pc);
        assert_eq!(emu.cpu.p, original_cpu.p);
        assert_eq!(emu.ram, original_ram);
        assert_eq!(cpu_cycle_guard, 0);
        assert_eq!(pending_ppu_cycles, 0);
    }

    fn assert_bounding_box_nibble_fast_forward_matches_interpreted(a: u8, p: u8) {
        let mut prg = vec![0xea; 32768];
        let offset = (SMB_BOUNDING_BOX_NIBBLE_PC - 0x8000) as usize;
        prg[offset..offset + 21].copy_from_slice(&[
            0x48, 0x4a, 0x4a, 0x4a, 0x4a, 0xa8, 0xb9, 0xdf, 0x9b, 0x85, 0x07, 0x68, 0x29, 0x0f,
            0x18, 0x79, 0xdd, 0x9b, 0x85, 0x06, 0x60,
        ]);
        let mut fast = NesEmulator::new_with_options(make_test_cart_with_prg(prg.clone()), true);
        let mut interpreted = NesEmulator::new_with_options(make_test_cart_with_prg(prg), true);
        for (index, value) in fast.ram.iter_mut().enumerate() {
            *value = ((index * 23 + 7) & 0xff) as u8;
        }
        interpreted.ram = fast.ram;
        for emu in [&mut fast, &mut interpreted] {
            emu.cpu.pc = SMB_BOUNDING_BOX_NIBBLE_PC;
            emu.cpu.a = a;
            emu.cpu.x = 0xa5;
            emu.cpu.y = 0x39;
            emu.cpu.p = p | FLAG_U;
            emu.cpu.sp = 0xf7;
            emu.push_u16(0x9122);
            emu.extra_cycles = 5;
            emu.ppu.set_dot(PPU_VBLANK_DOT);
        }

        let mut cpu_cycle_guard = 0usize;
        let mut pending_ppu_cycles = 0usize;
        assert!(fast
            .try_fast_forward_bounding_box_nibble(&mut cpu_cycle_guard, &mut pending_ppu_cycles));

        let mut interpreted_cycles = 0usize;
        while interpreted.cpu.pc != 0x9123 {
            interpreted_cycles += interpreted.interpret_one_for_test() as usize;
        }

        assert_eq!(interpreted_cycles, SMB_BOUNDING_BOX_NIBBLE_CPU_CYCLES + 5);
        assert_eq!(cpu_cycle_guard, interpreted_cycles);
        assert_eq!(pending_ppu_cycles, interpreted_cycles * 3);
        assert_eq!(fast.cpu.a, interpreted.cpu.a);
        assert_eq!(fast.cpu.x, interpreted.cpu.x);
        assert_eq!(fast.cpu.y, interpreted.cpu.y);
        assert_eq!(fast.cpu.sp, interpreted.cpu.sp);
        assert_eq!(fast.cpu.pc, interpreted.cpu.pc);
        assert_eq!(fast.cpu.p, interpreted.cpu.p);
        assert_eq!(fast.extra_cycles, interpreted.extra_cycles);
        assert_eq!(fast.ram, interpreted.ram);
    }

    #[test]
    fn bounding_box_nibble_fast_forward_matches_interpreted_low_nibble() {
        assert_bounding_box_nibble_fast_forward_matches_interpreted(0x0f, FLAG_C | FLAG_V | FLAG_N);
    }

    #[test]
    fn bounding_box_nibble_fast_forward_matches_interpreted_high_nibble() {
        assert_bounding_box_nibble_fast_forward_matches_interpreted(0xf3, FLAG_D | FLAG_I | FLAG_Z);
    }

    #[test]
    fn bounding_box_nibble_fast_forward_does_not_cross_ppu_event() {
        let mut prg = vec![0xea; 32768];
        let offset = (SMB_BOUNDING_BOX_NIBBLE_PC - 0x8000) as usize;
        prg[offset..offset + 21].copy_from_slice(&[
            0x48, 0x4a, 0x4a, 0x4a, 0x4a, 0xa8, 0xb9, 0xdf, 0x9b, 0x85, 0x07, 0x68, 0x29, 0x0f,
            0x18, 0x79, 0xdd, 0x9b, 0x85, 0x06, 0x60,
        ]);
        let mut emu = NesEmulator::new_with_options(make_test_cart_with_prg(prg), true);
        emu.cpu.pc = SMB_BOUNDING_BOX_NIBBLE_PC;
        emu.ppu
            .set_dot(PPU_PRERENDER_DOT - SMB_BOUNDING_BOX_NIBBLE_CPU_CYCLES * 3 + 1);
        let mut cpu_cycle_guard = 0usize;
        let mut pending_ppu_cycles = 0usize;

        assert!(!emu
            .try_fast_forward_bounding_box_nibble(&mut cpu_cycle_guard, &mut pending_ppu_cycles));
        assert_eq!(emu.cpu.pc, SMB_BOUNDING_BOX_NIBBLE_PC);
        assert_eq!(cpu_cycle_guard, 0);
        assert_eq!(pending_ppu_cycles, 0);
    }

    fn add_bounding_box_routines(prg: &mut [u8]) {
        let helper_offset = (SMB_BOUNDING_BOX_HELPER_PC - 0x8000) as usize;
        prg[helper_offset..helper_offset + 66].copy_from_slice(&[
            0x48, 0x84, 0x04, 0xb9, 0xb0, 0xe3, 0x18, 0x75, 0x86, 0x85, 0x05, 0xb5, 0x6d, 0x69,
            0x00, 0x29, 0x01, 0x4a, 0x05, 0x05, 0x6a, 0x4a, 0x4a, 0x4a, 0x20, 0xe1, 0x9b, 0xa4,
            0x04, 0xb5, 0xce, 0x18, 0x79, 0xcc, 0xe3, 0x29, 0xf0, 0x38, 0xe9, 0x20, 0x85, 0x02,
            0xa8, 0xb1, 0x06, 0x85, 0x03, 0xa4, 0x04, 0x68, 0xd0, 0x05, 0xb5, 0xce, 0x4c, 0x2b,
            0xe4, 0xb5, 0x86, 0x29, 0x0f, 0x85, 0x04, 0xa5, 0x03, 0x60,
        ]);
        let table_offset = (0x9bddu16 - 0x8000) as usize;
        prg[table_offset..table_offset + 4].copy_from_slice(&[0x00, 0xd0, 0x05, 0x05]);
        let nibble_offset = (SMB_BOUNDING_BOX_NIBBLE_PC - 0x8000) as usize;
        prg[nibble_offset..nibble_offset + 21].copy_from_slice(&[
            0x48, 0x4a, 0x4a, 0x4a, 0x4a, 0xa8, 0xb9, 0xdf, 0x9b, 0x85, 0x07, 0x68, 0x29, 0x0f,
            0x18, 0x79, 0xdd, 0x9b, 0x85, 0x06, 0x60,
        ]);
    }

    fn assert_bounding_box_helper_fast_forward_matches_interpreted(a: u8, x: u8, y: u8) {
        let mut prg = vec![0xea; 32768];
        add_bounding_box_routines(&mut prg);
        let mut fast = NesEmulator::new_with_options(make_test_cart_with_prg(prg.clone()), true);
        let mut interpreted = NesEmulator::new_with_options(make_test_cart_with_prg(prg), true);
        for (index, value) in fast.ram.iter_mut().enumerate() {
            *value = ((index * 29 + index / 5 + 13) & 0xff) as u8;
        }
        fast.ram[0x0006] = 0x40;
        fast.ram[0x0007] = 0x05;
        interpreted.ram = fast.ram;
        for emu in [&mut fast, &mut interpreted] {
            emu.cpu.pc = SMB_BOUNDING_BOX_HELPER_PC;
            emu.cpu.a = a;
            emu.cpu.x = x;
            emu.cpu.y = y;
            emu.cpu.p = FLAG_U | FLAG_C | FLAG_V | FLAG_N;
            emu.cpu.sp = 0xf3;
            emu.push_u16(0x9122);
            emu.extra_cycles = 7;
            emu.ppu.set_dot(PPU_VBLANK_DOT);
        }

        let mut cpu_cycle_guard = 0usize;
        let mut pending_ppu_cycles = 0usize;
        assert!(fast
            .try_fast_forward_bounding_box_helper(&mut cpu_cycle_guard, &mut pending_ppu_cycles));

        let mut interpreted_cycles = 0usize;
        while interpreted.cpu.pc != 0x9123 {
            interpreted_cycles += interpreted.interpret_one_for_test() as usize;
            assert!(interpreted_cycles < 300);
        }

        assert!(interpreted_cycles <= SMB_BOUNDING_BOX_HELPER_MAX_CPU_CYCLES + 7);
        assert_eq!(cpu_cycle_guard, interpreted_cycles);
        assert_eq!(pending_ppu_cycles, interpreted_cycles * 3);
        assert_eq!(fast.cpu.a, interpreted.cpu.a);
        assert_eq!(fast.cpu.x, interpreted.cpu.x);
        assert_eq!(fast.cpu.y, interpreted.cpu.y);
        assert_eq!(fast.cpu.sp, interpreted.cpu.sp);
        assert_eq!(fast.cpu.pc, interpreted.cpu.pc);
        assert_eq!(fast.cpu.p, interpreted.cpu.p);
        assert_eq!(fast.extra_cycles, interpreted.extra_cycles);
        assert_eq!(fast.ram, interpreted.ram);
    }

    #[test]
    fn bounding_box_helper_fast_forward_matches_interpreted_zero_a() {
        assert_bounding_box_helper_fast_forward_matches_interpreted(0x00, 0x03, 0x12);
    }

    #[test]
    fn bounding_box_helper_fast_forward_matches_interpreted_nonzero_a() {
        assert_bounding_box_helper_fast_forward_matches_interpreted(0x83, 0x07, 0x51);
    }

    #[test]
    fn bounding_box_helper_fast_forward_does_not_cross_ppu_event() {
        let mut prg = vec![0xea; 32768];
        add_bounding_box_routines(&mut prg);
        let mut emu = NesEmulator::new_with_options(make_test_cart_with_prg(prg), true);
        emu.cpu.pc = SMB_BOUNDING_BOX_HELPER_PC;
        emu.ppu
            .set_dot(PPU_PRERENDER_DOT - SMB_BOUNDING_BOX_HELPER_MAX_CPU_CYCLES * 3 + 1);
        let mut cpu_cycle_guard = 0usize;
        let mut pending_ppu_cycles = 0usize;

        assert!(!emu
            .try_fast_forward_bounding_box_helper(&mut cpu_cycle_guard, &mut pending_ppu_cycles));
        assert_eq!(emu.cpu.pc, SMB_BOUNDING_BOX_HELPER_PC);
        assert_eq!(cpu_cycle_guard, 0);
        assert_eq!(pending_ppu_cycles, 0);
    }
}
