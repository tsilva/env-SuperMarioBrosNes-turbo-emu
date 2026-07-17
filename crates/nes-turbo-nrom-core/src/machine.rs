use crate::{Cartridge, Profiler};
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::RwLock;
use std::time::Instant;
use thiserror::Error;

pub const NES_WIDTH: usize = 256;
pub const NES_HEIGHT: usize = 240;
pub const VISIBLE_FRAME_LEFT: usize = 8;
pub const VISIBLE_FRAME_TOP: usize = 8;
pub const VISIBLE_FRAME_WIDTH: usize = 240;
pub const VISIBLE_FRAME_HEIGHT: usize = 224;
pub const RGB_CHANNELS: usize = 3;
#[allow(dead_code)]
pub const FRAME_PIXELS_RGB: usize = NES_WIDTH * NES_HEIGHT * RGB_CHANNELS;
#[allow(dead_code)]
pub const FRAME_PIXELS_GRAY: usize = NES_WIDTH * NES_HEIGHT;

pub const CPU_CYCLES_PER_FRAME_GUARD: usize = 40_000;
pub const PPU_DOTS_PER_SCANLINE: usize = 341;
pub const PPU_SCANLINES_PER_FRAME: usize = 262;
pub const PPU_DOTS_PER_FRAME: usize = PPU_DOTS_PER_SCANLINE * PPU_SCANLINES_PER_FRAME;
pub const PPU_VBLANK_DOT: usize = PPU_DOTS_PER_SCANLINE;
pub const PPU_PRERENDER_DOT: usize = 21 * PPU_DOTS_PER_SCANLINE;
pub const PPU_VISIBLE_START_SCANLINE: usize = 22;
pub const DEFAULT_GRAY_CROP_TOP: usize = 32;
pub const DEFAULT_GRAY_CROP_HEIGHT: usize = VISIBLE_FRAME_HEIGHT - DEFAULT_GRAY_CROP_TOP;
pub const DEFAULT_GRAY_RESIZE_WIDTH: usize = 84;
pub const DEFAULT_GRAY_RESIZE_HEIGHT: usize = 84;
pub const DEFAULT_GRAY_RESIZE_PIXELS: usize =
    DEFAULT_GRAY_RESIZE_WIDTH * DEFAULT_GRAY_RESIZE_HEIGHT;

pub trait PpuTiming: Clone + Send + Sync + 'static {
    const SPRITE0_HIT_DOT: Option<usize>;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResumePolicy {
    Continue,
    FlushIfEventDue,
    FlushNow,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FastPathOutcome {
    Miss,
    Applied(ResumePolicy),
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FrameBudget {
    pub cpu_cycle_guard: usize,
    pub pending_ppu_cycles: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct StepResult {
    pub reward: f32,
    pub done: bool,
}

#[cfg(feature = "test-support")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CoreSnapshot {
    pub a: u8,
    pub x: u8,
    pub y: u8,
    pub sp: u8,
    pub pc: u16,
    pub status_flags: u8,
    pub ram: [u8; 2048],
    pub oam: [u8; 256],
    pub controller_state: u8,
    pub controller_shift: u8,
    pub controller_strobe: bool,
    pub extra_cycles: u16,
    pub ppu_status: u8,
    pub ppu_frame_dot: usize,
    pub ppu_next_event_dot: usize,
    pub ppu_frame: u64,
}

pub trait NromGame: Clone + Send + Sync + 'static {
    type Options: Clone + Send + Sync;
    type Signals: Clone + Default + Send + Sync;
    type StepContext: Copy + Send + Sync;
    type FastPaths: Clone + Default + Send + Sync;
    type PpuTiming: PpuTiming;

    const EXPECTED_ROM_SHA256: &'static str;

    fn detect_fast_paths(prg_rom: &[u8], prg_addr_mask: usize) -> Self::FastPaths;
    fn synchronize(core: &mut NromCore<Self::PpuTiming>, signals: &mut Self::Signals);
    fn pre_step(signals: &Self::Signals) -> Self::StepContext;
    fn post_step(
        options: &Self::Options,
        signals: &Self::Signals,
        context: Self::StepContext,
    ) -> StepResult;
    fn dispatch_fast_path(
        core: &mut NromCore<Self::PpuTiming>,
        fast_paths: &Self::FastPaths,
        budget: &mut FrameBudget,
    ) -> FastPathOutcome;
}

pub const FLAG_C: u8 = 0x01;
pub const FLAG_Z: u8 = 0x02;
pub const FLAG_I: u8 = 0x04;
pub const FLAG_D: u8 = 0x08;
pub const FLAG_B: u8 = 0x10;
pub const FLAG_U: u8 = 0x20;
pub const FLAG_V: u8 = 0x40;
pub const FLAG_N: u8 = 0x80;

#[derive(Debug, Error)]
pub enum StateLoadError {
    #[error("state field {name} with size {size} was not found")]
    MissingField { name: &'static str, size: usize },
}

#[derive(Clone, Copy)]
pub struct Cpu {
    pub a: u8,
    pub x: u8,
    pub y: u8,
    pub sp: u8,
    pub pc: u16,
    pub p: u8,
}

impl Cpu {
    pub(crate) fn new() -> Self {
        Self {
            a: 0,
            x: 0,
            y: 0,
            sp: 0xfd,
            pc: 0,
            p: FLAG_U | FLAG_I,
        }
    }
}

impl Default for Cpu {
    fn default() -> Self {
        Self::new()
    }
}

struct GrayBgQuadCache {
    tables: RwLock<[[u32; 256]; 4]>,
    dirty: AtomicU8,
    frame: RwLock<Option<GrayBgFrameCacheEntry>>,
}

#[derive(Clone, PartialEq, Eq)]
struct GrayBgFrameCacheKey {
    vram: [u8; 2048],
    palette_gray: [u8; 16],
    ctrl: u8,
    mask: u8,
    scroll_x_px: u16,
    scroll_y_px: u16,
    crop_top: usize,
    height: usize,
}

#[derive(Clone)]
struct GrayBgFrameCacheEntry {
    key: GrayBgFrameCacheKey,
    pixels: Vec<u8>,
}

impl GrayBgQuadCache {
    fn new(tables: [[u32; 256]; 4]) -> Self {
        Self {
            tables: RwLock::new(tables),
            dirty: AtomicU8::new(0),
            frame: RwLock::new(None),
        }
    }
}

impl Clone for GrayBgQuadCache {
    fn clone(&self) -> Self {
        Self {
            tables: RwLock::new(*self.tables.read().unwrap()),
            dirty: AtomicU8::new(self.dirty.load(Ordering::Relaxed)),
            frame: RwLock::new(None),
        }
    }
}

#[derive(Clone)]
pub struct Ppu {
    chr_rom: Vec<u8>,
    chr_row_pixels: Vec<u16>,
    chr_addr_mask: usize,
    vertical_mirroring: bool,
    ctrl: u8,
    mask: u8,
    pub status: u8,
    oam_addr: u8,
    oam: [u8; 256],
    vram: [u8; 2048],
    palette: [u8; 32],
    palette_gray: [u8; 32],
    gray_bg_quad_cache: GrayBgQuadCache,
    data_buffer: u8,
    addr: u16,
    temp_addr: u16,
    scroll_addr: u16,
    render_addr: u16,
    first_write: bool,
    fine_x: u8,
    scroll_x_px: u16,
    scroll_y_px: u16,
    scroll_x_low: u8,
    scroll_override_x_px: Option<u16>,
    frame_dot: usize,
    next_event_dot: usize,
    frame: u64,
    nmi_pending: bool,
    sprite0_hit_dot: Option<usize>,
}

impl Ppu {
    pub(crate) fn new(
        chr_rom: Vec<u8>,
        vertical_mirroring: bool,
        sprite0_hit_dot: Option<usize>,
    ) -> Self {
        let chr_addr_mask = chr_rom.len() - 1;
        let chr_row_pixels = decode_chr_row_pixels(&chr_rom);
        let palette_gray = [NES_GRAY_PALETTE[0]; 32];
        let gray_bg_quad_tables = build_gray_bg_quad_tables(&palette_gray);
        Self {
            chr_rom,
            chr_row_pixels,
            chr_addr_mask,
            vertical_mirroring,
            ctrl: 0,
            mask: 0,
            status: 0,
            oam_addr: 0,
            oam: [0; 256],
            vram: [0; 2048],
            palette: [0; 32],
            palette_gray,
            gray_bg_quad_cache: GrayBgQuadCache::new(gray_bg_quad_tables),
            data_buffer: 0,
            addr: 0,
            temp_addr: 0,
            scroll_addr: 0,
            render_addr: 0,
            first_write: true,
            fine_x: 0,
            scroll_x_px: 0,
            scroll_y_px: 0,
            scroll_x_low: 0,
            scroll_override_x_px: None,
            frame_dot: 0,
            next_event_dot: next_ppu_event_dot(0, sprite0_hit_dot),
            frame: 0,
            nmi_pending: false,
            sprite0_hit_dot,
        }
    }

    pub(crate) fn reset(&mut self) {
        self.ctrl = 0;
        self.mask = 0;
        self.status = 0;
        self.oam_addr = 0;
        self.oam = [0; 256];
        self.vram = [0; 2048];
        self.palette = [0; 32];
        self.refresh_gray_palette_cache();
        self.data_buffer = 0;
        self.addr = 0;
        self.temp_addr = 0;
        self.scroll_addr = 0;
        self.render_addr = 0;
        self.first_write = true;
        self.fine_x = 0;
        self.scroll_x_px = 0;
        self.scroll_y_px = 0;
        self.scroll_x_low = 0;
        self.scroll_override_x_px = None;
        self.frame_dot = 0;
        self.next_event_dot = next_ppu_event_dot(0, self.sprite0_hit_dot);
        self.frame = 0;
        self.nmi_pending = false;
    }

    pub(crate) fn oam(&self) -> &[u8; 256] {
        &self.oam
    }

    pub(crate) fn debug_bg_pixel(&self, x: usize, y: usize) -> (u8, bool) {
        (self.bg_color_index(x, y), self.bg_pixel_opaque(x, y))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn load_fceu_state(
        &mut self,
        ntar: &[u8],
        pram: &[u8],
        spra: &[u8],
        ppur: &[u8],
        radd: Option<&[u8]>,
        tadd: Option<&[u8]>,
        xoff: Option<&[u8]>,
    ) {
        self.vram.copy_from_slice(ntar);
        self.palette.copy_from_slice(pram);
        self.refresh_gray_palette_cache();
        self.oam.copy_from_slice(spra);
        self.ctrl = ppur[0];
        self.mask = ppur[1];
        self.status = ppur[2];
        self.oam_addr = ppur[3];
        self.addr = radd.and_then(read_u16_le).unwrap_or(0);
        self.temp_addr = tadd.and_then(read_u16_le).unwrap_or(0);
        self.scroll_addr = self.temp_addr;
        self.render_addr = self.addr;
        self.first_write = true;
        self.fine_x = xoff.and_then(|value| value.first().copied()).unwrap_or(0);
        self.frame_dot = 0;
        self.next_event_dot = next_ppu_event_dot(0, self.sprite0_hit_dot);
        self.nmi_pending = false;
        self.update_scroll_x_px();
    }

    #[inline]
    pub(crate) fn tick(&mut self, ppu_cycles: usize) -> bool {
        let mut completed_frame = false;
        let mut remaining = ppu_cycles;
        while remaining > 0 {
            let current = self.frame_dot;
            let next = self.next_event_dot;
            let advance = remaining.min(next - current);
            let dot = current + advance;
            self.frame_dot = dot;
            remaining -= advance;

            if Some(dot) == self.sprite0_hit_dot {
                self.status |= 0x40;
                self.next_event_dot = PPU_DOTS_PER_FRAME;
            } else {
                match dot {
                    PPU_VBLANK_DOT => {
                        self.status |= 0x80;
                        self.next_event_dot = PPU_PRERENDER_DOT;
                        if self.ctrl & 0x80 != 0 {
                            self.nmi_pending = true;
                        }
                    }
                    PPU_PRERENDER_DOT => {
                        self.status &= !0xc0;
                        self.next_event_dot = self.sprite0_hit_dot.unwrap_or(PPU_DOTS_PER_FRAME);
                    }
                    PPU_DOTS_PER_FRAME => {
                        self.frame_dot = 0;
                        self.next_event_dot = PPU_VBLANK_DOT;
                        self.frame = self.frame.wrapping_add(1);
                        completed_frame = true;
                    }
                    _ => {}
                }
            }
        }
        completed_frame
    }

    #[inline]
    pub(crate) fn tick_profiled(&mut self, ppu_cycles: usize, profiler: &mut Profiler) -> bool {
        let completed_frame = self.tick(ppu_cycles);
        profiler.record_ppu_tick(ppu_cycles, completed_frame);
        completed_frame
    }

    #[inline]
    pub fn cycles_until_next_event(&self) -> usize {
        self.next_event_dot - self.frame_dot
    }

    #[inline]
    pub fn sprite0_hit_set(&self) -> bool {
        self.status & 0x40 != 0
    }

    #[cfg(any(test, feature = "test-support"))]
    #[inline]
    pub fn set_dot(&mut self, dot: usize) {
        self.frame_dot = dot;
        self.next_event_dot = next_ppu_event_dot(dot, self.sprite0_hit_dot);
    }

    #[inline]
    pub(crate) fn take_nmi(&mut self) -> bool {
        let pending = self.nmi_pending;
        self.nmi_pending = false;
        pending
    }

    #[inline]
    pub fn cpu_read_register(&mut self, reg: u16) -> u8 {
        match reg & 7 {
            2 => {
                let value = self.status;
                self.status &= !0x80;
                self.first_write = true;
                value
            }
            4 => self.oam[self.oam_addr as usize],
            7 => self.read_data(),
            _ => 0,
        }
    }

    #[inline]
    pub(crate) fn cpu_write_register(&mut self, reg: u16, value: u8) {
        match reg & 7 {
            0 => {
                let old = self.ctrl;
                self.ctrl = value;
                self.temp_addr = (self.temp_addr & 0xf3ff) | (((value as u16) & 0x03) << 10);
                self.scroll_addr = (self.scroll_addr & 0xf3ff) | (((value as u16) & 0x03) << 10);
                self.update_scroll_x_px();
                if old & 0x80 == 0 && value & 0x80 != 0 && self.status & 0x80 != 0 {
                    self.nmi_pending = true;
                }
            }
            1 => self.mask = value,
            3 => self.oam_addr = value,
            4 => {
                self.oam[self.oam_addr as usize] = value;
                self.oam_addr = self.oam_addr.wrapping_add(1);
            }
            5 => {
                if self.first_write {
                    self.fine_x = value & 0x07;
                    self.scroll_x_low = value;
                    self.update_scroll_x_px();
                    self.temp_addr = (self.temp_addr & 0xffe0) | ((value as u16) >> 3);
                } else {
                    self.scroll_y_px = value as u16;
                    self.temp_addr = (self.temp_addr & 0x8fff) | (((value as u16) & 0x07) << 12);
                    self.temp_addr = (self.temp_addr & 0xfc1f) | (((value as u16) & 0xf8) << 2);
                }
                self.scroll_addr = self.temp_addr;
                self.first_write = !self.first_write;
            }
            6 => {
                if self.first_write {
                    self.temp_addr = (self.temp_addr & 0x00ff) | (((value as u16) & 0x3f) << 8);
                } else {
                    self.temp_addr = (self.temp_addr & 0xff00) | value as u16;
                    self.addr = self.temp_addr;
                }
                self.first_write = !self.first_write;
            }
            7 => self.write_data(value),
            _ => {}
        }
    }

    #[inline]
    pub(crate) fn read_data(&mut self) -> u8 {
        let addr = self.addr & 0x3fff;
        let inc = if self.ctrl & 0x04 != 0 { 32 } else { 1 };
        self.addr = self.addr.wrapping_add(inc) & 0x3fff;

        if addr >= 0x3f00 {
            self.ppu_read(addr)
        } else {
            let buffered = self.data_buffer;
            self.data_buffer = self.ppu_read(addr);
            buffered
        }
    }

    #[inline]
    pub(crate) fn write_data(&mut self, value: u8) {
        let addr = self.addr & 0x3fff;
        self.ppu_write(addr, value);
        let inc = if self.ctrl & 0x04 != 0 { 32 } else { 1 };
        self.addr = self.addr.wrapping_add(inc) & 0x3fff;
    }

    #[inline]
    pub(crate) fn chr_read(&self, addr: usize) -> u8 {
        let idx = addr & self.chr_addr_mask;
        // SAFETY: SMB/NROM CHR ROM sizes are power-of-two and chr_addr_mask is len - 1.
        unsafe { *self.chr_rom.get_unchecked(idx) }
    }

    #[inline]
    pub(crate) fn chr_row_pixels(&self, addr: usize) -> u16 {
        let idx = addr & self.chr_addr_mask;
        // SAFETY: SMB/NROM CHR ROM sizes are power-of-two, the decoded row
        // table has the same length, and chr_addr_mask is len - 1.
        unsafe { *self.chr_row_pixels.get_unchecked(idx) }
    }

    #[inline]
    pub(crate) fn ppu_read(&self, addr: u16) -> u8 {
        let addr = addr & 0x3fff;
        match addr {
            0x0000..=0x1fff => self.chr_read(addr as usize),
            0x2000..=0x3eff => {
                let idx = self.mirror_nametable_addr(addr);
                self.vram[idx]
            }
            0x3f00..=0x3fff => self.palette[self.mirror_palette_addr(addr)],
            _ => 0,
        }
    }

    #[inline(always)]
    pub(crate) fn nametable_read(&self, table: usize, offset: usize) -> u8 {
        let physical_table = if self.vertical_mirroring {
            table & 1
        } else {
            (table >> 1) & 1
        };
        // SAFETY: mirroring maps the four logical nametables into the two
        // physical 1 KiB VRAM pages, and the offset is masked to that page.
        unsafe {
            *self
                .vram
                .get_unchecked(physical_table * 0x400 + (offset & 0x3ff))
        }
    }

    #[inline]
    pub(crate) fn ppu_write(&mut self, addr: u16, value: u8) {
        let addr = addr & 0x3fff;
        match addr {
            0x0000..=0x1fff => {}
            0x2000..=0x3eff => {
                let idx = self.mirror_nametable_addr(addr);
                self.vram[idx] = value;
            }
            0x3f00..=0x3fff => {
                let idx = self.mirror_palette_addr(addr);
                self.palette[idx] = value;
                self.refresh_gray_palette_entry(idx);
            }
            _ => {}
        }
    }

    #[inline]
    pub(crate) fn mirror_nametable_addr(&self, addr: u16) -> usize {
        let v = (addr - 0x2000) as usize % 0x1000;
        let table = v / 0x400;
        let offset = v & 0x3ff;
        let physical_table = if self.vertical_mirroring {
            table & 1
        } else {
            (table >> 1) & 1
        };
        physical_table * 0x400 + offset
    }

    #[inline]
    pub(crate) fn mirror_palette_addr(&self, addr: u16) -> usize {
        let mut idx = (addr as usize - 0x3f00) & 0x1f;
        if idx == 0x10 {
            idx = 0x00;
        } else if idx == 0x14 {
            idx = 0x04;
        } else if idx == 0x18 {
            idx = 0x08;
        } else if idx == 0x1c {
            idx = 0x0c;
        }
        idx
    }

    #[allow(dead_code)]
    pub(crate) fn write_gray_frame(&self, dst: &mut [u8]) {
        debug_assert_eq!(dst.len(), FRAME_PIXELS_GRAY);
        for y in 0..NES_HEIGHT {
            for x in 0..NES_WIDTH {
                let color = self.bg_color_index(x, y);
                dst[y * NES_WIDTH + x] = NES_GRAY_PALETTE[color as usize];
            }
        }
        self.draw_sprites_gray(dst);
    }

    #[allow(dead_code)]
    pub(crate) fn write_gray_frame_cropped(&self, dst: &mut [u8], crop_top: usize, height: usize) {
        debug_assert_eq!(dst.len(), NES_WIDTH * height);
        self.write_bg_gray_cropped_tiled(dst, crop_top, height);
        self.draw_sprites_gray_cropped(dst, crop_top, height);
    }

    pub(crate) fn write_gray_frame_region(
        &self,
        dst: &mut [u8],
        crop_top: usize,
        crop_left: usize,
        width: usize,
        height: usize,
    ) {
        debug_assert_eq!(dst.len(), width * height);
        self.write_bg_gray_region_tiled(dst, crop_top, crop_left, width, height);
        self.draw_sprites_gray_region(dst, crop_top, crop_left, width, height);
    }

    pub(crate) fn write_gray_frame_cropped_area_84x84(
        &self,
        dst: &mut [u8],
        sprite_shadow: &mut [u8],
    ) {
        debug_assert_eq!(dst.len(), DEFAULT_GRAY_RESIZE_PIXELS);
        let native_len = VISIBLE_FRAME_WIDTH * DEFAULT_GRAY_CROP_HEIGHT;
        debug_assert!(sprite_shadow.len() >= native_len);
        let native = &mut sprite_shadow[..native_len];
        self.write_gray_frame_region(
            native,
            VISIBLE_FRAME_TOP + DEFAULT_GRAY_CROP_TOP,
            VISIBLE_FRAME_LEFT,
            VISIBLE_FRAME_WIDTH,
            DEFAULT_GRAY_CROP_HEIGHT,
        );

        for dy in 0..DEFAULT_GRAY_RESIZE_HEIGHT {
            let y0 = (dy * DEFAULT_GRAY_CROP_HEIGHT) / DEFAULT_GRAY_RESIZE_HEIGHT;
            let y1 = (((dy + 1) * DEFAULT_GRAY_CROP_HEIGHT) / DEFAULT_GRAY_RESIZE_HEIGHT)
                .max(y0 + 1)
                .min(DEFAULT_GRAY_CROP_HEIGHT);
            for dx in 0..DEFAULT_GRAY_RESIZE_WIDTH {
                let x0 = (dx * VISIBLE_FRAME_WIDTH) / DEFAULT_GRAY_RESIZE_WIDTH;
                let x1 = (((dx + 1) * VISIBLE_FRAME_WIDTH) / DEFAULT_GRAY_RESIZE_WIDTH)
                    .max(x0 + 1)
                    .min(VISIBLE_FRAME_WIDTH);
                let mut sum = 0u32;
                for sy in y0..y1 {
                    let src_row = sy * VISIBLE_FRAME_WIDTH;
                    for sx in x0..x1 {
                        sum += native[src_row + sx] as u32;
                    }
                }
                dst[dy * DEFAULT_GRAY_RESIZE_WIDTH + dx] =
                    (sum / ((x1 - x0) * (y1 - y0)) as u32) as u8;
            }
        }
    }

    pub(crate) fn write_gray_visible_mask_lower_frame(
        &self,
        dst: &mut [u8],
        crop_top: usize,
        height: usize,
    ) {
        debug_assert_eq!(dst.len(), VISIBLE_FRAME_WIDTH * height);
        self.write_bg_gray_visible_mask_lower_tiled(dst, crop_top, height);
        self.draw_sprites_gray_region(
            dst,
            crop_top,
            VISIBLE_FRAME_LEFT,
            VISIBLE_FRAME_WIDTH,
            height,
        );
    }

    #[allow(dead_code)]
    pub(crate) fn write_bg_gray_cropped_tiled(
        &self,
        dst: &mut [u8],
        crop_top: usize,
        height: usize,
    ) {
        let palette_gray = self.palette_gray();
        if self.mask & 0x08 == 0 {
            dst.fill(palette_gray[0]);
            return;
        }

        let pattern_base = if self.ctrl & 0x10 != 0 {
            0x1000
        } else {
            0x0000
        };
        let scroll_x = self.render_scroll_x_px() as usize;
        let scroll_y = self.scroll_y_px as usize;

        for out_y in 0..height {
            let y = crop_top + out_y;
            let world_y = if y < 32 { y } else { y + scroll_y };
            let table_y = (world_y / 240) & 1;
            let local_y = world_y % 240;
            let tile_y = local_y / 8;
            let fine_y = local_y & 7;
            let row_start = out_y * NES_WIDTH;
            let mut x = 0usize;

            while x < NES_WIDTH {
                let world_x = if y < 32 { x } else { x + scroll_x };
                let table_x = (world_x / 256) & 1;
                let table = table_y * 2 + table_x;
                let local_x = world_x & 0xff;
                let tile_x = local_x / 8;
                let fine_x = local_x & 7;
                let tile_id = self.nametable_read(table, tile_y * 32 + tile_x) as usize;
                let attr = self.nametable_read(table, 0x3c0 + ((tile_y / 4) * 8 + (tile_x / 4)));
                let shift = ((tile_y & 0x02) << 1) | (tile_x & 0x02);
                let palette_id = (attr >> shift) & 0x03;
                let pattern_addr = pattern_base + tile_id * 16 + fine_y;
                let lo = self.chr_read(pattern_addr);
                let hi = self.chr_read(pattern_addr + 8);
                let run = (8 - fine_x).min(NES_WIDTH - x);

                for col in 0..run {
                    let bit = 7 - (fine_x + col);
                    let pixel = ((lo >> bit) & 1) | (((hi >> bit) & 1) << 1);
                    let gray = if pixel == 0 {
                        palette_gray[0]
                    } else {
                        palette_gray[(palette_id as usize) * 4 + pixel as usize]
                    };
                    dst[row_start + x + col] = gray;
                }

                x += run;
            }
        }
    }

    pub(crate) fn write_bg_gray_region_tiled(
        &self,
        dst: &mut [u8],
        crop_top: usize,
        crop_left: usize,
        width: usize,
        height: usize,
    ) {
        let palette_gray = self.palette_gray();
        if self.mask & 0x08 == 0 {
            dst.fill(palette_gray[0]);
            return;
        }
        let pattern_base = if self.ctrl & 0x10 != 0 {
            0x1000
        } else {
            0x0000
        };
        let scroll_x = self.render_scroll_x_px() as usize;
        let scroll_y = self.scroll_y_px as usize;

        for out_y in 0..height {
            let y = crop_top + out_y;
            let world_y = if y < 32 { y } else { y + scroll_y };
            let table_y = (world_y / 240) & 1;
            let local_y = world_y % 240;
            let tile_y = local_y / 8;
            let fine_y = local_y & 7;
            let row_start = out_y * width;
            let mut out_x = 0usize;

            while out_x < width {
                let screen_x = crop_left + out_x;
                let world_x = if y < 32 {
                    screen_x
                } else {
                    screen_x + scroll_x
                };
                let table_x = (world_x / 256) & 1;
                let table = table_y * 2 + table_x;
                let local_x = world_x & 0xff;
                let tile_x = local_x / 8;
                let fine_x = local_x & 7;
                let tile_id = self.nametable_read(table, tile_y * 32 + tile_x) as usize;
                let attr = self.nametable_read(table, 0x3c0 + ((tile_y / 4) * 8 + (tile_x / 4)));
                let shift = ((tile_y & 0x02) << 1) | (tile_x & 0x02);
                let palette_id = (attr >> shift) & 0x03;
                let pattern_addr = pattern_base + tile_id * 16 + fine_y;
                let lo = self.chr_read(pattern_addr);
                let hi = self.chr_read(pattern_addr + 8);
                let run = (8 - fine_x).min(width - out_x);
                let bg_gray = palette_gray[0];
                let palette_base = (palette_id as usize) * 4;
                let colors = [
                    bg_gray,
                    palette_gray[palette_base + 1],
                    palette_gray[palette_base + 2],
                    palette_gray[palette_base + 3],
                ];

                if fine_x == 0 && run >= 8 {
                    write_full_bg_tile_gray(
                        &mut dst[row_start + out_x..row_start + out_x + 8],
                        lo,
                        hi,
                        colors,
                    );
                } else {
                    for col in 0..run {
                        let bit = 7 - (fine_x + col);
                        let pixel = ((lo >> bit) & 1) | (((hi >> bit) & 1) << 1);
                        dst[row_start + out_x + col] = colors[pixel as usize];
                    }
                }

                out_x += run;
            }
        }
    }

    pub(crate) fn write_bg_gray_visible_mask_lower_tiled(
        &self,
        dst: &mut [u8],
        crop_top: usize,
        height: usize,
    ) {
        let mut bg_palette = [0; 16];
        bg_palette.copy_from_slice(&self.palette_gray[..16]);
        let key = GrayBgFrameCacheKey {
            vram: self.vram,
            palette_gray: bg_palette,
            ctrl: self.ctrl,
            mask: self.mask & 0x08,
            scroll_x_px: self.render_scroll_x_px(),
            scroll_y_px: self.scroll_y_px,
            crop_top,
            height,
        };
        {
            let cached = self.gray_bg_quad_cache.frame.read().unwrap();
            if let Some(entry) = cached.as_ref().filter(|entry| entry.key == key) {
                dst.copy_from_slice(&entry.pixels);
                return;
            }
        }

        self.write_bg_gray_visible_mask_lower_tiled_uncached(dst, crop_top, height);
        *self.gray_bg_quad_cache.frame.write().unwrap() = Some(GrayBgFrameCacheEntry {
            key,
            pixels: dst.to_vec(),
        });
    }

    pub(crate) fn write_bg_gray_visible_mask_lower_tiled_uncached(
        &self,
        dst: &mut [u8],
        crop_top: usize,
        height: usize,
    ) {
        let palette_gray = self.palette_gray();
        if self.mask & 0x08 == 0 {
            dst.fill(palette_gray[0]);
            return;
        }
        self.refresh_gray_bg_quad_tables();
        let gray_quads = self.gray_bg_quad_cache.tables.read().unwrap();

        let pattern_base = if self.ctrl & 0x10 != 0 {
            0x1000
        } else {
            0x0000
        };
        let scroll_x = self.render_scroll_x_px() as usize;
        let scroll_y = self.scroll_y_px as usize;

        for out_y in 0..height {
            let y = crop_top + out_y;
            debug_assert!(y >= 32);
            let world_y = y + scroll_y;
            let table_y = (world_y / 240) & 1;
            let local_y = world_y % 240;
            let tile_y = local_y / 8;
            let fine_y = local_y & 7;
            let row_start = out_y * VISIBLE_FRAME_WIDTH;
            let mut out_x = 0usize;

            while out_x < VISIBLE_FRAME_WIDTH {
                let screen_x = VISIBLE_FRAME_LEFT + out_x;
                let world_x = screen_x + scroll_x;
                let table_x = (world_x / 256) & 1;
                let table = table_y * 2 + table_x;
                let local_x = world_x & 0xff;
                let tile_x = local_x / 8;
                let fine_x = local_x & 7;
                let tile_id = self.nametable_read(table, tile_y * 32 + tile_x) as usize;
                let attr = self.nametable_read(table, 0x3c0 + ((tile_y / 4) * 8 + (tile_x / 4)));
                let shift = ((tile_y & 0x02) << 1) | (tile_x & 0x02);
                let palette_id = (attr >> shift) & 0x03;
                let pattern_addr = pattern_base + tile_id * 16 + fine_y;
                let pixels = self.chr_row_pixels(pattern_addr);
                let run = (8 - fine_x).min(VISIBLE_FRAME_WIDTH - out_x);

                if fine_x == 0 && run >= 8 {
                    write_full_bg_tile_gray_pixels(
                        &mut dst[row_start + out_x..row_start + out_x + 8],
                        pixels,
                        &gray_quads[palette_id as usize],
                    );
                } else {
                    let palette_base = (palette_id as usize) * 4;
                    let colors = [
                        palette_gray[0],
                        palette_gray[palette_base + 1],
                        palette_gray[palette_base + 2],
                        palette_gray[palette_base + 3],
                    ];
                    for col in 0..run {
                        let pixel = ((pixels >> ((fine_x + col) * 2)) & 3) as usize;
                        dst[row_start + out_x + col] = colors[pixel];
                    }
                }

                out_x += run;
            }
        }
    }

    #[allow(dead_code)]
    pub(crate) fn write_rgb_frame(&self, dst: &mut [u8]) {
        debug_assert_eq!(dst.len(), FRAME_PIXELS_RGB);
        let plane = NES_WIDTH * NES_HEIGHT;
        for y in 0..NES_HEIGHT {
            for x in 0..NES_WIDTH {
                let color = nes_rgb(self.bg_color_index(x, y));
                let idx = y * NES_WIDTH + x;
                dst[idx] = color[0];
                dst[plane + idx] = color[1];
                dst[plane * 2 + idx] = color[2];
            }
        }
        self.draw_sprites_rgb(dst);
    }

    #[allow(dead_code)]
    pub(crate) fn write_rgb_frame_cropped(&self, dst: &mut [u8], crop_top: usize, height: usize) {
        debug_assert_eq!(dst.len(), NES_WIDTH * height * RGB_CHANNELS);
        let plane = NES_WIDTH * height;
        for out_y in 0..height {
            let y = crop_top + out_y;
            for x in 0..NES_WIDTH {
                let color = nes_rgb(self.bg_color_index(x, y));
                let idx = out_y * NES_WIDTH + x;
                dst[idx] = color[0];
                dst[plane + idx] = color[1];
                dst[plane * 2 + idx] = color[2];
            }
        }
        self.draw_sprites_rgb_cropped(dst, crop_top, height);
    }

    pub(crate) fn write_rgb_frame_region(
        &self,
        dst: &mut [u8],
        crop_top: usize,
        crop_left: usize,
        width: usize,
        height: usize,
    ) {
        debug_assert_eq!(dst.len(), width * height * RGB_CHANNELS);
        let plane = width * height;
        for out_y in 0..height {
            let y = crop_top + out_y;
            for out_x in 0..width {
                let x = crop_left + out_x;
                let color = nes_rgb(self.bg_color_index(x, y));
                let idx = out_y * width + out_x;
                dst[idx] = color[0];
                dst[plane + idx] = color[1];
                dst[plane * 2 + idx] = color[2];
            }
        }
        self.draw_sprites_rgb_region(dst, crop_top, crop_left, width, height);
    }

    #[inline]
    pub(crate) fn bg_color_index(&self, x: usize, y: usize) -> u8 {
        if self.mask & 0x08 == 0 {
            return self.palette[0];
        }

        let (world_x, world_y) = self.bg_world_pos(x, y);
        let table_x = (world_x / 256) & 1;
        let table_y = (world_y / 240) & 1;
        let table = table_y * 2 + table_x;

        let local_x = world_x & 0xff;
        let local_y = world_y % 240;
        let tile_x = local_x / 8;
        let tile_y = local_y / 8;
        let fine_x = local_x & 7;
        let fine_y = local_y & 7;

        let nt_base = 0x2000 + (table as u16) * 0x400;
        let tile_id = self.ppu_read(nt_base + (tile_y * 32 + tile_x) as u16) as usize;
        let attr = self.ppu_read(nt_base + 0x3c0 + ((tile_y / 4) * 8 + (tile_x / 4)) as u16);
        let shift = ((tile_y & 0x02) << 1) | (tile_x & 0x02);
        let palette_id = (attr >> shift) & 0x03;

        let pattern_base = if self.ctrl & 0x10 != 0 {
            0x1000
        } else {
            0x0000
        };
        let pattern_addr = pattern_base + tile_id * 16 + fine_y;
        let lo = self.chr_read(pattern_addr);
        let hi = self.chr_read(pattern_addr + 8);
        let bit = 7 - fine_x;
        let pixel = ((lo >> bit) & 1) | (((hi >> bit) & 1) << 1);
        if pixel == 0 {
            self.palette[0]
        } else {
            self.palette[(palette_id as usize) * 4 + pixel as usize]
        }
    }

    #[inline]
    pub(crate) fn bg_pixel_opaque(&self, x: usize, y: usize) -> bool {
        if self.mask & 0x08 == 0 {
            return false;
        }

        let (world_x, world_y) = self.bg_world_pos(x, y);
        let table_x = (world_x / 256) & 1;
        let table_y = (world_y / 240) & 1;
        let table = table_y * 2 + table_x;

        let local_x = world_x & 0xff;
        let local_y = world_y % 240;
        let tile_x = local_x / 8;
        let tile_y = local_y / 8;
        let fine_x = local_x & 7;
        let fine_y = local_y & 7;

        let nt_base = 0x2000 + (table as u16) * 0x400;
        let tile_id = self.ppu_read(nt_base + (tile_y * 32 + tile_x) as u16) as usize;
        let pattern_base = if self.ctrl & 0x10 != 0 {
            0x1000
        } else {
            0x0000
        };
        let pattern_addr = pattern_base + tile_id * 16 + fine_y;
        let lo = self.chr_read(pattern_addr);
        let hi = self.chr_read(pattern_addr + 8);
        let bit = 7 - fine_x;
        (((lo >> bit) & 1) | (((hi >> bit) & 1) << 1)) != 0
    }

    pub(crate) fn sprite_scanline_mask(&self) -> [u64; NES_HEIGHT] {
        let mut mask = [0u64; NES_HEIGHT];
        let mut counts = [0u8; NES_HEIGHT];
        for sprite in 0..64usize {
            let base = sprite * 4;
            let sprite_y = self.oam[base] as usize + 1;
            let sprite_bottom = (sprite_y + 8).min(NES_HEIGHT);
            for screen_y in sprite_y..sprite_bottom {
                if counts[screen_y] >= 8 {
                    continue;
                }
                counts[screen_y] += 1;
                mask[screen_y] |= 1u64 << sprite;
            }
        }
        mask
    }

    pub(crate) fn draw_sprites_gray(&self, dst: &mut [u8]) {
        if self.mask & 0x10 == 0 {
            return;
        }

        let palette_gray = self.palette_gray();
        let pattern_base = if self.ctrl & 0x08 != 0 {
            0x1000
        } else {
            0x0000
        };
        let sprite_scanline_mask = self.sprite_scanline_mask();
        for sprite in (0..64).rev() {
            let base = sprite * 4;
            let sprite_y = self.oam[base] as i16 + 1;
            let tile = self.oam[base + 1] as usize;
            let attr = self.oam[base + 2];
            let sprite_x = self.oam[base + 3] as i16;
            let palette_base = 0x10 + ((attr & 0x03) as usize) * 4;
            let flip_h = attr & 0x40 != 0;
            let flip_v = attr & 0x80 != 0;
            let behind_background = attr & 0x20 != 0;

            for row in 0..8usize {
                let screen_y = sprite_y + row as i16;
                if !(0..NES_HEIGHT as i16).contains(&screen_y) {
                    continue;
                }
                if sprite_scanline_mask[screen_y as usize] & (1u64 << sprite) == 0 {
                    continue;
                }
                let tile_row = if flip_v { 7 - row } else { row };
                let pattern_addr = pattern_base + tile * 16 + tile_row;
                let lo = self.chr_read(pattern_addr);
                let hi = self.chr_read(pattern_addr + 8);
                for col in 0..8usize {
                    let screen_x = sprite_x + col as i16;
                    if !(0..NES_WIDTH as i16).contains(&screen_x) {
                        continue;
                    }
                    let tile_col = if flip_h { col } else { 7 - col };
                    let pixel = ((lo >> tile_col) & 1) | (((hi >> tile_col) & 1) << 1);
                    if pixel == 0 {
                        continue;
                    }
                    let idx = screen_y as usize * NES_WIDTH + screen_x as usize;
                    if behind_background
                        && self.bg_pixel_opaque(screen_x as usize, screen_y as usize)
                    {
                        dst[idx] = NES_GRAY_PALETTE
                            [self.bg_color_index(screen_x as usize, screen_y as usize) as usize];
                        continue;
                    }
                    dst[idx] = palette_gray[palette_base + pixel as usize];
                }
            }
        }
    }

    #[allow(dead_code)]
    pub(crate) fn draw_sprites_gray_cropped(&self, dst: &mut [u8], crop_top: usize, height: usize) {
        if self.mask & 0x10 == 0 {
            return;
        }

        let palette_gray = self.palette_gray();
        let crop_top = crop_top as i16;
        let crop_bottom = crop_top + height as i16;
        let pattern_base = if self.ctrl & 0x08 != 0 {
            0x1000
        } else {
            0x0000
        };
        let sprite_scanline_mask = self.sprite_scanline_mask();
        for sprite in (0..64).rev() {
            let base = sprite * 4;
            let sprite_y = self.oam[base] as i16 + 1;
            let tile = self.oam[base + 1] as usize;
            let attr = self.oam[base + 2];
            let sprite_x = self.oam[base + 3] as i16;
            let palette_base = 0x10 + ((attr & 0x03) as usize) * 4;
            let flip_h = attr & 0x40 != 0;
            let flip_v = attr & 0x80 != 0;
            let behind_background = attr & 0x20 != 0;

            for row in 0..8usize {
                let screen_y = sprite_y + row as i16;
                if screen_y < crop_top || screen_y >= crop_bottom {
                    continue;
                }
                if sprite_scanline_mask[screen_y as usize] & (1u64 << sprite) == 0 {
                    continue;
                }
                let tile_row = if flip_v { 7 - row } else { row };
                let pattern_addr = pattern_base + tile * 16 + tile_row;
                let lo = self.chr_read(pattern_addr);
                let hi = self.chr_read(pattern_addr + 8);
                for col in 0..8usize {
                    let screen_x = sprite_x + col as i16;
                    if !(0..NES_WIDTH as i16).contains(&screen_x) {
                        continue;
                    }
                    let tile_col = if flip_h { col } else { 7 - col };
                    let pixel = ((lo >> tile_col) & 1) | (((hi >> tile_col) & 1) << 1);
                    if pixel == 0 {
                        continue;
                    }
                    let out_y = (screen_y - crop_top) as usize;
                    let idx = out_y * NES_WIDTH + screen_x as usize;
                    if behind_background
                        && self.bg_pixel_opaque(screen_x as usize, screen_y as usize)
                    {
                        dst[idx] = NES_GRAY_PALETTE
                            [self.bg_color_index(screen_x as usize, screen_y as usize) as usize];
                        continue;
                    }
                    dst[idx] = palette_gray[palette_base + pixel as usize];
                }
            }
        }
    }

    pub(crate) fn draw_sprites_gray_region(
        &self,
        dst: &mut [u8],
        crop_top: usize,
        crop_left: usize,
        width: usize,
        height: usize,
    ) {
        if self.mask & 0x10 == 0 {
            return;
        }

        let palette_gray = self.palette_gray();
        let crop_top_i = crop_top as i16;
        let crop_bottom = crop_top_i + height as i16;
        let crop_left_i = crop_left as i16;
        let crop_right = crop_left_i + width as i16;
        let pattern_base = if self.ctrl & 0x08 != 0 {
            0x1000
        } else {
            0x0000
        };
        let sprite_scanline_mask = self.sprite_scanline_mask();
        for sprite in (0..64).rev() {
            let base = sprite * 4;
            let sprite_y = self.oam[base] as i16 + 1;
            let tile = self.oam[base + 1] as usize;
            let attr = self.oam[base + 2];
            let sprite_x = self.oam[base + 3] as i16;
            let palette_base = 0x10 + ((attr & 0x03) as usize) * 4;
            let flip_h = attr & 0x40 != 0;
            let flip_v = attr & 0x80 != 0;
            let behind_background = attr & 0x20 != 0;
            let row_start = (crop_top_i - sprite_y).clamp(0, 8) as usize;
            let row_end = (crop_bottom - sprite_y).clamp(0, 8) as usize;
            let col_start = (crop_left_i - sprite_x).clamp(0, 8) as usize;
            let col_end = (crop_right - sprite_x).clamp(0, 8) as usize;
            if row_start >= row_end || col_start >= col_end {
                continue;
            }

            for row in row_start..row_end {
                let screen_y = sprite_y + row as i16;
                if sprite_scanline_mask[screen_y as usize] & (1u64 << sprite) == 0 {
                    continue;
                }
                let tile_row = if flip_v { 7 - row } else { row };
                let pattern_addr = pattern_base + tile * 16 + tile_row;
                let lo = self.chr_read(pattern_addr);
                let hi = self.chr_read(pattern_addr + 8);
                for col in col_start..col_end {
                    let screen_x = sprite_x + col as i16;
                    let tile_col = if flip_h { col } else { 7 - col };
                    let pixel = ((lo >> tile_col) & 1) | (((hi >> tile_col) & 1) << 1);
                    if pixel == 0 {
                        continue;
                    }
                    let out_y = (screen_y - crop_top_i) as usize;
                    let out_x = (screen_x - crop_left_i) as usize;
                    let idx = out_y * width + out_x;
                    if behind_background
                        && self.bg_pixel_opaque(screen_x as usize, screen_y as usize)
                    {
                        dst[idx] = NES_GRAY_PALETTE
                            [self.bg_color_index(screen_x as usize, screen_y as usize) as usize];
                        continue;
                    }
                    dst[idx] = palette_gray[palette_base + pixel as usize];
                }
            }
        }
    }

    #[allow(dead_code)]
    pub(crate) fn draw_sprites_rgb(&self, dst: &mut [u8]) {
        if self.mask & 0x10 == 0 {
            return;
        }

        let plane = NES_WIDTH * NES_HEIGHT;
        let pattern_base = if self.ctrl & 0x08 != 0 {
            0x1000
        } else {
            0x0000
        };
        let sprite_scanline_mask = self.sprite_scanline_mask();
        for sprite in (0..64).rev() {
            let base = sprite * 4;
            let sprite_y = self.oam[base] as i16 + 1;
            let tile = self.oam[base + 1] as usize;
            let attr = self.oam[base + 2];
            let sprite_x = self.oam[base + 3] as i16;
            let palette_base = 0x10 + ((attr & 0x03) as usize) * 4;
            let flip_h = attr & 0x40 != 0;
            let flip_v = attr & 0x80 != 0;
            let behind_background = attr & 0x20 != 0;

            for row in 0..8usize {
                let screen_y = sprite_y + row as i16;
                if !(0..NES_HEIGHT as i16).contains(&screen_y) {
                    continue;
                }
                if sprite_scanline_mask[screen_y as usize] & (1u64 << sprite) == 0 {
                    continue;
                }
                let tile_row = if flip_v { 7 - row } else { row };
                let pattern_addr = pattern_base + tile * 16 + tile_row;
                let lo = self.chr_read(pattern_addr);
                let hi = self.chr_read(pattern_addr + 8);
                for col in 0..8usize {
                    let screen_x = sprite_x + col as i16;
                    if !(0..NES_WIDTH as i16).contains(&screen_x) {
                        continue;
                    }
                    let tile_col = if flip_h { col } else { 7 - col };
                    let pixel = ((lo >> tile_col) & 1) | (((hi >> tile_col) & 1) << 1);
                    if pixel == 0 {
                        continue;
                    }
                    let idx = screen_y as usize * NES_WIDTH + screen_x as usize;
                    if behind_background
                        && self.bg_pixel_opaque(screen_x as usize, screen_y as usize)
                    {
                        let color =
                            nes_rgb(self.bg_color_index(screen_x as usize, screen_y as usize));
                        dst[idx] = color[0];
                        dst[plane + idx] = color[1];
                        dst[plane * 2 + idx] = color[2];
                        continue;
                    }
                    let color = nes_rgb(self.palette[palette_base + pixel as usize]);
                    dst[idx] = color[0];
                    dst[plane + idx] = color[1];
                    dst[plane * 2 + idx] = color[2];
                }
            }
        }
    }

    #[allow(dead_code)]
    pub(crate) fn draw_sprites_rgb_cropped(&self, dst: &mut [u8], crop_top: usize, height: usize) {
        if self.mask & 0x10 == 0 {
            return;
        }

        let crop_top = crop_top as i16;
        let crop_bottom = crop_top + height as i16;
        let plane = NES_WIDTH * height;
        let pattern_base = if self.ctrl & 0x08 != 0 {
            0x1000
        } else {
            0x0000
        };
        let sprite_scanline_mask = self.sprite_scanline_mask();
        for sprite in (0..64).rev() {
            let base = sprite * 4;
            let sprite_y = self.oam[base] as i16 + 1;
            let tile = self.oam[base + 1] as usize;
            let attr = self.oam[base + 2];
            let sprite_x = self.oam[base + 3] as i16;
            let palette_base = 0x10 + ((attr & 0x03) as usize) * 4;
            let flip_h = attr & 0x40 != 0;
            let flip_v = attr & 0x80 != 0;
            let behind_background = attr & 0x20 != 0;

            for row in 0..8usize {
                let screen_y = sprite_y + row as i16;
                if screen_y < crop_top || screen_y >= crop_bottom {
                    continue;
                }
                if sprite_scanline_mask[screen_y as usize] & (1u64 << sprite) == 0 {
                    continue;
                }
                let tile_row = if flip_v { 7 - row } else { row };
                let pattern_addr = pattern_base + tile * 16 + tile_row;
                let lo = self.chr_read(pattern_addr);
                let hi = self.chr_read(pattern_addr + 8);
                for col in 0..8usize {
                    let screen_x = sprite_x + col as i16;
                    if !(0..NES_WIDTH as i16).contains(&screen_x) {
                        continue;
                    }
                    let tile_col = if flip_h { col } else { 7 - col };
                    let pixel = ((lo >> tile_col) & 1) | (((hi >> tile_col) & 1) << 1);
                    if pixel == 0 {
                        continue;
                    }
                    let out_y = (screen_y - crop_top) as usize;
                    let idx = out_y * NES_WIDTH + screen_x as usize;
                    if behind_background
                        && self.bg_pixel_opaque(screen_x as usize, screen_y as usize)
                    {
                        let color =
                            nes_rgb(self.bg_color_index(screen_x as usize, screen_y as usize));
                        dst[idx] = color[0];
                        dst[plane + idx] = color[1];
                        dst[plane * 2 + idx] = color[2];
                        continue;
                    }
                    let color = nes_rgb(self.palette[palette_base + pixel as usize]);
                    dst[idx] = color[0];
                    dst[plane + idx] = color[1];
                    dst[plane * 2 + idx] = color[2];
                }
            }
        }
    }

    pub(crate) fn draw_sprites_rgb_region(
        &self,
        dst: &mut [u8],
        crop_top: usize,
        crop_left: usize,
        width: usize,
        height: usize,
    ) {
        if self.mask & 0x10 == 0 {
            return;
        }

        let crop_top_i = crop_top as i16;
        let crop_bottom = crop_top_i + height as i16;
        let crop_left_i = crop_left as i16;
        let crop_right = crop_left_i + width as i16;
        let plane = width * height;
        let pattern_base = if self.ctrl & 0x08 != 0 {
            0x1000
        } else {
            0x0000
        };
        let sprite_scanline_mask = self.sprite_scanline_mask();
        for sprite in (0..64).rev() {
            let base = sprite * 4;
            let sprite_y = self.oam[base] as i16 + 1;
            let tile = self.oam[base + 1] as usize;
            let attr = self.oam[base + 2];
            let sprite_x = self.oam[base + 3] as i16;
            let palette_base = 0x10 + ((attr & 0x03) as usize) * 4;
            let flip_h = attr & 0x40 != 0;
            let flip_v = attr & 0x80 != 0;
            let behind_background = attr & 0x20 != 0;

            for row in 0..8usize {
                let screen_y = sprite_y + row as i16;
                if screen_y < crop_top_i || screen_y >= crop_bottom {
                    continue;
                }
                if sprite_scanline_mask[screen_y as usize] & (1u64 << sprite) == 0 {
                    continue;
                }
                let tile_row = if flip_v { 7 - row } else { row };
                let pattern_addr = pattern_base + tile * 16 + tile_row;
                let lo = self.chr_read(pattern_addr);
                let hi = self.chr_read(pattern_addr + 8);
                for col in 0..8usize {
                    let screen_x = sprite_x + col as i16;
                    if screen_x < crop_left_i || screen_x >= crop_right {
                        continue;
                    }
                    let tile_col = if flip_h { col } else { 7 - col };
                    let pixel = ((lo >> tile_col) & 1) | (((hi >> tile_col) & 1) << 1);
                    if pixel == 0 {
                        continue;
                    }
                    let out_y = (screen_y - crop_top_i) as usize;
                    let out_x = (screen_x - crop_left_i) as usize;
                    let idx = out_y * width + out_x;
                    if behind_background
                        && self.bg_pixel_opaque(screen_x as usize, screen_y as usize)
                    {
                        let color =
                            nes_rgb(self.bg_color_index(screen_x as usize, screen_y as usize));
                        dst[idx] = color[0];
                        dst[plane + idx] = color[1];
                        dst[plane * 2 + idx] = color[2];
                        continue;
                    }
                    let color = nes_rgb(self.palette[palette_base + pixel as usize]);
                    dst[idx] = color[0];
                    dst[plane + idx] = color[1];
                    dst[plane * 2 + idx] = color[2];
                }
            }
        }
    }

    #[inline]
    pub(crate) fn update_scroll_x_px(&mut self) {
        self.scroll_x_px = (((self.ctrl & 0x01) as u16) << 8) | self.scroll_x_low as u16;
    }

    #[inline]
    pub fn set_scroll_override_x(&mut self, scroll_x_px: Option<u16>) {
        self.scroll_override_x_px = scroll_x_px;
    }

    #[inline]
    pub(crate) fn render_scroll_x_px(&self) -> u16 {
        self.scroll_override_x_px.unwrap_or(self.scroll_x_px)
    }

    #[inline]
    pub(crate) fn palette_gray(&self) -> [u8; 32] {
        self.palette_gray
    }

    pub(crate) fn refresh_gray_palette_cache(&mut self) {
        for (dst, &color) in self.palette_gray.iter_mut().zip(self.palette.iter()) {
            *dst = NES_GRAY_PALETTE[(color & 0x3f) as usize];
        }
        *self.gray_bg_quad_cache.tables.get_mut().unwrap() =
            build_gray_bg_quad_tables(&self.palette_gray);
        self.gray_bg_quad_cache.dirty.store(0, Ordering::Relaxed);
    }

    pub(crate) fn refresh_gray_palette_entry(&mut self, idx: usize) {
        self.palette_gray[idx] = NES_GRAY_PALETTE[(self.palette[idx] & 0x3f) as usize];
        if idx == 0 {
            self.gray_bg_quad_cache.dirty.store(0x0f, Ordering::Relaxed);
        } else if idx < 16 && idx & 3 != 0 {
            self.gray_bg_quad_cache
                .dirty
                .fetch_or(1 << (idx / 4), Ordering::Relaxed);
        }
    }

    pub(crate) fn refresh_gray_bg_quad_tables(&self) {
        if self.gray_bg_quad_cache.dirty.load(Ordering::Relaxed) == 0 {
            return;
        }
        let mut tables = self.gray_bg_quad_cache.tables.write().unwrap();
        let dirty = self.gray_bg_quad_cache.dirty.swap(0, Ordering::Relaxed);
        for (palette_id, table) in tables.iter_mut().enumerate() {
            if dirty & (1 << palette_id) != 0 {
                rebuild_gray_bg_quad_table(&self.palette_gray, palette_id, table);
            }
        }
    }

    #[inline]
    pub(crate) fn bg_world_pos(&self, x: usize, y: usize) -> (usize, usize) {
        if y < 32 {
            (x, y)
        } else {
            (
                x + self.render_scroll_x_px() as usize,
                y + self.scroll_y_px as usize,
            )
        }
    }
}

const NES_GRAY_PALETTE: [u8; 64] = build_nes_gray_palette();

const fn build_nes_gray_palette() -> [u8; 64] {
    let mut table = [0; 64];
    let mut color = 0;
    while color < 64 {
        let rgb = NES_RGB_PALETTE[color];
        table[color] = (((rgb[0] as u32) * 77 + (rgb[1] as u32) * 150 + (rgb[2] as u32) * 29 + 128)
            >> 8) as u8;
        color += 1;
    }
    table
}

#[inline]
fn nes_rgb(color: u8) -> [u8; 3] {
    NES_RGB_PALETTE[(color as usize) & 0x3f]
}

const NES_RGB_PALETTE: [[u8; 3]; 64] = [
    [112, 116, 112],
    [32, 24, 136],
    [0, 0, 168],
    [64, 0, 152],
    [136, 0, 112],
    [168, 0, 16],
    [160, 0, 0],
    [120, 8, 0],
    [64, 44, 0],
    [0, 68, 0],
    [0, 80, 0],
    [0, 60, 16],
    [24, 60, 88],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [184, 188, 184],
    [0, 112, 232],
    [32, 56, 232],
    [128, 0, 240],
    [184, 0, 184],
    [224, 0, 88],
    [216, 40, 0],
    [200, 76, 8],
    [136, 112, 0],
    [0, 148, 0],
    [0, 168, 0],
    [0, 144, 56],
    [0, 128, 136],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [248, 252, 248],
    [56, 188, 248],
    [88, 148, 248],
    [64, 136, 248],
    [240, 120, 248],
    [248, 116, 176],
    [248, 116, 96],
    [248, 152, 56],
    [240, 188, 56],
    [128, 208, 16],
    [72, 220, 72],
    [88, 248, 152],
    [0, 232, 216],
    [120, 120, 120],
    [0, 0, 0],
    [0, 0, 0],
    [248, 252, 248],
    [168, 228, 248],
    [192, 212, 248],
    [208, 200, 248],
    [248, 196, 248],
    [248, 196, 216],
    [248, 188, 176],
    [248, 216, 168],
    [248, 228, 160],
    [224, 252, 160],
    [168, 240, 184],
    [176, 252, 200],
    [152, 252, 240],
    [192, 196, 192],
    [0, 0, 0],
    [0, 0, 0],
];

#[inline]
fn next_ppu_event_dot(current: usize, sprite0_hit_dot: Option<usize>) -> usize {
    if current < PPU_VBLANK_DOT {
        PPU_VBLANK_DOT
    } else if current < PPU_PRERENDER_DOT {
        PPU_PRERENDER_DOT
    } else if let Some(dot) = sprite0_hit_dot.filter(|&dot| current < dot) {
        dot
    } else {
        PPU_DOTS_PER_FRAME
    }
}

fn required_field<'a>(
    state: &'a [u8],
    name: &'static [u8; 4],
    display_name: &'static str,
    size: usize,
) -> Result<&'a [u8], StateLoadError> {
    optional_field(state, name, size).ok_or(StateLoadError::MissingField {
        name: display_name,
        size,
    })
}

fn optional_field<'a>(state: &'a [u8], name: &[u8; 4], size: usize) -> Option<&'a [u8]> {
    let header_len = name.len() + 4;
    state
        .windows(header_len)
        .enumerate()
        .find_map(|(offset, window)| {
            if &window[..4] != name {
                return None;
            }
            let field_size = u32::from_le_bytes(window[4..8].try_into().ok()?) as usize;
            if field_size != size {
                return None;
            }
            let start = offset + header_len;
            let end = start.checked_add(field_size)?;
            state.get(start..end)
        })
}

fn read_u16_le(value: &[u8]) -> Option<u16> {
    Some(u16::from_le_bytes(value.get(..2)?.try_into().ok()?))
}

#[inline(always)]
fn write_full_bg_tile_gray(dst: &mut [u8], lo: u8, hi: u8, colors: [u8; 4]) {
    debug_assert!(dst.len() >= 8);
    // SAFETY: The caller passes at least eight destination pixels for a full tile.
    unsafe {
        *dst.get_unchecked_mut(0) = colors[(((lo >> 7) & 1) | (((hi >> 7) & 1) << 1)) as usize];
        *dst.get_unchecked_mut(1) = colors[(((lo >> 6) & 1) | (((hi >> 6) & 1) << 1)) as usize];
        *dst.get_unchecked_mut(2) = colors[(((lo >> 5) & 1) | (((hi >> 5) & 1) << 1)) as usize];
        *dst.get_unchecked_mut(3) = colors[(((lo >> 4) & 1) | (((hi >> 4) & 1) << 1)) as usize];
        *dst.get_unchecked_mut(4) = colors[(((lo >> 3) & 1) | (((hi >> 3) & 1) << 1)) as usize];
        *dst.get_unchecked_mut(5) = colors[(((lo >> 2) & 1) | (((hi >> 2) & 1) << 1)) as usize];
        *dst.get_unchecked_mut(6) = colors[(((lo >> 1) & 1) | (((hi >> 1) & 1) << 1)) as usize];
        *dst.get_unchecked_mut(7) = colors[((lo & 1) | ((hi & 1) << 1)) as usize];
    }
}

#[inline(always)]
fn write_full_bg_tile_gray_pixels(dst: &mut [u8], pixels: u16, quads: &[u32; 256]) {
    debug_assert!(dst.len() >= 8);
    let lower = quads[(pixels & 0xff) as usize];
    let upper = quads[(pixels >> 8) as usize];
    let packed = u64::from(lower) | (u64::from(upper) << 32);
    // SAFETY: The caller passes at least eight destination pixels. Writing the
    // little-endian packed value preserves left-to-right byte order.
    unsafe {
        dst.as_mut_ptr()
            .cast::<u64>()
            .write_unaligned(packed.to_le());
    }
}

fn build_gray_bg_quad_tables(palette_gray: &[u8; 32]) -> [[u32; 256]; 4] {
    let mut tables = [[0u32; 256]; 4];
    for (palette_id, table) in tables.iter_mut().enumerate() {
        rebuild_gray_bg_quad_table(palette_gray, palette_id, table);
    }
    tables
}

fn rebuild_gray_bg_quad_table(palette_gray: &[u8; 32], palette_id: usize, table: &mut [u32; 256]) {
    let base = palette_id * 4;
    let colors = [
        palette_gray[0],
        palette_gray[base + 1],
        palette_gray[base + 2],
        palette_gray[base + 3],
    ];
    for (pixels, packed) in table.iter_mut().enumerate() {
        *packed = u32::from_le_bytes([
            colors[pixels & 3],
            colors[(pixels >> 2) & 3],
            colors[(pixels >> 4) & 3],
            colors[(pixels >> 6) & 3],
        ]);
    }
}

fn decode_chr_row_pixels(chr_rom: &[u8]) -> Vec<u16> {
    let mut decoded = vec![0; chr_rom.len()];
    for tile_base in (0..chr_rom.len()).step_by(16) {
        for row in 0..8usize {
            let lo = chr_rom[tile_base + row];
            let hi = chr_rom[tile_base + row + 8];
            let mut pixels = 0u16;
            for col in 0..8usize {
                let bit = 7 - col;
                let pixel = ((lo >> bit) & 1) | (((hi >> bit) & 1) << 1);
                pixels |= u16::from(pixel) << (col * 2);
            }
            decoded[tile_base + row] = pixels;
        }
    }
    decoded
}

#[derive(Clone)]
pub struct NromCore<T: PpuTiming> {
    pub cpu: Cpu,
    pub ppu: Ppu,
    pub ram: [u8; 2048],
    pub prg_rom: Vec<u8>,
    pub prg_addr_mask: usize,
    pub controller_state: u8,
    pub controller_shift: u8,
    pub controller_strobe: bool,
    pub extra_cycles: u16,
    timing: PhantomData<T>,
}

impl<T: PpuTiming> NromCore<T> {
    pub fn new(cart: Cartridge) -> Self {
        let prg_addr_mask = cart.prg_rom.len() - 1;
        Self {
            cpu: Cpu::new(),
            ppu: Ppu::new(cart.chr_rom, cart.vertical_mirroring, T::SPRITE0_HIT_DOT),
            ram: [0; 2048],
            prg_rom: cart.prg_rom,
            prg_addr_mask,
            controller_state: 0,
            controller_shift: 0,
            controller_strobe: false,
            extra_cycles: 0,
            timing: PhantomData,
        }
    }

    pub fn reset(&mut self) {
        self.cpu = Cpu::new();
        self.ppu.reset();
        self.ram = [0xff; 2048];
        self.controller_state = 0;
        self.controller_shift = 0;
        self.controller_strobe = false;
        self.extra_cycles = 0;
        self.cpu.pc = self.cpu_read_u16(0xfffc);
    }

    pub fn load_fceu_state(&mut self, state: &[u8]) -> Result<(), StateLoadError> {
        let pc = required_field(state, b"PC\0\0", "PC", 2)?;
        let a = required_field(state, b"A\0\0\0", "A", 1)?[0];
        let x = required_field(state, b"X\0\0\0", "X", 1)?[0];
        let y = required_field(state, b"Y\0\0\0", "Y", 1)?[0];
        let sp = required_field(state, b"S\0\0\0", "S", 1)?[0];
        let p = required_field(state, b"P\0\0\0", "P", 1)?[0];
        let ram = required_field(state, b"RAM\0", "RAM", 2048)?;
        let ntar = required_field(state, b"NTAR", "NTAR", 2048)?;
        let pram = required_field(state, b"PRAM", "PRAM", 32)?;
        let spra = required_field(state, b"SPRA", "SPRA", 256)?;
        let ppur = required_field(state, b"PPUR", "PPUR", 4)?;
        self.cpu = Cpu {
            a,
            x,
            y,
            sp,
            pc: read_u16_le(pc).unwrap_or(0),
            p,
        };
        self.ppu.load_fceu_state(
            ntar,
            pram,
            spra,
            ppur,
            optional_field(state, b"RADD", 2),
            optional_field(state, b"TADD", 2),
            optional_field(state, b"XOFF", 1),
        );
        self.ram.copy_from_slice(ram);
        self.controller_state = 0;
        self.controller_shift = 0;
        self.controller_strobe = false;
        self.extra_cycles = 0;
        Ok(())
    }

    #[inline]
    pub fn write_rgb_frame(&self, dst: &mut [u8]) {
        self.ppu.write_rgb_frame(dst);
    }
    #[inline]
    pub fn write_rgb_frame_cropped(&self, dst: &mut [u8], crop_top: usize, height: usize) {
        self.ppu.write_rgb_frame_cropped(dst, crop_top, height);
    }
    #[inline]
    pub fn write_rgb_visible_frame_cropped(&self, dst: &mut [u8], crop_top: usize, height: usize) {
        self.write_rgb_visible_frame_region(dst, crop_top, 0, VISIBLE_FRAME_WIDTH, height);
    }
    #[inline]
    pub fn write_rgb_visible_frame_region(
        &self,
        dst: &mut [u8],
        crop_top: usize,
        crop_left: usize,
        width: usize,
        height: usize,
    ) {
        self.ppu.write_rgb_frame_region(
            dst,
            VISIBLE_FRAME_TOP + crop_top,
            VISIBLE_FRAME_LEFT + crop_left,
            width,
            height,
        );
    }
    #[inline]
    pub fn write_gray_frame(&self, dst: &mut [u8]) {
        self.ppu.write_gray_frame(dst);
    }
    #[inline]
    pub fn write_gray_frame_cropped(&self, dst: &mut [u8], crop_top: usize, height: usize) {
        self.ppu.write_gray_frame_cropped(dst, crop_top, height);
    }
    #[inline]
    pub fn write_gray_visible_frame_cropped(&self, dst: &mut [u8], crop_top: usize, height: usize) {
        self.write_gray_visible_frame_region(dst, crop_top, 0, VISIBLE_FRAME_WIDTH, height);
    }
    #[inline]
    pub fn write_gray_visible_frame_region(
        &self,
        dst: &mut [u8],
        crop_top: usize,
        crop_left: usize,
        width: usize,
        height: usize,
    ) {
        self.ppu.write_gray_frame_region(
            dst,
            VISIBLE_FRAME_TOP + crop_top,
            VISIBLE_FRAME_LEFT + crop_left,
            width,
            height,
        );
    }
    #[inline]
    pub fn write_gray_visible_mask_lower_frame(
        &self,
        dst: &mut [u8],
        crop_top: usize,
        height: usize,
    ) {
        self.ppu
            .write_gray_visible_mask_lower_frame(dst, VISIBLE_FRAME_TOP + crop_top, height);
    }
    #[inline]
    pub fn write_gray_frame_cropped_area_84x84(&self, dst: &mut [u8], sprite_shadow: &mut [u8]) {
        self.ppu
            .write_gray_frame_cropped_area_84x84(dst, sprite_shadow);
    }
    #[inline]
    pub fn ram(&self) -> &[u8; 2048] {
        &self.ram
    }
    #[inline]
    pub fn oam(&self) -> &[u8; 256] {
        self.ppu.oam()
    }
    #[inline]
    pub fn debug_bg_pixel(&self, x: usize, y: usize) -> (u8, bool) {
        self.ppu.debug_bg_pixel(x, y)
    }

    #[cfg(feature = "test-support")]
    pub fn snapshot(&self) -> CoreSnapshot {
        CoreSnapshot {
            a: self.cpu.a,
            x: self.cpu.x,
            y: self.cpu.y,
            sp: self.cpu.sp,
            pc: self.cpu.pc,
            status_flags: self.cpu.p,
            ram: self.ram,
            oam: self.ppu.oam,
            controller_state: self.controller_state,
            controller_shift: self.controller_shift,
            controller_strobe: self.controller_strobe,
            extra_cycles: self.extra_cycles,
            ppu_status: self.ppu.status,
            ppu_frame_dot: self.ppu.frame_dot,
            ppu_next_event_dot: self.ppu.next_event_dot,
            ppu_frame: self.ppu.frame,
        }
    }

    pub fn cpu_read(&mut self, addr: u16) -> u8 {
        match addr {
            0x0000..=0x1fff => self.ram[addr as usize & 0x07ff],
            0x2000..=0x3fff => self.ppu.cpu_read_register(addr),
            0x4016 => self.controller_read(),
            0x8000..=0xffff => self.prg_read(addr),
            _ => 0,
        }
    }

    #[inline]
    pub(crate) fn cpu_write(&mut self, addr: u16, value: u8) {
        match addr {
            0x0000..=0x1fff => self.ram[addr as usize & 0x07ff] = value,
            0x2000..=0x3fff => self.ppu.cpu_write_register(addr, value),
            0x4014 => self.oam_dma(value),
            0x4016 => self.controller_write(value),
            _ => {}
        }
    }

    #[inline]
    pub fn prg_read(&self, addr: u16) -> u8 {
        let idx = ((addr - 0x8000) as usize) & self.prg_addr_mask;
        // SAFETY: SMB/NROM PRG ROM sizes are power-of-two and prg_addr_mask is len - 1.
        unsafe { *self.prg_rom.get_unchecked(idx) }
    }

    #[inline]
    pub(crate) fn cpu_read_u16(&mut self, addr: u16) -> u16 {
        let lo = self.cpu_read(addr) as u16;
        let hi = self.cpu_read(addr.wrapping_add(1)) as u16;
        lo | (hi << 8)
    }

    #[inline]
    pub fn ram_read(&self, addr: usize) -> u8 {
        // SAFETY: Masking to 0x07ff keeps the index within the 2 KiB internal RAM.
        unsafe { *self.ram.get_unchecked(addr & 0x07ff) }
    }

    #[inline]
    pub fn ram_write(&mut self, addr: usize, value: u8) {
        // SAFETY: Masking to 0x07ff keeps the index within the 2 KiB internal RAM.
        unsafe {
            *self.ram.get_unchecked_mut(addr & 0x07ff) = value;
        }
    }

    #[inline]
    pub(crate) fn controller_write(&mut self, value: u8) {
        self.controller_strobe = value & 1 != 0;
        if self.controller_strobe {
            self.controller_shift = self.controller_state;
        }
    }

    #[inline]
    pub(crate) fn controller_read(&mut self) -> u8 {
        if self.controller_strobe {
            return 0x40 | (self.controller_state & 1);
        }
        let value = self.controller_shift & 1;
        self.controller_shift = (self.controller_shift >> 1) | 0x80;
        0x40 | value
    }

    pub(crate) fn oam_dma(&mut self, page: u8) {
        let base = (page as u16) << 8;
        for i in 0..256u16 {
            let value = self.cpu_read(base | i);
            let idx = self.ppu.oam_addr.wrapping_add(i as u8) as usize;
            self.ppu.oam[idx] = value;
        }
        self.extra_cycles = self.extra_cycles.wrapping_add(513);
    }

    #[inline]
    pub(crate) fn fetch_u8(&mut self) -> u8 {
        let value = if self.cpu.pc >= 0x8000 {
            self.prg_read(self.cpu.pc)
        } else {
            self.cpu_read(self.cpu.pc)
        };
        self.cpu.pc = self.cpu.pc.wrapping_add(1);
        value
    }

    #[inline]
    pub(crate) fn fetch_u16(&mut self) -> u16 {
        let lo = self.fetch_u8() as u16;
        let hi = self.fetch_u8() as u16;
        lo | (hi << 8)
    }

    #[inline]
    pub(crate) fn zp_index(&mut self) -> usize {
        self.fetch_u8() as usize
    }

    #[inline]
    pub(crate) fn zpx_index(&mut self) -> usize {
        self.fetch_u8().wrapping_add(self.cpu.x) as usize
    }

    #[inline]
    pub(crate) fn zpy_index(&mut self) -> usize {
        self.fetch_u8().wrapping_add(self.cpu.y) as usize
    }

    #[inline]
    pub(crate) fn abs(&mut self) -> u16 {
        self.fetch_u16()
    }

    #[inline]
    pub(crate) fn absx(&mut self) -> (u16, bool) {
        let base = self.fetch_u16();
        let addr = base.wrapping_add(self.cpu.x as u16);
        (addr, page_crossed(base, addr))
    }

    #[inline]
    pub(crate) fn absy(&mut self) -> (u16, bool) {
        let base = self.fetch_u16();
        let addr = base.wrapping_add(self.cpu.y as u16);
        (addr, page_crossed(base, addr))
    }

    #[inline]
    pub(crate) fn indx(&mut self) -> u16 {
        let ptr = self.fetch_u8().wrapping_add(self.cpu.x);
        let lo = self.ram_read(ptr as usize) as u16;
        let hi = self.ram_read(ptr.wrapping_add(1) as usize) as u16;
        lo | (hi << 8)
    }

    #[inline]
    pub(crate) fn indy(&mut self) -> (u16, bool) {
        let ptr = self.fetch_u8();
        let lo = self.ram_read(ptr as usize) as u16;
        let hi = self.ram_read(ptr.wrapping_add(1) as usize) as u16;
        let base = lo | (hi << 8);
        let addr = base.wrapping_add(self.cpu.y as u16);
        (addr, page_crossed(base, addr))
    }

    #[inline]
    pub fn set_flag(&mut self, flag: u8, value: bool) {
        if value {
            self.cpu.p |= flag;
        } else {
            self.cpu.p &= !flag;
        }
        self.cpu.p |= FLAG_U;
    }

    #[inline]
    pub fn flag(&self, flag: u8) -> bool {
        self.cpu.p & flag != 0
    }

    #[inline]
    pub fn set_zn(&mut self, value: u8) {
        let mut p = self.cpu.p & !(FLAG_Z | FLAG_N);
        if value == 0 {
            p |= FLAG_Z;
        }
        if value & 0x80 != 0 {
            p |= FLAG_N;
        }
        self.cpu.p = p | FLAG_U;
    }

    #[inline]
    pub fn push(&mut self, value: u8) {
        self.ram_write(0x0100 | self.cpu.sp as usize, value);
        self.cpu.sp = self.cpu.sp.wrapping_sub(1);
    }

    #[inline]
    pub fn pop(&mut self) -> u8 {
        self.cpu.sp = self.cpu.sp.wrapping_add(1);
        self.ram_read(0x0100 | self.cpu.sp as usize)
    }

    #[inline]
    pub fn push_u16(&mut self, value: u16) {
        self.push((value >> 8) as u8);
        self.push(value as u8);
    }

    #[inline]
    pub fn pop_u16(&mut self) -> u16 {
        let lo = self.pop() as u16;
        let hi = self.pop() as u16;
        lo | (hi << 8)
    }

    pub(crate) fn interrupt(&mut self, vector: u16, brk: bool) {
        self.push_u16(self.cpu.pc);
        let mut p = self.cpu.p | FLAG_U;
        if brk {
            p |= FLAG_B;
        } else {
            p &= !FLAG_B;
        }
        self.push(p);
        self.set_flag(FLAG_I, true);
        self.cpu.pc = self.cpu_read_u16(vector);
    }

    #[inline]
    pub(crate) fn cpu_step(&mut self) -> u16 {
        let opcode = self.fetch_u8();
        self.cpu_step_decoded(opcode)
    }

    #[cfg(feature = "test-support")]
    #[doc(hidden)]
    pub fn interpret_one_for_test(&mut self) -> u16 {
        self.cpu_step()
    }

    #[inline]
    pub(crate) fn cpu_step_profiled(&mut self, profiler: &mut Profiler) -> u16 {
        let pc = self.cpu.pc;
        let opcode = self.fetch_u8();
        profiler.record_cpu_step(pc, opcode);
        self.cpu_step_decoded(opcode)
    }

    pub(crate) fn cpu_step_decoded(&mut self, opcode: u8) -> u16 {
        let mut cycles = match opcode {
            0x00 => {
                self.cpu.pc = self.cpu.pc.wrapping_add(1);
                self.interrupt(0xfffe, true);
                7
            }
            0x01 => {
                let a = self.indx();
                let v = self.cpu_read(a);
                self.ora(v);
                6
            }
            0x05 => {
                let a = self.zp_index();
                let v = self.ram_read(a);
                self.ora(v);
                3
            }
            0x06 => {
                let a = self.zp_index();
                self.asl_ram(a);
                5
            }
            0x08 => {
                self.push(self.cpu.p | FLAG_B | FLAG_U);
                3
            }
            0x09 => {
                let v = self.fetch_u8();
                self.ora(v);
                2
            }
            0x0a => {
                self.cpu.a = self.asl(self.cpu.a);
                2
            }
            0x0d => {
                let a = self.abs();
                let v = self.cpu_read(a);
                self.ora(v);
                4
            }
            0x0e => {
                let a = self.abs();
                self.asl_mem(a);
                6
            }
            0x10 => self.branch(!self.flag(FLAG_N)),
            0x11 => {
                let (a, p) = self.indy();
                let v = self.cpu_read(a);
                self.ora(v);
                5 + p as u16
            }
            0x15 => {
                let a = self.zpx_index();
                let v = self.ram_read(a);
                self.ora(v);
                4
            }
            0x16 => {
                let a = self.zpx_index();
                self.asl_ram(a);
                6
            }
            0x18 => {
                self.set_flag(FLAG_C, false);
                2
            }
            0x19 => {
                let (a, p) = self.absy();
                let v = self.cpu_read(a);
                self.ora(v);
                4 + p as u16
            }
            0x1d => {
                let (a, p) = self.absx();
                let v = self.cpu_read(a);
                self.ora(v);
                4 + p as u16
            }
            0x1e => {
                let (a, _) = self.absx();
                self.asl_mem(a);
                7
            }
            0x20 => {
                let a = self.abs();
                self.push_u16(self.cpu.pc.wrapping_sub(1));
                self.cpu.pc = a;
                6
            }
            0x21 => {
                let a = self.indx();
                let v = self.cpu_read(a);
                self.and(v);
                6
            }
            0x24 => {
                let a = self.zp_index();
                let v = self.ram_read(a);
                self.bit(v);
                3
            }
            0x25 => {
                let a = self.zp_index();
                let v = self.ram_read(a);
                self.and(v);
                3
            }
            0x26 => {
                let a = self.zp_index();
                self.rol_ram(a);
                5
            }
            0x28 => {
                self.cpu.p = (self.pop() & !FLAG_B) | FLAG_U;
                4
            }
            0x29 => {
                let v = self.fetch_u8();
                self.and(v);
                2
            }
            0x2a => {
                self.cpu.a = self.rol(self.cpu.a);
                2
            }
            0x2c => {
                let a = self.abs();
                let v = self.cpu_read(a);
                self.bit(v);
                4
            }
            0x2d => {
                let a = self.abs();
                let v = self.cpu_read(a);
                self.and(v);
                4
            }
            0x2e => {
                let a = self.abs();
                self.rol_mem(a);
                6
            }
            0x30 => self.branch(self.flag(FLAG_N)),
            0x31 => {
                let (a, p) = self.indy();
                let v = self.cpu_read(a);
                self.and(v);
                5 + p as u16
            }
            0x35 => {
                let a = self.zpx_index();
                let v = self.ram_read(a);
                self.and(v);
                4
            }
            0x36 => {
                let a = self.zpx_index();
                self.rol_ram(a);
                6
            }
            0x38 => {
                self.set_flag(FLAG_C, true);
                2
            }
            0x39 => {
                let (a, p) = self.absy();
                let v = self.cpu_read(a);
                self.and(v);
                4 + p as u16
            }
            0x3d => {
                let (a, p) = self.absx();
                let v = self.cpu_read(a);
                self.and(v);
                4 + p as u16
            }
            0x3e => {
                let (a, _) = self.absx();
                self.rol_mem(a);
                7
            }
            0x40 => {
                self.cpu.p = (self.pop() & !FLAG_B) | FLAG_U;
                self.cpu.pc = self.pop_u16();
                6
            }
            0x41 => {
                let a = self.indx();
                let v = self.cpu_read(a);
                self.eor(v);
                6
            }
            0x45 => {
                let a = self.zp_index();
                let v = self.ram_read(a);
                self.eor(v);
                3
            }
            0x46 => {
                let a = self.zp_index();
                self.lsr_ram(a);
                5
            }
            0x48 => {
                self.push(self.cpu.a);
                3
            }
            0x49 => {
                let v = self.fetch_u8();
                self.eor(v);
                2
            }
            0x4a => {
                self.cpu.a = self.lsr(self.cpu.a);
                2
            }
            0x4c => {
                let a = self.abs();
                self.cpu.pc = a;
                3
            }
            0x4d => {
                let a = self.abs();
                let v = self.cpu_read(a);
                self.eor(v);
                4
            }
            0x4e => {
                let a = self.abs();
                self.lsr_mem(a);
                6
            }
            0x50 => self.branch(!self.flag(FLAG_V)),
            0x51 => {
                let (a, p) = self.indy();
                let v = self.cpu_read(a);
                self.eor(v);
                5 + p as u16
            }
            0x55 => {
                let a = self.zpx_index();
                let v = self.ram_read(a);
                self.eor(v);
                4
            }
            0x56 => {
                let a = self.zpx_index();
                self.lsr_ram(a);
                6
            }
            0x58 => {
                self.set_flag(FLAG_I, false);
                2
            }
            0x59 => {
                let (a, p) = self.absy();
                let v = self.cpu_read(a);
                self.eor(v);
                4 + p as u16
            }
            0x5d => {
                let (a, p) = self.absx();
                let v = self.cpu_read(a);
                self.eor(v);
                4 + p as u16
            }
            0x5e => {
                let (a, _) = self.absx();
                self.lsr_mem(a);
                7
            }
            0x60 => {
                self.cpu.pc = self.pop_u16().wrapping_add(1);
                6
            }
            0x61 => {
                let a = self.indx();
                let v = self.cpu_read(a);
                self.adc(v);
                6
            }
            0x65 => {
                let a = self.zp_index();
                let v = self.ram_read(a);
                self.adc(v);
                3
            }
            0x66 => {
                let a = self.zp_index();
                self.ror_ram(a);
                5
            }
            0x68 => {
                self.cpu.a = self.pop();
                self.set_zn(self.cpu.a);
                4
            }
            0x69 => {
                let v = self.fetch_u8();
                self.adc(v);
                2
            }
            0x6a => {
                self.cpu.a = self.ror(self.cpu.a);
                2
            }
            0x6c => {
                let ptr = self.abs();
                let lo = self.cpu_read(ptr) as u16;
                let hi_addr = (ptr & 0xff00) | ptr.wrapping_add(1) & 0x00ff;
                let hi = self.cpu_read(hi_addr) as u16;
                self.cpu.pc = lo | (hi << 8);
                5
            }
            0x6d => {
                let a = self.abs();
                let v = self.cpu_read(a);
                self.adc(v);
                4
            }
            0x6e => {
                let a = self.abs();
                self.ror_mem(a);
                6
            }
            0x70 => self.branch(self.flag(FLAG_V)),
            0x71 => {
                let (a, p) = self.indy();
                let v = self.cpu_read(a);
                self.adc(v);
                5 + p as u16
            }
            0x75 => {
                let a = self.zpx_index();
                let v = self.ram_read(a);
                self.adc(v);
                4
            }
            0x76 => {
                let a = self.zpx_index();
                self.ror_ram(a);
                6
            }
            0x78 => {
                self.set_flag(FLAG_I, true);
                2
            }
            0x79 => {
                let (a, p) = self.absy();
                let v = self.cpu_read(a);
                self.adc(v);
                4 + p as u16
            }
            0x7d => {
                let (a, p) = self.absx();
                let v = self.cpu_read(a);
                self.adc(v);
                4 + p as u16
            }
            0x7e => {
                let (a, _) = self.absx();
                self.ror_mem(a);
                7
            }
            0x81 => {
                let a = self.indx();
                self.cpu_write(a, self.cpu.a);
                6
            }
            0x84 => {
                let a = self.zp_index();
                self.ram_write(a, self.cpu.y);
                3
            }
            0x85 => {
                let a = self.zp_index();
                self.ram_write(a, self.cpu.a);
                3
            }
            0x86 => {
                let a = self.zp_index();
                self.ram_write(a, self.cpu.x);
                3
            }
            0x88 => {
                self.cpu.y = self.cpu.y.wrapping_sub(1);
                self.set_zn(self.cpu.y);
                2
            }
            0x8a => {
                self.cpu.a = self.cpu.x;
                self.set_zn(self.cpu.a);
                2
            }
            0x8c => {
                let a = self.abs();
                self.cpu_write(a, self.cpu.y);
                4
            }
            0x8d => {
                let a = self.abs();
                self.cpu_write(a, self.cpu.a);
                4
            }
            0x8e => {
                let a = self.abs();
                self.cpu_write(a, self.cpu.x);
                4
            }
            0x90 => self.branch(!self.flag(FLAG_C)),
            0x91 => {
                let (a, _) = self.indy();
                self.cpu_write(a, self.cpu.a);
                6
            }
            0x94 => {
                let a = self.zpx_index();
                self.ram_write(a, self.cpu.y);
                4
            }
            0x95 => {
                let a = self.zpx_index();
                self.ram_write(a, self.cpu.a);
                4
            }
            0x96 => {
                let a = self.zpy_index();
                self.ram_write(a, self.cpu.x);
                4
            }
            0x98 => {
                self.cpu.a = self.cpu.y;
                self.set_zn(self.cpu.a);
                2
            }
            0x99 => {
                let (a, _) = self.absy();
                self.cpu_write(a, self.cpu.a);
                5
            }
            0x9a => {
                self.cpu.sp = self.cpu.x;
                2
            }
            0x9d => {
                let (a, _) = self.absx();
                self.cpu_write(a, self.cpu.a);
                5
            }
            0xa0 => {
                let v = self.fetch_u8();
                self.cpu.y = v;
                self.set_zn(v);
                2
            }
            0xa1 => {
                let a = self.indx();
                let v = self.cpu_read(a);
                self.cpu.a = v;
                self.set_zn(v);
                6
            }
            0xa2 => {
                let v = self.fetch_u8();
                self.cpu.x = v;
                self.set_zn(v);
                2
            }
            0xa4 => {
                let a = self.zp_index();
                let v = self.ram_read(a);
                self.cpu.y = v;
                self.set_zn(v);
                3
            }
            0xa5 => {
                let a = self.zp_index();
                let v = self.ram_read(a);
                self.cpu.a = v;
                self.set_zn(v);
                3
            }
            0xa6 => {
                let a = self.zp_index();
                let v = self.ram_read(a);
                self.cpu.x = v;
                self.set_zn(v);
                3
            }
            0xa8 => {
                self.cpu.y = self.cpu.a;
                self.set_zn(self.cpu.y);
                2
            }
            0xa9 => {
                let v = self.fetch_u8();
                self.cpu.a = v;
                self.set_zn(v);
                2
            }
            0xaa => {
                self.cpu.x = self.cpu.a;
                self.set_zn(self.cpu.x);
                2
            }
            0xac => {
                let a = self.abs();
                let v = self.cpu_read(a);
                self.cpu.y = v;
                self.set_zn(v);
                4
            }
            0xad => {
                let a = self.abs();
                let v = self.cpu_read(a);
                self.cpu.a = v;
                self.set_zn(v);
                4
            }
            0xae => {
                let a = self.abs();
                let v = self.cpu_read(a);
                self.cpu.x = v;
                self.set_zn(v);
                4
            }
            0xb0 => self.branch(self.flag(FLAG_C)),
            0xb1 => {
                let (a, p) = self.indy();
                let v = self.cpu_read(a);
                self.cpu.a = v;
                self.set_zn(v);
                5 + p as u16
            }
            0xb4 => {
                let a = self.zpx_index();
                let v = self.ram_read(a);
                self.cpu.y = v;
                self.set_zn(v);
                4
            }
            0xb5 => {
                let a = self.zpx_index();
                let v = self.ram_read(a);
                self.cpu.a = v;
                self.set_zn(v);
                4
            }
            0xb6 => {
                let a = self.zpy_index();
                let v = self.ram_read(a);
                self.cpu.x = v;
                self.set_zn(v);
                4
            }
            0xb8 => {
                self.set_flag(FLAG_V, false);
                2
            }
            0xb9 => {
                let (a, p) = self.absy();
                let v = self.cpu_read(a);
                self.cpu.a = v;
                self.set_zn(v);
                4 + p as u16
            }
            0xba => {
                self.cpu.x = self.cpu.sp;
                self.set_zn(self.cpu.x);
                2
            }
            0xbc => {
                let (a, p) = self.absx();
                let v = self.cpu_read(a);
                self.cpu.y = v;
                self.set_zn(v);
                4 + p as u16
            }
            0xbd => {
                let (a, p) = self.absx();
                let v = self.cpu_read(a);
                self.cpu.a = v;
                self.set_zn(v);
                4 + p as u16
            }
            0xbe => {
                let (a, p) = self.absy();
                let v = self.cpu_read(a);
                self.cpu.x = v;
                self.set_zn(v);
                4 + p as u16
            }
            0xc0 => {
                let v = self.fetch_u8();
                self.cmp(self.cpu.y, v);
                2
            }
            0xc1 => {
                let a = self.indx();
                let v = self.cpu_read(a);
                self.cmp(self.cpu.a, v);
                6
            }
            0xc4 => {
                let a = self.zp_index();
                let v = self.ram_read(a);
                self.cmp(self.cpu.y, v);
                3
            }
            0xc5 => {
                let a = self.zp_index();
                let v = self.ram_read(a);
                self.cmp(self.cpu.a, v);
                3
            }
            0xc6 => {
                let a = self.zp_index();
                self.dec_ram(a);
                5
            }
            0xc8 => {
                self.cpu.y = self.cpu.y.wrapping_add(1);
                self.set_zn(self.cpu.y);
                2
            }
            0xc9 => {
                let v = self.fetch_u8();
                self.cmp(self.cpu.a, v);
                2
            }
            0xca => {
                self.cpu.x = self.cpu.x.wrapping_sub(1);
                self.set_zn(self.cpu.x);
                2
            }
            0xcc => {
                let a = self.abs();
                let v = self.cpu_read(a);
                self.cmp(self.cpu.y, v);
                4
            }
            0xcd => {
                let a = self.abs();
                let v = self.cpu_read(a);
                self.cmp(self.cpu.a, v);
                4
            }
            0xce => {
                let a = self.abs();
                self.dec_mem(a);
                6
            }
            0xd0 => self.branch(!self.flag(FLAG_Z)),
            0xd1 => {
                let (a, p) = self.indy();
                let v = self.cpu_read(a);
                self.cmp(self.cpu.a, v);
                5 + p as u16
            }
            0xd5 => {
                let a = self.zpx_index();
                let v = self.ram_read(a);
                self.cmp(self.cpu.a, v);
                4
            }
            0xd6 => {
                let a = self.zpx_index();
                self.dec_ram(a);
                6
            }
            0xd8 => {
                self.set_flag(FLAG_D, false);
                2
            }
            0xd9 => {
                let (a, p) = self.absy();
                let v = self.cpu_read(a);
                self.cmp(self.cpu.a, v);
                4 + p as u16
            }
            0xdd => {
                let (a, p) = self.absx();
                let v = self.cpu_read(a);
                self.cmp(self.cpu.a, v);
                4 + p as u16
            }
            0xde => {
                let (a, _) = self.absx();
                self.dec_mem(a);
                7
            }
            0xe0 => {
                let v = self.fetch_u8();
                self.cmp(self.cpu.x, v);
                2
            }
            0xe1 => {
                let a = self.indx();
                let v = self.cpu_read(a);
                self.sbc(v);
                6
            }
            0xe4 => {
                let a = self.zp_index();
                let v = self.ram_read(a);
                self.cmp(self.cpu.x, v);
                3
            }
            0xe5 => {
                let a = self.zp_index();
                let v = self.ram_read(a);
                self.sbc(v);
                3
            }
            0xe6 => {
                let a = self.zp_index();
                self.inc_ram(a);
                5
            }
            0xe8 => {
                self.cpu.x = self.cpu.x.wrapping_add(1);
                self.set_zn(self.cpu.x);
                2
            }
            0xe9 => {
                let v = self.fetch_u8();
                self.sbc(v);
                2
            }
            0xea => 2,
            0xec => {
                let a = self.abs();
                let v = self.cpu_read(a);
                self.cmp(self.cpu.x, v);
                4
            }
            0xed => {
                let a = self.abs();
                let v = self.cpu_read(a);
                self.sbc(v);
                4
            }
            0xee => {
                let a = self.abs();
                self.inc_mem(a);
                6
            }
            0xf0 => self.branch(self.flag(FLAG_Z)),
            0xf1 => {
                let (a, p) = self.indy();
                let v = self.cpu_read(a);
                self.sbc(v);
                5 + p as u16
            }
            0xf5 => {
                let a = self.zpx_index();
                let v = self.ram_read(a);
                self.sbc(v);
                4
            }
            0xf6 => {
                let a = self.zpx_index();
                self.inc_ram(a);
                6
            }
            0xf8 => {
                self.set_flag(FLAG_D, true);
                2
            }
            0xf9 => {
                let (a, p) = self.absy();
                let v = self.cpu_read(a);
                self.sbc(v);
                4 + p as u16
            }
            0xfd => {
                let (a, p) = self.absx();
                let v = self.cpu_read(a);
                self.sbc(v);
                4 + p as u16
            }
            0xfe => {
                let (a, _) = self.absx();
                self.inc_mem(a);
                7
            }
            _ => 2,
        };
        cycles = cycles.saturating_add(self.extra_cycles);
        self.extra_cycles = 0;
        cycles
    }

    #[inline]
    pub(crate) fn branch(&mut self, condition: bool) -> u16 {
        let offset = self.fetch_u8() as i8;
        if !condition {
            return 2;
        }
        let old_pc = self.cpu.pc;
        self.cpu.pc = self.cpu.pc.wrapping_add(offset as i16 as u16);
        3 + page_crossed(old_pc, self.cpu.pc) as u16
    }

    #[inline]
    pub fn adc(&mut self, value: u8) {
        let carry = u8::from(self.flag(FLAG_C));
        let a = self.cpu.a;
        let sum = a as u16 + value as u16 + carry as u16;
        let result = sum as u8;
        self.cpu.a = result;
        let mut p = self.cpu.p & !(FLAG_C | FLAG_V | FLAG_Z | FLAG_N);
        if sum > 0xff {
            p |= FLAG_C;
        }
        if (!(a ^ value) & (a ^ result) & 0x80) != 0 {
            p |= FLAG_V;
        }
        if result == 0 {
            p |= FLAG_Z;
        }
        if result & 0x80 != 0 {
            p |= FLAG_N;
        }
        self.cpu.p = p | FLAG_U;
    }

    #[inline]
    pub fn sbc(&mut self, value: u8) {
        self.adc(!value);
    }

    #[inline]
    pub fn cmp(&mut self, reg: u8, value: u8) {
        let result = reg.wrapping_sub(value);
        let mut p = self.cpu.p & !(FLAG_C | FLAG_Z | FLAG_N);
        if reg >= value {
            p |= FLAG_C;
        }
        if result == 0 {
            p |= FLAG_Z;
        }
        if result & 0x80 != 0 {
            p |= FLAG_N;
        }
        self.cpu.p = p | FLAG_U;
    }

    #[inline]
    pub fn ora(&mut self, value: u8) {
        self.cpu.a |= value;
        self.set_zn(self.cpu.a);
    }

    #[inline]
    pub fn and(&mut self, value: u8) {
        self.cpu.a &= value;
        self.set_zn(self.cpu.a);
    }

    #[inline]
    pub(crate) fn eor(&mut self, value: u8) {
        self.cpu.a ^= value;
        self.set_zn(self.cpu.a);
    }

    #[inline]
    pub(crate) fn bit(&mut self, value: u8) {
        let mut p = self.cpu.p & !(FLAG_Z | FLAG_V | FLAG_N);
        if self.cpu.a & value == 0 {
            p |= FLAG_Z;
        }
        p |= value & (FLAG_V | FLAG_N);
        self.cpu.p = p | FLAG_U;
    }

    #[inline]
    pub fn asl(&mut self, value: u8) -> u8 {
        let result = value << 1;
        let mut p = self.cpu.p & !(FLAG_C | FLAG_Z | FLAG_N);
        if value & 0x80 != 0 {
            p |= FLAG_C;
        }
        if result == 0 {
            p |= FLAG_Z;
        }
        if result & 0x80 != 0 {
            p |= FLAG_N;
        }
        self.cpu.p = p | FLAG_U;
        result
    }

    #[inline]
    pub fn lsr(&mut self, value: u8) -> u8 {
        let result = value >> 1;
        let mut p = self.cpu.p & !(FLAG_C | FLAG_Z | FLAG_N);
        if value & 1 != 0 {
            p |= FLAG_C;
        }
        if result == 0 {
            p |= FLAG_Z;
        }
        self.cpu.p = p | FLAG_U;
        result
    }

    #[inline]
    pub(crate) fn rol(&mut self, value: u8) -> u8 {
        let carry = u8::from(self.flag(FLAG_C));
        let result = (value << 1) | carry;
        let mut p = self.cpu.p & !(FLAG_C | FLAG_Z | FLAG_N);
        if value & 0x80 != 0 {
            p |= FLAG_C;
        }
        if result == 0 {
            p |= FLAG_Z;
        }
        if result & 0x80 != 0 {
            p |= FLAG_N;
        }
        self.cpu.p = p | FLAG_U;
        result
    }

    #[inline]
    pub fn ror(&mut self, value: u8) -> u8 {
        let carry = if self.flag(FLAG_C) { 0x80 } else { 0 };
        let result = (value >> 1) | carry;
        let mut p = self.cpu.p & !(FLAG_C | FLAG_Z | FLAG_N);
        if value & 1 != 0 {
            p |= FLAG_C;
        }
        if result == 0 {
            p |= FLAG_Z;
        }
        if result & 0x80 != 0 {
            p |= FLAG_N;
        }
        self.cpu.p = p | FLAG_U;
        result
    }

    #[inline]
    pub(crate) fn asl_mem(&mut self, addr: u16) {
        let value = self.cpu_read(addr);
        let result = self.asl(value);
        self.cpu_write(addr, result);
    }

    #[inline]
    pub(crate) fn asl_ram(&mut self, addr: usize) {
        let value = self.ram_read(addr);
        let result = self.asl(value);
        self.ram_write(addr, result);
    }

    #[inline]
    pub(crate) fn lsr_mem(&mut self, addr: u16) {
        let value = self.cpu_read(addr);
        let result = self.lsr(value);
        self.cpu_write(addr, result);
    }

    #[inline]
    pub(crate) fn lsr_ram(&mut self, addr: usize) {
        let value = self.ram_read(addr);
        let result = self.lsr(value);
        self.ram_write(addr, result);
    }

    #[inline]
    pub(crate) fn rol_mem(&mut self, addr: u16) {
        let value = self.cpu_read(addr);
        let result = self.rol(value);
        self.cpu_write(addr, result);
    }

    #[inline]
    pub(crate) fn rol_ram(&mut self, addr: usize) {
        let value = self.ram_read(addr);
        let result = self.rol(value);
        self.ram_write(addr, result);
    }

    #[inline]
    pub(crate) fn ror_mem(&mut self, addr: u16) {
        let value = self.cpu_read(addr);
        let result = self.ror(value);
        self.cpu_write(addr, result);
    }

    #[inline]
    pub(crate) fn ror_ram(&mut self, addr: usize) {
        let value = self.ram_read(addr);
        let result = self.ror(value);
        self.ram_write(addr, result);
    }

    #[inline]
    pub(crate) fn dec_mem(&mut self, addr: u16) {
        let result = self.cpu_read(addr).wrapping_sub(1);
        self.cpu_write(addr, result);
        self.set_zn(result);
    }

    #[inline]
    pub(crate) fn dec_ram(&mut self, addr: usize) {
        let result = self.ram_read(addr).wrapping_sub(1);
        self.ram_write(addr, result);
        self.set_zn(result);
    }

    #[inline]
    pub(crate) fn inc_mem(&mut self, addr: u16) {
        let result = self.cpu_read(addr).wrapping_add(1);
        self.cpu_write(addr, result);
        self.set_zn(result);
    }

    #[inline]
    pub(crate) fn inc_ram(&mut self, addr: usize) {
        let result = self.ram_read(addr).wrapping_add(1);
        self.ram_write(addr, result);
        self.set_zn(result);
    }
}

#[inline]
fn page_crossed(a: u16, b: u16) -> bool {
    (a & 0xff00) != (b & 0xff00)
}

#[derive(Clone)]
pub struct NromMachine<G: NromGame> {
    pub core: NromCore<G::PpuTiming>,
    pub options: G::Options,
    pub signals: G::Signals,
    pub fast_paths: G::FastPaths,
    done: bool,
    game: PhantomData<G>,
}

impl<G: NromGame> Deref for NromMachine<G> {
    type Target = NromCore<G::PpuTiming>;
    fn deref(&self) -> &Self::Target {
        &self.core
    }
}

impl<G: NromGame> DerefMut for NromMachine<G> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.core
    }
}

impl<G: NromGame> NromMachine<G> {
    pub fn new_with_options(cart: Cartridge, options: G::Options) -> Self {
        let fast_paths = G::detect_fast_paths(&cart.prg_rom, cart.prg_rom.len() - 1);
        let mut machine = Self {
            core: NromCore::new(cart),
            options,
            signals: G::Signals::default(),
            fast_paths,
            done: false,
            game: PhantomData,
        };
        machine.reset();
        machine
    }

    pub fn reset(&mut self) {
        self.core.reset();
        self.done = false;
        G::synchronize(&mut self.core, &mut self.signals);
    }

    pub fn prime_cold_boot(&mut self) {
        self.run_frame(0);
        G::synchronize(&mut self.core, &mut self.signals);
    }

    pub fn load_fceu_state(&mut self, state: &[u8]) -> Result<(), StateLoadError> {
        self.core.load_fceu_state(state)?;
        self.done = false;
        G::synchronize(&mut self.core, &mut self.signals);
        self.run_frame(0);
        G::synchronize(&mut self.core, &mut self.signals);
        Ok(())
    }

    #[inline]
    pub fn step_frame(&mut self, controller_state: u8) -> f32 {
        if self.done {
            return 0.0;
        }
        let context = G::pre_step(&self.signals);
        self.run_frame(controller_state);
        G::synchronize(&mut self.core, &mut self.signals);
        let result = G::post_step(&self.options, &self.signals, context);
        self.done |= result.done;
        result.reward
    }

    #[inline]
    pub fn step_frame_profiled(&mut self, controller_state: u8, profiler: &mut Profiler) -> f32 {
        if self.done {
            return 0.0;
        }
        let context = G::pre_step(&self.signals);
        let start = Instant::now();
        self.run_frame_profiled(controller_state, profiler);
        profiler.record_frame_step(start.elapsed());
        G::synchronize(&mut self.core, &mut self.signals);
        let result = G::post_step(&self.options, &self.signals, context);
        self.done |= result.done;
        result.reward
    }

    #[inline]
    pub fn signals(&self) -> &G::Signals {
        &self.signals
    }
    #[inline]
    pub fn is_done(&self) -> bool {
        self.done
    }

    #[cfg(feature = "test-support")]
    pub fn disable_fast_paths(&mut self) {
        self.fast_paths = G::FastPaths::default();
    }

    #[allow(clippy::collapsible_match)]
    fn run_frame(&mut self, buttons: u8) {
        self.core.controller_state = buttons;
        let mut budget = FrameBudget::default();
        loop {
            if self.core.ppu.take_nmi() {
                self.core.interrupt(0xfffa, false);
            }
            match G::dispatch_fast_path(&mut self.core, &self.fast_paths, &mut budget) {
                FastPathOutcome::Applied(policy) => {
                    let flush = policy == ResumePolicy::FlushNow
                        || (policy == ResumePolicy::FlushIfEventDue
                            && (budget.pending_ppu_cycles
                                >= self.core.ppu.cycles_until_next_event()
                                || budget.cpu_cycle_guard >= CPU_CYCLES_PER_FRAME_GUARD));
                    if flush {
                        let completed_frame = self.core.ppu.tick(budget.pending_ppu_cycles);
                        budget.pending_ppu_cycles = 0;
                        if completed_frame || budget.cpu_cycle_guard >= CPU_CYCLES_PER_FRAME_GUARD {
                            break;
                        }
                    }
                    continue;
                }
                FastPathOutcome::Miss => {}
            }
            let cycles = self.core.cpu_step() as usize;
            budget.cpu_cycle_guard += cycles;
            budget.pending_ppu_cycles += cycles * 3;
            if budget.pending_ppu_cycles >= self.core.ppu.cycles_until_next_event()
                || budget.cpu_cycle_guard >= CPU_CYCLES_PER_FRAME_GUARD
            {
                let completed_frame = self.core.ppu.tick(budget.pending_ppu_cycles);
                budget.pending_ppu_cycles = 0;
                if completed_frame || budget.cpu_cycle_guard >= CPU_CYCLES_PER_FRAME_GUARD {
                    break;
                }
            }
        }
        if budget.pending_ppu_cycles > 0 {
            self.core.ppu.tick(budget.pending_ppu_cycles);
        }
    }

    #[allow(clippy::collapsible_match)]
    fn run_frame_profiled(&mut self, buttons: u8, profiler: &mut Profiler) {
        self.core.controller_state = buttons;
        let mut budget = FrameBudget::default();
        loop {
            if self.core.ppu.take_nmi() {
                self.core.interrupt(0xfffa, false);
            }
            match G::dispatch_fast_path(&mut self.core, &self.fast_paths, &mut budget) {
                FastPathOutcome::Applied(policy) => {
                    let flush = policy == ResumePolicy::FlushNow
                        || (policy == ResumePolicy::FlushIfEventDue
                            && (budget.pending_ppu_cycles
                                >= self.core.ppu.cycles_until_next_event()
                                || budget.cpu_cycle_guard >= CPU_CYCLES_PER_FRAME_GUARD));
                    if flush {
                        let completed_frame = self
                            .core
                            .ppu
                            .tick_profiled(budget.pending_ppu_cycles, profiler);
                        budget.pending_ppu_cycles = 0;
                        if completed_frame || budget.cpu_cycle_guard >= CPU_CYCLES_PER_FRAME_GUARD {
                            break;
                        }
                    }
                    continue;
                }
                FastPathOutcome::Miss => {}
            }
            let cycles = self.core.cpu_step_profiled(profiler) as usize;
            budget.cpu_cycle_guard += cycles;
            budget.pending_ppu_cycles += cycles * 3;
            if budget.pending_ppu_cycles >= self.core.ppu.cycles_until_next_event()
                || budget.cpu_cycle_guard >= CPU_CYCLES_PER_FRAME_GUARD
            {
                let completed_frame = self
                    .core
                    .ppu
                    .tick_profiled(budget.pending_ppu_cycles, profiler);
                budget.pending_ppu_cycles = 0;
                if completed_frame || budget.cpu_cycle_guard >= CPU_CYCLES_PER_FRAME_GUARD {
                    break;
                }
            }
        }
        if budget.pending_ppu_cycles > 0 {
            self.core
                .ppu
                .tick_profiled(budget.pending_ppu_cycles, profiler);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PPU_SPRITE0_DOT: usize = (PPU_VISIBLE_START_SCANLINE + 30) * PPU_DOTS_PER_SCANLINE + 1;

    fn resize_default_area_reference(src: &[u8], dst: &mut [u8]) {
        for dst_y in 0..DEFAULT_GRAY_RESIZE_HEIGHT {
            let y0 = (dst_y * DEFAULT_GRAY_CROP_HEIGHT) / DEFAULT_GRAY_RESIZE_HEIGHT;
            let y1 = (((dst_y + 1) * DEFAULT_GRAY_CROP_HEIGHT) / DEFAULT_GRAY_RESIZE_HEIGHT)
                .max(y0 + 1)
                .min(DEFAULT_GRAY_CROP_HEIGHT);
            for dst_x in 0..DEFAULT_GRAY_RESIZE_WIDTH {
                let x0 = (dst_x * VISIBLE_FRAME_WIDTH) / DEFAULT_GRAY_RESIZE_WIDTH;
                let x1 = (((dst_x + 1) * VISIBLE_FRAME_WIDTH) / DEFAULT_GRAY_RESIZE_WIDTH)
                    .max(x0 + 1)
                    .min(VISIBLE_FRAME_WIDTH);
                let mut sum = 0u32;
                for sy in y0..y1 {
                    let row = sy * VISIBLE_FRAME_WIDTH;
                    for sx in x0..x1 {
                        sum += src[row + sx] as u32;
                    }
                }
                dst[dst_y * DEFAULT_GRAY_RESIZE_WIDTH + dst_x] =
                    (sum / ((x1 - x0) * (y1 - y0)) as u32) as u8;
            }
        }
    }

    fn set_sprite(ppu: &mut Ppu, sprite: usize, y: u8, tile: u8, attr: u8, x: u8) {
        let base = sprite * 4;
        ppu.oam[base] = y.wrapping_sub(1);
        ppu.oam[base + 1] = tile;
        ppu.oam[base + 2] = attr;
        ppu.oam[base + 3] = x;
    }

    fn make_test_ppu() -> Ppu {
        let chr_rom = (0..8192)
            .map(|idx| ((idx * 37 + idx / 11 + 23) & 0xff) as u8)
            .collect::<Vec<_>>();
        let mut ppu = Ppu::new(chr_rom, true, Some(PPU_SPRITE0_DOT));
        ppu.ctrl = 0x18;
        ppu.mask = 0x18;
        ppu.scroll_x_px = 37;
        ppu.scroll_x_low = 37;
        ppu.scroll_y_px = 11;
        for (idx, value) in ppu.vram.iter_mut().enumerate() {
            *value = ((idx * 13 + idx / 7 + 5) & 0xff) as u8;
        }
        for (idx, value) in ppu.palette.iter_mut().enumerate() {
            *value = ((idx * 3 + 7) & 0x3f) as u8;
        }
        ppu.refresh_gray_palette_cache();
        ppu.oam.fill(0xff);
        set_sprite(&mut ppu, 63, 70, 3, 0x00, 40);
        set_sprite(&mut ppu, 0, 72, 5, 0x01, 42);
        set_sprite(&mut ppu, 1, 74, 7, 0x22, 44);
        set_sprite(&mut ppu, 2, 190, 9, 0xc3, 250);
        ppu
    }
    #[test]
    fn gray_palette_cache_tracks_palette_writes() {
        let mut ppu = Ppu::new(vec![0; 8192], true, Some(PPU_SPRITE0_DOT));
        for (addr, value) in [
            (0x3f00, 0x01),
            (0x3f01, 0x11),
            (0x3f05, 0x21),
            (0x3f0f, 0x31),
            (0x3f11, 0x0f),
            (0x3f10, 0x30),
            (0x3f00, 0x40),
            (0x3f01, 0x7f),
            (0x3f05, 0xff),
        ] {
            ppu.ppu_write(addr, value);
            let mut expected_gray = [0; 32];
            for (dst, &color) in expected_gray.iter_mut().zip(ppu.palette.iter()) {
                *dst = NES_GRAY_PALETTE[(color & 0x3f) as usize];
            }
            assert_eq!(ppu.palette_gray, expected_gray);
            ppu.refresh_gray_bg_quad_tables();
            assert_eq!(
                *ppu.gray_bg_quad_cache.tables.read().unwrap(),
                build_gray_bg_quad_tables(&expected_gray)
            );
        }
    }

    #[test]
    fn gray_palette_cache_masks_high_bits_on_full_refresh() {
        let mut ppu = Ppu::new(vec![0; 8192], true, Some(PPU_SPRITE0_DOT));
        ppu.palette[..3].copy_from_slice(&[0x40, 0x7f, 0xff]);

        ppu.refresh_gray_palette_cache();

        assert_eq!(&ppu.palette[..3], &[0x40, 0x7f, 0xff]);
        assert_eq!(ppu.palette_gray[0], NES_GRAY_PALETTE[0x00]);
        assert_eq!(ppu.palette_gray[1], NES_GRAY_PALETTE[0x3f]);
        assert_eq!(ppu.palette_gray[2], NES_GRAY_PALETTE[0x3f]);
    }
    #[test]
    fn gray_background_frame_cache_tracks_all_render_inputs() {
        let mut ppu = make_test_ppu();
        let crop_top = VISIBLE_FRAME_TOP + DEFAULT_GRAY_CROP_TOP;
        let height = DEFAULT_GRAY_CROP_HEIGHT;
        let mut actual = vec![0; VISIBLE_FRAME_WIDTH * height];

        for change in 0..6 {
            if change == 1 {
                ppu.vram[37] ^= 0x5a;
            } else if change == 2 {
                ppu.palette[5] ^= 0x0f;
                ppu.refresh_gray_palette_cache();
            } else if change == 3 {
                ppu.scroll_x_px = ppu.scroll_x_px.wrapping_add(3);
            } else if change == 4 {
                ppu.ctrl ^= 0x10;
            } else if change == 5 {
                ppu.mask ^= 0x08;
            }

            actual.fill(0xa5);
            ppu.write_bg_gray_visible_mask_lower_tiled(&mut actual, crop_top, height);
            let uncached = ppu.clone();
            let mut expected = vec![0; actual.len()];
            uncached.write_bg_gray_visible_mask_lower_tiled(&mut expected, crop_top, height);
            assert_eq!(actual, expected);

            actual.fill(0x5a);
            ppu.write_bg_gray_visible_mask_lower_tiled(&mut actual, crop_top, height);
            assert_eq!(actual, expected);
        }
    }
    #[test]
    fn cached_next_ppu_event_tracks_frame_transitions() {
        let mut ppu = Ppu::new(vec![0; 8192], true, Some(PPU_SPRITE0_DOT));
        ppu.ctrl = 0x80;

        assert_eq!(ppu.cycles_until_next_event(), PPU_VBLANK_DOT);
        assert!(!ppu.tick(PPU_VBLANK_DOT));
        assert_eq!(ppu.frame_dot, PPU_VBLANK_DOT);
        assert_eq!(
            ppu.cycles_until_next_event(),
            PPU_PRERENDER_DOT - PPU_VBLANK_DOT
        );
        assert_ne!(ppu.status & 0x80, 0);
        assert!(ppu.take_nmi());

        assert!(!ppu.tick(PPU_PRERENDER_DOT - PPU_VBLANK_DOT));
        assert_eq!(ppu.frame_dot, PPU_PRERENDER_DOT);
        assert_eq!(
            ppu.cycles_until_next_event(),
            PPU_SPRITE0_DOT - PPU_PRERENDER_DOT
        );
        assert_eq!(ppu.status & 0xc0, 0);

        assert!(!ppu.tick(PPU_SPRITE0_DOT - PPU_PRERENDER_DOT));
        assert_eq!(ppu.frame_dot, PPU_SPRITE0_DOT);
        assert_eq!(
            ppu.cycles_until_next_event(),
            PPU_DOTS_PER_FRAME - PPU_SPRITE0_DOT
        );
        assert_ne!(ppu.status & 0x40, 0);

        assert!(ppu.tick(PPU_DOTS_PER_FRAME - PPU_SPRITE0_DOT));
        assert_eq!(ppu.frame_dot, 0);
        assert_eq!(ppu.cycles_until_next_event(), PPU_VBLANK_DOT);
    }

    #[test]
    fn no_sprite_timing_rebuilds_event_cache_across_reset_and_wrap() {
        let mut ppu = Ppu::new(vec![0; 8192], true, None);
        assert!(!ppu.tick(PPU_VBLANK_DOT));
        assert!(!ppu.tick(PPU_PRERENDER_DOT - PPU_VBLANK_DOT));
        assert_eq!(
            ppu.cycles_until_next_event(),
            PPU_DOTS_PER_FRAME - PPU_PRERENDER_DOT
        );
        assert!(ppu.tick(PPU_DOTS_PER_FRAME - PPU_PRERENDER_DOT));
        assert_eq!(ppu.cycles_until_next_event(), PPU_VBLANK_DOT);

        ppu.set_dot(PPU_PRERENDER_DOT);
        assert_eq!(
            ppu.cycles_until_next_event(),
            PPU_DOTS_PER_FRAME - PPU_PRERENDER_DOT
        );
        ppu.reset();
        assert_eq!(ppu.cycles_until_next_event(), PPU_VBLANK_DOT);
    }

    #[test]
    fn sprite_scanline_mask_limits_to_first_eight_oam_sprites() {
        let mut ppu = Ppu::new(vec![0; 8192], true, Some(PPU_SPRITE0_DOT));
        ppu.oam.fill(0xff);
        for sprite in 0..9usize {
            set_sprite(&mut ppu, sprite, 50, 1, 0x00, (sprite * 8) as u8);
        }

        let mask = ppu.sprite_scanline_mask();

        assert_eq!(mask[50].count_ones(), 8);
        for sprite in 0..8usize {
            assert_ne!(mask[50] & (1u64 << sprite), 0);
        }
        assert_eq!(mask[50] & (1u64 << 8), 0);
    }

    #[test]
    fn gray_region_matches_full_frame_crop_at_sprite_edges() {
        let mut ppu = make_test_ppu();
        set_sprite(&mut ppu, 3, 32, 11, 0x40, 0);
        set_sprite(&mut ppu, 4, 220, 13, 0x80, 252);
        let mut full = vec![0; NES_WIDTH * NES_HEIGHT];
        ppu.write_gray_frame_region(&mut full, 0, 0, NES_WIDTH, NES_HEIGHT);

        for (top, left, width, height) in [
            (31usize, 5usize, 240usize, 192usize),
            (32, VISIBLE_FRAME_LEFT, VISIBLE_FRAME_WIDTH, 192),
            (188, 245, 11, 36),
        ] {
            let mut region = vec![0; width * height];
            ppu.write_gray_frame_region(&mut region, top, left, width, height);
            for row in 0..height {
                assert_eq!(
                    &region[row * width..(row + 1) * width],
                    &full[(top + row) * NES_WIDTH + left..(top + row) * NES_WIDTH + left + width],
                );
            }
        }
    }

    #[test]
    fn behind_background_sprite_blocks_lower_priority_sprites() {
        let mut chr_rom = vec![0; 8192];
        for tile in [1usize, 2usize] {
            for row in 0..8usize {
                chr_rom[tile * 16 + row] = 0xff;
            }
        }
        let mut ppu = Ppu::new(chr_rom, true, Some(PPU_SPRITE0_DOT));
        ppu.mask = 0x18;
        ppu.vram[(40 / 8) * 32 + (40 / 8)] = 2;
        ppu.palette[1] = 0x0f;
        ppu.palette[0x11] = 0x30;
        ppu.refresh_gray_palette_cache();
        ppu.oam.fill(0xff);
        set_sprite(&mut ppu, 1, 40, 1, 0x00, 40);
        set_sprite(&mut ppu, 0, 40, 1, 0x20, 40);

        let mut dst =
            vec![NES_GRAY_PALETTE[ppu.bg_color_index(40, 40) as usize]; NES_WIDTH * NES_HEIGHT];
        ppu.draw_sprites_gray(&mut dst);

        assert_eq!(
            dst[40 * NES_WIDTH + 40],
            NES_GRAY_PALETTE[ppu.bg_color_index(40, 40) as usize],
        );
    }

    fn assert_default_area_writer_matches_scratch(ppu: &Ppu) {
        let mut native = vec![0; VISIBLE_FRAME_WIDTH * DEFAULT_GRAY_CROP_HEIGHT];
        let mut expected = vec![0; DEFAULT_GRAY_RESIZE_PIXELS];
        let mut actual = vec![0; DEFAULT_GRAY_RESIZE_PIXELS];
        let mut sprite_shadow = vec![0; VISIBLE_FRAME_WIDTH * DEFAULT_GRAY_CROP_HEIGHT];

        ppu.write_gray_frame_region(
            &mut native,
            VISIBLE_FRAME_TOP + DEFAULT_GRAY_CROP_TOP,
            VISIBLE_FRAME_LEFT,
            VISIBLE_FRAME_WIDTH,
            DEFAULT_GRAY_CROP_HEIGHT,
        );
        resize_default_area_reference(&native, &mut expected);
        ppu.write_gray_frame_cropped_area_84x84(&mut actual, &mut sprite_shadow);

        assert_eq!(actual, expected);
    }

    #[test]
    fn direct_nametable_read_matches_ppu_read_mirroring() {
        for vertical_mirroring in [true, false] {
            let chr_rom = vec![0; 8192];
            let mut ppu = Ppu::new(chr_rom, vertical_mirroring, Some(PPU_SPRITE0_DOT));
            for (idx, value) in ppu.vram.iter_mut().enumerate() {
                *value = ((idx * 29 + idx / 5 + 17) & 0xff) as u8;
            }

            for table in 0..4usize {
                for offset in 0..0x400usize {
                    let addr = 0x2000 + (table * 0x400 + offset) as u16;
                    assert_eq!(ppu.nametable_read(table, offset), ppu.ppu_read(addr));
                }
            }
        }
    }

    #[test]
    fn default_cropped_gray_area_writer_matches_scratch_resize() {
        let mut ppu = make_test_ppu();
        assert_default_area_writer_matches_scratch(&ppu);

        ppu.mask = 0x10;
        assert_default_area_writer_matches_scratch(&ppu);

        ppu.mask = 0x00;
        assert_default_area_writer_matches_scratch(&ppu);
    }

    #[derive(Clone)]
    struct NoSpriteTiming;

    impl PpuTiming for NoSpriteTiming {
        const SPRITE0_HIT_DOT: Option<usize> = None;
    }

    #[test]
    fn reset_uses_fceu_power_on_ram_pattern() {
        let mut prg_rom = vec![0xea; 32768];
        prg_rom[0x7ffc..0x7ffe].copy_from_slice(&0x8000u16.to_le_bytes());
        let cart = Cartridge {
            prg_rom,
            chr_rom: vec![0; 8192],
            vertical_mirroring: true,
        };
        let mut core = NromCore::<NoSpriteTiming>::new(cart);
        core.reset();
        assert!(core.ram.iter().all(|&value| value == 0xff));
    }
}
