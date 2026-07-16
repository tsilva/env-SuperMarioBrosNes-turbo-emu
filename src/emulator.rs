use crate::cartridge::Cartridge;
use crate::profiler::Profiler;
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

const CPU_CYCLES_PER_FRAME_GUARD: usize = 40_000;
const PPU_DOTS_PER_SCANLINE: usize = 341;
const PPU_SCANLINES_PER_FRAME: usize = 262;
const PPU_DOTS_PER_FRAME: usize = PPU_DOTS_PER_SCANLINE * PPU_SCANLINES_PER_FRAME;
const PPU_VBLANK_DOT: usize = PPU_DOTS_PER_SCANLINE;
const PPU_PRERENDER_DOT: usize = 21 * PPU_DOTS_PER_SCANLINE;
const PPU_VISIBLE_START_SCANLINE: usize = 22;
const PPU_SPRITE0_DOT: usize = (PPU_VISIBLE_START_SCANLINE + 30) * PPU_DOTS_PER_SCANLINE + 1;
const DEFAULT_GRAY_CROP_TOP: usize = 32;
const DEFAULT_GRAY_CROP_HEIGHT: usize = VISIBLE_FRAME_HEIGHT - DEFAULT_GRAY_CROP_TOP;
const DEFAULT_GRAY_RESIZE_WIDTH: usize = 84;
const DEFAULT_GRAY_RESIZE_HEIGHT: usize = 84;
const DEFAULT_GRAY_RESIZE_PIXELS: usize = DEFAULT_GRAY_RESIZE_WIDTH * DEFAULT_GRAY_RESIZE_HEIGHT;
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

const FLAG_C: u8 = 0x01;
const FLAG_Z: u8 = 0x02;
const FLAG_I: u8 = 0x04;
const FLAG_D: u8 = 0x08;
const FLAG_B: u8 = 0x10;
const FLAG_U: u8 = 0x20;
const FLAG_V: u8 = 0x40;
const FLAG_N: u8 = 0x80;

#[derive(Debug, Error)]
pub enum StateLoadError {
    #[error("state field {name} with size {size} was not found")]
    MissingField { name: &'static str, size: usize },
}

#[derive(Clone, Copy)]
struct Cpu {
    a: u8,
    x: u8,
    y: u8,
    sp: u8,
    pc: u16,
    p: u8,
}

impl Cpu {
    fn new() -> Self {
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
struct Ppu {
    chr_rom: Vec<u8>,
    chr_row_pixels: Vec<u16>,
    chr_addr_mask: usize,
    vertical_mirroring: bool,
    ctrl: u8,
    mask: u8,
    status: u8,
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
}

impl Ppu {
    fn new(chr_rom: Vec<u8>, vertical_mirroring: bool) -> Self {
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
            next_event_dot: next_ppu_event_dot(0),
            frame: 0,
            nmi_pending: false,
        }
    }

    fn reset(&mut self) {
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
        self.next_event_dot = next_ppu_event_dot(0);
        self.frame = 0;
        self.nmi_pending = false;
    }

    fn oam(&self) -> &[u8; 256] {
        &self.oam
    }

    fn debug_bg_pixel(&self, x: usize, y: usize) -> (u8, bool) {
        (self.bg_color_index(x, y), self.bg_pixel_opaque(x, y))
    }

    fn load_fceu_state(
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
        self.next_event_dot = next_ppu_event_dot(0);
        self.nmi_pending = false;
        self.update_scroll_x_px();
    }

    #[inline]
    fn tick(&mut self, ppu_cycles: usize) -> bool {
        let mut completed_frame = false;
        let mut remaining = ppu_cycles;
        while remaining > 0 {
            let current = self.frame_dot;
            let next = self.next_event_dot;
            let advance = remaining.min(next - current);
            let dot = current + advance;
            self.frame_dot = dot;
            remaining -= advance;

            match dot {
                PPU_SPRITE0_DOT => {
                    self.status |= 0x40;
                    self.next_event_dot = PPU_DOTS_PER_FRAME;
                }
                PPU_VBLANK_DOT => {
                    self.status |= 0x80;
                    self.next_event_dot = PPU_PRERENDER_DOT;
                    if self.ctrl & 0x80 != 0 {
                        self.nmi_pending = true;
                    }
                }
                PPU_PRERENDER_DOT => {
                    self.status &= !0xc0;
                    self.next_event_dot = PPU_SPRITE0_DOT;
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
        completed_frame
    }

    #[inline]
    fn tick_profiled(&mut self, ppu_cycles: usize, profiler: &mut Profiler) -> bool {
        let completed_frame = self.tick(ppu_cycles);
        profiler.record_ppu_tick(ppu_cycles, completed_frame);
        completed_frame
    }

    #[inline]
    fn cycles_until_next_event(&self) -> usize {
        self.next_event_dot - self.frame_dot
    }

    #[inline]
    fn sprite0_hit_set(&self) -> bool {
        self.status & 0x40 != 0
    }

    #[cfg(test)]
    #[inline]
    fn set_dot(&mut self, dot: usize) {
        self.frame_dot = dot;
        self.next_event_dot = next_ppu_event_dot(dot);
    }

    #[inline]
    fn take_nmi(&mut self) -> bool {
        let pending = self.nmi_pending;
        self.nmi_pending = false;
        pending
    }

    #[inline]
    fn cpu_read_register(&mut self, reg: u16) -> u8 {
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
    fn cpu_write_register(&mut self, reg: u16, value: u8) {
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
    fn read_data(&mut self) -> u8 {
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
    fn write_data(&mut self, value: u8) {
        let addr = self.addr & 0x3fff;
        self.ppu_write(addr, value);
        let inc = if self.ctrl & 0x04 != 0 { 32 } else { 1 };
        self.addr = self.addr.wrapping_add(inc) & 0x3fff;
    }

    #[inline]
    fn chr_read(&self, addr: usize) -> u8 {
        let idx = addr & self.chr_addr_mask;
        // SAFETY: SMB/NROM CHR ROM sizes are power-of-two and chr_addr_mask is len - 1.
        unsafe { *self.chr_rom.get_unchecked(idx) }
    }

    #[inline]
    fn chr_row_pixels(&self, addr: usize) -> u16 {
        let idx = addr & self.chr_addr_mask;
        // SAFETY: SMB/NROM CHR ROM sizes are power-of-two, the decoded row
        // table has the same length, and chr_addr_mask is len - 1.
        unsafe { *self.chr_row_pixels.get_unchecked(idx) }
    }

    #[inline]
    fn ppu_read(&self, addr: u16) -> u8 {
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
    fn nametable_read(&self, table: usize, offset: usize) -> u8 {
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
    fn ppu_write(&mut self, addr: u16, value: u8) {
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
    fn mirror_nametable_addr(&self, addr: u16) -> usize {
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
    fn mirror_palette_addr(&self, addr: u16) -> usize {
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
    fn write_gray_frame(&self, dst: &mut [u8]) {
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
    fn write_gray_frame_cropped(&self, dst: &mut [u8], crop_top: usize, height: usize) {
        debug_assert_eq!(dst.len(), NES_WIDTH * height);
        self.write_bg_gray_cropped_tiled(dst, crop_top, height);
        self.draw_sprites_gray_cropped(dst, crop_top, height);
    }

    fn write_gray_frame_region(
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

    fn write_gray_frame_cropped_area_84x84(&self, dst: &mut [u8], sprite_shadow: &mut [u8]) {
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

    fn write_gray_visible_mask_lower_frame(&self, dst: &mut [u8], crop_top: usize, height: usize) {
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
    fn write_bg_gray_cropped_tiled(&self, dst: &mut [u8], crop_top: usize, height: usize) {
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

    fn write_bg_gray_region_tiled(
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

    fn write_bg_gray_visible_mask_lower_tiled(
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

    fn write_bg_gray_visible_mask_lower_tiled_uncached(
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
    fn write_rgb_frame(&self, dst: &mut [u8]) {
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
    fn write_rgb_frame_cropped(&self, dst: &mut [u8], crop_top: usize, height: usize) {
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

    fn write_rgb_frame_region(
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
    fn bg_color_index(&self, x: usize, y: usize) -> u8 {
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
    fn bg_pixel_opaque(&self, x: usize, y: usize) -> bool {
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

    fn sprite_scanline_mask(&self) -> [u64; NES_HEIGHT] {
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

    fn draw_sprites_gray(&self, dst: &mut [u8]) {
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
    fn draw_sprites_gray_cropped(&self, dst: &mut [u8], crop_top: usize, height: usize) {
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

    fn draw_sprites_gray_region(
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
    fn draw_sprites_rgb(&self, dst: &mut [u8]) {
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
    fn draw_sprites_rgb_cropped(&self, dst: &mut [u8], crop_top: usize, height: usize) {
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

    fn draw_sprites_rgb_region(
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
    fn update_scroll_x_px(&mut self) {
        self.scroll_x_px = (((self.ctrl & 0x01) as u16) << 8) | self.scroll_x_low as u16;
    }

    #[inline]
    fn set_scroll_override_x(&mut self, scroll_x_px: Option<u16>) {
        self.scroll_override_x_px = scroll_x_px;
    }

    #[inline]
    fn render_scroll_x_px(&self) -> u16 {
        self.scroll_override_x_px.unwrap_or(self.scroll_x_px)
    }

    #[inline]
    fn palette_gray(&self) -> [u8; 32] {
        self.palette_gray
    }

    fn refresh_gray_palette_cache(&mut self) {
        for (dst, &color) in self.palette_gray.iter_mut().zip(self.palette.iter()) {
            *dst = NES_GRAY_PALETTE[color as usize];
        }
        *self.gray_bg_quad_cache.tables.get_mut().unwrap() =
            build_gray_bg_quad_tables(&self.palette_gray);
        self.gray_bg_quad_cache.dirty.store(0, Ordering::Relaxed);
    }

    fn refresh_gray_palette_entry(&mut self, idx: usize) {
        self.palette_gray[idx] = NES_GRAY_PALETTE[self.palette[idx] as usize];
        if idx == 0 {
            self.gray_bg_quad_cache.dirty.store(0x0f, Ordering::Relaxed);
        } else if idx < 16 && idx & 3 != 0 {
            self.gray_bg_quad_cache
                .dirty
                .fetch_or(1 << (idx / 4), Ordering::Relaxed);
        }
    }

    fn refresh_gray_bg_quad_tables(&self) {
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
    fn bg_world_pos(&self, x: usize, y: usize) -> (usize, usize) {
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
fn next_ppu_event_dot(current: usize) -> usize {
    if current < PPU_VBLANK_DOT {
        PPU_VBLANK_DOT
    } else if current < PPU_PRERENDER_DOT {
        PPU_PRERENDER_DOT
    } else if current < PPU_SPRITE0_DOT {
        PPU_SPRITE0_DOT
    } else {
        PPU_DOTS_PER_FRAME
    }
}

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

#[inline]
fn sign_extend_u8(value: u8) -> i16 {
    (value as i8) as i16
}

#[derive(Clone)]
pub struct NesEmulator {
    cpu: Cpu,
    ppu: Ppu,
    ram: [u8; 2048],
    prg_rom: Vec<u8>,
    prg_addr_mask: usize,
    smb_idle_jmp_supported: bool,
    smb_sprite0_poll_supported: bool,
    smb_sprite0_poll_exit_supported: bool,
    smb_timer_control_loop_supported: bool,
    smb_oam_clear_supported: bool,
    smb_scroll_slot_loop_supported: bool,
    smb_controller_read_supported: bool,
    smb_digit_math_loop_supported: bool,
    smb_bounding_box_nibble_supported: bool,
    smb_bounding_box_helper_supported: bool,
    smb_offscreen_bits_subs_supported: bool,
    smb_relative_position_helper_supported: bool,
    smb_draw_sprite_object_supported: bool,
    controller_state: u8,
    controller_shift: u8,
    controller_strobe: bool,
    extra_cycles: u16,
    x_pos: u16,
    coins: u8,
    level_hi: i16,
    level_lo: i16,
    lives: i16,
    score: u32,
    scrolling: i16,
    time: u16,
    xscroll_hi: u8,
    xscroll_lo: u8,
    terminate_on_flag: bool,
    done: bool,
}

impl NesEmulator {
    pub fn new_with_options(cart: Cartridge, terminate_on_flag: bool) -> Self {
        let prg_addr_mask = cart.prg_rom.len() - 1;
        let smb_idle_jmp_supported = prg_rom_supports_smb_idle_jmp(&cart.prg_rom, prg_addr_mask);
        let smb_sprite0_poll_supported =
            prg_rom_supports_smb_sprite0_poll(&cart.prg_rom, prg_addr_mask);
        let smb_sprite0_poll_exit_supported =
            prg_rom_supports_smb_sprite0_poll_exit(&cart.prg_rom, prg_addr_mask);
        let smb_timer_control_loop_supported =
            prg_rom_supports_smb_timer_control_loop(&cart.prg_rom, prg_addr_mask);
        let smb_oam_clear_supported = prg_rom_supports_smb_oam_clear(&cart.prg_rom, prg_addr_mask);
        let smb_scroll_slot_loop_supported =
            prg_rom_supports_smb_scroll_slot_loop(&cart.prg_rom, prg_addr_mask);
        let smb_controller_read_supported =
            prg_rom_supports_smb_controller_read(&cart.prg_rom, prg_addr_mask);
        let smb_digit_math_loop_supported =
            prg_rom_supports_smb_digit_math_loop(&cart.prg_rom, prg_addr_mask);
        let smb_bounding_box_nibble_supported =
            prg_rom_supports_smb_bounding_box_nibble(&cart.prg_rom, prg_addr_mask);
        let smb_bounding_box_helper_supported =
            prg_rom_supports_smb_bounding_box_helper(&cart.prg_rom, prg_addr_mask);
        let smb_offscreen_bits_subs_supported =
            prg_rom_supports_smb_offscreen_bits_subs(&cart.prg_rom, prg_addr_mask);
        let smb_relative_position_helper_supported =
            prg_rom_supports_smb_relative_position_helper(&cart.prg_rom, prg_addr_mask);
        let smb_draw_sprite_object_supported =
            prg_rom_supports_smb_draw_sprite_object(&cart.prg_rom, prg_addr_mask);
        let ppu = Ppu::new(cart.chr_rom, cart.vertical_mirroring);
        let mut emu = Self {
            cpu: Cpu::new(),
            ppu,
            ram: [0; 2048],
            prg_rom: cart.prg_rom,
            prg_addr_mask,
            smb_idle_jmp_supported,
            smb_sprite0_poll_supported,
            smb_sprite0_poll_exit_supported,
            smb_timer_control_loop_supported,
            smb_oam_clear_supported,
            smb_scroll_slot_loop_supported,
            smb_controller_read_supported,
            smb_digit_math_loop_supported,
            smb_bounding_box_nibble_supported,
            smb_bounding_box_helper_supported,
            smb_offscreen_bits_subs_supported,
            smb_relative_position_helper_supported,
            smb_draw_sprite_object_supported,
            controller_state: 0,
            controller_shift: 0,
            controller_strobe: false,
            extra_cycles: 0,
            x_pos: 0,
            coins: 0,
            level_hi: 0,
            level_lo: 0,
            lives: 0,
            score: 0,
            scrolling: 0,
            time: 0,
            xscroll_hi: 0,
            xscroll_lo: 0,
            terminate_on_flag,
            done: false,
        };
        emu.reset();
        emu
    }

    pub fn reset(&mut self) {
        self.cpu = Cpu::new();
        self.ppu.reset();
        self.ram = [0; 2048];
        self.controller_state = 0;
        self.controller_shift = 0;
        self.controller_strobe = false;
        self.extra_cycles = 0;
        self.done = false;
        self.cpu.pc = self.cpu_read_u16(0xfffc);
        self.refresh_smb_state();
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
        self.done = false;
        self.refresh_smb_state();
        // FCEU state files resume close to, but not exactly at, this simplified
        // PPU frame boundary. One no-op frame matches stable-retro's first
        // visible gameplay frame without changing the reset observation.
        self.run_frame(0);
        self.refresh_smb_state();
        Ok(())
    }

    #[inline]
    pub fn step_frame(&mut self, controller_state: u8) -> f32 {
        if self.done {
            return 0.0;
        }

        let before = self.xscroll_lo;
        self.run_frame(controller_state);
        self.refresh_smb_state();
        if self.native_scenario_done() {
            self.done = true;
        }
        if self.terminate_on_flag && self.x_pos >= 3160 {
            self.done = true;
        }
        (self.xscroll_lo as i16 - before as i16).max(0) as f32
    }

    #[inline]
    pub fn step_frame_profiled(&mut self, controller_state: u8, profiler: &mut Profiler) -> f32 {
        if self.done {
            return 0.0;
        }

        let before = self.xscroll_lo;
        let start = Instant::now();
        self.run_frame_profiled(controller_state, profiler);
        profiler.record_frame_step(start.elapsed());
        self.refresh_smb_state();
        if self.native_scenario_done() {
            self.done = true;
        }
        if self.terminate_on_flag && self.x_pos >= 3160 {
            self.done = true;
        }
        (self.xscroll_lo as i16 - before as i16).max(0) as f32
    }

    #[inline]
    #[allow(dead_code)]
    pub fn write_rgb_frame(&self, dst: &mut [u8]) {
        self.ppu.write_rgb_frame(dst);
    }

    #[inline]
    #[allow(dead_code)]
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
    #[allow(dead_code)]
    pub fn write_gray_frame(&self, dst: &mut [u8]) {
        self.ppu.write_gray_frame(dst);
    }

    #[inline]
    #[allow(dead_code)]
    pub fn write_gray_frame_cropped(&self, dst: &mut [u8], crop_top: usize, height: usize) {
        self.ppu.write_gray_frame_cropped(dst, crop_top, height);
    }

    #[inline]
    #[allow(dead_code)]
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
    pub fn x_pos(&self) -> u16 {
        self.x_pos
    }

    #[inline]
    pub fn coins(&self) -> u8 {
        self.coins
    }

    #[inline]
    pub fn level_hi(&self) -> i16 {
        self.level_hi
    }

    #[inline]
    pub fn level_lo(&self) -> i16 {
        self.level_lo
    }

    #[inline]
    pub fn lives(&self) -> i16 {
        self.lives
    }

    #[inline]
    pub fn score(&self) -> u32 {
        self.score
    }

    #[inline]
    pub fn scrolling(&self) -> i16 {
        self.scrolling
    }

    #[inline]
    pub fn time(&self) -> u16 {
        self.time
    }

    #[inline]
    pub fn xscroll_hi(&self) -> u8 {
        self.xscroll_hi
    }

    #[inline]
    pub fn xscroll_lo(&self) -> u8 {
        self.xscroll_lo
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

    #[inline]
    pub fn is_done(&self) -> bool {
        self.done
    }

    #[inline]
    fn native_scenario_done(&self) -> bool {
        self.lives == -1
    }

    fn run_frame(&mut self, buttons: u8) {
        self.controller_state = buttons;
        let mut cpu_cycle_guard = 0usize;
        let mut pending_ppu_cycles = 0usize;
        loop {
            if self.ppu.take_nmi() {
                self.interrupt(0xfffa, false);
            }
            match self.cpu.pc {
                SMB_IDLE_JMP_PC => {
                    if self.try_fast_forward_idle_jmp(&mut cpu_cycle_guard, &mut pending_ppu_cycles)
                    {
                        if self.ppu.tick(pending_ppu_cycles)
                            || cpu_cycle_guard >= CPU_CYCLES_PER_FRAME_GUARD
                        {
                            pending_ppu_cycles = 0;
                            break;
                        }
                        pending_ppu_cycles = 0;
                        continue;
                    }
                }
                SMB_SPRITE0_POLL_PC => {
                    if self.try_fast_forward_sprite0_poll(
                        &mut cpu_cycle_guard,
                        &mut pending_ppu_cycles,
                    ) {
                        if self.ppu.tick(pending_ppu_cycles)
                            || cpu_cycle_guard >= CPU_CYCLES_PER_FRAME_GUARD
                        {
                            pending_ppu_cycles = 0;
                            break;
                        }
                        pending_ppu_cycles = 0;
                        continue;
                    }
                }
                SMB_TIMER_CONTROL_LOOP_PC => {
                    if self.try_fast_forward_timer_control_loop(
                        &mut cpu_cycle_guard,
                        &mut pending_ppu_cycles,
                    ) {
                        continue;
                    }
                }
                SMB_OAM_CLEAR_PC => {
                    if self
                        .try_fast_forward_oam_clear(&mut cpu_cycle_guard, &mut pending_ppu_cycles)
                    {
                        if pending_ppu_cycles >= self.ppu.cycles_until_next_event()
                            || cpu_cycle_guard >= CPU_CYCLES_PER_FRAME_GUARD
                        {
                            if self.ppu.tick(pending_ppu_cycles)
                                || cpu_cycle_guard >= CPU_CYCLES_PER_FRAME_GUARD
                            {
                                pending_ppu_cycles = 0;
                                break;
                            }
                            pending_ppu_cycles = 0;
                        }
                        continue;
                    }
                }
                SMB_SCROLL_SLOT_LOOP_PC => {
                    if self.try_fast_forward_scroll_slot_loop(
                        &mut cpu_cycle_guard,
                        &mut pending_ppu_cycles,
                    ) {
                        continue;
                    }
                }
                SMB_CONTROLLER_READ_PC => {
                    if self.try_fast_forward_controller_read(
                        &mut cpu_cycle_guard,
                        &mut pending_ppu_cycles,
                    ) {
                        continue;
                    }
                }
                SMB_DIGIT_MATH_LOOP_PC => {
                    if self.try_fast_forward_digit_math_loop(
                        &mut cpu_cycle_guard,
                        &mut pending_ppu_cycles,
                    ) {
                        continue;
                    }
                }
                SMB_BOUNDING_BOX_HELPER_PC => {
                    if self.try_fast_forward_bounding_box_helper(
                        &mut cpu_cycle_guard,
                        &mut pending_ppu_cycles,
                    ) {
                        continue;
                    }
                }
                SMB_BOUNDING_BOX_NIBBLE_PC => {
                    if self.try_fast_forward_bounding_box_nibble(
                        &mut cpu_cycle_guard,
                        &mut pending_ppu_cycles,
                    ) {
                        continue;
                    }
                }
                SMB_OFFSCREEN_BITS_SUBS_PC => {
                    if self.try_fast_forward_offscreen_bits_subs(
                        &mut cpu_cycle_guard,
                        &mut pending_ppu_cycles,
                    ) {
                        continue;
                    }
                }
                SMB_RELATIVE_POSITION_HELPER_PC => {
                    if self.try_fast_forward_relative_position_helper(
                        &mut cpu_cycle_guard,
                        &mut pending_ppu_cycles,
                    ) {
                        continue;
                    }
                }
                SMB_DRAW_SPRITE_OBJECT_PC => {
                    if self.try_fast_forward_draw_sprite_object(
                        &mut cpu_cycle_guard,
                        &mut pending_ppu_cycles,
                    ) {
                        continue;
                    }
                }
                _ => {}
            }
            let cycles = self.cpu_step() as usize;
            cpu_cycle_guard += cycles;
            pending_ppu_cycles += cycles * 3;
            let must_flush_ppu = pending_ppu_cycles >= self.ppu.cycles_until_next_event()
                || cpu_cycle_guard >= CPU_CYCLES_PER_FRAME_GUARD;
            if must_flush_ppu {
                if self.ppu.tick(pending_ppu_cycles)
                    || cpu_cycle_guard >= CPU_CYCLES_PER_FRAME_GUARD
                {
                    pending_ppu_cycles = 0;
                    break;
                }
                pending_ppu_cycles = 0;
            }
        }
        if pending_ppu_cycles > 0 {
            self.ppu.tick(pending_ppu_cycles);
        }
    }

    fn run_frame_profiled(&mut self, buttons: u8, profiler: &mut Profiler) {
        self.controller_state = buttons;
        let mut cpu_cycle_guard = 0usize;
        let mut pending_ppu_cycles = 0usize;
        loop {
            if self.ppu.take_nmi() {
                self.interrupt(0xfffa, false);
            }
            match self.cpu.pc {
                SMB_IDLE_JMP_PC => {
                    if self.try_fast_forward_idle_jmp(&mut cpu_cycle_guard, &mut pending_ppu_cycles)
                    {
                        if self.ppu.tick_profiled(pending_ppu_cycles, profiler)
                            || cpu_cycle_guard >= CPU_CYCLES_PER_FRAME_GUARD
                        {
                            pending_ppu_cycles = 0;
                            break;
                        }
                        pending_ppu_cycles = 0;
                        continue;
                    }
                }
                SMB_SPRITE0_POLL_PC => {
                    if self.try_fast_forward_sprite0_poll(
                        &mut cpu_cycle_guard,
                        &mut pending_ppu_cycles,
                    ) {
                        if self.ppu.tick_profiled(pending_ppu_cycles, profiler)
                            || cpu_cycle_guard >= CPU_CYCLES_PER_FRAME_GUARD
                        {
                            pending_ppu_cycles = 0;
                            break;
                        }
                        pending_ppu_cycles = 0;
                        continue;
                    }
                }
                SMB_TIMER_CONTROL_LOOP_PC => {
                    if self.try_fast_forward_timer_control_loop(
                        &mut cpu_cycle_guard,
                        &mut pending_ppu_cycles,
                    ) {
                        continue;
                    }
                }
                SMB_OAM_CLEAR_PC => {
                    if self
                        .try_fast_forward_oam_clear(&mut cpu_cycle_guard, &mut pending_ppu_cycles)
                    {
                        if pending_ppu_cycles >= self.ppu.cycles_until_next_event()
                            || cpu_cycle_guard >= CPU_CYCLES_PER_FRAME_GUARD
                        {
                            if self.ppu.tick_profiled(pending_ppu_cycles, profiler)
                                || cpu_cycle_guard >= CPU_CYCLES_PER_FRAME_GUARD
                            {
                                pending_ppu_cycles = 0;
                                break;
                            }
                            pending_ppu_cycles = 0;
                        }
                        continue;
                    }
                }
                SMB_SCROLL_SLOT_LOOP_PC => {
                    if self.try_fast_forward_scroll_slot_loop(
                        &mut cpu_cycle_guard,
                        &mut pending_ppu_cycles,
                    ) {
                        continue;
                    }
                }
                SMB_CONTROLLER_READ_PC => {
                    if self.try_fast_forward_controller_read(
                        &mut cpu_cycle_guard,
                        &mut pending_ppu_cycles,
                    ) {
                        continue;
                    }
                }
                SMB_DIGIT_MATH_LOOP_PC => {
                    if self.try_fast_forward_digit_math_loop(
                        &mut cpu_cycle_guard,
                        &mut pending_ppu_cycles,
                    ) {
                        continue;
                    }
                }
                SMB_BOUNDING_BOX_HELPER_PC => {
                    if self.try_fast_forward_bounding_box_helper(
                        &mut cpu_cycle_guard,
                        &mut pending_ppu_cycles,
                    ) {
                        continue;
                    }
                }
                SMB_BOUNDING_BOX_NIBBLE_PC => {
                    if self.try_fast_forward_bounding_box_nibble(
                        &mut cpu_cycle_guard,
                        &mut pending_ppu_cycles,
                    ) {
                        continue;
                    }
                }
                SMB_OFFSCREEN_BITS_SUBS_PC => {
                    if self.try_fast_forward_offscreen_bits_subs(
                        &mut cpu_cycle_guard,
                        &mut pending_ppu_cycles,
                    ) {
                        continue;
                    }
                }
                SMB_RELATIVE_POSITION_HELPER_PC => {
                    if self.try_fast_forward_relative_position_helper(
                        &mut cpu_cycle_guard,
                        &mut pending_ppu_cycles,
                    ) {
                        continue;
                    }
                }
                SMB_DRAW_SPRITE_OBJECT_PC => {
                    if self.try_fast_forward_draw_sprite_object(
                        &mut cpu_cycle_guard,
                        &mut pending_ppu_cycles,
                    ) {
                        continue;
                    }
                }
                _ => {}
            }
            let cycles = self.cpu_step_profiled(profiler) as usize;
            cpu_cycle_guard += cycles;
            pending_ppu_cycles += cycles * 3;
            let must_flush_ppu = pending_ppu_cycles >= self.ppu.cycles_until_next_event()
                || cpu_cycle_guard >= CPU_CYCLES_PER_FRAME_GUARD;
            if must_flush_ppu {
                if self.ppu.tick_profiled(pending_ppu_cycles, profiler)
                    || cpu_cycle_guard >= CPU_CYCLES_PER_FRAME_GUARD
                {
                    pending_ppu_cycles = 0;
                    break;
                }
                pending_ppu_cycles = 0;
            }
        }
        if pending_ppu_cycles > 0 {
            self.ppu.tick_profiled(pending_ppu_cycles, profiler);
        }
    }

    #[inline(never)]
    fn try_fast_forward_idle_jmp(
        &mut self,
        cpu_cycle_guard: &mut usize,
        pending_ppu_cycles: &mut usize,
    ) -> bool {
        if self.cpu.pc != SMB_IDLE_JMP_PC || !self.smb_idle_jmp_supported {
            return false;
        }

        let ppu_cycles_until_event = self.ppu.cycles_until_next_event();
        let remaining = ppu_cycles_until_event.saturating_sub(*pending_ppu_cycles);
        let jumps = remaining.div_ceil(SMB_IDLE_JMP_PPU_CYCLES).max(1);
        *cpu_cycle_guard += jumps * 3;
        *pending_ppu_cycles += jumps * SMB_IDLE_JMP_PPU_CYCLES;
        *pending_ppu_cycles >= ppu_cycles_until_event
            || *cpu_cycle_guard >= CPU_CYCLES_PER_FRAME_GUARD
    }

    #[inline(never)]
    fn try_fast_forward_sprite0_poll(
        &mut self,
        cpu_cycle_guard: &mut usize,
        pending_ppu_cycles: &mut usize,
    ) -> bool {
        if self.cpu.pc != SMB_SPRITE0_POLL_PC || !self.smb_sprite0_poll_supported {
            return false;
        }

        if self.ppu.sprite0_hit_set() {
            return self.try_fast_forward_sprite0_poll_exit(cpu_cycle_guard, pending_ppu_cycles);
        }

        let ppu_cycles_until_event = self.ppu.cycles_until_next_event();
        let remaining = ppu_cycles_until_event.saturating_sub(*pending_ppu_cycles);
        let loops = remaining.div_ceil(SMB_SPRITE0_POLL_PPU_CYCLES).max(1);
        self.cpu.a = 0;
        self.set_zn(0);
        *cpu_cycle_guard += loops * 9;
        *pending_ppu_cycles += loops * SMB_SPRITE0_POLL_PPU_CYCLES;
        *pending_ppu_cycles >= ppu_cycles_until_event
            || *cpu_cycle_guard >= CPU_CYCLES_PER_FRAME_GUARD
    }

    #[inline(never)]
    fn try_fast_forward_sprite0_poll_exit(
        &mut self,
        cpu_cycle_guard: &mut usize,
        pending_ppu_cycles: &mut usize,
    ) -> bool {
        if !self.smb_sprite0_poll_exit_supported {
            return false;
        }

        let ppu_cycles_until_event = self.ppu.cycles_until_next_event();
        let remaining = ppu_cycles_until_event.saturating_sub(*pending_ppu_cycles);
        if SMB_SPRITE0_POLL_EXIT_PPU_CYCLES > remaining {
            return false;
        }

        let status = self.ppu.cpu_read_register(0x2002);
        self.cpu.a = status & 0x40;
        self.set_zn(self.cpu.a);
        self.cpu.y = 0;
        self.set_zn(0);
        self.cpu.pc = SMB_SPRITE0_POLL_PC + 12;
        *cpu_cycle_guard += SMB_SPRITE0_POLL_EXIT_CPU_CYCLES;
        *pending_ppu_cycles += SMB_SPRITE0_POLL_EXIT_PPU_CYCLES;
        true
    }

    #[inline(never)]
    fn try_fast_forward_timer_control_loop(
        &mut self,
        cpu_cycle_guard: &mut usize,
        pending_ppu_cycles: &mut usize,
    ) -> bool {
        if self.cpu.pc != SMB_TIMER_CONTROL_LOOP_PC
            || !self.smb_timer_control_loop_supported
            || self.cpu.x & 0x80 != 0
        {
            return false;
        }

        let (cycles, last_a) = self.timer_control_loop_cycles_and_last_a(self.cpu.x);
        let ppu_cycles = cycles * 3;
        let remaining = self
            .ppu
            .cycles_until_next_event()
            .saturating_sub(*pending_ppu_cycles);
        if ppu_cycles >= remaining {
            return false;
        }

        let mut x = self.cpu.x;
        loop {
            let timer = &mut self.ram[0x0780 + x as usize];
            if *timer != 0 {
                *timer = timer.wrapping_sub(1);
            }
            x = x.wrapping_sub(1);
            if x & 0x80 != 0 {
                break;
            }
        }
        self.cpu.a = last_a;
        self.cpu.x = x;
        self.set_zn(x);
        self.cpu.pc = SMB_TIMER_CONTROL_LOOP_PC + 11;
        *cpu_cycle_guard += cycles;
        *pending_ppu_cycles += ppu_cycles;
        true
    }

    fn timer_control_loop_cycles_and_last_a(&self, start_x: u8) -> (usize, u8) {
        let mut cycles = 0usize;
        let mut x = start_x;
        let last_a = loop {
            let value = self.ram[0x0780 + x as usize];
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
        &mut self,
        cpu_cycle_guard: &mut usize,
        pending_ppu_cycles: &mut usize,
    ) -> bool {
        if self.cpu.pc != SMB_OAM_CLEAR_PC || !self.smb_oam_clear_supported {
            return false;
        }

        let ppu_cycles_until_event = self.ppu.cycles_until_next_event();
        let remaining = ppu_cycles_until_event.saturating_sub(*pending_ppu_cycles);
        if SMB_OAM_CLEAR_PPU_CYCLES > remaining {
            return false;
        }

        for offset in (4usize..=252).step_by(4) {
            self.ram[0x0200 + offset] = 0xf8;
        }
        self.cpu.a = 0xf8;
        self.cpu.y = 0;
        self.set_zn(0);
        self.cpu.pc = self.pop_u16().wrapping_add(1);
        *cpu_cycle_guard += SMB_OAM_CLEAR_CPU_CYCLES;
        *pending_ppu_cycles += SMB_OAM_CLEAR_PPU_CYCLES;
        true
    }

    #[inline(never)]
    fn try_fast_forward_scroll_slot_loop(
        &mut self,
        cpu_cycle_guard: &mut usize,
        pending_ppu_cycles: &mut usize,
    ) -> bool {
        if self.cpu.pc != SMB_SCROLL_SLOT_LOOP_PC || !self.smb_scroll_slot_loop_supported {
            return false;
        }

        let iterations = if self.cpu.x < 0x80 {
            self.cpu.x as usize + 1
        } else {
            1
        };
        let max_ppu_cycles = iterations * 40 * 3;
        let remaining = self
            .ppu
            .cycles_until_next_event()
            .saturating_sub(*pending_ppu_cycles);
        if max_ppu_cycles > remaining {
            return false;
        }

        let mut cycles = 0usize;
        loop {
            let x = self.cpu.x;
            let slot_addr = 0x06e4usize + x as usize;
            self.cpu.a = self.ram_read(slot_addr);
            cycles += 4;

            let threshold = self.ram_read(0);
            self.cmp(self.cpu.a, threshold);
            cycles += 3;
            if !self.flag(FLAG_C) {
                cycles += 3;
            } else {
                cycles += 2;
                self.cpu.y = self.ram_read(0x06e0);
                self.set_zn(self.cpu.y);
                cycles += 4;
                self.set_flag(FLAG_C, false);
                cycles += 2;
                let add_addr = 0x06e1u16.wrapping_add(self.cpu.y as u16);
                let addend = self.cpu_read(add_addr);
                self.adc(addend);
                cycles += 4 + page_crossed(0x06e1, add_addr) as usize;
                if !self.flag(FLAG_C) {
                    cycles += 3;
                } else {
                    cycles += 2;
                    self.set_flag(FLAG_C, false);
                    cycles += 2;
                    self.adc(threshold);
                    cycles += 3;
                }
                self.ram_write(slot_addr, self.cpu.a);
                cycles += 5;
            }

            self.cpu.x = x.wrapping_sub(1);
            self.set_zn(self.cpu.x);
            cycles += 2;
            if self.flag(FLAG_N) {
                cycles += 2;
                break;
            }
            cycles += 3;
        }

        self.cpu.pc = 0x81e8;
        *cpu_cycle_guard += cycles;
        *pending_ppu_cycles += cycles * 3;
        true
    }

    #[inline(never)]
    fn try_fast_forward_controller_read(
        &mut self,
        cpu_cycle_guard: &mut usize,
        pending_ppu_cycles: &mut usize,
    ) -> bool {
        if self.cpu.pc != SMB_CONTROLLER_READ_PC
            || !self.smb_controller_read_supported
            || self.cpu.x > 1
        {
            return false;
        }

        let x = self.cpu.x as usize;
        let mut a = self.cpu.a;
        let mut last_read = 0u8;
        let mut controller_shift = self.controller_shift;
        let mut carry = self.flag(FLAG_C);
        for _ in 0..8 {
            let value = if x == 0 {
                let bit = if self.controller_strobe {
                    self.controller_state & 1
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

        let prior_buttons = self.ram_read(0x074a + x);
        let duplicate_start_select = (a & 0x30) & prior_buttons;
        let branch_taken = duplicate_start_select == 0;
        let routine_cycles = if branch_taken {
            SMB_CONTROLLER_READ_TAKEN_CPU_CYCLES
        } else {
            SMB_CONTROLLER_READ_NOT_TAKEN_CPU_CYCLES
        };
        let extra_cycles = self.extra_cycles as usize;
        let total_cycles = routine_cycles + extra_cycles;
        let ppu_cycles = total_cycles * 3;
        let ppu_cycles_until_event = self.ppu.cycles_until_next_event();
        let remaining = ppu_cycles_until_event.saturating_sub(*pending_ppu_cycles);
        if ppu_cycles > remaining {
            return false;
        }

        self.controller_shift = controller_shift;
        self.extra_cycles = 0;
        self.ram_write(0x0000, last_read);
        self.ram_write(0x06fc + x, a);
        self.ram_write(0x0100 | self.cpu.sp as usize, a);
        if carry {
            self.cpu.p |= FLAG_C;
        } else {
            self.cpu.p &= !FLAG_C;
        }
        self.cpu.p |= FLAG_U;
        if branch_taken {
            self.cpu.a = a;
            self.set_zn(a);
            self.ram_write(0x074a + x, a);
        } else {
            self.cpu.a = a & 0xcf;
            self.set_zn(self.cpu.a);
            self.ram_write(0x06fc + x, self.cpu.a);
        }
        self.cpu.y = 0;
        self.cpu.pc = self.pop_u16().wrapping_add(1);
        *cpu_cycle_guard += total_cycles;
        *pending_ppu_cycles += ppu_cycles;
        true
    }

    fn digit_math_loop_cycles(&self) -> usize {
        let mut cycles = self.extra_cycles as usize;
        let mut x = self.cpu.x;
        let mut y = self.cpu.y;
        let mut carry = self.flag(FLAG_C);

        loop {
            let source_base = 0x07ddu16;
            let source_addr = source_base.wrapping_add(x as u16);
            let a = self.ram_read(source_addr as usize);
            cycles += 4 + page_crossed(source_base, source_addr) as usize;

            let sub_base = 0x07d7u16;
            let sub_addr = sub_base.wrapping_add(y as u16);
            let value = self.ram_read(sub_addr as usize);
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
        &mut self,
        cpu_cycle_guard: &mut usize,
        pending_ppu_cycles: &mut usize,
    ) -> bool {
        if self.cpu.pc != SMB_DIGIT_MATH_LOOP_PC || !self.smb_digit_math_loop_supported {
            return false;
        }

        let total_cycles = self.digit_math_loop_cycles();
        let ppu_cycles = total_cycles * 3;
        let remaining = self
            .ppu
            .cycles_until_next_event()
            .saturating_sub(*pending_ppu_cycles);
        if ppu_cycles >= remaining || *cpu_cycle_guard + total_cycles >= CPU_CYCLES_PER_FRAME_GUARD
        {
            return false;
        }

        let mut cycles = self.extra_cycles as usize;
        loop {
            let source_base = 0x07ddu16;
            let source_addr = source_base.wrapping_add(self.cpu.x as u16);
            self.cpu.a = self.ram_read(source_addr as usize);
            self.set_zn(self.cpu.a);
            cycles += 4 + page_crossed(source_base, source_addr) as usize;

            let sub_base = 0x07d7u16;
            let sub_addr = sub_base.wrapping_add(self.cpu.y as u16);
            let value = self.ram_read(sub_addr as usize);
            self.sbc(value);
            cycles += 4 + page_crossed(sub_base, sub_addr) as usize;

            self.cpu.x = self.cpu.x.wrapping_sub(1);
            self.set_zn(self.cpu.x);
            self.cpu.y = self.cpu.y.wrapping_sub(1);
            self.set_zn(self.cpu.y);
            cycles += 4;
            if self.flag(FLAG_N) {
                cycles += 2;
                break;
            }
            cycles += 3;
        }

        if !self.flag(FLAG_C) {
            cycles += 3;
        } else {
            cycles += 2;
            self.cpu.x = self.cpu.x.wrapping_add(1);
            self.set_zn(self.cpu.x);
            self.cpu.y = self.cpu.y.wrapping_add(1);
            self.set_zn(self.cpu.y);
            cycles += 4;

            loop {
                let source_base = 0x07ddu16;
                let source_addr = source_base.wrapping_add(self.cpu.x as u16);
                self.cpu.a = self.ram_read(source_addr as usize);
                self.set_zn(self.cpu.a);
                cycles += 4 + page_crossed(source_base, source_addr) as usize;

                let destination = 0x07d7u16.wrapping_add(self.cpu.y as u16);
                self.ram_write(destination as usize, self.cpu.a);
                cycles += 5;

                self.cpu.x = self.cpu.x.wrapping_add(1);
                self.set_zn(self.cpu.x);
                self.cpu.y = self.cpu.y.wrapping_add(1);
                self.set_zn(self.cpu.y);
                self.cmp(self.cpu.y, 6);
                cycles += 6;
                if self.flag(FLAG_C) {
                    cycles += 2;
                    break;
                }
                cycles += 3;
            }
        }

        self.cpu.pc = self.pop_u16().wrapping_add(1);
        cycles += 6;
        self.extra_cycles = 0;
        debug_assert_eq!(cycles, total_cycles);
        *cpu_cycle_guard += total_cycles;
        *pending_ppu_cycles += ppu_cycles;
        true
    }

    #[inline(never)]
    fn try_fast_forward_bounding_box_nibble(
        &mut self,
        cpu_cycle_guard: &mut usize,
        pending_ppu_cycles: &mut usize,
    ) -> bool {
        if self.cpu.pc != SMB_BOUNDING_BOX_NIBBLE_PC || !self.smb_bounding_box_nibble_supported {
            return false;
        }

        let extra_cycles = self.extra_cycles as usize;
        let total_cycles = SMB_BOUNDING_BOX_NIBBLE_CPU_CYCLES + extra_cycles;
        let ppu_cycles = total_cycles * 3;
        let remaining = self
            .ppu
            .cycles_until_next_event()
            .saturating_sub(*pending_ppu_cycles);
        if ppu_cycles > remaining {
            return false;
        }

        self.push(self.cpu.a);
        self.cpu.a = self.lsr(self.cpu.a);
        self.cpu.a = self.lsr(self.cpu.a);
        self.cpu.a = self.lsr(self.cpu.a);
        self.cpu.a = self.lsr(self.cpu.a);
        self.cpu.y = self.cpu.a;
        self.set_zn(self.cpu.y);
        let high_addr = 0x9bdfu16.wrapping_add(self.cpu.y as u16);
        self.cpu.a = self.prg_read(high_addr);
        self.set_zn(self.cpu.a);
        self.ram_write(0x0007, self.cpu.a);
        self.cpu.a = self.pop();
        self.set_zn(self.cpu.a);
        self.cpu.a &= 0x0f;
        self.set_zn(self.cpu.a);
        self.set_flag(FLAG_C, false);
        let low_addr = 0x9bddu16.wrapping_add(self.cpu.y as u16);
        let addend = self.prg_read(low_addr);
        self.adc(addend);
        self.ram_write(0x0006, self.cpu.a);
        self.cpu.pc = self.pop_u16().wrapping_add(1);
        self.extra_cycles = 0;
        *cpu_cycle_guard += total_cycles;
        *pending_ppu_cycles += ppu_cycles;
        true
    }

    #[inline(never)]
    fn try_fast_forward_bounding_box_helper(
        &mut self,
        cpu_cycle_guard: &mut usize,
        pending_ppu_cycles: &mut usize,
    ) -> bool {
        if self.cpu.pc != SMB_BOUNDING_BOX_HELPER_PC || !self.smb_bounding_box_helper_supported {
            return false;
        }

        let extra_cycles = self.extra_cycles as usize;
        let max_ppu_cycles = (SMB_BOUNDING_BOX_HELPER_MAX_CPU_CYCLES + extra_cycles) * 3;
        let remaining = self
            .ppu
            .cycles_until_next_event()
            .saturating_sub(*pending_ppu_cycles);
        if max_ppu_cycles > remaining {
            return false;
        }

        let mut cycles = extra_cycles;
        let x = self.cpu.x;
        self.push(self.cpu.a);
        cycles += 3;
        self.ram_write(0x0004, self.cpu.y);
        cycles += 3;

        let table1_base = 0xe3b0u16;
        let table1_addr = table1_base.wrapping_add(self.cpu.y as u16);
        self.cpu.a = self.prg_read(table1_addr);
        self.set_zn(self.cpu.a);
        cycles += 4 + page_crossed(table1_base, table1_addr) as usize;
        self.set_flag(FLAG_C, false);
        cycles += 2;
        let x86 = 0x86u8.wrapping_add(x) as usize;
        let addend = self.ram_read(x86);
        self.adc(addend);
        cycles += 4;
        self.ram_write(0x0005, self.cpu.a);
        cycles += 3;
        let x6d = 0x6du8.wrapping_add(x) as usize;
        self.cpu.a = self.ram_read(x6d);
        self.set_zn(self.cpu.a);
        cycles += 4;
        self.adc(0);
        cycles += 2;
        self.cpu.a &= 0x01;
        self.set_zn(self.cpu.a);
        cycles += 2;
        self.cpu.a = self.lsr(self.cpu.a);
        cycles += 2;
        self.ora(self.ram_read(0x0005));
        cycles += 3;
        self.cpu.a = self.ror(self.cpu.a);
        cycles += 2;
        self.cpu.a = self.lsr(self.cpu.a);
        self.cpu.a = self.lsr(self.cpu.a);
        self.cpu.a = self.lsr(self.cpu.a);
        cycles += 6;

        self.push_u16(0xe40a);
        cycles += 6;
        self.push(self.cpu.a);
        self.cpu.a = self.lsr(self.cpu.a);
        self.cpu.a = self.lsr(self.cpu.a);
        self.cpu.a = self.lsr(self.cpu.a);
        self.cpu.a = self.lsr(self.cpu.a);
        self.cpu.y = self.cpu.a;
        self.set_zn(self.cpu.y);
        let high_addr = 0x9bdfu16.wrapping_add(self.cpu.y as u16);
        self.cpu.a = self.prg_read(high_addr);
        self.set_zn(self.cpu.a);
        self.ram_write(0x0007, self.cpu.a);
        self.cpu.a = self.pop();
        self.set_zn(self.cpu.a);
        self.cpu.a &= 0x0f;
        self.set_zn(self.cpu.a);
        self.set_flag(FLAG_C, false);
        let low_addr = 0x9bddu16.wrapping_add(self.cpu.y as u16);
        let addend = self.prg_read(low_addr);
        self.adc(addend);
        self.ram_write(0x0006, self.cpu.a);
        self.cpu.pc = self.pop_u16().wrapping_add(1);
        debug_assert_eq!(self.cpu.pc, 0xe40b);
        cycles += SMB_BOUNDING_BOX_NIBBLE_CPU_CYCLES;

        self.cpu.y = self.ram_read(0x0004);
        self.set_zn(self.cpu.y);
        cycles += 3;
        let xce = 0xceu8.wrapping_add(x) as usize;
        self.cpu.a = self.ram_read(xce);
        self.set_zn(self.cpu.a);
        cycles += 4;
        self.set_flag(FLAG_C, false);
        cycles += 2;
        let table2_base = 0xe3ccu16;
        let table2_addr = table2_base.wrapping_add(self.cpu.y as u16);
        let addend = self.prg_read(table2_addr);
        self.adc(addend);
        cycles += 4 + page_crossed(table2_base, table2_addr) as usize;
        self.cpu.a &= 0xf0;
        self.set_zn(self.cpu.a);
        cycles += 2;
        self.set_flag(FLAG_C, true);
        cycles += 2;
        self.sbc(0x20);
        cycles += 2;
        self.ram_write(0x0002, self.cpu.a);
        cycles += 3;
        self.cpu.y = self.cpu.a;
        self.set_zn(self.cpu.y);
        cycles += 2;
        let ptr = self.ram_read(0x0006) as u16 | ((self.ram_read(0x0007) as u16) << 8);
        let indirect_addr = ptr.wrapping_add(self.cpu.y as u16);
        self.cpu.a = self.cpu_read(indirect_addr);
        self.set_zn(self.cpu.a);
        cycles += 5 + page_crossed(ptr, indirect_addr) as usize;
        self.ram_write(0x0003, self.cpu.a);
        cycles += 3;
        self.cpu.y = self.ram_read(0x0004);
        self.set_zn(self.cpu.y);
        cycles += 3;
        self.cpu.a = self.pop();
        self.set_zn(self.cpu.a);
        cycles += 4;
        if self.cpu.a != 0 {
            cycles += 3;
            self.cpu.a = self.ram_read(x86);
            self.set_zn(self.cpu.a);
            cycles += 4;
        } else {
            cycles += 2;
            self.cpu.a = self.ram_read(xce);
            self.set_zn(self.cpu.a);
            cycles += 4;
            self.cpu.pc = 0xe42b;
            cycles += 3;
        }
        self.cpu.a &= 0x0f;
        self.set_zn(self.cpu.a);
        cycles += 2;
        self.ram_write(0x0004, self.cpu.a);
        cycles += 3;
        self.cpu.a = self.ram_read(0x0003);
        self.set_zn(self.cpu.a);
        cycles += 3;
        self.cpu.pc = self.pop_u16().wrapping_add(1);
        cycles += 6;
        self.extra_cycles = 0;
        *cpu_cycle_guard += cycles;
        *pending_ppu_cycles += cycles * 3;
        true
    }

    #[inline]
    fn fast_forward_divide_pdiff_body(&mut self) -> usize {
        let mut cycles = 0;
        self.ram_write(0x0005, self.cpu.a);
        cycles += 3;
        self.cpu.a = self.ram_read(0x0007);
        self.set_zn(self.cpu.a);
        cycles += 3;
        self.cmp(self.cpu.a, self.ram_read(0x0006));
        cycles += 3;
        if self.flag(FLAG_C) {
            cycles += 3;
            self.pop_u16();
            return cycles + 6;
        }
        cycles += 2;
        self.cpu.a = self.lsr(self.cpu.a);
        self.cpu.a = self.lsr(self.cpu.a);
        self.cpu.a = self.lsr(self.cpu.a);
        cycles += 6;
        self.cpu.a &= 0x07;
        self.set_zn(self.cpu.a);
        cycles += 2;
        self.cmp(self.cpu.y, 1);
        cycles += 2;
        if self.flag(FLAG_C) {
            cycles += 3;
        } else {
            cycles += 2;
            self.adc(self.ram_read(0x0005));
            cycles += 3;
        }
        self.cpu.x = self.cpu.a;
        self.set_zn(self.cpu.x);
        cycles += 2;
        self.pop_u16();
        cycles + 6
    }

    #[inline]
    fn fast_forward_x_offscreen_bits_body(&mut self) -> usize {
        let mut cycles = 0;
        self.ram_write(0x0004, self.cpu.x);
        cycles += 3;
        self.cpu.y = 1;
        self.set_zn(self.cpu.y);
        cycles += 2;
        loop {
            let y = self.cpu.y as usize;
            self.cpu.a = self.ram_read(0x071c + y);
            self.set_zn(self.cpu.a);
            cycles += 4;
            self.set_flag(FLAG_C, true);
            cycles += 2;
            let object_x = self.ram_read(0x86u8.wrapping_add(self.cpu.x) as usize);
            self.sbc(object_x);
            cycles += 4;
            self.ram_write(0x0007, self.cpu.a);
            cycles += 3;
            self.cpu.a = self.ram_read(0x071a + y);
            self.set_zn(self.cpu.a);
            cycles += 4;
            let object_page = self.ram_read(0x6du8.wrapping_add(self.cpu.x) as usize);
            self.sbc(object_page);
            cycles += 4;
            self.cpu.x = self.prg_read(0xf1f3u16.wrapping_add(self.cpu.y as u16));
            self.set_zn(self.cpu.x);
            cycles += 4;
            self.cmp(self.cpu.a, 0);
            cycles += 2;
            if self.flag(FLAG_N) {
                cycles += 3;
            } else {
                cycles += 2;
                self.cpu.x = self.prg_read(0xf1f4u16.wrapping_add(self.cpu.y as u16));
                self.set_zn(self.cpu.x);
                cycles += 4;
                self.cmp(self.cpu.a, 1);
                cycles += 2;
                if !self.flag(FLAG_N) {
                    cycles += 3;
                } else {
                    cycles += 2;
                    self.cpu.a = 0x38;
                    self.set_zn(self.cpu.a);
                    cycles += 2;
                    self.ram_write(0x0006, self.cpu.a);
                    cycles += 3;
                    self.cpu.a = 0x08;
                    self.set_zn(self.cpu.a);
                    cycles += 2;
                    self.push_u16(0xf21d);
                    cycles += 6;
                    cycles += self.fast_forward_divide_pdiff_body();
                }
            }
            self.cpu.a = self.prg_read(0xf1e3u16.wrapping_add(self.cpu.x as u16));
            self.set_zn(self.cpu.a);
            cycles += 4;
            self.cpu.x = self.ram_read(0x0004);
            self.set_zn(self.cpu.x);
            cycles += 3;
            self.cmp(self.cpu.a, 0);
            cycles += 2;
            if !self.flag(FLAG_Z) {
                cycles += 3;
                self.pop_u16();
                return cycles + 6;
            }
            cycles += 2;
            self.cpu.y = self.cpu.y.wrapping_sub(1);
            self.set_zn(self.cpu.y);
            cycles += 2;
            if !self.flag(FLAG_N) {
                cycles += 4;
            } else {
                cycles += 2;
                self.pop_u16();
                return cycles + 6;
            }
        }
    }

    #[inline]
    fn fast_forward_y_offscreen_bits_body(&mut self) -> (usize, u16) {
        let mut cycles = 0;
        self.ram_write(0x0004, self.cpu.x);
        cycles += 3;
        self.cpu.y = 1;
        self.set_zn(self.cpu.y);
        cycles += 2;
        loop {
            self.cpu.a = self.prg_read(0xf237u16.wrapping_add(self.cpu.y as u16));
            self.set_zn(self.cpu.a);
            cycles += 4;
            self.set_flag(FLAG_C, true);
            cycles += 2;
            let object_y = self.ram_read(0xceu8.wrapping_add(self.cpu.x) as usize);
            self.sbc(object_y);
            cycles += 4;
            self.ram_write(0x0007, self.cpu.a);
            cycles += 3;
            self.cpu.a = 1;
            self.set_zn(self.cpu.a);
            cycles += 2;
            let object_high = self.ram_read(0xb5u8.wrapping_add(self.cpu.x) as usize);
            self.sbc(object_high);
            cycles += 4;
            self.cpu.x = self.prg_read(0xf234u16.wrapping_add(self.cpu.y as u16));
            self.set_zn(self.cpu.x);
            cycles += 4;
            self.cmp(self.cpu.a, 0);
            cycles += 2;
            if self.flag(FLAG_N) {
                cycles += 3;
            } else {
                cycles += 2;
                self.cpu.x = self.prg_read(0xf235u16.wrapping_add(self.cpu.y as u16));
                self.set_zn(self.cpu.x);
                cycles += 4;
                self.cmp(self.cpu.a, 1);
                cycles += 2;
                if !self.flag(FLAG_N) {
                    cycles += 3;
                } else {
                    cycles += 2;
                    self.cpu.a = 0x20;
                    self.set_zn(self.cpu.a);
                    cycles += 2;
                    self.ram_write(0x0006, self.cpu.a);
                    cycles += 3;
                    self.cpu.a = 0x04;
                    self.set_zn(self.cpu.a);
                    cycles += 2;
                    self.push_u16(0xf25f);
                    cycles += 6;
                    cycles += self.fast_forward_divide_pdiff_body();
                }
            }
            self.cpu.a = self.prg_read(0xf22bu16.wrapping_add(self.cpu.x as u16));
            self.set_zn(self.cpu.a);
            cycles += 4;
            self.cpu.x = self.ram_read(0x0004);
            self.set_zn(self.cpu.x);
            cycles += 3;
            self.cmp(self.cpu.a, 0);
            cycles += 2;
            if !self.flag(FLAG_Z) {
                cycles += 3;
                let return_address = self.pop_u16();
                return (cycles + 6, return_address);
            }
            cycles += 2;
            self.cpu.y = self.cpu.y.wrapping_sub(1);
            self.set_zn(self.cpu.y);
            cycles += 2;
            if !self.flag(FLAG_N) {
                cycles += 3;
            } else {
                cycles += 2;
                let return_address = self.pop_u16();
                return (cycles + 6, return_address);
            }
        }
    }

    #[inline(never)]
    fn try_fast_forward_offscreen_bits_subs(
        &mut self,
        cpu_cycle_guard: &mut usize,
        pending_ppu_cycles: &mut usize,
    ) -> bool {
        if self.cpu.pc != SMB_OFFSCREEN_BITS_SUBS_PC || !self.smb_offscreen_bits_subs_supported {
            return false;
        }
        let max_cycles = SMB_OFFSCREEN_BITS_SUBS_MAX_CPU_CYCLES + self.extra_cycles as usize;
        let remaining = self
            .ppu
            .cycles_until_next_event()
            .saturating_sub(*pending_ppu_cycles);
        if max_cycles * 3 >= remaining
            || *cpu_cycle_guard + max_cycles >= CPU_CYCLES_PER_FRAME_GUARD
        {
            return false;
        }

        let mut total_cycles = self.extra_cycles as usize;
        self.extra_cycles = 0;
        self.cpu.a = self.cpu.y;
        self.set_zn(self.cpu.a);
        total_cycles += 2;
        self.push(self.cpu.a);
        total_cycles += 3;
        self.push_u16(0xf1c4);
        total_cycles += 6;
        self.push_u16(0xf1d9);
        total_cycles += 6;
        total_cycles += self.fast_forward_x_offscreen_bits_body();
        self.cpu.a = self.lsr(self.cpu.a);
        self.cpu.a = self.lsr(self.cpu.a);
        self.cpu.a = self.lsr(self.cpu.a);
        self.cpu.a = self.lsr(self.cpu.a);
        total_cycles += 8;
        self.ram_write(0x0000, self.cpu.a);
        total_cycles += 3;
        total_cycles += 3;
        let (y_cycles, return_address) = self.fast_forward_y_offscreen_bits_body();
        total_cycles += y_cycles;
        debug_assert_eq!(return_address, 0xf1c4);
        self.cpu.a = self.asl(self.cpu.a);
        self.cpu.a = self.asl(self.cpu.a);
        self.cpu.a = self.asl(self.cpu.a);
        self.cpu.a = self.asl(self.cpu.a);
        total_cycles += 8;
        self.ora(self.ram_read(0x0000));
        total_cycles += 3;
        self.ram_write(0x0000, self.cpu.a);
        total_cycles += 3;
        self.cpu.a = self.pop();
        total_cycles += 4;
        self.cpu.y = self.cpu.a;
        self.set_zn(self.cpu.y);
        total_cycles += 2;
        self.cpu.a = self.ram_read(0x0000);
        self.set_zn(self.cpu.a);
        total_cycles += 3;
        self.ram_write(0x03d0 + self.cpu.y as usize, self.cpu.a);
        total_cycles += 5;
        self.cpu.x = self.ram_read(0x0008);
        self.set_zn(self.cpu.x);
        total_cycles += 3;
        let return_address = self.pop_u16();
        total_cycles += 6;
        debug_assert!(total_cycles <= max_cycles);
        self.cpu.pc = return_address.wrapping_add(1);
        *cpu_cycle_guard += total_cycles;
        *pending_ppu_cycles += total_cycles * 3;
        true
    }

    #[inline(never)]
    fn try_fast_forward_relative_position_helper(
        &mut self,
        cpu_cycle_guard: &mut usize,
        pending_ppu_cycles: &mut usize,
    ) -> bool {
        if self.cpu.pc != SMB_RELATIVE_POSITION_HELPER_PC
            || !self.smb_relative_position_helper_supported
        {
            return false;
        }

        let routine_cycles = if self.ram_read(0x0007) >= self.ram_read(0x0006) {
            18
        } else if self.cpu.y >= 1 {
            32
        } else {
            34
        };
        let total_cycles = routine_cycles + self.extra_cycles as usize;
        let ppu_cycles = total_cycles * 3;
        let remaining = self
            .ppu
            .cycles_until_next_event()
            .saturating_sub(*pending_ppu_cycles);
        if ppu_cycles >= remaining || *cpu_cycle_guard + total_cycles >= CPU_CYCLES_PER_FRAME_GUARD
        {
            return false;
        }

        self.ram_write(0x0005, self.cpu.a);
        self.cpu.a = self.ram_read(0x0007);
        self.set_zn(self.cpu.a);
        self.cmp(self.cpu.a, self.ram_read(0x0006));
        if !self.flag(FLAG_C) {
            self.cpu.a = self.lsr(self.cpu.a);
            self.cpu.a = self.lsr(self.cpu.a);
            self.cpu.a = self.lsr(self.cpu.a);
            self.and(0x07);
            self.cmp(self.cpu.y, 1);
            if !self.flag(FLAG_C) {
                self.adc(self.ram_read(0x0005));
            }
            self.cpu.x = self.cpu.a;
            self.set_zn(self.cpu.x);
        }
        self.cpu.pc = self.pop_u16().wrapping_add(1);
        self.extra_cycles = 0;
        *cpu_cycle_guard += total_cycles;
        *pending_ppu_cycles += ppu_cycles;
        true
    }

    #[inline(never)]
    fn try_fast_forward_draw_sprite_object(
        &mut self,
        cpu_cycle_guard: &mut usize,
        pending_ppu_cycles: &mut usize,
    ) -> bool {
        if self.cpu.pc != SMB_DRAW_SPRITE_OBJECT_PC || !self.smb_draw_sprite_object_supported {
            return false;
        }

        let flip = self.ram_read(0x0003) & 0x02 != 0;
        let routine_cycles = if flip { 101 } else { 99 };
        let total_cycles = routine_cycles + self.extra_cycles as usize;
        let ppu_cycles = total_cycles * 3;
        let remaining = self
            .ppu
            .cycles_until_next_event()
            .saturating_sub(*pending_ppu_cycles);
        if ppu_cycles >= remaining || *cpu_cycle_guard + total_cycles >= CPU_CYCLES_PER_FRAME_GUARD
        {
            return false;
        }

        let y = self.cpu.y as usize;
        let tile_0 = self.ram_read(0x0000);
        let tile_1 = self.ram_read(0x0001);
        if flip {
            self.ram_write(0x0205 + y, tile_0);
            self.ram_write(0x0201 + y, tile_1);
        } else {
            self.ram_write(0x0201 + y, tile_0);
            self.ram_write(0x0205 + y, tile_1);
        }
        let attributes = self.ram_read(0x0004) | if flip { 0x40 } else { 0 };
        self.ram_write(0x0202 + y, attributes);
        self.ram_write(0x0206 + y, attributes);
        let sprite_y = self.ram_read(0x0002);
        self.ram_write(0x0200 + y, sprite_y);
        self.ram_write(0x0204 + y, sprite_y);
        let sprite_x = self.ram_read(0x0005);
        self.ram_write(0x0203 + y, sprite_x);
        self.ram_write(0x0207 + y, sprite_x.wrapping_add(8));
        self.ram_write(0x0002, sprite_y.wrapping_add(8));

        self.cpu.a = self.cpu.y;
        self.set_flag(FLAG_C, false);
        self.adc(8);
        self.cpu.y = self.cpu.a;
        self.set_zn(self.cpu.y);
        self.cpu.x = self.cpu.x.wrapping_add(1);
        self.set_zn(self.cpu.x);
        self.cpu.x = self.cpu.x.wrapping_add(1);
        self.set_zn(self.cpu.x);
        self.cpu.pc = self.pop_u16().wrapping_add(1);
        self.extra_cycles = 0;
        *cpu_cycle_guard += total_cycles;
        *pending_ppu_cycles += ppu_cycles;
        true
    }

    #[inline]
    fn refresh_smb_state(&mut self) {
        self.x_pos = ((self.ram[0x006d] as u16) << 8) | self.ram[0x0086] as u16;
        self.coins = self.ram[0x075e];
        self.level_hi = sign_extend_u8(self.ram[0x075f]);
        self.level_lo = sign_extend_u8(self.ram[0x075c]);
        self.lives = sign_extend_u8(self.ram[0x075a]);
        self.score = u32::from(self.ram[0x07dd] & 0x0f) * 100_000
            + u32::from(self.ram[0x07de] & 0x0f) * 10_000
            + u32::from(self.ram[0x07df] & 0x0f) * 1_000
            + u32::from(self.ram[0x07e0] & 0x0f) * 100
            + u32::from(self.ram[0x07e1] & 0x0f) * 10
            + u32::from(self.ram[0x07e2] & 0x0f);
        self.scrolling = sign_extend_u8(self.ram[0x0778]);
        self.time = u16::from(self.ram[0x07f8] & 0x0f) * 100
            + u16::from(self.ram[0x07f9] & 0x0f) * 10
            + u16::from(self.ram[0x07fa] & 0x0f);
        self.xscroll_hi = self.ram[0x071a];
        self.xscroll_lo = self.ram[0x071c];
        self.ppu.set_scroll_override_x(None);
    }

    #[inline]
    fn cpu_read(&mut self, addr: u16) -> u8 {
        match addr {
            0x0000..=0x1fff => self.ram[addr as usize & 0x07ff],
            0x2000..=0x3fff => self.ppu.cpu_read_register(addr),
            0x4016 => self.controller_read(),
            0x8000..=0xffff => self.prg_read(addr),
            _ => 0,
        }
    }

    #[inline]
    fn cpu_write(&mut self, addr: u16, value: u8) {
        match addr {
            0x0000..=0x1fff => self.ram[addr as usize & 0x07ff] = value,
            0x2000..=0x3fff => self.ppu.cpu_write_register(addr, value),
            0x4014 => self.oam_dma(value),
            0x4016 => self.controller_write(value),
            _ => {}
        }
    }

    #[inline]
    fn prg_read(&self, addr: u16) -> u8 {
        let idx = ((addr - 0x8000) as usize) & self.prg_addr_mask;
        // SAFETY: SMB/NROM PRG ROM sizes are power-of-two and prg_addr_mask is len - 1.
        unsafe { *self.prg_rom.get_unchecked(idx) }
    }

    #[inline]
    fn cpu_read_u16(&mut self, addr: u16) -> u16 {
        let lo = self.cpu_read(addr) as u16;
        let hi = self.cpu_read(addr.wrapping_add(1)) as u16;
        lo | (hi << 8)
    }

    #[inline]
    fn ram_read(&self, addr: usize) -> u8 {
        // SAFETY: Masking to 0x07ff keeps the index within the 2 KiB internal RAM.
        unsafe { *self.ram.get_unchecked(addr & 0x07ff) }
    }

    #[inline]
    fn ram_write(&mut self, addr: usize, value: u8) {
        // SAFETY: Masking to 0x07ff keeps the index within the 2 KiB internal RAM.
        unsafe {
            *self.ram.get_unchecked_mut(addr & 0x07ff) = value;
        }
    }

    #[inline]
    fn controller_write(&mut self, value: u8) {
        self.controller_strobe = value & 1 != 0;
        if self.controller_strobe {
            self.controller_shift = self.controller_state;
        }
    }

    #[inline]
    fn controller_read(&mut self) -> u8 {
        if self.controller_strobe {
            return 0x40 | (self.controller_state & 1);
        }
        let value = self.controller_shift & 1;
        self.controller_shift = (self.controller_shift >> 1) | 0x80;
        0x40 | value
    }

    fn oam_dma(&mut self, page: u8) {
        let base = (page as u16) << 8;
        for i in 0..256u16 {
            let value = self.cpu_read(base | i);
            let idx = self.ppu.oam_addr.wrapping_add(i as u8) as usize;
            self.ppu.oam[idx] = value;
        }
        self.extra_cycles = self.extra_cycles.wrapping_add(513);
    }

    #[inline]
    fn fetch_u8(&mut self) -> u8 {
        let value = if self.cpu.pc >= 0x8000 {
            self.prg_read(self.cpu.pc)
        } else {
            self.cpu_read(self.cpu.pc)
        };
        self.cpu.pc = self.cpu.pc.wrapping_add(1);
        value
    }

    #[inline]
    fn fetch_u16(&mut self) -> u16 {
        let lo = self.fetch_u8() as u16;
        let hi = self.fetch_u8() as u16;
        lo | (hi << 8)
    }

    #[inline]
    fn zp_index(&mut self) -> usize {
        self.fetch_u8() as usize
    }

    #[inline]
    fn zpx_index(&mut self) -> usize {
        self.fetch_u8().wrapping_add(self.cpu.x) as usize
    }

    #[inline]
    fn zpy_index(&mut self) -> usize {
        self.fetch_u8().wrapping_add(self.cpu.y) as usize
    }

    #[inline]
    fn abs(&mut self) -> u16 {
        self.fetch_u16()
    }

    #[inline]
    fn absx(&mut self) -> (u16, bool) {
        let base = self.fetch_u16();
        let addr = base.wrapping_add(self.cpu.x as u16);
        (addr, page_crossed(base, addr))
    }

    #[inline]
    fn absy(&mut self) -> (u16, bool) {
        let base = self.fetch_u16();
        let addr = base.wrapping_add(self.cpu.y as u16);
        (addr, page_crossed(base, addr))
    }

    #[inline]
    fn indx(&mut self) -> u16 {
        let ptr = self.fetch_u8().wrapping_add(self.cpu.x);
        let lo = self.ram_read(ptr as usize) as u16;
        let hi = self.ram_read(ptr.wrapping_add(1) as usize) as u16;
        lo | (hi << 8)
    }

    #[inline]
    fn indy(&mut self) -> (u16, bool) {
        let ptr = self.fetch_u8();
        let lo = self.ram_read(ptr as usize) as u16;
        let hi = self.ram_read(ptr.wrapping_add(1) as usize) as u16;
        let base = lo | (hi << 8);
        let addr = base.wrapping_add(self.cpu.y as u16);
        (addr, page_crossed(base, addr))
    }

    #[inline]
    fn set_flag(&mut self, flag: u8, value: bool) {
        if value {
            self.cpu.p |= flag;
        } else {
            self.cpu.p &= !flag;
        }
        self.cpu.p |= FLAG_U;
    }

    #[inline]
    fn flag(&self, flag: u8) -> bool {
        self.cpu.p & flag != 0
    }

    #[inline]
    fn set_zn(&mut self, value: u8) {
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
    fn push(&mut self, value: u8) {
        self.ram_write(0x0100 | self.cpu.sp as usize, value);
        self.cpu.sp = self.cpu.sp.wrapping_sub(1);
    }

    #[inline]
    fn pop(&mut self) -> u8 {
        self.cpu.sp = self.cpu.sp.wrapping_add(1);
        self.ram_read(0x0100 | self.cpu.sp as usize)
    }

    #[inline]
    fn push_u16(&mut self, value: u16) {
        self.push((value >> 8) as u8);
        self.push(value as u8);
    }

    #[inline]
    fn pop_u16(&mut self) -> u16 {
        let lo = self.pop() as u16;
        let hi = self.pop() as u16;
        lo | (hi << 8)
    }

    fn interrupt(&mut self, vector: u16, brk: bool) {
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
    fn cpu_step(&mut self) -> u16 {
        let opcode = self.fetch_u8();
        self.cpu_step_decoded(opcode)
    }

    #[inline]
    fn cpu_step_profiled(&mut self, profiler: &mut Profiler) -> u16 {
        let pc = self.cpu.pc;
        let opcode = self.fetch_u8();
        profiler.record_cpu_step(pc, opcode);
        self.cpu_step_decoded(opcode)
    }

    fn cpu_step_decoded(&mut self, opcode: u8) -> u16 {
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
    fn branch(&mut self, condition: bool) -> u16 {
        let offset = self.fetch_u8() as i8;
        if !condition {
            return 2;
        }
        let old_pc = self.cpu.pc;
        self.cpu.pc = self.cpu.pc.wrapping_add(offset as i16 as u16);
        3 + page_crossed(old_pc, self.cpu.pc) as u16
    }

    #[inline]
    fn adc(&mut self, value: u8) {
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
    fn sbc(&mut self, value: u8) {
        self.adc(!value);
    }

    #[inline]
    fn cmp(&mut self, reg: u8, value: u8) {
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
    fn ora(&mut self, value: u8) {
        self.cpu.a |= value;
        self.set_zn(self.cpu.a);
    }

    #[inline]
    fn and(&mut self, value: u8) {
        self.cpu.a &= value;
        self.set_zn(self.cpu.a);
    }

    #[inline]
    fn eor(&mut self, value: u8) {
        self.cpu.a ^= value;
        self.set_zn(self.cpu.a);
    }

    #[inline]
    fn bit(&mut self, value: u8) {
        let mut p = self.cpu.p & !(FLAG_Z | FLAG_V | FLAG_N);
        if self.cpu.a & value == 0 {
            p |= FLAG_Z;
        }
        p |= value & (FLAG_V | FLAG_N);
        self.cpu.p = p | FLAG_U;
    }

    #[inline]
    fn asl(&mut self, value: u8) -> u8 {
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
    fn lsr(&mut self, value: u8) -> u8 {
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
    fn rol(&mut self, value: u8) -> u8 {
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
    fn ror(&mut self, value: u8) -> u8 {
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
    fn asl_mem(&mut self, addr: u16) {
        let value = self.cpu_read(addr);
        let result = self.asl(value);
        self.cpu_write(addr, result);
    }

    #[inline]
    fn asl_ram(&mut self, addr: usize) {
        let value = self.ram_read(addr);
        let result = self.asl(value);
        self.ram_write(addr, result);
    }

    #[inline]
    fn lsr_mem(&mut self, addr: u16) {
        let value = self.cpu_read(addr);
        let result = self.lsr(value);
        self.cpu_write(addr, result);
    }

    #[inline]
    fn lsr_ram(&mut self, addr: usize) {
        let value = self.ram_read(addr);
        let result = self.lsr(value);
        self.ram_write(addr, result);
    }

    #[inline]
    fn rol_mem(&mut self, addr: u16) {
        let value = self.cpu_read(addr);
        let result = self.rol(value);
        self.cpu_write(addr, result);
    }

    #[inline]
    fn rol_ram(&mut self, addr: usize) {
        let value = self.ram_read(addr);
        let result = self.rol(value);
        self.ram_write(addr, result);
    }

    #[inline]
    fn ror_mem(&mut self, addr: u16) {
        let value = self.cpu_read(addr);
        let result = self.ror(value);
        self.cpu_write(addr, result);
    }

    #[inline]
    fn ror_ram(&mut self, addr: usize) {
        let value = self.ram_read(addr);
        let result = self.ror(value);
        self.ram_write(addr, result);
    }

    #[inline]
    fn dec_mem(&mut self, addr: u16) {
        let result = self.cpu_read(addr).wrapping_sub(1);
        self.cpu_write(addr, result);
        self.set_zn(result);
    }

    #[inline]
    fn dec_ram(&mut self, addr: usize) {
        let result = self.ram_read(addr).wrapping_sub(1);
        self.ram_write(addr, result);
        self.set_zn(result);
    }

    #[inline]
    fn inc_mem(&mut self, addr: u16) {
        let result = self.cpu_read(addr).wrapping_add(1);
        self.cpu_write(addr, result);
        self.set_zn(result);
    }

    #[inline]
    fn inc_ram(&mut self, addr: usize) {
        let result = self.ram_read(addr).wrapping_add(1);
        self.ram_write(addr, result);
        self.set_zn(result);
    }
}

#[inline]
fn page_crossed(a: u16, b: u16) -> bool {
    (a & 0xff00) != (b & 0xff00)
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let mut ppu = Ppu::new(chr_rom, true);
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

    fn make_test_cart_with_prg(prg_rom: Vec<u8>) -> Cartridge {
        Cartridge {
            prg_rom,
            chr_rom: vec![0; 8192],
            vertical_mirroring: true,
        }
    }

    #[test]
    fn step_frame_preserves_the_raw_controller_byte() {
        let mut emu =
            NesEmulator::new_with_options(make_test_cart_with_prg(vec![0xea; 32768]), true);
        let controller_state = 0b1011_0111;

        emu.step_frame(controller_state);

        assert_eq!(emu.controller_state, controller_state);
    }

    #[test]
    fn gray_palette_cache_tracks_palette_writes() {
        let mut ppu = Ppu::new(vec![0; 8192], true);
        for (addr, value) in [
            (0x3f00, 0x01),
            (0x3f01, 0x11),
            (0x3f05, 0x21),
            (0x3f0f, 0x31),
            (0x3f11, 0x0f),
            (0x3f10, 0x30),
        ] {
            ppu.ppu_write(addr, value);
            let mut expected_gray = [0; 32];
            for (dst, &color) in expected_gray.iter_mut().zip(ppu.palette.iter()) {
                *dst = NES_GRAY_PALETTE[color as usize];
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
            interpreted_cycles += interpreted.cpu_step() as usize;
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
            interpreted_cycles += interpreted.cpu_step() as usize;
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
            interpreted_cycles += interpreted.cpu_step() as usize;
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
            interpreted_cycles += interpreted.cpu_step() as usize;
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
            interpreted_cycles += interpreted.cpu_step() as usize;
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
            interpreted_cycles += interpreted.cpu_step() as usize;
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
                interpreted_cycles += interpreted.cpu_step() as usize;
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
            interpreted_cycles += interpreted.cpu_step() as usize;
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
            interpreted_cycles += interpreted.cpu_step() as usize;
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
            interpreted_cycles += interpreted.cpu_step() as usize;
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
            interpreted_cycles += interpreted.cpu_step() as usize;
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

    #[test]
    fn cached_next_ppu_event_tracks_frame_transitions() {
        let mut ppu = Ppu::new(vec![0; 8192], true);
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
    fn sprite_scanline_mask_limits_to_first_eight_oam_sprites() {
        let mut ppu = Ppu::new(vec![0; 8192], true);
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
        let mut ppu = Ppu::new(chr_rom, true);
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
            let mut ppu = Ppu::new(chr_rom, vertical_mirroring);
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
}
