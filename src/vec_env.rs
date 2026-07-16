use crate::cartridge::Cartridge;
use crate::emulator::{
    NesEmulator, StateLoadError, RGB_CHANNELS, VISIBLE_FRAME_HEIGHT, VISIBLE_FRAME_WIDTH,
};
use crate::profiler::Profiler;
use rayon::prelude::*;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const PARALLEL_ENV_THRESHOLD: usize = 4;
const DEFAULT_MASK_CROP_TOP: usize = 32;
const DEFAULT_AREA_SRC_WIDTH: usize = VISIBLE_FRAME_WIDTH;
const DEFAULT_AREA_SRC_HEIGHT: usize = VISIBLE_FRAME_HEIGHT - DEFAULT_MASK_CROP_TOP;
const DEFAULT_AREA_DST_WIDTH: usize = 84;
const DEFAULT_AREA_DST_HEIGHT: usize = 84;
const DEFAULT_AREA_Y0: [usize; DEFAULT_AREA_DST_HEIGHT] =
    build_default_area_axis_start(DEFAULT_AREA_SRC_HEIGHT);
const DEFAULT_AREA_Y1: [usize; DEFAULT_AREA_DST_HEIGHT] =
    build_default_area_axis_end(DEFAULT_AREA_SRC_HEIGHT);
const DEFAULT_AREA_Y_COUNT: [u16; DEFAULT_AREA_DST_HEIGHT] =
    build_default_area_axis_count(DEFAULT_AREA_SRC_HEIGHT);
const FULL_AREA_SRC_HEIGHT: usize = VISIBLE_FRAME_HEIGHT;
const FULL_AREA_Y0: [usize; DEFAULT_AREA_DST_HEIGHT] =
    build_default_area_axis_start(FULL_AREA_SRC_HEIGHT);
const FULL_AREA_Y1: [usize; DEFAULT_AREA_DST_HEIGHT] =
    build_default_area_axis_end(FULL_AREA_SRC_HEIGHT);
const FULL_AREA_Y_COUNT: [u16; DEFAULT_AREA_DST_HEIGHT] =
    build_default_area_axis_count(FULL_AREA_SRC_HEIGHT);

#[derive(Clone, Copy, Debug)]
pub struct VecEnvConfig {
    pub num_envs: usize,
    pub frame_skip: usize,
    pub grayscale: bool,
    pub frame_stack: usize,
    pub frame_maxpool: bool,
    pub noop_reset_max: usize,
    pub sticky_action_prob: f64,
    pub terminate_on_flag: bool,
    pub crop_top: usize,
    pub crop_bottom: usize,
    pub crop_left: usize,
    pub crop_right: usize,
    pub crop_mode: CropMode,
    pub crop_fill: u8,
    pub resize_width: usize,
    pub resize_height: usize,
    pub resize_algorithm: ResizeAlgorithm,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CropMode {
    Remove,
    Mask,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResizeAlgorithm {
    Area,
    Nearest,
    Bilinear,
}

impl VecEnvConfig {
    pub fn source_width(&self) -> usize {
        match self.crop_mode {
            CropMode::Remove => VISIBLE_FRAME_WIDTH - self.crop_left - self.crop_right,
            CropMode::Mask => VISIBLE_FRAME_WIDTH,
        }
    }

    pub fn source_height(&self) -> usize {
        match self.crop_mode {
            CropMode::Remove => VISIBLE_FRAME_HEIGHT - self.crop_top - self.crop_bottom,
            CropMode::Mask => VISIBLE_FRAME_HEIGHT,
        }
    }

    pub fn obs_width(&self) -> usize {
        self.resize_width
    }

    pub fn obs_height(&self) -> usize {
        self.resize_height
    }

    pub fn channels(&self) -> usize {
        if self.grayscale {
            self.frame_stack
        } else {
            self.frame_stack * RGB_CHANNELS
        }
    }

    pub fn obs_len_per_env(&self) -> usize {
        self.channels() * self.obs_height() * self.obs_width()
    }

    fn needs_resize(&self) -> bool {
        self.resize_width != self.source_width() || self.resize_height != self.source_height()
    }

    fn uses_default_gray_area_resize(&self) -> bool {
        false
    }

    fn uses_default_gray_mask_top_area_resize(&self) -> bool {
        self.grayscale
            && self.crop_mode == CropMode::Mask
            && self.crop_top == DEFAULT_MASK_CROP_TOP
            && self.crop_bottom == 0
            && self.crop_left == 0
            && self.crop_right == 0
            && self.resize_algorithm == ResizeAlgorithm::Area
            && self.source_width() == VISIBLE_FRAME_WIDTH
            && self.source_height() == VISIBLE_FRAME_HEIGHT
            && self.resize_width == DEFAULT_AREA_DST_WIDTH
            && self.resize_height == DEFAULT_AREA_DST_HEIGHT
    }
}

pub struct MarioVecEnv {
    config: VecEnvConfig,
    resize_plan: AreaResizePlan,
    envs: Vec<NesEmulator>,
    initial_states: Vec<InitialState>,
    weighted_initial_states: bool,
    active_state_indices: Vec<i32>,
    initial_state_names: Vec<String>,
    last_actions: Vec<u8>,
    rngs: Vec<XorShift64>,
    pending_reset: Vec<bool>,
    any_pending_reset: bool,
    scratch: Vec<Vec<u8>>,
    profiler: Option<Profiler>,
    profile_shards: Vec<Profiler>,
}

impl MarioVecEnv {
    pub fn new(
        cart: Cartridge,
        config: VecEnvConfig,
        initial_states: Vec<InitialState>,
        weighted_initial_states: bool,
        seed: u64,
    ) -> Result<Self, StateLoadError> {
        let resize_plan = AreaResizePlan::new(
            config.source_width(),
            config.source_height(),
            config.resize_width,
            config.resize_height,
        );
        let scratch_len = scratch_len(config);
        let envs = (0..config.num_envs)
            .map(|_| NesEmulator::new_with_options(cart.clone(), config.terminate_on_flag))
            .collect::<Vec<_>>();
        let scratch = (0..config.num_envs)
            .map(|_| vec![0; scratch_len])
            .collect::<Vec<_>>();
        let mut env = Self {
            config,
            resize_plan,
            envs,
            initial_states,
            weighted_initial_states,
            active_state_indices: vec![-1; config.num_envs],
            initial_state_names: Vec::new(),
            last_actions: vec![0; config.num_envs],
            rngs: (0..config.num_envs)
                .map(|env_idx| XorShift64::new(seed.wrapping_add(env_idx as u64)))
                .collect(),
            pending_reset: vec![false; config.num_envs],
            any_pending_reset: false,
            scratch,
            profiler: None,
            profile_shards: Vec::new(),
        };
        env.intern_initial_state_names();
        env.reset_envs()?;
        Ok(env)
    }

    fn intern_initial_state_names(&mut self) {
        for state in &mut self.initial_states {
            if let Some(index) = self
                .initial_state_names
                .iter()
                .position(|name| name == &state.name)
            {
                state.name_index = index as i32;
            } else {
                state.name_index = self.initial_state_names.len() as i32;
                self.initial_state_names.push(state.name.clone());
            }
        }
    }

    fn reset_envs(&mut self) -> Result<(), StateLoadError> {
        if self.initial_states.is_empty() {
            for env_idx in 0..self.config.num_envs {
                self.envs[env_idx].reset();
                self.last_actions[env_idx] = 0;
                self.apply_noop_reset(env_idx);
            }
            self.active_state_indices.fill(-1);
            return Ok(());
        }

        for env_idx in 0..self.config.num_envs {
            let state_index = self.initial_state_index_for_env(env_idx);
            self.active_state_indices[env_idx] = self.initial_states[state_index].name_index;
            self.envs[env_idx].load_fceu_state(&self.initial_states[state_index].data)?;
            self.last_actions[env_idx] = 0;
            self.apply_noop_reset(env_idx);
        }
        Ok(())
    }

    fn initial_state_index_for_env(&mut self, env_idx: usize) -> usize {
        if self.weighted_initial_states {
            let sample = self.rngs[env_idx].next_unit_f64();
            for (idx, state) in self.initial_states.iter().enumerate() {
                if sample < state.cumulative_weight {
                    return idx;
                }
            }
            self.initial_states.len() - 1
        } else if self.initial_states.len() == 1 {
            0
        } else {
            env_idx
        }
    }

    fn apply_noop_reset(&mut self, env_idx: usize) {
        if self.config.noop_reset_max == 0 {
            return;
        }
        let noop_count = self.rngs[env_idx].next_bounded_usize(self.config.noop_reset_max + 1);
        for _ in 0..noop_count {
            self.envs[env_idx].step_frame(0);
            if self.envs[env_idx].is_done() {
                break;
            }
        }
    }

    fn effective_actions(&mut self, actions: &[u8]) -> Vec<u8> {
        if self.config.sticky_action_prob <= 0.0 {
            self.last_actions.copy_from_slice(actions);
            return actions.to_vec();
        }

        let mut effective = Vec::with_capacity(actions.len());
        for (env_idx, &action) in actions.iter().enumerate() {
            let chosen = if self.rngs[env_idx].next_unit_f64() < self.config.sticky_action_prob {
                self.last_actions[env_idx]
            } else {
                action
            };
            self.last_actions[env_idx] = chosen;
            effective.push(chosen);
        }
        effective
    }

    pub fn config(&self) -> VecEnvConfig {
        self.config
    }

    pub fn reset_into(&mut self, obs: &mut [u8]) -> Result<(), StateLoadError> {
        let reset_mask = vec![true; self.config.num_envs];
        let start_indices = vec![-1; self.config.num_envs];
        let seeds = vec![None; self.config.num_envs];
        self.reset_masked_into(obs, &reset_mask, &start_indices, &seeds)
    }

    pub fn reset_masked_into(
        &mut self,
        obs: &mut [u8],
        reset_mask: &[bool],
        start_indices: &[i32],
        seeds: &[Option<u64>],
    ) -> Result<(), StateLoadError> {
        let config = self.config;
        let obs_stride = config.obs_len_per_env();
        debug_assert_eq!(reset_mask.len(), config.num_envs);
        debug_assert_eq!(start_indices.len(), config.num_envs);
        debug_assert_eq!(seeds.len(), config.num_envs);

        for env_idx in 0..config.num_envs {
            if !reset_mask[env_idx] {
                continue;
            }
            if let Some(seed) = seeds[env_idx] {
                self.rngs[env_idx] = XorShift64::from_explicit_seed(seed);
            }
            let explicit_index =
                (start_indices[env_idx] >= 0).then_some(start_indices[env_idx] as usize);
            self.reset_one_env(env_idx, explicit_index)?;
            self.pending_reset[env_idx] = false;
        }

        if reset_mask.iter().filter(|&&selected| selected).count() >= PARALLEL_ENV_THRESHOLD {
            let resize_plan = &self.resize_plan;
            self.envs
                .par_iter()
                .zip(self.scratch.par_iter_mut())
                .zip(obs.par_chunks_mut(obs_stride))
                .zip(reset_mask.par_iter())
                .for_each(|(((env, scratch), obs_chunk), &selected)| {
                    if selected {
                        write_reset_stack(config, resize_plan, env, scratch, obs_chunk);
                    }
                });
        } else {
            for (env_idx, &selected) in reset_mask.iter().enumerate() {
                if !selected {
                    continue;
                }
                let start = env_idx * obs_stride;
                let end = start + obs_stride;
                write_reset_stack(
                    config,
                    &self.resize_plan,
                    &self.envs[env_idx],
                    &mut self.scratch[env_idx],
                    &mut obs[start..end],
                );
            }
        }
        self.any_pending_reset = self.pending_reset.iter().any(|pending| *pending);
        Ok(())
    }

    pub fn initial_state_names(&self) -> Vec<String> {
        self.initial_state_names.clone()
    }

    pub fn initial_state_policy_names(&self) -> Vec<String> {
        self.initial_states
            .iter()
            .map(|state| state.name.clone())
            .collect()
    }

    pub fn initial_state_weights(&self) -> Vec<f64> {
        if self.initial_states.is_empty() {
            return Vec::new();
        }
        if !self.weighted_initial_states {
            return vec![1.0 / self.initial_states.len() as f64; self.initial_states.len()];
        }
        let mut weights = Vec::with_capacity(self.initial_states.len());
        let mut previous = 0.0;
        for state in &self.initial_states {
            weights.push((state.cumulative_weight - previous).max(0.0));
            previous = state.cumulative_weight;
        }
        weights
    }

    pub fn active_state_indices(&self) -> &[i32] {
        &self.active_state_indices
    }

    pub fn rgb_frames_hwc_into(&self, dst: &mut [u8]) {
        let frame_stride = visual_rgb_frame_len();
        debug_assert_eq!(dst.len(), self.config.num_envs * frame_stride);
        let mut planar = vec![0; frame_stride];

        for env_idx in 0..self.config.num_envs {
            let start = env_idx * frame_stride;
            write_visible_rgb_hwc(
                &self.envs[env_idx],
                &mut planar,
                &mut dst[start..start + frame_stride],
            );
        }
    }

    pub fn seed(&mut self, seed: u64) {
        for (env_idx, rng) in self.rngs.iter_mut().enumerate() {
            *rng = XorShift64::from_explicit_seed(seed.wrapping_add(env_idx as u64));
        }
    }

    pub fn has_pending_reset(&self) -> bool {
        self.any_pending_reset
    }

    pub fn enable_profiler(&mut self) {
        self.profiler = Some(Profiler::new());
        self.profile_shards = (0..self.config.num_envs).map(|_| Profiler::new()).collect();
    }

    pub fn reset_profiler(&mut self) {
        if let Some(profiler) = &mut self.profiler {
            profiler.clear();
        }
        for shard in &mut self.profile_shards {
            shard.clear();
        }
    }

    pub fn disable_profiler(&mut self) {
        self.profiler = None;
        self.profile_shards.clear();
    }

    pub fn profiler_snapshot_json(&self, top_n: usize) -> Option<String> {
        let mut merged = self.profiler.clone()?;
        for shard in &self.profile_shards {
            merged.add(shard);
        }
        Some(merged.to_json(top_n))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn info_into(
        &self,
        x_pos: &mut [u16],
        coins: &mut [u8],
        level_hi: &mut [i16],
        level_lo: &mut [i16],
        lives: &mut [i16],
        score: &mut [u32],
        scrolling: &mut [i16],
        time: &mut [u16],
        xscroll_hi: &mut [u8],
        xscroll_lo: &mut [u8],
    ) {
        for env_idx in 0..self.config.num_envs {
            write_info_from_env(
                &self.envs[env_idx],
                &mut x_pos[env_idx],
                &mut coins[env_idx],
                &mut level_hi[env_idx],
                &mut level_lo[env_idx],
                &mut lives[env_idx],
                &mut score[env_idx],
                &mut scrolling[env_idx],
                &mut time[env_idx],
                &mut xscroll_hi[env_idx],
                &mut xscroll_lo[env_idx],
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn step_into(
        &mut self,
        actions: &[u8],
        obs: &mut [u8],
        rewards: &mut [f32],
        terminated: &mut [bool],
        truncated: &mut [bool],
        x_pos: &mut [u16],
        coins: &mut [u8],
        level_hi: &mut [i16],
        level_lo: &mut [i16],
        lives: &mut [i16],
        score: &mut [u32],
        scrolling: &mut [i16],
        time: &mut [u16],
        xscroll_hi: &mut [u8],
        xscroll_lo: &mut [u8],
    ) {
        if self.profiler.is_some() {
            self.step_into_profiled(
                actions, obs, rewards, terminated, truncated, x_pos, coins, level_hi, level_lo,
                lives, score, scrolling, time, xscroll_hi, xscroll_lo,
            );
            return;
        }

        let config = self.config;
        let obs_stride = config.obs_len_per_env();
        let effective_actions_storage;
        let actions = if config.sticky_action_prob > 0.0 {
            effective_actions_storage = self.effective_actions(actions);
            effective_actions_storage.as_slice()
        } else {
            self.last_actions.copy_from_slice(actions);
            actions
        };
        if config.num_envs >= PARALLEL_ENV_THRESHOLD {
            let resize_plan = &self.resize_plan;
            self.envs
                .par_iter_mut()
                .zip(self.scratch.par_iter_mut())
                .zip(actions.par_iter())
                .zip(obs.par_chunks_mut(obs_stride))
                .zip(rewards.par_iter_mut())
                .zip(terminated.par_iter_mut())
                .zip(truncated.par_iter_mut())
                .zip(x_pos.par_iter_mut())
                .zip(coins.par_iter_mut())
                .zip(level_hi.par_iter_mut())
                .zip(level_lo.par_iter_mut())
                .zip(lives.par_iter_mut())
                .zip(score.par_iter_mut())
                .zip(scrolling.par_iter_mut())
                .zip(time.par_iter_mut())
                .zip(xscroll_hi.par_iter_mut())
                .zip(xscroll_lo.par_iter_mut())
                .for_each(|zipped| {
                    let (zipped, xscroll_lo_out) = zipped;
                    let (zipped, xscroll_hi_out) = zipped;
                    let (zipped, time_out) = zipped;
                    let (zipped, scrolling_out) = zipped;
                    let (zipped, score_out) = zipped;
                    let (zipped, lives_out) = zipped;
                    let (zipped, level_lo_out) = zipped;
                    let (zipped, level_hi_out) = zipped;
                    let (zipped, coins_out) = zipped;
                    let (zipped, x_out) = zipped;
                    let (zipped, truncated_out) = zipped;
                    let (zipped, terminated_out) = zipped;
                    let (zipped, reward_out) = zipped;
                    let (zipped, obs_chunk) = zipped;
                    let ((env, scratch), action) = zipped;
                    step_one(
                        config,
                        resize_plan,
                        env,
                        scratch,
                        *action,
                        obs_chunk,
                        reward_out,
                        terminated_out,
                        truncated_out,
                        x_out,
                        coins_out,
                        level_hi_out,
                        level_lo_out,
                        lives_out,
                        score_out,
                        scrolling_out,
                        time_out,
                        xscroll_hi_out,
                        xscroll_lo_out,
                    );
                });
        } else {
            for env_idx in 0..config.num_envs {
                let start = env_idx * obs_stride;
                let end = start + obs_stride;
                step_one(
                    config,
                    &self.resize_plan,
                    &mut self.envs[env_idx],
                    &mut self.scratch[env_idx],
                    actions[env_idx],
                    &mut obs[start..end],
                    &mut rewards[env_idx],
                    &mut terminated[env_idx],
                    &mut truncated[env_idx],
                    &mut x_pos[env_idx],
                    &mut coins[env_idx],
                    &mut level_hi[env_idx],
                    &mut level_lo[env_idx],
                    &mut lives[env_idx],
                    &mut score[env_idx],
                    &mut scrolling[env_idx],
                    &mut time[env_idx],
                    &mut xscroll_hi[env_idx],
                    &mut xscroll_lo[env_idx],
                );
            }
        }

        if terminated.iter().any(|done| *done) || truncated.iter().any(|done| *done) {
            for env_idx in 0..config.num_envs {
                self.pending_reset[env_idx] = terminated[env_idx] || truncated[env_idx];
            }
            self.any_pending_reset = true;
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn step_into_profiled(
        &mut self,
        actions: &[u8],
        obs: &mut [u8],
        rewards: &mut [f32],
        terminated: &mut [bool],
        truncated: &mut [bool],
        x_pos: &mut [u16],
        coins: &mut [u8],
        level_hi: &mut [i16],
        level_lo: &mut [i16],
        lives: &mut [i16],
        score: &mut [u32],
        scrolling: &mut [i16],
        time: &mut [u16],
        xscroll_hi: &mut [u8],
        xscroll_lo: &mut [u8],
    ) {
        if self.profile_shards.len() != self.config.num_envs {
            self.profile_shards = (0..self.config.num_envs).map(|_| Profiler::new()).collect();
        }
        if let Some(profiler) = &mut self.profiler {
            profiler.record_batch_step(self.config.num_envs);
        }

        let config = self.config;
        let obs_stride = config.obs_len_per_env();
        let effective_actions_storage;
        let actions = if config.sticky_action_prob > 0.0 {
            effective_actions_storage = self.effective_actions(actions);
            effective_actions_storage.as_slice()
        } else {
            self.last_actions.copy_from_slice(actions);
            actions
        };
        if config.num_envs >= PARALLEL_ENV_THRESHOLD {
            let resize_plan = &self.resize_plan;
            self.envs
                .par_iter_mut()
                .zip(self.scratch.par_iter_mut())
                .zip(actions.par_iter())
                .zip(obs.par_chunks_mut(obs_stride))
                .zip(rewards.par_iter_mut())
                .zip(terminated.par_iter_mut())
                .zip(truncated.par_iter_mut())
                .zip(x_pos.par_iter_mut())
                .zip(coins.par_iter_mut())
                .zip(level_hi.par_iter_mut())
                .zip(level_lo.par_iter_mut())
                .zip(lives.par_iter_mut())
                .zip(score.par_iter_mut())
                .zip(scrolling.par_iter_mut())
                .zip(time.par_iter_mut())
                .zip(xscroll_hi.par_iter_mut())
                .zip(xscroll_lo.par_iter_mut())
                .zip(self.profile_shards.par_iter_mut())
                .for_each(|zipped| {
                    let (zipped, profiler) = zipped;
                    let (zipped, xscroll_lo_out) = zipped;
                    let (zipped, xscroll_hi_out) = zipped;
                    let (zipped, time_out) = zipped;
                    let (zipped, scrolling_out) = zipped;
                    let (zipped, score_out) = zipped;
                    let (zipped, lives_out) = zipped;
                    let (zipped, level_lo_out) = zipped;
                    let (zipped, level_hi_out) = zipped;
                    let (zipped, coins_out) = zipped;
                    let (zipped, x_out) = zipped;
                    let (zipped, truncated_out) = zipped;
                    let (zipped, terminated_out) = zipped;
                    let (zipped, reward_out) = zipped;
                    let (zipped, obs_chunk) = zipped;
                    let ((env, scratch), action) = zipped;
                    step_one_profiled(
                        config,
                        resize_plan,
                        env,
                        scratch,
                        *action,
                        obs_chunk,
                        reward_out,
                        terminated_out,
                        truncated_out,
                        x_out,
                        coins_out,
                        level_hi_out,
                        level_lo_out,
                        lives_out,
                        score_out,
                        scrolling_out,
                        time_out,
                        xscroll_hi_out,
                        xscroll_lo_out,
                        profiler,
                    );
                });
        } else {
            for env_idx in 0..config.num_envs {
                let start = env_idx * obs_stride;
                let end = start + obs_stride;
                step_one_profiled(
                    config,
                    &self.resize_plan,
                    &mut self.envs[env_idx],
                    &mut self.scratch[env_idx],
                    actions[env_idx],
                    &mut obs[start..end],
                    &mut rewards[env_idx],
                    &mut terminated[env_idx],
                    &mut truncated[env_idx],
                    &mut x_pos[env_idx],
                    &mut coins[env_idx],
                    &mut level_hi[env_idx],
                    &mut level_lo[env_idx],
                    &mut lives[env_idx],
                    &mut score[env_idx],
                    &mut scrolling[env_idx],
                    &mut time[env_idx],
                    &mut xscroll_hi[env_idx],
                    &mut xscroll_lo[env_idx],
                    &mut self.profile_shards[env_idx],
                );
            }
        }

        if terminated.iter().any(|done| *done) || truncated.iter().any(|done| *done) {
            for env_idx in 0..config.num_envs {
                self.pending_reset[env_idx] = terminated[env_idx] || truncated[env_idx];
            }
            self.any_pending_reset = true;
        }
    }

    fn reset_one_env(
        &mut self,
        env_idx: usize,
        explicit_state_index: Option<usize>,
    ) -> Result<(), StateLoadError> {
        if self.initial_states.is_empty() {
            self.envs[env_idx].reset();
            self.active_state_indices[env_idx] = -1;
            self.last_actions[env_idx] = 0;
            self.apply_noop_reset(env_idx);
            return Ok(());
        }

        let state_index =
            explicit_state_index.unwrap_or_else(|| self.initial_state_index_for_env(env_idx));
        self.active_state_indices[env_idx] = self.initial_states[state_index].name_index;
        self.envs[env_idx].load_fceu_state(&self.initial_states[state_index].data)?;
        self.last_actions[env_idx] = 0;
        self.apply_noop_reset(env_idx);
        Ok(())
    }

    pub fn env_ram(&self, env_idx: usize) -> Option<&[u8; 2048]> {
        self.envs.get(env_idx).map(NesEmulator::ram)
    }

    pub fn env_oam(&self, env_idx: usize) -> Option<&[u8; 256]> {
        self.envs.get(env_idx).map(NesEmulator::oam)
    }

    pub fn env_bg_pixel(&self, env_idx: usize, x: usize, y: usize) -> Option<(u8, bool)> {
        self.envs.get(env_idx).map(|env| env.debug_bg_pixel(x, y))
    }
}

#[derive(Clone)]
pub struct InitialState {
    name: String,
    data: Vec<u8>,
    cumulative_weight: f64,
    name_index: i32,
}

impl InitialState {
    pub fn new(name: String, data: Vec<u8>, cumulative_weight: f64) -> Self {
        Self {
            name,
            data,
            cumulative_weight,
            name_index: -1,
        }
    }
}

struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn from_explicit_seed(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }

    fn new(seed: u64) -> Self {
        let state = if seed == 0 {
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|duration| duration.as_nanos() as u64)
                .unwrap_or(0x9e37_79b9_7f4a_7c15)
                ^ 0x9e37_79b9_7f4a_7c15
        } else {
            seed
        };
        Self {
            state: state.max(1),
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut value = self.state;
        value ^= value << 13;
        value ^= value >> 7;
        value ^= value << 17;
        self.state = value;
        value
    }

    fn next_unit_f64(&mut self) -> f64 {
        const DENOM: f64 = (1u64 << 53) as f64;
        ((self.next_u64() >> 11) as f64) / DENOM
    }

    fn next_bounded_usize(&mut self, upper_exclusive: usize) -> usize {
        if upper_exclusive <= 1 {
            return 0;
        }
        (self.next_u64() % upper_exclusive as u64) as usize
    }
}

fn write_visible_rgb_hwc(env: &NesEmulator, planar: &mut [u8], dst: &mut [u8]) {
    debug_assert_eq!(planar.len(), visual_rgb_frame_len());
    debug_assert_eq!(dst.len(), visual_rgb_frame_len());
    env.write_rgb_visible_frame_cropped(planar, 0, VISIBLE_FRAME_HEIGHT);
    planar_rgb_to_hwc(planar, dst);
}

fn planar_rgb_to_hwc(src: &[u8], dst: &mut [u8]) {
    let plane = VISIBLE_FRAME_WIDTH * VISIBLE_FRAME_HEIGHT;
    for idx in 0..plane {
        let out = idx * RGB_CHANNELS;
        dst[out] = src[idx];
        dst[out + 1] = src[plane + idx];
        dst[out + 2] = src[plane * 2 + idx];
    }
}

#[inline]
fn visual_rgb_frame_len() -> usize {
    VISIBLE_FRAME_WIDTH * VISIBLE_FRAME_HEIGHT * RGB_CHANNELS
}

#[allow(clippy::too_many_arguments)]
fn write_info_from_env(
    env: &NesEmulator,
    x_out: &mut u16,
    coins_out: &mut u8,
    level_hi_out: &mut i16,
    level_lo_out: &mut i16,
    lives_out: &mut i16,
    score_out: &mut u32,
    scrolling_out: &mut i16,
    time_out: &mut u16,
    xscroll_hi_out: &mut u8,
    xscroll_lo_out: &mut u8,
) {
    *x_out = env.x_pos();
    *coins_out = env.coins();
    *level_hi_out = env.level_hi();
    *level_lo_out = env.level_lo();
    *lives_out = env.lives();
    *score_out = env.score();
    *scrolling_out = env.scrolling();
    *time_out = env.time();
    *xscroll_hi_out = env.xscroll_hi();
    *xscroll_lo_out = env.xscroll_lo();
}

#[allow(clippy::too_many_arguments)]
fn step_one(
    config: VecEnvConfig,
    resize_plan: &AreaResizePlan,
    env: &mut NesEmulator,
    scratch: &mut [u8],
    controller_state: u8,
    obs_chunk: &mut [u8],
    reward_out: &mut f32,
    terminated_out: &mut bool,
    truncated_out: &mut bool,
    x_out: &mut u16,
    coins_out: &mut u8,
    level_hi_out: &mut i16,
    level_lo_out: &mut i16,
    lives_out: &mut i16,
    score_out: &mut u32,
    scrolling_out: &mut i16,
    time_out: &mut u16,
    xscroll_hi_out: &mut u8,
    xscroll_lo_out: &mut u8,
) {
    let mut reward = 0.0;
    let mut done = false;
    if config.frame_maxpool {
        let source_len = rgb_source_frame_len(config);
        let (frame_a, frame_b_rest) = scratch.split_at_mut(source_len);
        let frame_b = &mut frame_b_rest[..source_len];
        let mut recent_count = 0usize;
        for _ in 0..config.frame_skip {
            reward += env.step_frame(controller_state);
            let target = if recent_count % 2 == 0 {
                &mut *frame_a
            } else {
                &mut *frame_b
            };
            write_rgb_source_frame(config, env, target);
            recent_count += 1;
            done = env.is_done();
            if done {
                break;
            }
        }
        shift_stack_left(config, obs_chunk);
        write_maxpooled_rgb_frame_to_last_stack_slot(
            config,
            resize_plan,
            frame_a,
            frame_b,
            recent_count,
            obs_chunk,
        );
    } else {
        for _ in 0..config.frame_skip {
            reward += env.step_frame(controller_state);
            done = env.is_done();
            if done {
                break;
            }
        }
        shift_stack_left(config, obs_chunk);
        write_current_frame_to_last_stack_slot(config, resize_plan, env, scratch, obs_chunk);
    }

    *reward_out = reward;
    *terminated_out = done;
    *truncated_out = false;
    write_info_from_env(
        env,
        x_out,
        coins_out,
        level_hi_out,
        level_lo_out,
        lives_out,
        score_out,
        scrolling_out,
        time_out,
        xscroll_hi_out,
        xscroll_lo_out,
    );
}

#[allow(clippy::too_many_arguments)]
fn step_one_profiled(
    config: VecEnvConfig,
    resize_plan: &AreaResizePlan,
    env: &mut NesEmulator,
    scratch: &mut [u8],
    controller_state: u8,
    obs_chunk: &mut [u8],
    reward_out: &mut f32,
    terminated_out: &mut bool,
    truncated_out: &mut bool,
    x_out: &mut u16,
    coins_out: &mut u8,
    level_hi_out: &mut i16,
    level_lo_out: &mut i16,
    lives_out: &mut i16,
    score_out: &mut u32,
    scrolling_out: &mut i16,
    time_out: &mut u16,
    xscroll_hi_out: &mut u8,
    xscroll_lo_out: &mut u8,
    profiler: &mut Profiler,
) {
    let mut reward = 0.0;
    let mut done = false;
    if config.frame_maxpool {
        let source_len = rgb_source_frame_len(config);
        let (frame_a, frame_b_rest) = scratch.split_at_mut(source_len);
        let frame_b = &mut frame_b_rest[..source_len];
        let mut recent_count = 0usize;
        for _ in 0..config.frame_skip {
            reward += env.step_frame_profiled(controller_state, profiler);
            let target = if recent_count % 2 == 0 {
                &mut *frame_a
            } else {
                &mut *frame_b
            };
            let render_start = Instant::now();
            write_rgb_source_frame(config, env, target);
            profiler.record_render(render_start.elapsed());
            recent_count += 1;
            done = env.is_done();
            if done {
                break;
            }
        }
        let shift_start = Instant::now();
        shift_stack_left(config, obs_chunk);
        profiler.record_stack_shift(shift_start.elapsed());
        let resize_start = Instant::now();
        write_maxpooled_rgb_frame_to_last_stack_slot(
            config,
            resize_plan,
            frame_a,
            frame_b,
            recent_count,
            obs_chunk,
        );
        profiler.record_resize(resize_start.elapsed());
    } else {
        for _ in 0..config.frame_skip {
            reward += env.step_frame_profiled(controller_state, profiler);
            done = env.is_done();
            if done {
                break;
            }
        }
        let shift_start = Instant::now();
        shift_stack_left(config, obs_chunk);
        profiler.record_stack_shift(shift_start.elapsed());
        write_current_frame_to_last_stack_slot_profiled(
            config,
            resize_plan,
            env,
            scratch,
            obs_chunk,
            profiler,
        );
    }

    *reward_out = reward;
    *terminated_out = done;
    *truncated_out = false;
    write_info_from_env(
        env,
        x_out,
        coins_out,
        level_hi_out,
        level_lo_out,
        lives_out,
        score_out,
        scrolling_out,
        time_out,
        xscroll_hi_out,
        xscroll_lo_out,
    );
}

fn write_reset_stack(
    config: VecEnvConfig,
    resize_plan: &AreaResizePlan,
    env: &NesEmulator,
    scratch: &mut [u8],
    obs_chunk: &mut [u8],
) {
    let frame_len = frame_len(config);
    for stack_i in 0..config.frame_stack {
        let dst_start = stack_i * frame_len;
        let dst_end = dst_start + frame_len;
        write_current_frame(
            config,
            resize_plan,
            env,
            scratch,
            &mut obs_chunk[dst_start..dst_end],
        );
    }
}

fn shift_stack_left(config: VecEnvConfig, obs_chunk: &mut [u8]) {
    if config.frame_stack <= 1 {
        return;
    }

    let frame_len = frame_len(config);
    let move_len = (config.frame_stack - 1) * frame_len;
    obs_chunk.copy_within(frame_len..frame_len + move_len, 0);
}

fn write_current_frame_to_last_stack_slot(
    config: VecEnvConfig,
    resize_plan: &AreaResizePlan,
    env: &NesEmulator,
    scratch: &mut [u8],
    obs_chunk: &mut [u8],
) {
    let frame_len = frame_len(config);
    let dst_start = (config.frame_stack - 1) * frame_len;
    let dst_end = dst_start + frame_len;
    write_current_frame(
        config,
        resize_plan,
        env,
        scratch,
        &mut obs_chunk[dst_start..dst_end],
    );
}

fn write_current_frame_to_last_stack_slot_profiled(
    config: VecEnvConfig,
    resize_plan: &AreaResizePlan,
    env: &NesEmulator,
    scratch: &mut [u8],
    obs_chunk: &mut [u8],
    profiler: &mut Profiler,
) {
    let frame_len = frame_len(config);
    let dst_start = (config.frame_stack - 1) * frame_len;
    let dst_end = dst_start + frame_len;
    write_current_frame_profiled(
        config,
        resize_plan,
        env,
        scratch,
        &mut obs_chunk[dst_start..dst_end],
        profiler,
    );
}

fn write_maxpooled_rgb_frame_to_last_stack_slot(
    config: VecEnvConfig,
    resize_plan: &AreaResizePlan,
    frame_a: &mut [u8],
    frame_b: &[u8],
    recent_count: usize,
    obs_chunk: &mut [u8],
) {
    let frame_len = frame_len(config);
    let dst_start = (config.frame_stack - 1) * frame_len;
    let dst = &mut obs_chunk[dst_start..dst_start + frame_len];
    if recent_count == 0 {
        return;
    }
    if recent_count == 1 {
        let src = if recent_count % 2 == 1 {
            &frame_a[..]
        } else {
            frame_b
        };
        process_rgb_source_frame(config, resize_plan, src, dst);
        return;
    }
    let source_len = rgb_source_frame_len(config);
    for idx in 0..source_len {
        frame_a[idx] = frame_a[idx].max(frame_b[idx]);
    }
    process_rgb_source_frame(config, resize_plan, &frame_a[..source_len], dst);
}

fn write_current_frame(
    config: VecEnvConfig,
    resize_plan: &AreaResizePlan,
    env: &NesEmulator,
    scratch: &mut [u8],
    dst: &mut [u8],
) {
    if config.uses_default_gray_area_resize() {
        env.write_gray_frame_cropped_area_84x84(dst, scratch);
        return;
    }
    if config.uses_default_gray_mask_top_area_resize() {
        write_default_gray_mask_top_area_frame(config, resize_plan, env, scratch, dst);
        return;
    }

    if config.needs_resize() {
        let native_len = native_frame_len(config);
        let native = &mut scratch[..native_len];
        write_native_frame(config, env, native);
        resize_frame(config, resize_plan, native, dst);
    } else {
        write_native_frame(config, env, dst);
    }
}

fn write_current_frame_profiled(
    config: VecEnvConfig,
    resize_plan: &AreaResizePlan,
    env: &NesEmulator,
    scratch: &mut [u8],
    dst: &mut [u8],
    profiler: &mut Profiler,
) {
    if config.uses_default_gray_area_resize() {
        let start = Instant::now();
        env.write_gray_frame_cropped_area_84x84(dst, scratch);
        profiler.record_render(start.elapsed());
        return;
    }
    if config.uses_default_gray_mask_top_area_resize() {
        let render_start = Instant::now();
        write_default_gray_mask_top_area_frame(config, resize_plan, env, scratch, dst);
        profiler.record_render(render_start.elapsed());
        return;
    }

    if config.needs_resize() {
        let native_len = native_frame_len(config);
        let native = &mut scratch[..native_len];
        let render_start = Instant::now();
        write_native_frame(config, env, native);
        profiler.record_render(render_start.elapsed());
        let resize_start = Instant::now();
        resize_frame(config, resize_plan, native, dst);
        profiler.record_resize(resize_start.elapsed());
    } else {
        let start = Instant::now();
        write_native_frame(config, env, dst);
        profiler.record_render(start.elapsed());
    }
}

fn write_default_gray_mask_top_area_frame(
    config: VecEnvConfig,
    resize_plan: &AreaResizePlan,
    env: &NesEmulator,
    scratch: &mut [u8],
    dst: &mut [u8],
) {
    debug_assert_eq!(resize_plan.src_width, DEFAULT_AREA_SRC_WIDTH);
    debug_assert_eq!(resize_plan.src_height, FULL_AREA_SRC_HEIGHT);
    debug_assert_eq!(resize_plan.dst_width, DEFAULT_AREA_DST_WIDTH);
    debug_assert_eq!(resize_plan.dst_height, DEFAULT_AREA_DST_HEIGHT);
    let native_len = VISIBLE_FRAME_WIDTH * VISIBLE_FRAME_HEIGHT;
    let top_len = config.crop_top * VISIBLE_FRAME_WIDTH;
    let native = &mut scratch[..native_len];
    env.write_gray_visible_mask_lower_frame(
        &mut native[top_len..],
        config.crop_top,
        VISIBLE_FRAME_HEIGHT - config.crop_top,
    );
    resize_full_gray_area_mask_top_32(native, dst, config.crop_fill);
}

fn write_native_frame(config: VecEnvConfig, env: &NesEmulator, dst: &mut [u8]) {
    let crop_top = native_crop_top(config);
    let crop_left = native_crop_left(config);
    let width = config.source_width();
    let height = config.source_height();
    if config.grayscale {
        env.write_gray_visible_frame_region(dst, crop_top, crop_left, width, height);
    } else {
        env.write_rgb_visible_frame_region(dst, crop_top, crop_left, width, height);
    }
    if config.crop_mode == CropMode::Mask {
        mask_native_frame(config, dst);
    }
}

fn write_rgb_source_frame(config: VecEnvConfig, env: &NesEmulator, dst: &mut [u8]) {
    let crop_top = native_crop_top(config);
    let crop_left = native_crop_left(config);
    env.write_rgb_visible_frame_region(
        dst,
        crop_top,
        crop_left,
        config.source_width(),
        config.source_height(),
    );
    if config.crop_mode == CropMode::Mask {
        mask_native_rgb_frame(config, dst);
    }
}

fn process_rgb_source_frame(
    config: VecEnvConfig,
    plan: &AreaResizePlan,
    src: &[u8],
    dst: &mut [u8],
) {
    if config.grayscale {
        if config.needs_resize() {
            resize_gray_from_rgb_source(config, src, dst, plan);
        } else {
            gray_from_rgb_source(src, dst);
        }
    } else if config.needs_resize() {
        resize_frame(config, plan, src, dst);
    } else {
        dst.copy_from_slice(&src[..frame_len(config)]);
    }
}

#[inline]
fn native_crop_top(config: VecEnvConfig) -> usize {
    match config.crop_mode {
        CropMode::Remove => config.crop_top,
        CropMode::Mask => 0,
    }
}

#[inline]
fn native_crop_left(config: VecEnvConfig) -> usize {
    match config.crop_mode {
        CropMode::Remove => config.crop_left,
        CropMode::Mask => 0,
    }
}

fn mask_native_frame(config: VecEnvConfig, dst: &mut [u8]) {
    if config.grayscale {
        mask_native_plane(config, dst, 0);
    } else {
        mask_native_rgb_frame(config, dst);
    }
}

fn mask_native_rgb_frame(config: VecEnvConfig, dst: &mut [u8]) {
    let plane = config.source_width() * config.source_height();
    for channel in 0..RGB_CHANNELS {
        mask_native_plane(config, dst, channel * plane);
    }
}

fn mask_native_plane(config: VecEnvConfig, dst: &mut [u8], offset: usize) {
    let width = config.source_width();
    let height = config.source_height();
    let fill = config.crop_fill;
    let plane = &mut dst[offset..offset + width * height];

    if config.crop_top > 0 {
        plane[..config.crop_top * width].fill(fill);
    }
    if config.crop_bottom > 0 {
        let start = (height - config.crop_bottom) * width;
        plane[start..].fill(fill);
    }
    if config.crop_left > 0 || config.crop_right > 0 {
        for row in plane.chunks_exact_mut(width) {
            if config.crop_left > 0 {
                row[..config.crop_left].fill(fill);
            }
            if config.crop_right > 0 {
                let start = width - config.crop_right;
                row[start..].fill(fill);
            }
        }
    }
}

#[inline]
fn frame_len(config: VecEnvConfig) -> usize {
    if config.grayscale {
        config.obs_width() * config.obs_height()
    } else {
        config.obs_width() * config.obs_height() * RGB_CHANNELS
    }
}

#[inline]
fn native_frame_len(config: VecEnvConfig) -> usize {
    if config.grayscale {
        config.source_width() * config.source_height()
    } else {
        config.source_width() * config.source_height() * RGB_CHANNELS
    }
}

#[inline]
fn rgb_source_frame_len(config: VecEnvConfig) -> usize {
    config.source_width() * config.source_height() * RGB_CHANNELS
}

#[inline]
fn native_scratch_len(config: VecEnvConfig) -> usize {
    if config.needs_resize() {
        native_frame_len(config)
    } else {
        0
    }
}

#[inline]
fn scratch_len(config: VecEnvConfig) -> usize {
    if config.frame_maxpool {
        2 * rgb_source_frame_len(config)
    } else {
        native_scratch_len(config)
    }
}

fn resize_frame_area(config: VecEnvConfig, plan: &AreaResizePlan, src: &[u8], dst: &mut [u8]) {
    debug_assert_eq!(config.resize_algorithm, ResizeAlgorithm::Area);
    if config.grayscale {
        if plan.src_width == DEFAULT_AREA_SRC_WIDTH
            && plan.src_height == DEFAULT_AREA_SRC_HEIGHT
            && plan.dst_width == DEFAULT_AREA_DST_WIDTH
            && plan.dst_height == DEFAULT_AREA_DST_HEIGHT
        {
            resize_default_gray_area(src, dst);
            return;
        }
        if plan.src_width == DEFAULT_AREA_SRC_WIDTH
            && plan.src_height == FULL_AREA_SRC_HEIGHT
            && plan.dst_width == DEFAULT_AREA_DST_WIDTH
            && plan.dst_height == DEFAULT_AREA_DST_HEIGHT
        {
            resize_full_gray_area(src, dst);
            return;
        }
        resize_plane_area(src, dst, plan, 0, 0);
    } else {
        let src_plane = plan.src_width * plan.src_height;
        let dst_plane = plan.dst_width * plan.dst_height;
        for channel in 0..RGB_CHANNELS {
            resize_plane_area(src, dst, plan, channel * src_plane, channel * dst_plane);
        }
    }
}

fn resize_default_gray_area(src: &[u8], dst: &mut [u8]) {
    debug_assert!(src.len() >= DEFAULT_AREA_SRC_WIDTH * DEFAULT_AREA_SRC_HEIGHT);
    debug_assert!(dst.len() >= DEFAULT_AREA_DST_WIDTH * DEFAULT_AREA_DST_HEIGHT);

    for dy in 0..DEFAULT_AREA_DST_HEIGHT {
        let mut sums = [0u16; DEFAULT_AREA_DST_WIDTH];
        for sy in DEFAULT_AREA_Y0[dy]..DEFAULT_AREA_Y1[dy] {
            let row_start = sy * DEFAULT_AREA_SRC_WIDTH;
            accumulate_default_area_row(
                &src[row_start..row_start + DEFAULT_AREA_SRC_WIDTH],
                &mut sums,
            );
        }

        let dst_row = dy * DEFAULT_AREA_DST_WIDTH;
        let y_count = DEFAULT_AREA_Y_COUNT[dy];
        if y_count == 2 {
            for group in 0..12usize {
                let out = dst_row + group * 7;
                let sum = group * 7;
                dst[out] = (sums[sum] / 4) as u8;
                dst[out + 1] = (sums[sum + 1] / 6) as u8;
                dst[out + 2] = (sums[sum + 2] / 6) as u8;
                dst[out + 3] = (sums[sum + 3] / 6) as u8;
                dst[out + 4] = (sums[sum + 4] / 6) as u8;
                dst[out + 5] = (sums[sum + 5] / 6) as u8;
                dst[out + 6] = (sums[sum + 6] / 6) as u8;
            }
        } else {
            for group in 0..12usize {
                let out = dst_row + group * 7;
                let sum = group * 7;
                dst[out] = (sums[sum] / 6) as u8;
                dst[out + 1] = (sums[sum + 1] / 9) as u8;
                dst[out + 2] = (sums[sum + 2] / 9) as u8;
                dst[out + 3] = (sums[sum + 3] / 9) as u8;
                dst[out + 4] = (sums[sum + 4] / 9) as u8;
                dst[out + 5] = (sums[sum + 5] / 9) as u8;
                dst[out + 6] = (sums[sum + 6] / 9) as u8;
            }
        }
    }
}

fn resize_full_gray_area(src: &[u8], dst: &mut [u8]) {
    debug_assert!(src.len() >= DEFAULT_AREA_SRC_WIDTH * FULL_AREA_SRC_HEIGHT);
    debug_assert!(dst.len() >= DEFAULT_AREA_DST_WIDTH * DEFAULT_AREA_DST_HEIGHT);

    for dy in 0..DEFAULT_AREA_DST_HEIGHT {
        let mut sums = [0u16; DEFAULT_AREA_DST_WIDTH];
        for sy in FULL_AREA_Y0[dy]..FULL_AREA_Y1[dy] {
            let row_start = sy * DEFAULT_AREA_SRC_WIDTH;
            accumulate_default_area_row(
                &src[row_start..row_start + DEFAULT_AREA_SRC_WIDTH],
                &mut sums,
            );
        }

        let dst_row = dy * DEFAULT_AREA_DST_WIDTH;
        write_default_area_sum_row(&sums, FULL_AREA_Y_COUNT[dy], &mut dst[dst_row..]);
    }
}

fn resize_full_gray_area_mask_top_32(src: &[u8], dst: &mut [u8], fill: u8) {
    debug_assert!(src.len() >= DEFAULT_AREA_SRC_WIDTH * FULL_AREA_SRC_HEIGHT);
    debug_assert!(dst.len() >= DEFAULT_AREA_DST_WIDTH * DEFAULT_AREA_DST_HEIGHT);

    for dy in 0..DEFAULT_AREA_DST_HEIGHT {
        let y0 = FULL_AREA_Y0[dy];
        let y1 = FULL_AREA_Y1[dy];
        let mut sums = [0u16; DEFAULT_AREA_DST_WIDTH];
        if y0 < DEFAULT_MASK_CROP_TOP && fill != 0 {
            let masked_rows = y1.min(DEFAULT_MASK_CROP_TOP) - y0;
            add_default_area_fill_rows(fill, masked_rows as u16, &mut sums);
        }
        for sy in y0.max(DEFAULT_MASK_CROP_TOP)..y1 {
            let row_start = sy * DEFAULT_AREA_SRC_WIDTH;
            accumulate_default_area_row(
                &src[row_start..row_start + DEFAULT_AREA_SRC_WIDTH],
                &mut sums,
            );
        }

        let dst_row = dy * DEFAULT_AREA_DST_WIDTH;
        write_default_area_sum_row(&sums, FULL_AREA_Y_COUNT[dy], &mut dst[dst_row..]);
    }
}

fn write_default_area_sum_row(sums: &[u16; DEFAULT_AREA_DST_WIDTH], y_count: u16, dst: &mut [u8]) {
    debug_assert!(dst.len() >= DEFAULT_AREA_DST_WIDTH);
    if y_count == 2 {
        for group in 0..12usize {
            let out = group * 7;
            let sum = group * 7;
            dst[out] = (sums[sum] / 4) as u8;
            dst[out + 1] = (sums[sum + 1] / 6) as u8;
            dst[out + 2] = (sums[sum + 2] / 6) as u8;
            dst[out + 3] = (sums[sum + 3] / 6) as u8;
            dst[out + 4] = (sums[sum + 4] / 6) as u8;
            dst[out + 5] = (sums[sum + 5] / 6) as u8;
            dst[out + 6] = (sums[sum + 6] / 6) as u8;
        }
    } else {
        for group in 0..12usize {
            let out = group * 7;
            let sum = group * 7;
            dst[out] = (sums[sum] / 6) as u8;
            dst[out + 1] = (sums[sum + 1] / 9) as u8;
            dst[out + 2] = (sums[sum + 2] / 9) as u8;
            dst[out + 3] = (sums[sum + 3] / 9) as u8;
            dst[out + 4] = (sums[sum + 4] / 9) as u8;
            dst[out + 5] = (sums[sum + 5] / 9) as u8;
            dst[out + 6] = (sums[sum + 6] / 9) as u8;
        }
    }
}

fn add_default_area_fill_rows(fill: u8, row_count: u16, sums: &mut [u16; DEFAULT_AREA_DST_WIDTH]) {
    let fill = u16::from(fill);
    for group in 0..12usize {
        let dst = group * 7;
        sums[dst] += fill * row_count * 2;
        sums[dst + 1] += fill * row_count * 3;
        sums[dst + 2] += fill * row_count * 3;
        sums[dst + 3] += fill * row_count * 3;
        sums[dst + 4] += fill * row_count * 3;
        sums[dst + 5] += fill * row_count * 3;
        sums[dst + 6] += fill * row_count * 3;
    }
}

#[inline(always)]
fn accumulate_default_area_row(row: &[u8], sums: &mut [u16; DEFAULT_AREA_DST_WIDTH]) {
    debug_assert!(row.len() >= DEFAULT_AREA_SRC_WIDTH);
    for group in 0..12usize {
        let src = group * 20;
        let dst = group * 7;
        // SAFETY: callers pass exactly one 240-pixel default source row; each
        // 20-pixel group maps into seven bins and the loop covers 12 groups.
        unsafe {
            sums[dst] += *row.get_unchecked(src) as u16 + *row.get_unchecked(src + 1) as u16;
            sums[dst + 1] += *row.get_unchecked(src + 2) as u16
                + *row.get_unchecked(src + 3) as u16
                + *row.get_unchecked(src + 4) as u16;
            sums[dst + 2] += *row.get_unchecked(src + 5) as u16
                + *row.get_unchecked(src + 6) as u16
                + *row.get_unchecked(src + 7) as u16;
            sums[dst + 3] += *row.get_unchecked(src + 8) as u16
                + *row.get_unchecked(src + 9) as u16
                + *row.get_unchecked(src + 10) as u16;
            sums[dst + 4] += *row.get_unchecked(src + 11) as u16
                + *row.get_unchecked(src + 12) as u16
                + *row.get_unchecked(src + 13) as u16;
            sums[dst + 5] += *row.get_unchecked(src + 14) as u16
                + *row.get_unchecked(src + 15) as u16
                + *row.get_unchecked(src + 16) as u16;
            sums[dst + 6] += *row.get_unchecked(src + 17) as u16
                + *row.get_unchecked(src + 18) as u16
                + *row.get_unchecked(src + 19) as u16;
        }
    }
}

fn resize_frame(config: VecEnvConfig, plan: &AreaResizePlan, src: &[u8], dst: &mut [u8]) {
    match config.resize_algorithm {
        ResizeAlgorithm::Area => resize_frame_area(config, plan, src, dst),
        ResizeAlgorithm::Nearest => resize_frame_nearest(config, plan, src, dst),
        ResizeAlgorithm::Bilinear => resize_frame_bilinear(config, plan, src, dst),
    }
}

fn gray_from_rgb_source(src: &[u8], dst: &mut [u8]) {
    let plane = dst.len();
    for idx in 0..plane {
        dst[idx] = rgb_to_gray(src[idx], src[plane + idx], src[2 * plane + idx]);
    }
}

fn resize_gray_from_rgb_source(
    config: VecEnvConfig,
    src: &[u8],
    dst: &mut [u8],
    plan: &AreaResizePlan,
) {
    match config.resize_algorithm {
        ResizeAlgorithm::Area => resize_gray_from_rgb_source_area(src, dst, plan),
        ResizeAlgorithm::Nearest => resize_gray_from_rgb_source_nearest(src, dst, plan),
        ResizeAlgorithm::Bilinear => resize_gray_from_rgb_source_bilinear(src, dst, plan),
    }
}

fn resize_gray_from_rgb_source_area(src: &[u8], dst: &mut [u8], plan: &AreaResizePlan) {
    let src_plane = plan.src_width * plan.src_height;
    for (dst_idx, bin) in plan.bins.iter().enumerate() {
        let mut sum = 0u32;
        for sy in bin.y0..bin.y1 {
            let src_row = sy * plan.src_width;
            for sx in bin.x0..bin.x1 {
                let idx = src_row + sx;
                sum += rgb_to_gray(src[idx], src[src_plane + idx], src[2 * src_plane + idx]) as u32;
            }
        }
        dst[dst_idx] = (sum / bin.count) as u8;
    }
}

#[inline]
fn rgb_to_gray(r: u8, g: u8, b: u8) -> u8 {
    (((r as u32) * 77 + (g as u32) * 150 + (b as u32) * 29 + 128) >> 8) as u8
}

fn resize_frame_nearest(config: VecEnvConfig, plan: &AreaResizePlan, src: &[u8], dst: &mut [u8]) {
    let channels = if config.grayscale { 1 } else { RGB_CHANNELS };
    for channel in 0..channels {
        resize_plane_nearest(
            src,
            dst,
            plan,
            channel * plan.src_width * plan.src_height,
            channel * plan.dst_width * plan.dst_height,
        );
    }
}

fn resize_frame_bilinear(config: VecEnvConfig, plan: &AreaResizePlan, src: &[u8], dst: &mut [u8]) {
    let channels = if config.grayscale { 1 } else { RGB_CHANNELS };
    for channel in 0..channels {
        resize_plane_bilinear(
            src,
            dst,
            plan,
            channel * plan.src_width * plan.src_height,
            channel * plan.dst_width * plan.dst_height,
        );
    }
}

fn resize_gray_from_rgb_source_nearest(src: &[u8], dst: &mut [u8], plan: &AreaResizePlan) {
    let src_plane = plan.src_width * plan.src_height;
    for dy in 0..plan.dst_height {
        let sy = (dy * plan.src_height / plan.dst_height).min(plan.src_height - 1);
        for dx in 0..plan.dst_width {
            let sx = (dx * plan.src_width / plan.dst_width).min(plan.src_width - 1);
            let src_idx = sy * plan.src_width + sx;
            let dst_idx = dy * plan.dst_width + dx;
            dst[dst_idx] = rgb_to_gray(
                src[src_idx],
                src[src_plane + src_idx],
                src[2 * src_plane + src_idx],
            );
        }
    }
}

fn resize_gray_from_rgb_source_bilinear(src: &[u8], dst: &mut [u8], plan: &AreaResizePlan) {
    let src_plane = plan.src_width * plan.src_height;
    for dy in 0..plan.dst_height {
        let (y0, y1, wy) = bilinear_axis(dy, plan.dst_height, plan.src_height);
        for dx in 0..plan.dst_width {
            let (x0, x1, wx) = bilinear_axis(dx, plan.dst_width, plan.src_width);
            let top = rgb_gray_lerp(src, src_plane, y0, x0, x1, plan.src_width, wx);
            let bottom = rgb_gray_lerp(src, src_plane, y1, x0, x1, plan.src_width, wx);
            dst[dy * plan.dst_width + dx] = round_u8(top * (1.0 - wy) + bottom * wy);
        }
    }
}

fn rgb_gray_lerp(
    src: &[u8],
    src_plane: usize,
    y: usize,
    x0: usize,
    x1: usize,
    width: usize,
    wx: f32,
) -> f32 {
    let left_idx = y * width + x0;
    let right_idx = y * width + x1;
    let left = rgb_to_gray(
        src[left_idx],
        src[src_plane + left_idx],
        src[2 * src_plane + left_idx],
    ) as f32;
    let right = rgb_to_gray(
        src[right_idx],
        src[src_plane + right_idx],
        src[2 * src_plane + right_idx],
    ) as f32;
    left * (1.0 - wx) + right * wx
}

fn resize_plane_area(
    src: &[u8],
    dst: &mut [u8],
    plan: &AreaResizePlan,
    src_offset: usize,
    dst_offset: usize,
) {
    debug_assert!(src.len() >= src_offset + plan.src_width * plan.src_height);
    debug_assert!(dst.len() >= dst_offset + plan.dst_width * plan.dst_height);

    for (dst_i, bin) in plan.bins.iter().enumerate() {
        let mut sum = 0u32;
        for sy in bin.y0..bin.y1 {
            let src_row = src_offset + sy * plan.src_width;
            for sx in bin.x0..bin.x1 {
                // SAFETY: AreaResizePlan bins are built from dimensions validated above.
                sum += unsafe { *src.get_unchecked(src_row + sx) } as u32;
            }
        }
        // SAFETY: dst_i iterates over exactly dst_width * dst_height planned pixels.
        unsafe {
            *dst.get_unchecked_mut(dst_offset + dst_i) = (sum / bin.count) as u8;
        }
    }
}

fn resize_plane_nearest(
    src: &[u8],
    dst: &mut [u8],
    plan: &AreaResizePlan,
    src_offset: usize,
    dst_offset: usize,
) {
    debug_assert!(src.len() >= src_offset + plan.src_width * plan.src_height);
    debug_assert!(dst.len() >= dst_offset + plan.dst_width * plan.dst_height);

    for dy in 0..plan.dst_height {
        let sy = (dy * plan.src_height / plan.dst_height).min(plan.src_height - 1);
        for dx in 0..plan.dst_width {
            let sx = (dx * plan.src_width / plan.dst_width).min(plan.src_width - 1);
            dst[dst_offset + dy * plan.dst_width + dx] = src[src_offset + sy * plan.src_width + sx];
        }
    }
}

fn resize_plane_bilinear(
    src: &[u8],
    dst: &mut [u8],
    plan: &AreaResizePlan,
    src_offset: usize,
    dst_offset: usize,
) {
    debug_assert!(src.len() >= src_offset + plan.src_width * plan.src_height);
    debug_assert!(dst.len() >= dst_offset + plan.dst_width * plan.dst_height);

    for dy in 0..plan.dst_height {
        let (y0, y1, wy) = bilinear_axis(dy, plan.dst_height, plan.src_height);
        for dx in 0..plan.dst_width {
            let (x0, x1, wx) = bilinear_axis(dx, plan.dst_width, plan.src_width);
            let top_left = src[src_offset + y0 * plan.src_width + x0] as f32;
            let top_right = src[src_offset + y0 * plan.src_width + x1] as f32;
            let bottom_left = src[src_offset + y1 * plan.src_width + x0] as f32;
            let bottom_right = src[src_offset + y1 * plan.src_width + x1] as f32;
            let top = top_left * (1.0 - wx) + top_right * wx;
            let bottom = bottom_left * (1.0 - wx) + bottom_right * wx;
            dst[dst_offset + dy * plan.dst_width + dx] = round_u8(top * (1.0 - wy) + bottom * wy);
        }
    }
}

fn bilinear_axis(dst_index: usize, dst_len: usize, src_len: usize) -> (usize, usize, f32) {
    if dst_len == 1 {
        return (0, 0, 0.0);
    }
    let pos = (dst_index as f32) * ((src_len - 1) as f32) / ((dst_len - 1) as f32);
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(src_len - 1);
    (lo, hi, pos - lo as f32)
}

fn round_u8(value: f32) -> u8 {
    value.clamp(0.0, 255.0).round() as u8
}

struct AreaResizePlan {
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
    bins: Vec<AreaResizeBin>,
}

impl AreaResizePlan {
    fn new(src_width: usize, src_height: usize, dst_width: usize, dst_height: usize) -> Self {
        let mut bins = Vec::with_capacity(dst_width * dst_height);
        for dy in 0..dst_height {
            let y0 = (dy * src_height) / dst_height;
            let y1 = (((dy + 1) * src_height) / dst_height)
                .max(y0 + 1)
                .min(src_height);
            for dx in 0..dst_width {
                let x0 = (dx * src_width) / dst_width;
                let x1 = (((dx + 1) * src_width) / dst_width)
                    .max(x0 + 1)
                    .min(src_width);
                bins.push(AreaResizeBin {
                    x0,
                    x1,
                    y0,
                    y1,
                    count: ((x1 - x0) * (y1 - y0)) as u32,
                });
            }
        }
        Self {
            src_width,
            src_height,
            dst_width,
            dst_height,
            bins,
        }
    }
}

struct AreaResizeBin {
    x0: usize,
    x1: usize,
    y0: usize,
    y1: usize,
    count: u32,
}

const fn build_default_area_axis_start(src_len: usize) -> [usize; DEFAULT_AREA_DST_WIDTH] {
    let mut out = [0usize; DEFAULT_AREA_DST_WIDTH];
    let mut idx = 0usize;
    while idx < DEFAULT_AREA_DST_WIDTH {
        out[idx] = (idx * src_len) / DEFAULT_AREA_DST_WIDTH;
        idx += 1;
    }
    out
}

const fn build_default_area_axis_end(src_len: usize) -> [usize; DEFAULT_AREA_DST_WIDTH] {
    let mut out = [0usize; DEFAULT_AREA_DST_WIDTH];
    let mut idx = 0usize;
    while idx < DEFAULT_AREA_DST_WIDTH {
        let start = (idx * src_len) / DEFAULT_AREA_DST_WIDTH;
        let mut end = ((idx + 1) * src_len) / DEFAULT_AREA_DST_WIDTH;
        if end < start + 1 {
            end = start + 1;
        }
        if end > src_len {
            end = src_len;
        }
        out[idx] = end;
        idx += 1;
    }
    out
}

const fn build_default_area_axis_count(src_len: usize) -> [u16; DEFAULT_AREA_DST_WIDTH] {
    let mut out = [0u16; DEFAULT_AREA_DST_WIDTH];
    let mut idx = 0usize;
    while idx < DEFAULT_AREA_DST_WIDTH {
        let start = (idx * src_len) / DEFAULT_AREA_DST_WIDTH;
        let mut end = ((idx + 1) * src_len) / DEFAULT_AREA_DST_WIDTH;
        if end < start + 1 {
            end = start + 1;
        }
        if end > src_len {
            end = src_len;
        }
        out[idx] = (end - start) as u16;
        idx += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_render_test_env() -> NesEmulator {
        let prg_rom = vec![0; 32768];
        let chr_rom = (0..8192)
            .map(|idx| ((idx * 17 + idx / 13 + 29) & 0xff) as u8)
            .collect::<Vec<_>>();
        NesEmulator::new_with_options(
            Cartridge {
                prg_rom,
                chr_rom,
                vertical_mirroring: true,
            },
            true,
        )
    }

    #[test]
    fn sticky_actions_replay_the_exact_raw_controller_state() {
        let config = VecEnvConfig {
            num_envs: 1,
            frame_skip: 1,
            grayscale: true,
            frame_stack: 1,
            frame_maxpool: false,
            noop_reset_max: 0,
            sticky_action_prob: 1.0,
            terminate_on_flag: true,
            crop_top: 0,
            crop_bottom: 0,
            crop_left: 0,
            crop_right: 0,
            crop_mode: CropMode::Remove,
            crop_fill: 0,
            resize_width: VISIBLE_FRAME_WIDTH,
            resize_height: VISIBLE_FRAME_HEIGHT,
            resize_algorithm: ResizeAlgorithm::Nearest,
        };
        let cart = Cartridge {
            prg_rom: vec![0; 32768],
            chr_rom: vec![0; 8192],
            vertical_mirroring: true,
        };
        let mut env = MarioVecEnv::new(cart, config, Vec::new(), false, 1).unwrap();
        let previous_controller_state = 0b1011_0111;
        env.last_actions[0] = previous_controller_state;

        let effective = env.effective_actions(&[0b0100_1000]);

        assert_eq!(effective, vec![previous_controller_state]);
        assert_eq!(env.last_actions, vec![previous_controller_state]);
    }

    fn reference_resize_plane_area(
        src: &[u8],
        dst: &mut [u8],
        src_width: usize,
        src_height: usize,
        dst_width: usize,
        dst_height: usize,
        src_offset: usize,
        dst_offset: usize,
    ) {
        let plan = AreaResizePlan::new(src_width, src_height, dst_width, dst_height);
        for (dst_i, bin) in plan.bins.iter().enumerate() {
            let mut sum = 0u32;
            for sy in bin.y0..bin.y1 {
                let src_row = src_offset + sy * src_width;
                for sx in bin.x0..bin.x1 {
                    sum += src[src_row + sx] as u32;
                }
            }
            dst[dst_offset + dst_i] = (sum / bin.count) as u8;
        }
    }

    #[test]
    fn precomputed_area_resize_matches_reference_default_grayscale() {
        let config = VecEnvConfig {
            num_envs: 16,
            frame_skip: 4,
            grayscale: true,
            frame_stack: 4,
            frame_maxpool: false,
            noop_reset_max: 0,
            sticky_action_prob: 0.0,
            terminate_on_flag: true,
            crop_top: 32,
            crop_bottom: 0,
            crop_left: 0,
            crop_right: 0,
            crop_mode: CropMode::Remove,
            crop_fill: 0,
            resize_width: 84,
            resize_height: 84,
            resize_algorithm: ResizeAlgorithm::Area,
        };
        let plan = AreaResizePlan::new(config.source_width(), config.source_height(), 84, 84);
        let src_len = config.source_width() * config.source_height();
        let src = (0..src_len)
            .map(|idx| ((idx * 37 + idx / 251 + 19) & 0xff) as u8)
            .collect::<Vec<_>>();
        let mut optimized = vec![0; 84 * 84];
        let mut reference = vec![0; 84 * 84];

        resize_frame_area(config, &plan, &src, &mut optimized);
        reference_resize_plane_area(
            &src,
            &mut reference,
            config.source_width(),
            config.source_height(),
            84,
            84,
            0,
            0,
        );

        assert_eq!(optimized, reference);
    }

    #[test]
    fn precomputed_area_resize_matches_reference_full_mask_grayscale() {
        let config = VecEnvConfig {
            num_envs: 16,
            frame_skip: 4,
            grayscale: true,
            frame_stack: 4,
            frame_maxpool: false,
            noop_reset_max: 0,
            sticky_action_prob: 0.0,
            terminate_on_flag: true,
            crop_top: 32,
            crop_bottom: 0,
            crop_left: 0,
            crop_right: 0,
            crop_mode: CropMode::Mask,
            crop_fill: 7,
            resize_width: 84,
            resize_height: 84,
            resize_algorithm: ResizeAlgorithm::Area,
        };
        let plan = AreaResizePlan::new(config.source_width(), config.source_height(), 84, 84);
        let src_len = config.source_width() * config.source_height();
        let mut src = (0..src_len)
            .map(|idx| ((idx * 41 + idx / 199 + 23) & 0xff) as u8)
            .collect::<Vec<_>>();
        mask_native_frame(config, &mut src);
        let mut optimized = vec![0; 84 * 84];
        let mut reference = vec![0; 84 * 84];

        resize_frame_area(config, &plan, &src, &mut optimized);
        reference_resize_plane_area(
            &src,
            &mut reference,
            config.source_width(),
            config.source_height(),
            84,
            84,
            0,
            0,
        );

        assert_eq!(optimized, reference);
    }

    #[test]
    fn default_gray_mask_top_area_frame_matches_legacy_path() {
        let config = VecEnvConfig {
            num_envs: 16,
            frame_skip: 4,
            grayscale: true,
            frame_stack: 4,
            frame_maxpool: false,
            noop_reset_max: 0,
            sticky_action_prob: 0.0,
            terminate_on_flag: true,
            crop_top: 32,
            crop_bottom: 0,
            crop_left: 0,
            crop_right: 0,
            crop_mode: CropMode::Mask,
            crop_fill: 7,
            resize_width: 84,
            resize_height: 84,
            resize_algorithm: ResizeAlgorithm::Area,
        };
        let resize_plan = AreaResizePlan::new(
            config.source_width(),
            config.source_height(),
            config.resize_width,
            config.resize_height,
        );
        let env = make_render_test_env();
        let mut native = vec![0; native_frame_len(config)];
        let mut scratch = vec![0; native_frame_len(config)];
        let mut expected = vec![0; frame_len(config)];
        let mut actual = vec![0; frame_len(config)];

        assert!(config.uses_default_gray_mask_top_area_resize());
        write_native_frame(config, &env, &mut native);
        resize_frame(config, &resize_plan, &native, &mut expected);
        write_default_gray_mask_top_area_frame(
            config,
            &resize_plan,
            &env,
            &mut scratch,
            &mut actual,
        );

        assert_eq!(actual, expected);
    }

    #[test]
    fn precomputed_area_resize_matches_reference_rgb_planes() {
        let src_width = VISIBLE_FRAME_WIDTH;
        let src_height = VISIBLE_FRAME_HEIGHT - 32;
        let dst_width = 84;
        let dst_height = 84;
        let config = VecEnvConfig {
            num_envs: 1,
            frame_skip: 4,
            grayscale: false,
            frame_stack: 1,
            frame_maxpool: false,
            noop_reset_max: 0,
            sticky_action_prob: 0.0,
            terminate_on_flag: true,
            crop_top: 32,
            crop_bottom: 0,
            crop_left: 0,
            crop_right: 0,
            crop_mode: CropMode::Remove,
            crop_fill: 0,
            resize_width: dst_width,
            resize_height: dst_height,
            resize_algorithm: ResizeAlgorithm::Area,
        };
        let plan = AreaResizePlan::new(src_width, src_height, dst_width, dst_height);
        let src_plane = src_width * src_height;
        let dst_plane = dst_width * dst_height;
        let src = (0..src_plane * RGB_CHANNELS)
            .map(|idx| ((idx * 17 + idx / 97 + 31) & 0xff) as u8)
            .collect::<Vec<_>>();
        let mut optimized = vec![0; dst_plane * RGB_CHANNELS];
        let mut reference = vec![0; dst_plane * RGB_CHANNELS];

        resize_frame_area(config, &plan, &src, &mut optimized);
        for channel in 0..RGB_CHANNELS {
            reference_resize_plane_area(
                &src,
                &mut reference,
                src_width,
                src_height,
                dst_width,
                dst_height,
                channel * src_plane,
                channel * dst_plane,
            );
        }

        assert_eq!(optimized, reference);
    }

    #[test]
    fn mask_native_frame_fills_crop_margins_without_changing_visible_center() {
        let config = VecEnvConfig {
            num_envs: 1,
            frame_skip: 4,
            grayscale: true,
            frame_stack: 1,
            frame_maxpool: false,
            noop_reset_max: 0,
            sticky_action_prob: 0.0,
            terminate_on_flag: true,
            crop_top: 3,
            crop_bottom: 5,
            crop_left: 7,
            crop_right: 11,
            crop_mode: CropMode::Mask,
            crop_fill: 42,
            resize_width: VISIBLE_FRAME_WIDTH,
            resize_height: VISIBLE_FRAME_HEIGHT,
            resize_algorithm: ResizeAlgorithm::Area,
        };
        let width = config.source_width();
        let height = config.source_height();
        let mut frame = (0..width * height)
            .map(|idx| (idx % 251) as u8)
            .collect::<Vec<_>>();
        let center_idx = config.crop_top * width + config.crop_left;
        let center_before = frame[center_idx];

        mask_native_frame(config, &mut frame);

        assert_eq!(config.source_width(), VISIBLE_FRAME_WIDTH);
        assert_eq!(config.source_height(), VISIBLE_FRAME_HEIGHT);
        for y in 0..height {
            for x in 0..width {
                let pixel = frame[y * width + x];
                let masked = y < config.crop_top
                    || y >= height - config.crop_bottom
                    || x < config.crop_left
                    || x >= width - config.crop_right;
                if masked {
                    assert_eq!(pixel, config.crop_fill);
                }
            }
        }
        assert_eq!(frame[center_idx], center_before);
    }

    #[test]
    fn remove_crop_source_geometry_accounts_for_all_sides() {
        let config = VecEnvConfig {
            num_envs: 1,
            frame_skip: 4,
            grayscale: true,
            frame_stack: 1,
            frame_maxpool: false,
            noop_reset_max: 0,
            sticky_action_prob: 0.0,
            terminate_on_flag: true,
            crop_top: 32,
            crop_bottom: 4,
            crop_left: 10,
            crop_right: 6,
            crop_mode: CropMode::Remove,
            crop_fill: 0,
            resize_width: VISIBLE_FRAME_WIDTH - 16,
            resize_height: VISIBLE_FRAME_HEIGHT - 36,
            resize_algorithm: ResizeAlgorithm::Area,
        };

        assert_eq!(config.source_width(), VISIBLE_FRAME_WIDTH - 16);
        assert_eq!(config.source_height(), VISIBLE_FRAME_HEIGHT - 36);
        assert!(!config.needs_resize());
    }
}
