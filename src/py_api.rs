// PyO3's generated return conversion code triggers false-positive
// useless-conversion warnings for PyResult methods.
#![allow(clippy::useless_conversion)]

use numpy::{PyReadonlyArray1, PyReadwriteArray1, PyReadwriteArray4, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyModule;
use rayon::{ThreadPool, ThreadPoolBuilder};
use std::thread;

use crate::cartridge::Cartridge;
use crate::emulator::{
    NES_HEIGHT, NES_WIDTH, RGB_CHANNELS, VISIBLE_FRAME_HEIGHT, VISIBLE_FRAME_WIDTH,
};
use crate::vec_env::{CropMode, InitialState, MarioVecEnv, ResizeAlgorithm, VecEnvConfig};
use smb_turbo_driver::EXPECTED_SMB_ROM_SHA256;

#[pyclass(name = "_RetroVecEnv")]
pub struct RetroVecEnv {
    inner: MarioVecEnv,
    thread_pool: Option<ThreadPool>,
    num_threads: usize,
}

// PyO3 exposes wide buffer-oriented methods directly to Python.
#[allow(clippy::too_many_arguments)]
#[pymethods]
impl RetroVecEnv {
    #[new]
    #[pyo3(signature = (rom_path, num_envs, frame_skip=4, grayscale=true, frame_stack=4, terminate_on_flag=true, crop_top=0, crop_bottom=0, resize_width=84, resize_height=84, initial_states=None, initial_state_names=None, initial_state_weights=None, seed=0, frame_maxpool=false, noop_reset_max=0, sticky_action_prob=0.0, crop_left=0, crop_right=0, crop_mode="remove", crop_fill=0, resize_algorithm="area", num_threads=None))]
    pub fn new(
        rom_path: String,
        num_envs: usize,
        frame_skip: usize,
        grayscale: bool,
        frame_stack: usize,
        terminate_on_flag: bool,
        crop_top: usize,
        crop_bottom: usize,
        resize_width: usize,
        resize_height: usize,
        initial_states: Option<Vec<Vec<u8>>>,
        initial_state_names: Option<Vec<String>>,
        initial_state_weights: Option<Vec<f64>>,
        seed: u64,
        frame_maxpool: bool,
        noop_reset_max: isize,
        sticky_action_prob: f64,
        crop_left: usize,
        crop_right: usize,
        crop_mode: &str,
        crop_fill: u8,
        resize_algorithm: &str,
        num_threads: Option<isize>,
    ) -> PyResult<Self> {
        if num_envs == 0 {
            return Err(PyValueError::new_err("num_envs must be > 0"));
        }
        if frame_skip == 0 {
            return Err(PyValueError::new_err("frame_skip must be > 0"));
        }
        if frame_stack == 0 {
            return Err(PyValueError::new_err("frame_stack must be > 0"));
        }
        if crop_top + crop_bottom >= VISIBLE_FRAME_HEIGHT {
            return Err(PyValueError::new_err(format!(
                "crop_top + crop_bottom must be less than {VISIBLE_FRAME_HEIGHT}, got {}",
                crop_top + crop_bottom
            )));
        }
        if crop_left + crop_right >= VISIBLE_FRAME_WIDTH {
            return Err(PyValueError::new_err(format!(
                "crop_left + crop_right must be less than {VISIBLE_FRAME_WIDTH}, got {}",
                crop_left + crop_right
            )));
        }
        if resize_width == 0 || resize_height == 0 {
            return Err(PyValueError::new_err(
                "resize_width and resize_height must be > 0",
            ));
        }
        if noop_reset_max < 0 {
            return Err(PyValueError::new_err("noop_reset_max must be non-negative"));
        }
        if !(0.0..=1.0).contains(&sticky_action_prob) {
            return Err(PyValueError::new_err(
                "sticky_action_prob must be between 0.0 and 1.0",
            ));
        }
        if matches!(num_threads, Some(value) if value <= 0) {
            return Err(PyValueError::new_err("num_threads must be > 0"));
        }
        let crop_mode = build_crop_mode(crop_mode)?;
        let resize_algorithm = build_resize_algorithm(resize_algorithm)?;
        let available_threads = thread::available_parallelism()
            .map(|count| count.get())
            .unwrap_or(1);
        let (effective_num_threads, parallel_env_threshold, thread_pool) =
            if let Some(requested) = num_threads {
                let effective =
                    effective_private_num_threads(requested as usize, num_envs, available_threads);
                let pool = if effective > 1 {
                    Some(
                        ThreadPoolBuilder::new()
                            .num_threads(effective)
                            .build()
                            .map_err(|err| {
                                PyRuntimeError::new_err(format!(
                                    "failed to create Rayon thread pool: {err}"
                                ))
                            })?,
                    )
                } else {
                    None
                };
                (effective, 2, pool)
            } else {
                (
                    rayon::current_num_threads().min(num_envs).max(1),
                    crate::vec_env::PARALLEL_ENV_THRESHOLD,
                    None,
                )
            };

        let cart = Cartridge::load_ines_for(rom_path, EXPECTED_SMB_ROM_SHA256)
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
        let (initial_states, weighted_initial_states) = build_initial_states(
            initial_states.unwrap_or_default(),
            initial_state_names.unwrap_or_default(),
            initial_state_weights,
            num_envs,
        )?;
        let config = VecEnvConfig {
            num_envs,
            num_threads: effective_num_threads,
            parallel_env_threshold,
            frame_skip,
            grayscale,
            frame_stack,
            frame_maxpool,
            terminate_on_flag,
            crop_top,
            crop_bottom,
            crop_left,
            crop_right,
            crop_mode,
            crop_fill,
            resize_width,
            resize_height,
            resize_algorithm,
            noop_reset_max: noop_reset_max as usize,
            sticky_action_prob,
        };
        Ok(Self {
            inner: MarioVecEnv::new(cart, config, initial_states, weighted_initial_states, seed)
                .map_err(|err| PyValueError::new_err(err.to_string()))?,
            thread_pool,
            num_threads: effective_num_threads,
        })
    }

    #[getter]
    pub fn num_envs(&self) -> usize {
        self.inner.config().num_envs
    }

    #[getter]
    pub fn num_threads(&self) -> usize {
        self.num_threads
    }

    #[getter]
    pub fn frame_skip(&self) -> usize {
        self.inner.config().frame_skip
    }

    #[getter]
    pub fn grayscale(&self) -> bool {
        self.inner.config().grayscale
    }

    #[getter]
    pub fn frame_stack(&self) -> usize {
        self.inner.config().frame_stack
    }

    #[getter]
    pub fn frame_maxpool(&self) -> bool {
        self.inner.config().frame_maxpool
    }

    #[getter]
    pub fn noop_reset_max(&self) -> usize {
        self.inner.config().noop_reset_max
    }

    #[getter]
    pub fn sticky_action_prob(&self) -> f64 {
        self.inner.config().sticky_action_prob
    }

    #[getter]
    pub fn crop_top(&self) -> usize {
        self.inner.config().crop_top
    }

    #[getter]
    pub fn crop_bottom(&self) -> usize {
        self.inner.config().crop_bottom
    }

    #[getter]
    pub fn resize_width(&self) -> usize {
        self.inner.config().resize_width
    }

    #[getter]
    pub fn resize_height(&self) -> usize {
        self.inner.config().resize_height
    }

    pub fn obs_shape(&self) -> (usize, usize, usize, usize) {
        (
            self.inner.config().num_envs,
            self.inner.config().channels(),
            self.inner.config().obs_height(),
            self.inner.config().obs_width(),
        )
    }

    #[getter]
    pub fn initial_state_names(&self) -> Vec<String> {
        self.inner.initial_state_names()
    }

    pub fn initial_state_policy_names(&self) -> Vec<String> {
        self.inner.initial_state_policy_names()
    }

    pub fn initial_state_weights(&self) -> Vec<f64> {
        self.inner.initial_state_weights()
    }

    pub fn active_state_indices(&self) -> Vec<i32> {
        self.inner.active_state_indices().to_vec()
    }

    pub fn rgb_frame_shape(&self) -> (usize, usize, usize, usize) {
        (
            self.inner.config().num_envs,
            VISIBLE_FRAME_HEIGHT,
            VISIBLE_FRAME_WIDTH,
            RGB_CHANNELS,
        )
    }

    pub fn rgb_frames_into<'py>(
        &self,
        py: Python<'py>,
        mut frames: PyReadwriteArray4<'py, u8>,
    ) -> PyResult<()> {
        self.validate_rgb_frame_shape(&frames)?;
        let mut frames_rw = frames.as_array_mut();
        let frames_slice = frames_rw
            .as_slice_mut()
            .ok_or_else(|| PyValueError::new_err("frames must be C-contiguous"))?;
        py.allow_threads(|| {
            self.inner.rgb_frames_hwc_into(frames_slice);
        });
        Ok(())
    }

    pub fn seed(&mut self, seed: u64) {
        self.inner.seed(seed);
    }

    pub fn enable_profiler(&mut self) {
        self.inner.enable_profiler();
    }

    pub fn reset_profiler(&mut self) {
        self.inner.reset_profiler();
    }

    pub fn disable_profiler(&mut self) {
        self.inner.disable_profiler();
    }

    #[pyo3(signature = (top_n=64))]
    pub fn profiler_snapshot(&self, top_n: usize) -> PyResult<String> {
        self.inner
            .profiler_snapshot_json(top_n)
            .ok_or_else(|| PyValueError::new_err("profiler is not enabled"))
    }

    pub fn _debug_ram(&self, env_idx: usize) -> PyResult<Vec<u8>> {
        self.inner
            .env_ram(env_idx)
            .map(|ram| ram.to_vec())
            .ok_or_else(|| PyValueError::new_err(format!("env_idx out of range: {env_idx}")))
    }

    pub fn _debug_oam(&self, env_idx: usize) -> PyResult<Vec<u8>> {
        self.inner
            .env_oam(env_idx)
            .map(|oam| oam.to_vec())
            .ok_or_else(|| PyValueError::new_err(format!("env_idx out of range: {env_idx}")))
    }

    pub fn _debug_bg_pixel(&self, env_idx: usize, x: usize, y: usize) -> PyResult<(u8, bool)> {
        self.inner
            .env_bg_pixel(env_idx, x, y)
            .ok_or_else(|| PyValueError::new_err(format!("env_idx out of range: {env_idx}")))
    }

    pub fn reset_into<'py>(
        &mut self,
        py: Python<'py>,
        mut obs: PyReadwriteArray4<'py, u8>,
    ) -> PyResult<()> {
        self.validate_obs_shape(&obs)?;
        let mut obs_rw = obs.as_array_mut();
        let obs_slice = obs_rw
            .as_slice_mut()
            .ok_or_else(|| PyValueError::new_err("obs must be C-contiguous"))?;
        let pool = self.thread_pool.as_ref();
        let inner = &mut self.inner;
        py.allow_threads(|| install_in_pool(pool, || inner.reset_into(obs_slice)))
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        Ok(())
    }

    pub fn reset_masked_into<'py>(
        &mut self,
        py: Python<'py>,
        mut obs: PyReadwriteArray4<'py, u8>,
        reset_mask: PyReadonlyArray1<'py, bool>,
        start_indices: PyReadonlyArray1<'py, i32>,
        seeds: Vec<Option<u64>>,
    ) -> PyResult<()> {
        self.validate_obs_shape(&obs)?;
        self.validate_vec_len(reset_mask.len(), "reset_mask")?;
        self.validate_vec_len(start_indices.len(), "start_indices")?;
        self.validate_vec_len(seeds.len(), "seeds")?;
        let reset_mask_ro = reset_mask.as_array();
        let reset_mask_slice = reset_mask_ro
            .as_slice()
            .ok_or_else(|| PyValueError::new_err("reset_mask must be C-contiguous"))?;
        let start_indices_ro = start_indices.as_array();
        let start_indices_slice = start_indices_ro
            .as_slice()
            .ok_or_else(|| PyValueError::new_err("start_indices must be C-contiguous"))?;
        if !reset_mask_slice.iter().any(|selected| *selected) {
            return Err(PyValueError::new_err(
                "reset_mask must select at least one lane",
            ));
        }
        let state_count = self.inner.initial_state_policy_names().len();
        for (env_idx, (&selected, &start_index)) in reset_mask_slice
            .iter()
            .zip(start_indices_slice.iter())
            .enumerate()
        {
            if !selected {
                continue;
            }
            if start_index < -1 || (start_index >= 0 && start_index as usize >= state_count) {
                return Err(PyValueError::new_err(format!(
                    "start_indices[{env_idx}] must be -1 or a valid initial-state catalog index",
                )));
            }
        }
        let mut obs_rw = obs.as_array_mut();
        let obs_slice = obs_rw
            .as_slice_mut()
            .ok_or_else(|| PyValueError::new_err("obs must be C-contiguous"))?;
        let pool = self.thread_pool.as_ref();
        let inner = &mut self.inner;
        py.allow_threads(|| {
            install_in_pool(pool, || {
                inner.reset_masked_into(obs_slice, reset_mask_slice, start_indices_slice, &seeds)
            })
        })
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
        Ok(())
    }

    pub fn info_into<'py>(
        &self,
        py: Python<'py>,
        mut x_pos: PyReadwriteArray1<'py, u16>,
        mut coins: PyReadwriteArray1<'py, u8>,
        mut level_hi: PyReadwriteArray1<'py, i16>,
        mut level_lo: PyReadwriteArray1<'py, i16>,
        mut lives: PyReadwriteArray1<'py, i16>,
        mut score: PyReadwriteArray1<'py, u32>,
        mut scrolling: PyReadwriteArray1<'py, i16>,
        mut time: PyReadwriteArray1<'py, u16>,
        mut xscroll_hi: PyReadwriteArray1<'py, u8>,
        mut xscroll_lo: PyReadwriteArray1<'py, u8>,
    ) -> PyResult<()> {
        self.validate_vec_len(x_pos.len(), "x_pos")?;
        self.validate_vec_len(coins.len(), "coins")?;
        self.validate_vec_len(level_hi.len(), "level_hi")?;
        self.validate_vec_len(level_lo.len(), "level_lo")?;
        self.validate_vec_len(lives.len(), "lives")?;
        self.validate_vec_len(score.len(), "score")?;
        self.validate_vec_len(scrolling.len(), "scrolling")?;
        self.validate_vec_len(time.len(), "time")?;
        self.validate_vec_len(xscroll_hi.len(), "xscroll_hi")?;
        self.validate_vec_len(xscroll_lo.len(), "xscroll_lo")?;
        let mut x_pos_rw = x_pos.as_array_mut();
        let x_pos_slice = x_pos_rw
            .as_slice_mut()
            .ok_or_else(|| PyValueError::new_err("x_pos must be C-contiguous"))?;
        let mut coins_rw = coins.as_array_mut();
        let coins_slice = coins_rw
            .as_slice_mut()
            .ok_or_else(|| PyValueError::new_err("coins must be C-contiguous"))?;
        let mut level_hi_rw = level_hi.as_array_mut();
        let level_hi_slice = level_hi_rw
            .as_slice_mut()
            .ok_or_else(|| PyValueError::new_err("level_hi must be C-contiguous"))?;
        let mut level_lo_rw = level_lo.as_array_mut();
        let level_lo_slice = level_lo_rw
            .as_slice_mut()
            .ok_or_else(|| PyValueError::new_err("level_lo must be C-contiguous"))?;
        let mut lives_rw = lives.as_array_mut();
        let lives_slice = lives_rw
            .as_slice_mut()
            .ok_or_else(|| PyValueError::new_err("lives must be C-contiguous"))?;
        let mut score_rw = score.as_array_mut();
        let score_slice = score_rw
            .as_slice_mut()
            .ok_or_else(|| PyValueError::new_err("score must be C-contiguous"))?;
        let mut scrolling_rw = scrolling.as_array_mut();
        let scrolling_slice = scrolling_rw
            .as_slice_mut()
            .ok_or_else(|| PyValueError::new_err("scrolling must be C-contiguous"))?;
        let mut time_rw = time.as_array_mut();
        let time_slice = time_rw
            .as_slice_mut()
            .ok_or_else(|| PyValueError::new_err("time must be C-contiguous"))?;
        let mut xscroll_hi_rw = xscroll_hi.as_array_mut();
        let xscroll_hi_slice = xscroll_hi_rw
            .as_slice_mut()
            .ok_or_else(|| PyValueError::new_err("xscroll_hi must be C-contiguous"))?;
        let mut xscroll_lo_rw = xscroll_lo.as_array_mut();
        let xscroll_lo_slice = xscroll_lo_rw
            .as_slice_mut()
            .ok_or_else(|| PyValueError::new_err("xscroll_lo must be C-contiguous"))?;
        py.allow_threads(|| {
            self.inner.info_into(
                x_pos_slice,
                coins_slice,
                level_hi_slice,
                level_lo_slice,
                lives_slice,
                score_slice,
                scrolling_slice,
                time_slice,
                xscroll_hi_slice,
                xscroll_lo_slice,
            );
        });
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn step_into<'py>(
        &mut self,
        py: Python<'py>,
        actions: PyReadonlyArray1<'py, u8>,
        mut obs: PyReadwriteArray4<'py, u8>,
        mut rewards: PyReadwriteArray1<'py, f32>,
        mut terminated: PyReadwriteArray1<'py, bool>,
        mut truncated: PyReadwriteArray1<'py, bool>,
        mut x_pos: PyReadwriteArray1<'py, u16>,
        mut coins: PyReadwriteArray1<'py, u8>,
        mut level_hi: PyReadwriteArray1<'py, i16>,
        mut level_lo: PyReadwriteArray1<'py, i16>,
        mut lives: PyReadwriteArray1<'py, i16>,
        mut score: PyReadwriteArray1<'py, u32>,
        mut scrolling: PyReadwriteArray1<'py, i16>,
        mut time: PyReadwriteArray1<'py, u16>,
        mut xscroll_hi: PyReadwriteArray1<'py, u8>,
        mut xscroll_lo: PyReadwriteArray1<'py, u8>,
    ) -> PyResult<()> {
        if self.inner.has_pending_reset() {
            return Err(PyRuntimeError::new_err(
                "cannot step while a terminated lane is pending reset; call reset(options={'reset_mask': ...}) first",
            ));
        }
        self.validate_obs_shape(&obs)?;
        self.validate_vec_len(actions.len(), "actions")?;
        self.validate_vec_len(rewards.len(), "rewards")?;
        self.validate_vec_len(terminated.len(), "terminated")?;
        self.validate_vec_len(truncated.len(), "truncated")?;
        self.validate_vec_len(x_pos.len(), "x_pos")?;
        self.validate_vec_len(coins.len(), "coins")?;
        self.validate_vec_len(level_hi.len(), "level_hi")?;
        self.validate_vec_len(level_lo.len(), "level_lo")?;
        self.validate_vec_len(lives.len(), "lives")?;
        self.validate_vec_len(score.len(), "score")?;
        self.validate_vec_len(scrolling.len(), "scrolling")?;
        self.validate_vec_len(time.len(), "time")?;
        self.validate_vec_len(xscroll_hi.len(), "xscroll_hi")?;
        self.validate_vec_len(xscroll_lo.len(), "xscroll_lo")?;

        let actions_ro = actions.as_array();
        let actions_slice = actions_ro
            .as_slice()
            .ok_or_else(|| PyValueError::new_err("actions must be C-contiguous"))?;
        let mut obs_rw = obs.as_array_mut();
        let obs_slice = obs_rw
            .as_slice_mut()
            .ok_or_else(|| PyValueError::new_err("obs must be C-contiguous"))?;
        let mut rewards_rw = rewards.as_array_mut();
        let rewards_slice = rewards_rw
            .as_slice_mut()
            .ok_or_else(|| PyValueError::new_err("rewards must be C-contiguous"))?;
        let mut terminated_rw = terminated.as_array_mut();
        let terminated_slice = terminated_rw
            .as_slice_mut()
            .ok_or_else(|| PyValueError::new_err("terminated must be C-contiguous"))?;
        let mut truncated_rw = truncated.as_array_mut();
        let truncated_slice = truncated_rw
            .as_slice_mut()
            .ok_or_else(|| PyValueError::new_err("truncated must be C-contiguous"))?;
        let mut x_pos_rw = x_pos.as_array_mut();
        let x_pos_slice = x_pos_rw
            .as_slice_mut()
            .ok_or_else(|| PyValueError::new_err("x_pos must be C-contiguous"))?;
        let mut coins_rw = coins.as_array_mut();
        let coins_slice = coins_rw
            .as_slice_mut()
            .ok_or_else(|| PyValueError::new_err("coins must be C-contiguous"))?;
        let mut level_hi_rw = level_hi.as_array_mut();
        let level_hi_slice = level_hi_rw
            .as_slice_mut()
            .ok_or_else(|| PyValueError::new_err("level_hi must be C-contiguous"))?;
        let mut level_lo_rw = level_lo.as_array_mut();
        let level_lo_slice = level_lo_rw
            .as_slice_mut()
            .ok_or_else(|| PyValueError::new_err("level_lo must be C-contiguous"))?;
        let mut lives_rw = lives.as_array_mut();
        let lives_slice = lives_rw
            .as_slice_mut()
            .ok_or_else(|| PyValueError::new_err("lives must be C-contiguous"))?;
        let mut score_rw = score.as_array_mut();
        let score_slice = score_rw
            .as_slice_mut()
            .ok_or_else(|| PyValueError::new_err("score must be C-contiguous"))?;
        let mut scrolling_rw = scrolling.as_array_mut();
        let scrolling_slice = scrolling_rw
            .as_slice_mut()
            .ok_or_else(|| PyValueError::new_err("scrolling must be C-contiguous"))?;
        let mut time_rw = time.as_array_mut();
        let time_slice = time_rw
            .as_slice_mut()
            .ok_or_else(|| PyValueError::new_err("time must be C-contiguous"))?;
        let mut xscroll_hi_rw = xscroll_hi.as_array_mut();
        let xscroll_hi_slice = xscroll_hi_rw
            .as_slice_mut()
            .ok_or_else(|| PyValueError::new_err("xscroll_hi must be C-contiguous"))?;
        let mut xscroll_lo_rw = xscroll_lo.as_array_mut();
        let xscroll_lo_slice = xscroll_lo_rw
            .as_slice_mut()
            .ok_or_else(|| PyValueError::new_err("xscroll_lo must be C-contiguous"))?;

        let pool = self.thread_pool.as_ref();
        let inner = &mut self.inner;
        py.allow_threads(|| {
            install_in_pool(pool, || {
                inner.step_into(
                    actions_slice,
                    obs_slice,
                    rewards_slice,
                    terminated_slice,
                    truncated_slice,
                    x_pos_slice,
                    coins_slice,
                    level_hi_slice,
                    level_lo_slice,
                    lives_slice,
                    score_slice,
                    scrolling_slice,
                    time_slice,
                    xscroll_hi_slice,
                    xscroll_lo_slice,
                )
            });
        });
        Ok(())
    }
}

fn effective_private_num_threads(
    requested: usize,
    num_envs: usize,
    available_threads: usize,
) -> usize {
    requested.min(num_envs).min(available_threads).max(1)
}

fn install_in_pool<OP, R>(pool: Option<&ThreadPool>, op: OP) -> R
where
    OP: FnOnce() -> R + Send,
    R: Send,
{
    match pool {
        Some(pool) => pool.install(op),
        None => op(),
    }
}

#[cfg(test)]
mod thread_pool_tests {
    use super::*;

    #[test]
    fn explicit_pool_runs_work_with_its_configured_size() {
        let pool = ThreadPoolBuilder::new().num_threads(2).build().unwrap();
        let observed = install_in_pool(Some(&pool), rayon::current_num_threads);
        assert_eq!(observed, 2);
    }

    #[test]
    fn explicit_thread_count_is_capped_by_lanes_and_available_parallelism() {
        assert_eq!(effective_private_num_threads(16, 8, 4), 4);
        assert_eq!(effective_private_num_threads(16, 2, 8), 2);
        assert_eq!(effective_private_num_threads(1, 8, 8), 1);
    }
}

impl RetroVecEnv {
    fn validate_obs_shape(&self, obs: &PyReadwriteArray4<'_, u8>) -> PyResult<()> {
        let shape = obs.shape();
        let expected = self.obs_shape();
        if shape != [expected.0, expected.1, expected.2, expected.3] {
            return Err(PyValueError::new_err(format!(
                "obs shape must be {:?}, got {:?}",
                expected, shape
            )));
        }
        Ok(())
    }

    fn validate_vec_len(&self, len: usize, name: &str) -> PyResult<()> {
        if len != self.inner.config().num_envs {
            return Err(PyValueError::new_err(format!(
                "{name} length must be {}, got {len}",
                self.inner.config().num_envs
            )));
        }
        Ok(())
    }

    fn validate_rgb_frame_shape(&self, frames: &PyReadwriteArray4<'_, u8>) -> PyResult<()> {
        let shape = frames.shape();
        let expected = self.rgb_frame_shape();
        if shape != [expected.0, expected.1, expected.2, expected.3] {
            return Err(PyValueError::new_err(format!(
                "frames shape must be {:?}, got {:?}",
                expected, shape
            )));
        }
        Ok(())
    }
}

fn build_initial_states(
    state_data: Vec<Vec<u8>>,
    state_names: Vec<String>,
    state_weights: Option<Vec<f64>>,
    num_envs: usize,
) -> PyResult<(Vec<InitialState>, bool)> {
    if state_data.is_empty() {
        if !state_names.is_empty() {
            return Err(PyValueError::new_err(
                "initial_state_names requires initial_states",
            ));
        }
        if state_weights.is_some() {
            return Err(PyValueError::new_err(
                "initial_state_weights requires initial_states",
            ));
        }
        return Ok((Vec::new(), false));
    }
    if state_data.iter().any(Vec::is_empty) {
        return Err(PyValueError::new_err(
            "initial_states entries must not be empty",
        ));
    }
    if !state_names.is_empty() && state_names.len() != state_data.len() {
        return Err(PyValueError::new_err(
            "initial_state_names length must match initial_states length",
        ));
    }

    let names = if state_names.is_empty() {
        (0..state_data.len())
            .map(|idx| format!("state-{idx}"))
            .collect::<Vec<_>>()
    } else {
        state_names
    };

    if let Some(weights) = state_weights {
        if weights.len() != state_data.len() {
            return Err(PyValueError::new_err(
                "initial_state_weights length must match initial_states length",
            ));
        }
        let total = weights.iter().try_fold(0.0, |acc, weight| {
            if !weight.is_finite() || *weight < 0.0 {
                Err(PyValueError::new_err(
                    "initial_state_weights must contain non-negative finite values",
                ))
            } else {
                Ok(acc + *weight)
            }
        })?;
        if !total.is_finite() || total <= 0.0 {
            return Err(PyValueError::new_err(
                "initial_state_weights must sum to a positive finite value",
            ));
        }

        let mut cumulative = 0.0;
        let mut states = Vec::with_capacity(state_data.len());
        for ((name, data), weight) in names.into_iter().zip(state_data).zip(weights) {
            cumulative += weight / total;
            states.push(InitialState::new(name, data, cumulative.min(1.0)));
        }
        return Ok((states, true));
    }

    if state_data.len() != 1 && state_data.len() != num_envs {
        return Err(PyValueError::new_err(format!(
            "initial_states length must be 1 or num_envs when weights are not provided: got {} for num_envs={num_envs}",
            state_data.len(),
        )));
    }

    Ok((
        names
            .into_iter()
            .zip(state_data)
            .map(|(name, data)| InitialState::new(name, data, 0.0))
            .collect(),
        false,
    ))
}

fn build_crop_mode(mode: &str) -> PyResult<CropMode> {
    match mode {
        "remove" => Ok(CropMode::Remove),
        "mask" => Ok(CropMode::Mask),
        _ => Err(PyValueError::new_err(
            "crop_mode must be 'remove' or 'mask'",
        )),
    }
}

fn build_resize_algorithm(algorithm: &str) -> PyResult<ResizeAlgorithm> {
    match algorithm {
        "area" => Ok(ResizeAlgorithm::Area),
        "nearest" => Ok(ResizeAlgorithm::Nearest),
        "bilinear" => Ok(ResizeAlgorithm::Bilinear),
        _ => Err(PyValueError::new_err(
            "resize_algorithm must be one of: nearest, bilinear, area",
        )),
    }
}

#[pymodule]
fn _supermariobrosnes_turbo(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RetroVecEnv>()?;
    m.add("NES_WIDTH", NES_WIDTH)?;
    m.add("NES_HEIGHT", NES_HEIGHT)?;
    m.add("VISIBLE_FRAME_WIDTH", VISIBLE_FRAME_WIDTH)?;
    m.add("VISIBLE_FRAME_HEIGHT", VISIBLE_FRAME_HEIGHT)?;
    Ok(())
}
