// PyO3's generated return conversion code triggers false-positive
// useless-conversion warnings for PyResult methods.
#![allow(clippy::useless_conversion)]

use numpy::{
    PyReadonlyArray1, PyReadonlyArray4, PyReadwriteArray1, PyReadwriteArray2, PyReadwriteArray4,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyModule;
use rayon::{ThreadPool, ThreadPoolBuilder};
use std::sync::Arc;
use std::thread;

use crate::cartridge::Cartridge;
use crate::emulator::{
    NES_HEIGHT, NES_WIDTH, RGB_CHANNELS, VISIBLE_FRAME_HEIGHT, VISIBLE_FRAME_WIDTH,
};
use crate::vec_env::{
    CropMode, InitialState, LiveSnapshot, MarioVecEnv, ResizeAlgorithm, VecEnvConfig,
};
use smb_turbo_driver::{
    selected_extra_info_width, EXPECTED_SMB_ROM_SHA256, EXTRA_INFO_DESCRIPTORS,
};

#[pyclass(name = "_RetroVecEnv")]
pub struct RetroVecEnv {
    inner: MarioVecEnv,
    thread_pool: Option<ThreadPool>,
    num_threads: usize,
    snapshot_owner: Arc<()>,
}

#[pyclass(frozen, module = "supermariobrosnes_turbo._supermariobrosnes_turbo")]
pub struct MarioLiveSnapshot {
    owner: Arc<()>,
    snapshot: LiveSnapshot,
}

#[pymethods]
impl MarioLiveSnapshot {
    #[getter]
    fn nbytes(&self) -> usize {
        self.snapshot.nbytes()
    }

    fn __reduce__(&self) -> PyResult<()> {
        Err(PyTypeError::new_err(
            "live snapshot handles are session-local and cannot be pickled",
        ))
    }
}

// PyO3 exposes wide buffer-oriented methods directly to Python.
#[allow(clippy::too_many_arguments)]
#[pymethods]
impl RetroVecEnv {
    #[new]
    #[pyo3(signature = (rom_path, num_envs, frame_skip=4, grayscale=true, frame_stack=4, terminate_on_flag=true, crop_top=0, crop_bottom=0, resize_width=84, resize_height=84, state_catalog_data=None, state_catalog_names=None, seed=0, frame_maxpool=false, noop_reset_max=0, sticky_action_prob=0.0, crop_left=0, crop_right=0, crop_mode="remove", crop_fill=0, resize_algorithm="area", num_threads=None, extra_info_ids=None))]
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
        state_catalog_data: Option<Vec<Vec<u8>>>,
        state_catalog_names: Option<Vec<String>>,
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
        extra_info_ids: Option<Vec<u8>>,
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
        let extra_info_ids = extra_info_ids.unwrap_or_default();
        if selected_extra_info_width(&extra_info_ids).is_none() {
            return Err(PyValueError::new_err(
                "extra_info_ids contains an unknown id",
            ));
        }
        let mut unique_extra_ids = std::collections::HashSet::new();
        if extra_info_ids
            .iter()
            .any(|id| !unique_extra_ids.insert(*id))
        {
            return Err(PyValueError::new_err(
                "extra_info_ids must not contain duplicates",
            ));
        }
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
        let state_catalog = build_state_catalog(
            state_catalog_data.unwrap_or_default(),
            state_catalog_names.unwrap_or_default(),
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
            inner: MarioVecEnv::new(cart, config, state_catalog, seed, extra_info_ids)
                .map_err(|err| PyValueError::new_err(err.to_string()))?,
            thread_pool,
            num_threads: effective_num_threads,
            snapshot_owner: Arc::new(()),
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
    pub fn state_catalog(&self) -> Vec<String> {
        self.inner.state_catalog()
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

    pub fn extra_info_shape(&self) -> (usize, usize) {
        (self.inner.config().num_envs, self.inner.extra_info_width())
    }

    pub fn extra_info_into<'py>(
        &self,
        py: Python<'py>,
        mut output: PyReadwriteArray2<'py, i64>,
    ) -> PyResult<()> {
        if output.shape() != [self.inner.config().num_envs, self.inner.extra_info_width()] {
            return Err(PyValueError::new_err(format!(
                "extra info output shape must be {:?}, got {:?}",
                self.extra_info_shape(),
                output.shape(),
            )));
        }
        let mut output_rw = output.as_array_mut();
        let output_slice = output_rw
            .as_slice_mut()
            .ok_or_else(|| PyValueError::new_err("extra info output must be C-contiguous"))?;
        let pool = self.thread_pool.as_ref();
        let inner = &self.inner;
        py.allow_threads(|| install_in_pool(pool, || inner.extra_info_into(output_slice)));
        Ok(())
    }

    pub fn ram_shape(&self) -> (usize, usize) {
        (self.inner.config().num_envs, 2048)
    }

    pub fn ram_into<'py>(
        &self,
        py: Python<'py>,
        mut output: PyReadwriteArray2<'py, u8>,
    ) -> PyResult<()> {
        if output.shape() != [self.inner.config().num_envs, 2048] {
            return Err(PyValueError::new_err(format!(
                "RAM output shape must be {:?}, got {:?}",
                self.ram_shape(),
                output.shape(),
            )));
        }
        let mut output_rw = output.as_array_mut();
        let output_slice = output_rw
            .as_slice_mut()
            .ok_or_else(|| PyValueError::new_err("RAM output must be C-contiguous"))?;
        py.allow_threads(|| self.inner.ram_into(output_slice));
        Ok(())
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
        state_indices: PyReadonlyArray1<'py, i32>,
        seeds: Vec<Option<u64>>,
    ) -> PyResult<()> {
        self.validate_obs_shape(&obs)?;
        self.validate_vec_len(reset_mask.len(), "reset_mask")?;
        self.validate_vec_len(state_indices.len(), "state_indices")?;
        self.validate_vec_len(seeds.len(), "seeds")?;
        let reset_mask_ro = reset_mask.as_array();
        let reset_mask_slice = reset_mask_ro
            .as_slice()
            .ok_or_else(|| PyValueError::new_err("reset_mask must be C-contiguous"))?;
        let state_indices_ro = state_indices.as_array();
        let state_indices_slice = state_indices_ro
            .as_slice()
            .ok_or_else(|| PyValueError::new_err("state_indices must be C-contiguous"))?;
        if !reset_mask_slice.iter().any(|selected| *selected) {
            return Err(PyValueError::new_err(
                "reset_mask must select at least one lane",
            ));
        }
        let state_count = self.inner.state_catalog().len();
        for (env_idx, (&selected, &state_index)) in reset_mask_slice
            .iter()
            .zip(state_indices_slice.iter())
            .enumerate()
        {
            if !selected {
                continue;
            }
            let valid = if state_count == 0 {
                state_index == -1
            } else {
                state_index >= 0 && (state_index as usize) < state_count
            };
            if !valid {
                return Err(PyValueError::new_err(format!(
                    "state_indices[{env_idx}] must index state_catalog",
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
                inner.reset_masked_into(obs_slice, reset_mask_slice, state_indices_slice, &seeds)
            })
        })
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
        Ok(())
    }

    pub fn capture_snapshots<'py>(
        &self,
        py: Python<'py>,
        obs: PyReadonlyArray4<'py, u8>,
        capture_mask: PyReadonlyArray1<'py, bool>,
    ) -> PyResult<Vec<Option<Py<MarioLiveSnapshot>>>> {
        let expected = self.obs_shape();
        if obs.shape() != [expected.0, expected.1, expected.2, expected.3] {
            return Err(PyValueError::new_err("obs has an incorrect shape"));
        }
        self.validate_vec_len(capture_mask.len(), "capture_mask")?;
        let obs_ro = obs.as_array();
        let obs_slice = obs_ro
            .as_slice()
            .ok_or_else(|| PyValueError::new_err("obs must be C-contiguous"))?;
        let mask_ro = capture_mask.as_array();
        let mask_slice = mask_ro
            .as_slice()
            .ok_or_else(|| PyValueError::new_err("capture_mask must be C-contiguous"))?;
        let snapshots = self
            .inner
            .capture_snapshots(obs_slice, mask_slice)
            .map_err(PyRuntimeError::new_err)?;
        snapshots
            .into_iter()
            .map(|snapshot| {
                snapshot
                    .map(|snapshot| {
                        Py::new(
                            py,
                            MarioLiveSnapshot {
                                owner: Arc::clone(&self.snapshot_owner),
                                snapshot,
                            },
                        )
                    })
                    .transpose()
            })
            .collect()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn reset_mixed_into<'py>(
        &mut self,
        py: Python<'py>,
        mut obs: PyReadwriteArray4<'py, u8>,
        reset_mask: PyReadonlyArray1<'py, bool>,
        state_indices: PyReadonlyArray1<'py, i32>,
        seeds: Vec<Option<u64>>,
        snapshots: Vec<Option<Py<MarioLiveSnapshot>>>,
    ) -> PyResult<()> {
        self.validate_obs_shape(&obs)?;
        self.validate_vec_len(reset_mask.len(), "reset_mask")?;
        self.validate_vec_len(state_indices.len(), "state_indices")?;
        self.validate_vec_len(seeds.len(), "seeds")?;
        self.validate_vec_len(snapshots.len(), "snapshots")?;
        let reset_mask_ro = reset_mask.as_array();
        let reset_mask_slice = reset_mask_ro
            .as_slice()
            .ok_or_else(|| PyValueError::new_err("reset_mask must be C-contiguous"))?;
        let state_indices_ro = state_indices.as_array();
        let state_indices_slice = state_indices_ro
            .as_slice()
            .ok_or_else(|| PyValueError::new_err("state_indices must be C-contiguous"))?;
        if !reset_mask_slice.iter().any(|&selected| selected) {
            return Err(PyValueError::new_err(
                "reset_mask must select at least one lane",
            ));
        }

        let state_count = self.inner.state_catalog().len();
        let mut native_snapshots = Vec::with_capacity(snapshots.len());
        for env_idx in 0..snapshots.len() {
            match (reset_mask_slice[env_idx], snapshots[env_idx].as_ref()) {
                (false, Some(_)) => {
                    return Err(PyValueError::new_err(
                        "snapshots may only be supplied for selected reset lanes",
                    ));
                }
                (false, None) => native_snapshots.push(None),
                (true, Some(snapshot)) => {
                    if state_indices_slice[env_idx] != -1 {
                        return Err(PyValueError::new_err(
                            "snapshot reset lanes must use the -1 state-index sentinel",
                        ));
                    }
                    if seeds[env_idx].is_some() {
                        return Err(PyValueError::new_err(
                            "snapshot reset lanes cannot also specify a seed",
                        ));
                    }
                    let snapshot = snapshot.bind(py).borrow();
                    if !Arc::ptr_eq(&snapshot.owner, &self.snapshot_owner) {
                        return Err(PyValueError::new_err(
                            "snapshot belongs to a different environment instance",
                        ));
                    }
                    native_snapshots.push(Some(snapshot.snapshot.clone()));
                }
                (true, None) => {
                    let state_index = state_indices_slice[env_idx];
                    let valid = if state_count == 0 {
                        state_index == -1
                    } else {
                        state_index >= 0 && (state_index as usize) < state_count
                    };
                    if !valid {
                        return Err(PyValueError::new_err(format!(
                            "state_indices[{env_idx}] must index state_catalog",
                        )));
                    }
                    native_snapshots.push(None);
                }
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
                inner.reset_mixed_into(
                    obs_slice,
                    reset_mask_slice,
                    state_indices_slice,
                    &seeds,
                    &native_snapshots,
                )
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

fn build_state_catalog(
    state_data: Vec<Vec<u8>>,
    state_names: Vec<String>,
) -> PyResult<Vec<InitialState>> {
    if state_data.is_empty() {
        if !state_names.is_empty() {
            return Err(PyValueError::new_err(
                "state_catalog_names requires state_catalog_data",
            ));
        }
        return Ok(Vec::new());
    }
    if state_data.iter().any(Vec::is_empty) {
        return Err(PyValueError::new_err(
            "state_catalog_data entries must not be empty",
        ));
    }
    if !state_names.is_empty() && state_names.len() != state_data.len() {
        return Err(PyValueError::new_err(
            "state_catalog_names length must match state_catalog_data length",
        ));
    }

    let names = if state_names.is_empty() {
        (0..state_data.len())
            .map(|idx| format!("state-{idx}"))
            .collect::<Vec<_>>()
    } else {
        state_names
    };

    let mut unique_names = std::collections::HashSet::with_capacity(names.len());
    if names.iter().any(|name| !unique_names.insert(name.clone())) {
        return Err(PyValueError::new_err(
            "state_catalog_names must not contain duplicates",
        ));
    }

    Ok(names
        .into_iter()
        .zip(state_data)
        .map(|(name, data)| InitialState::new(name, data))
        .collect())
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

#[pyfunction]
fn extra_info_descriptors() -> Vec<(u8, String, usize, String, Option<String>)> {
    EXTRA_INFO_DESCRIPTORS
        .iter()
        .map(|descriptor| {
            (
                descriptor.id,
                descriptor.name.to_string(),
                descriptor.width,
                descriptor.dtype.as_str().to_string(),
                descriptor.enum_name.map(str::to_string),
            )
        })
        .collect()
}

#[pymodule]
fn _supermariobrosnes_turbo(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<MarioLiveSnapshot>()?;
    m.add_class::<RetroVecEnv>()?;
    m.add_function(wrap_pyfunction!(extra_info_descriptors, m)?)?;
    m.add("NES_WIDTH", NES_WIDTH)?;
    m.add("NES_HEIGHT", NES_HEIGHT)?;
    m.add("VISIBLE_FRAME_WIDTH", VISIBLE_FRAME_WIDTH)?;
    m.add("VISIBLE_FRAME_HEIGHT", VISIBLE_FRAME_HEIGHT)?;
    Ok(())
}
