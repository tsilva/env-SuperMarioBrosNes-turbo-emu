#!/usr/bin/env python3
"""Single-file DreamerV3 trainer for Super Mario Bros NES.

The implementation follows the core DreamerV3 design: a categorical RSSM world
model, image reconstruction and reward/continuation prediction, KL balancing
with free nats, latent imagination, lambda returns, percentile return
normalization, and separate actor/critic learning.  It intentionally uses only
task-general Mario signals and exports a successful rollout as the repository's
portable positive-duration action-run policy.

Run from this checkout with:

    uv run --with 'torch>=2.7,<3' python dreamerv3_train.py Level1-1

The ROM is discovered through the same RETRO_DATA_PATH contract as the package.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import signal
import time
from typing import Any, Iterable, Sequence

import numpy as np

try:
    import torch
    from torch import Tensor, nn
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover - exercised without the train extra
    raise SystemExit(
        "DreamerV3 requires PyTorch. Run with: "
        "uv run --with 'torch>=2.7,<3' python dreamerv3_train.py Level1-1"
    ) from exc

from supermariobrosnes_turbo import (
    ACTION_SETS,
    SuperMarioBrosNesTurboVecEnv,
    list_available_states,
)
from supermariobrosnes_turbo.jerk import ActionRun, JerkPolicy, canonicalize_runs


ALGORITHM = "dreamerv3"
ACTION_SET = "standard"
NOOP_ACTION = ACTION_SETS[ACTION_SET].index("noop")
OBSERVATION_SHAPE = (1, 64, 64)


@dataclass(frozen=True)
class Config:
    state: str = "Level1-1"
    seed: int = 0
    envs: int = 16
    steps: int = 2_000_000
    prefill: int = 20_000
    replay_size: int = 200_000
    batch_size: int = 8
    batch_length: int = 32
    train_ratio: float = 8.0
    eval_every: int = 50_000
    checkpoint_every: int = 100_000
    log_every: int = 10_000
    max_episode_steps: int = 4_500
    stall_steps: int = 450
    frame_skip: int = 4
    noop_reset_max: int = 0
    learning_rate: float = 3e-4
    model_learning_rate: float = 3e-4
    gamma: float = 0.997
    lambda_: float = 0.95
    imagination_horizon: int = 15
    imagination_starts: int = 128
    entropy_scale: float = 3e-3
    exploration: float = 0.05
    exploration_run_mean: float = 4.0
    exploration_run_max: int = 32
    free_nats: float = 1.0
    dyn_scale: float = 1.0
    rep_scale: float = 0.1
    unimix: float = 0.01
    deter: int = 512
    hidden: int = 256
    stoch: int = 16
    classes: int = 16
    embed: int = 256
    cnn_depth: int = 24
    device: str = "mps"
    output: str = "runs/dreamerv3/Level1-1"
    rom: str | None = None
    state_dir: str | None = None
    stop_on_success: bool = True


@dataclass
class RSSMState:
    deter: Tensor
    stoch: Tensor

    def detach(self) -> "RSSMState":
        return RSSMState(self.deter.detach(), self.stoch.detach())

    def index(self, indices: Tensor) -> "RSSMState":
        return RSSMState(self.deter[indices], self.stoch[indices])


@dataclass
class Episode:
    obs: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    cont: np.ndarray
    first: np.ndarray

    def __len__(self) -> int:
        return int(self.action.shape[0])


@dataclass(frozen=True)
class Evaluation:
    success: bool
    episode_return: float
    progress: float
    steps: int
    actions: tuple[int, ...]
    life_loss: bool
    stalled: bool


def symlog(x: Tensor) -> Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: Tensor) -> Tensor:
    return torch.sign(x) * torch.expm1(torch.abs(x))


def _unimix_probs(logits: Tensor, amount: float) -> Tensor:
    probs = logits.softmax(-1)
    if amount:
        probs = (1.0 - amount) * probs + amount / logits.shape[-1]
    return probs


def sample_onehot(logits: Tensor, unimix: float, *, straight_through: bool) -> Tensor:
    probs = _unimix_probs(logits, unimix)
    noise = -torch.log(-torch.log(torch.rand_like(probs).clamp_(1e-6, 1 - 1e-6)))
    indices = (torch.log(probs.clamp_min(1e-8)) + noise).argmax(-1)
    hard = F.one_hot(indices, probs.shape[-1]).to(probs.dtype)
    return hard + probs - probs.detach() if straight_through else hard


def categorical_kl(lhs_logits: Tensor, rhs_logits: Tensor, unimix: float) -> Tensor:
    lhs = _unimix_probs(lhs_logits, unimix)
    rhs = _unimix_probs(rhs_logits, unimix)
    kl = lhs * (torch.log(lhs.clamp_min(1e-8)) - torch.log(rhs.clamp_min(1e-8)))
    return kl.sum((-1, -2))


class TwoHot:
    def __init__(self, device: torch.device, bins: int = 255, limit: float = 20.0):
        self.bins = torch.linspace(-limit, limit, bins, device=device)

    def target(self, values: Tensor) -> Tensor:
        values = symlog(values).clamp(self.bins[0], self.bins[-1])
        position = (values - self.bins[0]) / (self.bins[-1] - self.bins[0])
        position = position * (self.bins.numel() - 1)
        lower = position.floor().long()
        upper = position.ceil().long()
        upper_weight = position - lower.to(position.dtype)
        lower_weight = 1.0 - upper_weight
        target = torch.zeros(*values.shape, self.bins.numel(), device=values.device)
        target.scatter_add_(-1, lower.unsqueeze(-1), lower_weight.unsqueeze(-1))
        target.scatter_add_(-1, upper.unsqueeze(-1), upper_weight.unsqueeze(-1))
        return target

    def loss(self, logits: Tensor, values: Tensor) -> Tensor:
        return -(self.target(values) * logits.log_softmax(-1)).sum(-1)

    def mean(self, logits: Tensor) -> Tensor:
        return symexp((logits.softmax(-1) * self.bins).sum(-1))


class RMSNorm(nn.Module):
    def __init__(self, size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        scale = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
        return x * scale * self.weight


class MLP(nn.Module):
    def __init__(self, input_size: int, hidden: int, output_size: int, layers: int):
        super().__init__()
        modules: list[nn.Module] = []
        size = input_size
        for _ in range(layers):
            modules.extend((nn.Linear(size, hidden), RMSNorm(hidden), nn.SiLU()))
            size = hidden
        modules.append(nn.Linear(size, output_size))
        self.net = nn.Sequential(*modules)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class Encoder(nn.Module):
    def __init__(self, depth: int, embed: int):
        super().__init__()
        channels = (depth, depth * 2, depth * 4, depth * 8)
        layers: list[nn.Module] = []
        incoming = OBSERVATION_SHAPE[0]
        for outgoing in channels:
            layers.extend(
                (
                    nn.Conv2d(incoming, outgoing, kernel_size=4, stride=2, padding=1),
                    nn.GroupNorm(1, outgoing),
                    nn.SiLU(),
                )
            )
            incoming = outgoing
        self.conv = nn.Sequential(*layers)
        self.out = nn.Sequential(nn.Flatten(), nn.Linear(channels[-1] * 4 * 4, embed), RMSNorm(embed), nn.SiLU())

    def forward(self, obs: Tensor) -> Tensor:
        return self.out(self.conv(obs.float() / 255.0 - 0.5))


class Decoder(nn.Module):
    def __init__(self, feature_size: int, depth: int):
        super().__init__()
        channels = depth * 8
        self.start = nn.Sequential(
            nn.Linear(feature_size, channels * 4 * 4),
            RMSNorm(channels * 4 * 4),
            nn.SiLU(),
        )
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(channels, depth * 4, 4, 2, 1),
            nn.GroupNorm(1, depth * 4),
            nn.SiLU(),
            nn.ConvTranspose2d(depth * 4, depth * 2, 4, 2, 1),
            nn.GroupNorm(1, depth * 2),
            nn.SiLU(),
            nn.ConvTranspose2d(depth * 2, depth, 4, 2, 1),
            nn.GroupNorm(1, depth),
            nn.SiLU(),
            nn.ConvTranspose2d(depth, OBSERVATION_SHAPE[0], 4, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, feature: Tensor) -> Tensor:
        channels = self.conv[0].in_channels
        return self.conv(self.start(feature).reshape(-1, channels, 4, 4))


class RSSM(nn.Module):
    def __init__(self, config: Config, action_count: int):
        super().__init__()
        self.deter = config.deter
        self.stoch = config.stoch
        self.classes = config.classes
        self.unimix = config.unimix
        stochastic_size = config.stoch * config.classes
        self.gru = nn.GRUCell(stochastic_size + action_count, config.deter)
        self.prior = MLP(config.deter, config.hidden, stochastic_size, 2)
        self.posterior = MLP(config.deter + config.embed, config.hidden, stochastic_size, 1)

    def initial(self, batch: int, device: torch.device) -> RSSMState:
        return RSSMState(
            torch.zeros(batch, self.deter, device=device),
            torch.zeros(batch, self.stoch, self.classes, device=device),
        )

    def _logits(self, flat: Tensor) -> Tensor:
        return flat.reshape(*flat.shape[:-1], self.stoch, self.classes)

    def img_step(self, state: RSSMState, action: Tensor) -> tuple[RSSMState, Tensor]:
        inputs = torch.cat((state.stoch.flatten(-2), action), -1)
        deter = self.gru(inputs, state.deter)
        logits = self._logits(self.prior(deter))
        stoch = sample_onehot(logits, self.unimix, straight_through=True)
        return RSSMState(deter, stoch), logits

    def obs_step(
        self,
        state: RSSMState,
        prev_action: Tensor,
        embed: Tensor,
        first: Tensor,
    ) -> tuple[RSSMState, Tensor, Tensor]:
        keep = (~first).to(state.deter.dtype).unsqueeze(-1)
        state = RSSMState(state.deter * keep, state.stoch * keep.unsqueeze(-1))
        prev_action = prev_action * keep
        prior_state, prior_logits = self.img_step(state, prev_action)
        post_logits = self._logits(self.posterior(torch.cat((prior_state.deter, embed), -1)))
        stoch = sample_onehot(post_logits, self.unimix, straight_through=True)
        return RSSMState(prior_state.deter, stoch), prior_logits, post_logits


class DreamerV3(nn.Module):
    def __init__(self, config: Config, action_count: int, device: torch.device):
        super().__init__()
        self.config = config
        self.action_count = action_count
        self.encoder = Encoder(config.cnn_depth, config.embed)
        self.rssm = RSSM(config, action_count)
        feature_size = config.deter + config.stoch * config.classes
        self.decoder = Decoder(feature_size, config.cnn_depth)
        self.reward = MLP(feature_size, config.hidden, 255, 1)
        self.cont = MLP(feature_size, config.hidden, 1, 1)
        self.actor = MLP(feature_size, config.hidden, action_count, 3)
        self.critic = MLP(feature_size, config.hidden, 255, 3)
        self.slow_critic = MLP(feature_size, config.hidden, 255, 3)
        self.twohot = TwoHot(device)
        self.to(device)
        self.slow_critic.load_state_dict(self.critic.state_dict())
        for parameter in self.slow_critic.parameters():
            parameter.requires_grad_(False)

    @staticmethod
    def feature(state: RSSMState) -> Tensor:
        return torch.cat((state.deter, state.stoch.flatten(-2)), -1)

    def observe(
        self, obs: Tensor, prev_action: Tensor, first: Tensor
    ) -> tuple[RSSMState, Tensor, Tensor, Tensor]:
        batch, length = obs.shape[:2]
        embeds = self.encoder(obs.reshape(batch * length, *obs.shape[2:])).reshape(batch, length, -1)
        state = self.rssm.initial(batch, obs.device)
        states: list[RSSMState] = []
        priors: list[Tensor] = []
        posts: list[Tensor] = []
        for index in range(length):
            state, prior, post = self.rssm.obs_step(
                state, prev_action[:, index], embeds[:, index], first[:, index]
            )
            states.append(state)
            priors.append(prior)
            posts.append(post)
        sequence = RSSMState(
            torch.stack([item.deter for item in states], 1),
            torch.stack([item.stoch for item in states], 1),
        )
        return sequence, self.feature(sequence), torch.stack(priors, 1), torch.stack(posts, 1)

    def observe_step(
        self,
        state: RSSMState,
        obs: Tensor,
        prev_action: Tensor,
        first: Tensor,
    ) -> RSSMState:
        embed = self.encoder(obs)
        state, _, _ = self.rssm.obs_step(state, prev_action, embed, first)
        return state

    def policy(self, state: RSSMState, *, deterministic: bool) -> tuple[Tensor, Tensor]:
        logits = self.actor(self.feature(state))
        if deterministic:
            action = logits.argmax(-1)
        else:
            probs = _unimix_probs(logits, self.config.unimix)
            noise = -torch.log(-torch.log(torch.rand_like(probs).clamp_(1e-6, 1 - 1e-6)))
            action = (torch.log(probs.clamp_min(1e-8)) + noise).argmax(-1)
        return action, logits


class Replay:
    """Episode replay with bounded image storage and live-episode sampling."""

    def __init__(self, envs: int, capacity: int, sequence_length: int):
        self.envs = envs
        self.capacity = capacity
        self.sequence_length = sequence_length
        self.episodes: deque[Episode] = deque()
        self.episode_rows = 0
        self.current: list[dict[str, list[Any]]] = [self._empty() for _ in range(envs)]

    @staticmethod
    def _empty() -> dict[str, list[Any]]:
        return {name: [] for name in ("obs", "action", "reward", "cont", "first")}

    def begin(self, observations: np.ndarray, mask: np.ndarray | None = None) -> None:
        selected = np.ones(self.envs, dtype=np.bool_) if mask is None else np.asarray(mask, dtype=np.bool_)
        for lane in np.flatnonzero(selected):
            index = int(lane)
            self.current[index] = self._empty()
            self._append(index, observations[index], NOOP_ACTION, 0.0, 1.0, True)

    def _append(
        self, lane: int, obs: np.ndarray, action: int, reward: float, cont: float, first: bool
    ) -> None:
        target = self.current[lane]
        target["obs"].append(np.asarray(obs, dtype=np.uint8).copy())
        target["action"].append(np.uint8(action))
        target["reward"].append(np.float32(reward))
        target["cont"].append(np.float32(cont))
        target["first"].append(np.bool_(first))

    def append_step(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        for lane in range(self.envs):
            self._append(
                lane,
                observations[lane],
                int(actions[lane]),
                float(rewards[lane]),
                0.0 if dones[lane] else 1.0,
                False,
            )

    def finish(self, mask: np.ndarray) -> None:
        for lane in np.flatnonzero(mask):
            source = self.current[int(lane)]
            episode = Episode(
                obs=np.stack(source["obs"]),
                action=np.asarray(source["action"], dtype=np.uint8),
                reward=np.asarray(source["reward"], dtype=np.float32),
                cont=np.asarray(source["cont"], dtype=np.float32),
                first=np.asarray(source["first"], dtype=np.bool_),
            )
            self.episodes.append(episode)
            self.episode_rows += len(episode)
            self.current[int(lane)] = self._empty()
        while self.episodes and self.rows > self.capacity:
            removed = self.episodes.popleft()
            self.episode_rows -= len(removed)

    @property
    def rows(self) -> int:
        return self.episode_rows + sum(len(lane["action"]) for lane in self.current)

    def _sources(self) -> list[Episode | dict[str, list[Any]]]:
        length = self.sequence_length
        sources: list[Episode | dict[str, list[Any]]] = [
            episode for episode in self.episodes if len(episode) >= length
        ]
        sources.extend(lane for lane in self.current if len(lane["action"]) >= length)
        return sources

    def ready(self, batch_size: int) -> bool:
        return self.rows >= batch_size * self.sequence_length and bool(self._sources())

    def sample(self, batch_size: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        sources = self._sources()
        if not sources:
            raise RuntimeError("replay does not yet contain a complete sequence")
        batches: dict[str, list[np.ndarray]] = {
            name: [] for name in ("obs", "action", "reward", "cont", "first")
        }
        length = self.sequence_length
        weights = np.asarray(
            [
                (len(source) if isinstance(source, Episode) else len(source["action"])) - length + 1
                for source in sources
            ],
            dtype=np.float64,
        )
        weights /= weights.sum()
        for _ in range(batch_size):
            source = sources[int(rng.choice(len(sources), p=weights))]
            size = len(source) if isinstance(source, Episode) else len(source["action"])
            start = int(rng.integers(0, size - length + 1))
            end = start + length
            for name in batches:
                values = getattr(source, name) if isinstance(source, Episode) else source[name]
                batches[name].append(np.asarray(values[start:end]))
        result = {name: np.stack(values) for name, values in batches.items()}
        result["first"][:, 0] = True
        return result


class ProgressReward:
    """Task-general progress/score shaping and strict level-completion detection."""

    def __init__(self, env: SuperMarioBrosNesTurboVecEnv, config: Config):
        self.env = env
        self.envs = config.envs
        self.max_steps = config.max_episode_steps
        self.stall_steps = config.stall_steps
        self.steps = np.zeros(self.envs, dtype=np.int64)
        self.last_progress = np.zeros(self.envs, dtype=np.int64)
        self.returns = np.zeros(self.envs, dtype=np.float64)
        self.previous_lives = np.zeros(self.envs, dtype=np.int16)
        self.previous_level_hi = np.zeros(self.envs, dtype=np.int16)
        self.previous_level_lo = np.zeros(self.envs, dtype=np.int16)
        self.previous_score = np.zeros(self.envs, dtype=np.int64)
        self.level_max_x = np.zeros(self.envs, dtype=np.int64)
        self.completed_base = np.zeros(self.envs, dtype=np.int64)
        self.max_global_x = np.zeros(self.envs, dtype=np.int64)
        self.previous_x = np.zeros(self.envs, dtype=np.int64)
        self.seen_transitions: list[set[tuple[int, int]]] = [set() for _ in range(self.envs)]

    def reset(self, mask: np.ndarray | None = None) -> None:
        selected = np.ones(self.envs, dtype=np.bool_) if mask is None else np.asarray(mask, dtype=np.bool_)
        x = self.env.xscroll_hi.astype(np.int64) * 256 + self.env.xscroll_lo.astype(np.int64)
        x = np.where(x >= 0xFF00, 0, x)
        self.steps[selected] = 0
        self.last_progress[selected] = 0
        self.returns[selected] = 0.0
        self.previous_lives[selected] = self.env.lives[selected]
        self.previous_level_hi[selected] = self.env.level_hi[selected]
        self.previous_level_lo[selected] = self.env.level_lo[selected]
        self.previous_score[selected] = self.env.score[selected]
        self.level_max_x[selected] = x[selected]
        self.completed_base[selected] = 0
        self.max_global_x[selected] = x[selected]
        self.previous_x[selected] = x[selected]
        for lane in np.flatnonzero(selected):
            self.seen_transitions[int(lane)].clear()

    def step(
        self, native_terminated: np.ndarray, native_truncated: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        lives = self.env.lives.astype(np.int64, copy=False)
        level_hi = self.env.level_hi.astype(np.int64, copy=False)
        level_lo = self.env.level_lo.astype(np.int64, copy=False)
        score = self.env.score.astype(np.int64, copy=False)
        x = self.env.xscroll_hi.astype(np.int64) * 256 + self.env.xscroll_lo.astype(np.int64)
        x = np.where(x >= 0xFF00, self.level_max_x, x)
        life_loss = lives < self.previous_lives
        level_changed = (level_hi != self.previous_level_hi) | (level_lo != self.previous_level_lo)
        completed = level_changed & ~life_loss

        scroll = (self.previous_x - x >= 128) & ~life_loss & ~level_changed
        novel_scroll = np.zeros(self.envs, dtype=np.bool_)
        for lane in np.flatnonzero(scroll):
            index = int(lane)
            signature = (int(self.previous_x[index]) // 64, int(x[index]) // 64)
            if signature not in self.seen_transitions[index]:
                self.seen_transitions[index].add(signature)
                novel_scroll[index] = True
        segment = completed | novel_scroll
        self.completed_base[segment] += self.level_max_x[segment]
        self.level_max_x[segment] = 0
        self.level_max_x = np.maximum(self.level_max_x, np.where(level_changed, 0, x))
        global_x = self.completed_base + self.level_max_x
        progress_delta = np.maximum(global_x - self.max_global_x, 0)
        self.max_global_x = np.maximum(self.max_global_x, global_x)
        score_delta = np.maximum(score - self.previous_score, 0)

        self.steps += 1
        progressed = progress_delta > 0
        self.last_progress[progressed] = self.steps[progressed]
        stalled = (self.stall_steps > 0) & (self.steps - self.last_progress >= self.stall_steps)
        timed_out = self.steps >= self.max_steps
        rewards = progress_delta.astype(np.float32) + 0.01 * score_delta.astype(np.float32) - 0.1
        rewards -= 25.0 * life_loss.astype(np.float32)
        self.returns += rewards

        unexpected_terminal = native_terminated & ~completed & ~life_loss
        failure = life_loss | stalled | timed_out | native_truncated | unexpected_terminal
        done = failure | completed
        info = {
            "life_loss": life_loss.copy(),
            "stalled": stalled.copy(),
            "timed_out": timed_out.copy(),
            "progress": self.max_global_x.copy(),
            "return": self.returns.copy(),
            "length": self.steps.copy(),
        }
        self.previous_lives[:] = lives
        self.previous_level_hi[:] = level_hi
        self.previous_level_lo[:] = level_lo
        self.previous_score[:] = score
        self.previous_x[:] = x
        return rewards, done, completed, info


class ReturnNormalizer:
    def __init__(self, rate: float = 0.01):
        self.rate = rate
        self.low = 0.0
        self.high = 0.0
        self.initialized = False

    def update(self, returns: Tensor) -> float:
        values = returns.detach().float().cpu().numpy().reshape(-1)
        low, high = np.percentile(values, (5, 95)).tolist()
        if not self.initialized:
            self.low, self.high, self.initialized = float(low), float(high), True
        else:
            self.low += self.rate * (float(low) - self.low)
            self.high += self.rate * (float(high) - self.high)
        return max(self.high - self.low, 1.0)


def make_env(config: Config, envs: int | None = None) -> SuperMarioBrosNesTurboVecEnv:
    count = config.envs if envs is None else envs
    return SuperMarioBrosNesTurboVecEnv(
        "SuperMarioBros-Nes-v0",
        state=config.state,
        state_dir=config.state_dir,
        num_envs=count,
        num_threads=count,
        rom_path=config.rom,
        render_mode=None,
        use_restricted_actions=ACTION_SET,
        obs_copy="unsafe_view",
        obs_grayscale=True,
        obs_layout="chw",
        obs_crop=(32, 0, 0, 0),
        obs_crop_mode="mask",
        obs_resize=(64, 64),
        obs_resize_algorithm="area",
        frame_skip=config.frame_skip,
        frame_stack=1,
        maxpool_last_two=True,
        noop_reset_max=config.noop_reset_max,
        sticky_action_prob=0.0,
        reward_clip=False,
        info_filter="none",
    )


def onehot_actions(actions: np.ndarray | Tensor, count: int, device: torch.device) -> Tensor:
    tensor = torch.as_tensor(actions, dtype=torch.long, device=device)
    return F.one_hot(tensor, count).float()


def to_torch(batch: dict[str, np.ndarray], device: torch.device, action_count: int) -> dict[str, Tensor]:
    return {
        "obs": torch.as_tensor(batch["obs"], device=device),
        "action": onehot_actions(batch["action"], action_count, device),
        "reward": torch.as_tensor(batch["reward"], dtype=torch.float32, device=device),
        "cont": torch.as_tensor(batch["cont"], dtype=torch.float32, device=device),
        "first": torch.as_tensor(batch["first"], dtype=torch.bool, device=device),
    }


def agc(parameters: Iterable[nn.Parameter], clipping: float = 0.3, eps: float = 1e-3) -> None:
    for parameter in parameters:
        if parameter.grad is None or parameter.ndim <= 1:
            continue
        parameter_norm = parameter.detach().norm().clamp_min(eps)
        gradient_norm = parameter.grad.detach().norm().clamp_min(1e-6)
        maximum = clipping * parameter_norm
        if gradient_norm > maximum:
            parameter.grad.mul_(maximum / gradient_norm)


def lambda_returns(reward: Tensor, discount: Tensor, value: Tensor, lambda_: float) -> Tensor:
    result: list[Tensor] = []
    carry = value[-1]
    for index in range(reward.shape[0] - 1, -1, -1):
        carry = reward[index] + discount[index] * (
            (1.0 - lambda_) * value[index + 1] + lambda_ * carry
        )
        result.append(carry)
    return torch.stack(result[::-1])


def world_model_loss(agent: DreamerV3, batch: dict[str, Tensor]) -> tuple[Tensor, RSSMState, dict[str, float]]:
    states, features, prior, post = agent.observe(batch["obs"], batch["action"], batch["first"])
    flat_features = features.reshape(-1, features.shape[-1])
    target_obs = batch["obs"].float().reshape(-1, *OBSERVATION_SHAPE) / 255.0
    reconstruction = agent.decoder(flat_features)
    rec_loss = (reconstruction - target_obs).square().mean((1, 2, 3)).reshape(
        batch["reward"].shape
    )
    reward_logits = agent.reward(flat_features).reshape(*batch["reward"].shape, -1)
    reward_loss = agent.twohot.loss(reward_logits, batch["reward"])
    cont_logits = agent.cont(flat_features).reshape(batch["cont"].shape)
    cont_loss = F.binary_cross_entropy_with_logits(cont_logits, batch["cont"], reduction="none")
    dyn_loss = categorical_kl(post.detach(), prior, agent.config.unimix).clamp_min(agent.config.free_nats)
    rep_loss = categorical_kl(post, prior.detach(), agent.config.unimix).clamp_min(agent.config.free_nats)
    loss = (
        rec_loss
        + reward_loss
        + cont_loss
        + agent.config.dyn_scale * dyn_loss
        + agent.config.rep_scale * rep_loss
    ).mean()
    metrics = {
        "model": float(loss.detach().cpu()),
        "recon": float(rec_loss.mean().detach().cpu()),
        "reward": float(reward_loss.mean().detach().cpu()),
        "cont": float(cont_loss.mean().detach().cpu()),
        "dyn_kl": float(dyn_loss.mean().detach().cpu()),
        "rep_kl": float(rep_loss.mean().detach().cpu()),
    }
    return loss, states, metrics


def imagine(
    agent: DreamerV3,
    starts: RSSMState,
    horizon: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    states = starts.detach()
    features: list[Tensor] = [agent.feature(states)]
    actions: list[Tensor] = []
    rewards: list[Tensor] = []
    continuations: list[Tensor] = []
    with torch.no_grad():
        for _ in range(horizon):
            action_index, _ = agent.policy(states, deterministic=False)
            action = F.one_hot(action_index, agent.action_count).float()
            states, _ = agent.rssm.img_step(states, action)
            feature = agent.feature(states)
            features.append(feature)
            actions.append(action_index)
            rewards.append(agent.twohot.mean(agent.reward(feature)))
            continuations.append(torch.sigmoid(agent.cont(feature).squeeze(-1)))
        stacked_features = torch.stack(features)
        values = agent.twohot.mean(agent.slow_critic(stacked_features))
    return (
        stacked_features,
        torch.stack(actions),
        torch.stack(rewards),
        torch.stack(continuations),
        values,
    )


def actor_critic_losses(
    agent: DreamerV3,
    posterior: RSSMState,
    replay_batch: dict[str, Tensor],
    normalizer: ReturnNormalizer,
    rng: torch.Generator,
) -> tuple[Tensor, Tensor, dict[str, float]]:
    flat = RSSMState(
        posterior.deter.reshape(-1, posterior.deter.shape[-1]),
        posterior.stoch.reshape(-1, *posterior.stoch.shape[-2:]),
    )
    count = min(agent.config.imagination_starts, flat.deter.shape[0])
    indices = torch.randperm(flat.deter.shape[0], generator=rng, device=flat.deter.device)[:count]
    features, actions, rewards, continuations, slow_values = imagine(
        agent, flat.index(indices), agent.config.imagination_horizon
    )
    discounts = agent.config.gamma * continuations
    returns = lambda_returns(rewards, discounts, slow_values, agent.config.lambda_)
    weights = torch.cat(
        (torch.ones_like(discounts[:1]), torch.cumprod(discounts[:-1], 0)), 0
    ).detach()
    scale = normalizer.update(returns)

    policy_logits = agent.actor(features[:-1].detach())
    probs = _unimix_probs(policy_logits, agent.config.unimix)
    log_policy = torch.log(probs.clamp_min(1e-8))
    log_probs = log_policy.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    entropy = -(probs * log_policy).sum(-1)
    baseline = agent.twohot.mean(agent.critic(features[:-1].detach())).detach()
    advantage = ((returns - baseline) / scale).detach()
    actor_loss = -(weights * (log_probs * advantage + agent.config.entropy_scale * entropy)).mean()

    critic_logits = agent.critic(features[:-1].detach())
    value_loss = agent.twohot.loss(critic_logits, returns.detach())
    with torch.no_grad():
        slow_probs = agent.slow_critic(features[:-1].detach()).softmax(-1)
    slow_reg = -(slow_probs * critic_logits.log_softmax(-1)).sum(-1)
    imagination_critic_loss = (weights * (value_loss + slow_reg)).mean()

    replay_features = agent.feature(posterior).detach()
    with torch.no_grad():
        replay_values = agent.twohot.mean(agent.slow_critic(replay_features)).transpose(0, 1)
        replay_rewards = replay_batch["reward"][:, 1:].transpose(0, 1)
        replay_discounts = (
            agent.config.gamma * replay_batch["cont"][:, 1:]
        ).transpose(0, 1)
        replay_returns = lambda_returns(
            replay_rewards, replay_discounts, replay_values, agent.config.lambda_
        ).transpose(0, 1)
    replay_logits = agent.critic(replay_features[:, :-1])
    replay_value_loss = agent.twohot.loss(replay_logits, replay_returns).mean()
    critic_loss = imagination_critic_loss + 0.3 * replay_value_loss
    metrics = {
        "actor": float(actor_loss.detach().cpu()),
        "critic": float(critic_loss.detach().cpu()),
        "replay_critic": float(replay_value_loss.detach().cpu()),
        "imag_return": float(returns.mean().detach().cpu()),
        "imag_reward": float(rewards.mean().detach().cpu()),
        "entropy": float(entropy.mean().detach().cpu()),
        "return_scale": float(scale),
    }
    return actor_loss, critic_loss, metrics


def update_slow_critic(agent: DreamerV3, rate: float = 0.02) -> None:
    with torch.no_grad():
        for slow, online in zip(agent.slow_critic.parameters(), agent.critic.parameters()):
            slow.lerp_(online, rate)


def optimize(
    agent: DreamerV3,
    batch: dict[str, Tensor],
    optimizers: tuple[torch.optim.Optimizer, torch.optim.Optimizer, torch.optim.Optimizer],
    normalizer: ReturnNormalizer,
    torch_rng: torch.Generator,
) -> dict[str, float]:
    model_optimizer, actor_optimizer, critic_optimizer = optimizers
    model_optimizer.zero_grad(set_to_none=True)
    model_loss, posterior, metrics = world_model_loss(agent, batch)
    model_loss.backward()
    model_parameters = list(agent.encoder.parameters()) + list(agent.rssm.parameters()) + list(agent.decoder.parameters()) + list(agent.reward.parameters()) + list(agent.cont.parameters())
    agc(model_parameters)
    torch.nn.utils.clip_grad_norm_(model_parameters, 100.0)
    model_optimizer.step()

    actor_loss, critic_loss, behavior_metrics = actor_critic_losses(
        agent, posterior.detach(), batch, normalizer, torch_rng
    )
    actor_optimizer.zero_grad(set_to_none=True)
    actor_loss.backward()
    agc(agent.actor.parameters())
    torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 100.0)
    actor_optimizer.step()

    critic_optimizer.zero_grad(set_to_none=True)
    critic_loss.backward()
    agc(agent.critic.parameters())
    torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 100.0)
    critic_optimizer.step()
    update_slow_critic(agent)
    metrics.update(behavior_metrics)
    return metrics


def _new_optimizers(agent: DreamerV3, config: Config) -> tuple[torch.optim.Optimizer, ...]:
    model_parameters = list(agent.encoder.parameters()) + list(agent.rssm.parameters()) + list(agent.decoder.parameters()) + list(agent.reward.parameters()) + list(agent.cont.parameters())
    return (
        torch.optim.AdamW(model_parameters, lr=config.model_learning_rate, eps=1e-8),
        torch.optim.AdamW(agent.actor.parameters(), lr=config.learning_rate, eps=1e-8),
        torch.optim.AdamW(agent.critic.parameters(), lr=config.learning_rate, eps=1e-8),
    )


def create_policy(
    actions: Sequence[int],
    *,
    steps: int,
    episodes: int,
    episode_return: float,
    config: Config,
) -> JerkPolicy:
    runs: list[ActionRun] = []
    for action in actions:
        if runs and runs[-1].action == int(action):
            previous = runs[-1]
            runs[-1] = ActionRun(previous.action, previous.duration + 1)
        else:
            runs.append(ActionRun(int(action), 1))
    return JerkPolicy(
        action_names=ACTION_SETS[ACTION_SET],
        action_runs=canonicalize_runs(runs),
        fallback_action=NOOP_ACTION,
        timesteps=steps,
        episodes=episodes,
        best_reward=episode_return,
        metadata={
            "source_algorithm": ALGORITHM,
            "state": config.state,
            "seed": config.seed,
            "frame_skip": config.frame_skip,
            "observation": "grayscale-64x64-hud-mask32-chw",
            "success": "level-change-without-life-loss",
        },
    )


def save_policy(policy: JerkPolicy, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    policy.save(temporary)
    temporary.replace(path)


def evaluate_agent(
    agent: DreamerV3,
    config: Config,
    device: torch.device,
    *,
    deterministic: bool = True,
) -> Evaluation:
    eval_config = Config(**{**asdict(config), "envs": 1})
    env = make_env(eval_config, 1)
    tracker = ProgressReward(env, eval_config)
    actions: list[int] = []
    try:
        obs, _ = env.reset(seed=config.seed + 10_000)
        tracker.reset()
        state = agent.rssm.initial(1, device)
        previous = onehot_actions(np.asarray([NOOP_ACTION]), agent.action_count, device)
        first = torch.ones(1, dtype=torch.bool, device=device)
        with torch.inference_mode():
            for _ in range(config.max_episode_steps):
                tensor = torch.as_tensor(obs.copy(), device=device)
                state = agent.observe_step(state, tensor, previous, first)
                action, _ = agent.policy(state, deterministic=deterministic)
                action_np = action.cpu().numpy().astype(np.int64)
                actions.append(int(action_np[0]))
                obs, _, terminated, truncated, _ = env.step(action_np)
                _, done, completed, info = tracker.step(terminated, truncated)
                if completed[0] or done[0]:
                    return Evaluation(
                        success=bool(completed[0]),
                        episode_return=float(info["return"][0]),
                        progress=float(info["progress"][0]),
                        steps=len(actions),
                        actions=tuple(actions),
                        life_loss=bool(info["life_loss"][0]),
                        stalled=bool(info["stalled"][0]),
                    )
                previous = F.one_hot(action, agent.action_count).float()
                first.fill_(False)
    finally:
        env.close()
    return Evaluation(False, float(tracker.returns[0]), float(tracker.max_global_x[0]), len(actions), tuple(actions), False, False)


def verify_policy(path: Path, config: Config) -> Evaluation:
    policy = JerkPolicy.load(path)
    actions = tuple(
        action
        for run in policy.action_runs
        for action in (run.action,) * run.duration
    )
    eval_config = Config(**{**asdict(config), "envs": 1})
    env = make_env(eval_config, 1)
    tracker = ProgressReward(env, eval_config)
    try:
        env.reset(seed=config.seed + 20_000)
        tracker.reset()
        for step, action in enumerate(actions, 1):
            _, _, terminated, truncated, _ = env.step(np.asarray([action], dtype=np.int64))
            _, done, completed, info = tracker.step(terminated, truncated)
            if completed[0] or done[0]:
                return Evaluation(
                    bool(completed[0]),
                    float(info["return"][0]),
                    float(info["progress"][0]),
                    step,
                    actions[:step],
                    bool(info["life_loss"][0]),
                    bool(info["stalled"][0]),
                )
        return Evaluation(False, float(tracker.returns[0]), float(tracker.max_global_x[0]), len(actions), actions, False, False)
    finally:
        env.close()


def save_checkpoint(
    path: Path,
    agent: DreamerV3,
    optimizers: Sequence[torch.optim.Optimizer],
    config: Config,
    steps: int,
    episodes: int,
    updates: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(
        {
            "algorithm": ALGORITHM,
            "config": asdict(config),
            "agent": agent.state_dict(),
            "optimizers": [optimizer.state_dict() for optimizer in optimizers],
            "steps": steps,
            "episodes": episodes,
            "updates": updates,
        },
        temporary,
    )
    temporary.replace(path)


def load_checkpoint(
    path: Path,
    agent: DreamerV3,
    optimizers: Sequence[torch.optim.Optimizer] | None,
) -> tuple[int, int, int]:
    payload = torch.load(path, map_location=next(agent.parameters()).device, weights_only=False)
    if payload.get("algorithm") != ALGORITHM:
        raise ValueError(f"{path} is not a {ALGORITHM} checkpoint")
    agent.load_state_dict(payload["agent"])
    if optimizers is not None:
        for optimizer, state in zip(optimizers, payload["optimizers"]):
            optimizer.load_state_dict(state)
    return int(payload.get("steps", 0)), int(payload.get("episodes", 0)), int(payload.get("updates", 0))


def _metric_line(metrics: dict[str, Any]) -> str:
    order = ("steps", "updates", "episodes", "fps", "model", "actor", "critic", "imag_return", "entropy", "replay", "best_progress")
    parts = []
    for key in order:
        if key not in metrics:
            continue
        value = metrics[key]
        if isinstance(value, int):
            parts.append(f"{key}={value:,}")
        elif key in {"steps", "updates", "episodes", "replay"}:
            parts.append(f"{key}={int(value):,}")
        else:
            parts.append(f"{key}={float(value):.3f}")
    return " ".join(parts)


def train(config: Config, *, resume: Path | None = None) -> Path:
    if config.state not in list_available_states(config.state_dir):
        raise SystemExit(f"unknown state {config.state!r}")
    device = torch.device(config.device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS was requested but is unavailable; pass --device cpu explicitly")
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    rng = np.random.default_rng(config.seed)
    torch_rng = torch.Generator(device=device).manual_seed(config.seed)

    output = Path(config.output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "dreamerv3.pt"
    policy_path = output / f"{config.state}.zip"
    metrics_path = output / "metrics.jsonl"
    config_path = output / "run_config.json"
    config_path.write_text(json.dumps(asdict(config), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    action_count = len(ACTION_SETS[ACTION_SET])
    agent = DreamerV3(config, action_count, device)
    optimizers = _new_optimizers(agent, config)
    total_steps = episodes = updates = 0
    if resume is not None:
        total_steps, episodes, updates = load_checkpoint(resume, agent, optimizers)
        print(f"resumed {resume} at {total_steps:,} transitions", flush=True)

    replay = Replay(config.envs, config.replay_size, config.batch_length)
    normalizer = ReturnNormalizer()
    env = make_env(config)
    tracker = ProgressReward(env, config)
    stop = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    start_time = time.monotonic()
    interval_time = start_time
    interval_steps = total_steps
    next_log = ((total_steps // config.log_every) + 1) * config.log_every
    next_eval = ((total_steps // config.eval_every) + 1) * config.eval_every
    next_checkpoint = ((total_steps // config.checkpoint_every) + 1) * config.checkpoint_every
    update_credit = 0.0
    recent_metrics: dict[str, float] = {}
    best_progress = 0.0
    trajectories: list[list[int]] = [[] for _ in range(config.envs)]
    exploration_actions = np.full(config.envs, NOOP_ACTION, dtype=np.int64)
    exploration_remaining = np.zeros(config.envs, dtype=np.int64)
    state = agent.rssm.initial(config.envs, device)

    try:
        observations, _ = env.reset(seed=config.seed)
        observations = observations.copy()
        tracker.reset()
        replay.begin(observations)
        previous_actions = onehot_actions(
            np.full(config.envs, NOOP_ACTION, dtype=np.int64), action_count, device
        )
        first = torch.ones(config.envs, dtype=torch.bool, device=device)
        while total_steps < config.steps and not stop:
            with torch.inference_mode():
                tensor = torch.as_tensor(observations, device=device)
                state = agent.observe_step(state, tensor, previous_actions, first)
                if total_steps < config.prefill:
                    actions = rng.integers(0, action_count, config.envs, dtype=np.int64)
                else:
                    sampled, _ = agent.policy(state, deterministic=False)
                    actions = sampled.cpu().numpy().astype(np.int64)
                    continuing = exploration_remaining > 0
                    actions[continuing] = exploration_actions[continuing]
                    exploration_remaining[continuing] -= 1
                    starting = (~continuing) & (
                        rng.random(config.envs) < config.exploration
                    )
                    count = int(np.count_nonzero(starting))
                    if count:
                        exploration_actions[starting] = rng.integers(
                            0, action_count, count, dtype=np.int64
                        )
                        durations = np.minimum(
                            rng.geometric(1.0 / config.exploration_run_mean, count),
                            config.exploration_run_max,
                        )
                        exploration_remaining[starting] = durations - 1
                        actions[starting] = exploration_actions[starting]
            for lane, action in enumerate(actions):
                trajectories[lane].append(int(action))

            next_observations, _, terminated, truncated, _ = env.step(actions)
            next_observations = next_observations.copy()
            rewards, dones, completed, episode_info = tracker.step(terminated, truncated)
            replay.append_step(next_observations, actions, rewards, dones)
            total_steps += config.envs
            best_progress = max(best_progress, float(np.max(episode_info["progress"])))

            if np.any(completed):
                lane = int(np.flatnonzero(completed)[np.argmax(episode_info["return"][completed])])
                policy = create_policy(
                    trajectories[lane],
                    steps=total_steps,
                    episodes=episodes + int(np.count_nonzero(dones)),
                    episode_return=float(episode_info["return"][lane]),
                    config=config,
                )
                save_policy(policy, policy_path)
                save_checkpoint(checkpoint_path, agent, optimizers, config, total_steps, episodes, updates)
                verified = verify_policy(policy_path, config)
                success_row = {
                    "event": "success",
                    "steps": total_steps,
                    "lane": lane,
                    "episode_return": float(episode_info["return"][lane]),
                    "episode_length": int(episode_info["length"][lane]),
                    "policy": str(policy_path),
                    "verified": asdict(verified),
                }
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(success_row, sort_keys=True) + "\n")
                print(json.dumps(success_row, sort_keys=True), flush=True)
                if not verified.success:
                    raise RuntimeError("a completed training trajectory failed clean replay verification")
                if config.stop_on_success:
                    return policy_path

            if np.any(dones):
                replay.finish(dones)
                episodes += int(np.count_nonzero(dones))
                reset_obs, _ = env.reset(options={"reset_mask": dones})
                reset_obs = reset_obs.copy()
                tracker.reset(dones)
                replay.begin(reset_obs, dones)
                next_observations[dones] = reset_obs[dones]
                for lane in np.flatnonzero(dones):
                    trajectories[int(lane)].clear()
                exploration_remaining[dones] = 0

            observations = next_observations
            previous_actions = onehot_actions(actions, action_count, device)
            previous_actions[dones] = onehot_actions(
                np.full(int(np.count_nonzero(dones)), NOOP_ACTION, dtype=np.int64),
                action_count,
                device,
            )
            first = torch.as_tensor(dones.copy(), dtype=torch.bool, device=device)

            if replay.rows >= config.prefill and replay.ready(config.batch_size):
                update_credit += config.envs * config.train_ratio
                batch_cost = config.batch_size * config.batch_length
                while update_credit >= batch_cost:
                    batch = to_torch(replay.sample(config.batch_size, rng), device, action_count)
                    recent_metrics = optimize(agent, batch, optimizers, normalizer, torch_rng)
                    updates += 1
                    update_credit -= batch_cost

            if total_steps >= next_log:
                now = time.monotonic()
                row: dict[str, Any] = {
                    "event": "train",
                    "steps": total_steps,
                    "updates": updates,
                    "episodes": episodes,
                    "fps": (total_steps - interval_steps) / max(now - interval_time, 1e-6),
                    "elapsed": now - start_time,
                    "replay": replay.rows,
                    "best_progress": best_progress,
                    **recent_metrics,
                }
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                print(_metric_line(row), flush=True)
                interval_time, interval_steps = now, total_steps
                next_log += config.log_every

            if total_steps >= next_eval:
                evaluation = evaluate_agent(agent, config, device)
                row = {"event": "evaluation", "steps": total_steps, **asdict(evaluation)}
                row["actions"] = len(evaluation.actions)
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                print(json.dumps(row, sort_keys=True), flush=True)
                if evaluation.success:
                    policy = create_policy(
                        evaluation.actions,
                        steps=total_steps,
                        episodes=episodes,
                        episode_return=evaluation.episode_return,
                        config=config,
                    )
                    save_policy(policy, policy_path)
                    save_checkpoint(checkpoint_path, agent, optimizers, config, total_steps, episodes, updates)
                    verified = verify_policy(policy_path, config)
                    if not verified.success:
                        raise RuntimeError("deterministic evaluation failed action-run verification")
                    return policy_path
                next_eval += config.eval_every

            if total_steps >= next_checkpoint:
                save_checkpoint(checkpoint_path, agent, optimizers, config, total_steps, episodes, updates)
                print(f"checkpoint={checkpoint_path} steps={total_steps:,}", flush=True)
                next_checkpoint += config.checkpoint_every
    finally:
        env.close()
        if not policy_path.is_file():
            save_checkpoint(checkpoint_path, agent, optimizers, config, total_steps, episodes, updates)
    raise RuntimeError(
        f"DreamerV3 stopped after {total_steps:,} transitions without beating {config.state}; "
        f"resume with --resume {checkpoint_path} and a larger --steps budget"
    )


def self_test(device_name: str) -> None:
    device = torch.device(device_name)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS is unavailable")
    config = Config(
        envs=2,
        batch_size=2,
        batch_length=4,
        deter=64,
        hidden=32,
        stoch=4,
        classes=4,
        embed=32,
        cnn_depth=4,
        imagination_starts=4,
        imagination_horizon=3,
        device=device_name,
    )
    agent = DreamerV3(config, len(ACTION_SETS[ACTION_SET]), device)
    optimizers = _new_optimizers(agent, config)
    rng = torch.Generator(device=device).manual_seed(0)
    obs = torch.randint(0, 256, (2, 4, *OBSERVATION_SHAPE), dtype=torch.uint8, device=device)
    actions = F.one_hot(torch.randint(0, agent.action_count, (2, 4), device=device), agent.action_count).float()
    batch = {
        "obs": obs,
        "action": actions,
        "reward": torch.randn(2, 4, device=device),
        "cont": torch.ones(2, 4, device=device),
        "first": torch.tensor([[True, False, False, False], [True, False, False, False]], device=device),
    }
    metrics = optimize(agent, batch, optimizers, ReturnNormalizer(), rng)
    values = torch.linspace(-100, 100, 101, device=device)
    roundtrip = symexp(symlog(values))
    if not torch.allclose(values, roundtrip, atol=1e-4, rtol=1e-4):
        raise AssertionError("symlog/symexp roundtrip failed")
    if not all(math.isfinite(value) for value in metrics.values()):
        raise AssertionError(f"non-finite self-test metrics: {metrics}")
    print(json.dumps({"self_test": "passed", "device": device_name, **metrics}, sort_keys=True))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("state", nargs="?", default="Level1-1")
    parser.add_argument("--steps", type=int, default=2_000_000)
    parser.add_argument("--envs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="mps", choices=("mps", "cpu"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--rom")
    parser.add_argument("--state-dir")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--prefill", type=int, default=20_000)
    parser.add_argument("--replay-size", type=int, default=200_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--batch-length", type=int, default=32)
    parser.add_argument("--train-ratio", type=float, default=8.0)
    parser.add_argument("--exploration", type=float, default=0.05)
    parser.add_argument("--exploration-run-mean", type=float, default=4.0)
    parser.add_argument("--exploration-run-max", type=int, default=32)
    parser.add_argument("--entropy-scale", type=float, default=3e-3)
    parser.add_argument("--eval-every", type=int, default=50_000)
    parser.add_argument("--checkpoint-every", type=int, default=100_000)
    parser.add_argument("--log-every", type=int, default=10_000)
    parser.add_argument("--max-episode-steps", type=int, default=4_500)
    parser.add_argument("--stall-steps", type=int, default=450)
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument(
        "--noop-reset-max",
        type=int,
        default=0,
        metavar="FRAMES",
        help=(
            "maximum seeded random raw emulator frames applied after ordinary "
            "state resets (default: 0, disabled)"
        ),
    )
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--verify-policy", type=Path)
    parser.add_argument("--no-stop-on-success", action="store_true")
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> Config:
    if not 0.0 <= args.exploration <= 1.0:
        raise SystemExit("--exploration must be between zero and one")
    if args.exploration_run_mean < 1.0:
        raise SystemExit("--exploration-run-mean must be at least one")
    if args.exploration_run_max < 1:
        raise SystemExit("--exploration-run-max must be positive")
    if args.noop_reset_max < 0:
        raise SystemExit("--noop-reset-max must be non-negative")
    output = args.output or Path("runs") / "dreamerv3" / args.state
    return Config(
        state=args.state,
        seed=args.seed,
        envs=args.envs,
        steps=args.steps,
        prefill=args.prefill,
        replay_size=args.replay_size,
        batch_size=args.batch_size,
        batch_length=args.batch_length,
        train_ratio=args.train_ratio,
        exploration=args.exploration,
        exploration_run_mean=args.exploration_run_mean,
        exploration_run_max=args.exploration_run_max,
        entropy_scale=args.entropy_scale,
        eval_every=args.eval_every,
        checkpoint_every=args.checkpoint_every,
        log_every=args.log_every,
        max_episode_steps=args.max_episode_steps,
        stall_steps=args.stall_steps,
        frame_skip=args.frame_skip,
        noop_reset_max=args.noop_reset_max,
        device=args.device,
        output=str(output),
        rom=args.rom,
        state_dir=args.state_dir,
        stop_on_success=not args.no_stop_on_success,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        self_test(args.device)
        return 0
    config = config_from_args(args)
    if args.verify_policy:
        result = verify_policy(args.verify_policy, config)
        print(json.dumps(asdict(result), sort_keys=True))
        return 0 if result.success else 1
    if args.eval_only:
        checkpoint = args.resume or Path(config.output) / "dreamerv3.pt"
        device = torch.device(config.device)
        agent = DreamerV3(config, len(ACTION_SETS[ACTION_SET]), device)
        load_checkpoint(checkpoint, agent, None)
        result = evaluate_agent(agent, config, device)
        print(json.dumps(asdict(result), sort_keys=True))
        return 0 if result.success else 1
    policy = train(config, resume=args.resume)
    print(f"verified_policy={policy}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
