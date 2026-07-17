"""Provider-owned Super Mario Bros task, reward, and episode semantics."""

from __future__ import annotations

import copy
from collections.abc import Mapping

import gymnasium as gym


class DeclarativeTaskEnv(gym.Wrapper):
    """Apply a declarative Mario reward and episode contract to one environment."""

    def __init__(self, env, task):
        super().__init__(env)
        self.task = copy.deepcopy(dict(task))
        self.signals = self.task.get("signals") or {}
        self.events = self.task.get("events") or {}
        self.termination = self.task.get("termination") or {}
        self.reward_config = self.task.get("reward") or {}
        self._validate_contract()
        self._episode_steps = 0
        self._last_progress_step = 0

    def _validate_contract(self):
        expected_events = {
            "life_loss": ("lives", "decrease"),
            "level_change": ("level", "change"),
            "stalled": ("x", "unchanged_for"),
        }
        unknown = sorted(set(self.events) - set(expected_events))
        if unknown:
            raise ValueError(f"Unsupported Mario task event(s): {', '.join(unknown)}")
        for name, expected in expected_events.items():
            if name not in self.events:
                continue
            rule = self.events[name]
            if not isinstance(rule, Mapping):
                raise ValueError(f"Mario task event {name!r} must be an object")
            actual = (rule.get("signal"), rule.get("operation"))
            if actual != expected:
                raise ValueError(
                    f"Mario task event {name!r} requires signal={expected[0]!r}, "
                    f"operation={expected[1]!r}"
                )
        reward_mode = self.reward_config.get("reward_mode", "native")
        if reward_mode not in {"native", "bounded", "baseline", "score", "additive"}:
            raise ValueError(f"Unsupported Mario task reward mode {reward_mode!r}")

    @staticmethod
    def _signal(info, source, *, pair=False):
        names = (source,) if isinstance(source, str) else tuple(source)
        try:
            values = tuple(int(info[name]) for name in names)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Mario task signal {source!r} is absent from provider info"
            ) from exc
        if pair:
            if len(values) != 2:
                raise ValueError(
                    f"Mario task pair signal {source!r} must contain two fields"
                )
            return values
        if len(values) == 1:
            return values[0]
        if len(values) == 2:
            return values[0] * 256 + values[1]
        raise ValueError(
            f"Mario task signal {source!r} has unsupported width {len(values)}"
        )

    def _read_state(self, info):
        return {
            "x": self._signal(info, self.signals.get("x", ("xscrollHi", "xscrollLo"))),
            "score": self._signal(info, self.signals.get("score", "score")),
            "lives": self._signal(info, self.signals.get("lives", "lives")),
            "level": self._signal(
                info,
                self.signals.get("level", ("levelHi", "levelLo")),
                pair=True,
            ),
        }

    def reset(self, *, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)
        state = self._read_state(info)
        self._level_max_x = state["x"]
        self._completed_level_base = 0
        self._max_global_x = state["x"]
        self._score = state["score"]
        self._lives = state["lives"]
        self._level = state["level"]
        self._episode_steps = 0
        self._last_progress_step = 0
        return observation, info

    def step(self, action):
        observation, native_reward, provider_terminated, provider_truncated, info = (
            self.env.step(action)
        )
        info = dict(info)
        state = self._read_state(info)
        lost_life = state["lives"] < self._lives
        changed_level = state["level"] != self._level
        completed = changed_level and not lost_life

        if completed:
            self._completed_level_base += self._level_max_x
            self._level_max_x = 0
        effective_x = 0 if changed_level else state["x"]
        self._level_max_x = max(self._level_max_x, effective_x)
        global_x = self._completed_level_base + effective_x
        global_max_x = self._completed_level_base + self._level_max_x
        progress_delta = max(global_max_x - self._max_global_x, 0)
        self._max_global_x = max(self._max_global_x, global_max_x)
        score_delta = max(state["score"] - self._score, 0)

        no_progress_min_delta = int(self.termination.get("no_progress_min_delta", 0))
        if progress_delta > no_progress_min_delta:
            self._last_progress_step = self._episode_steps
        self._episode_steps += 1
        stalled_rule = self.events.get("stalled") or {}
        stalled_steps = int(stalled_rule.get("steps", 0))
        stalled = bool(
            stalled_steps > 0
            and self._episode_steps - self._last_progress_step >= stalled_steps
        )

        failure_events = set(self.termination.get("failure") or ())
        success_events = set(self.termination.get("success") or ())
        task_terminated = (
            (lost_life and "life_loss" in failure_events)
            or (completed and "level_change" in success_events)
            or (stalled and "stalled" in failure_events)
        )
        max_episode_steps = int(self.termination.get("max_episode_steps", 0))
        task_truncated = (stalled and "stalled" not in failure_events) or (
            max_episode_steps > 0 and self._episode_steps >= max_episode_steps
        )
        if task_terminated:
            task_truncated = False

        terminated = bool(provider_terminated or task_terminated)
        truncated = bool((provider_truncated or task_truncated) and not terminated)
        reward = self._shape_reward(
            float(native_reward),
            progress_delta=progress_delta,
            score_delta=score_delta,
            completed=completed,
            lost_life=lost_life,
            done=terminated or truncated,
        )

        emitted_events = []
        if lost_life and "life_loss" in self.events:
            emitted_events.append("life_loss")
        if changed_level and "level_change" in self.events:
            emitted_events.append("level_change")
        if stalled and "stalled" in self.events:
            emitted_events.append("stalled")
        outcome = "neutral"
        if truncated:
            outcome = "timeout"
        if completed:
            outcome = "success"
        if (
            lost_life
            or (stalled and "stalled" in failure_events)
            or (provider_terminated and not completed)
        ):
            outcome = "failure"
        info.update(
            {
                "task_events": emitted_events,
                "task_outcome": outcome,
                "level_complete": completed,
                "life_loss": lost_life,
                "progress_delta": progress_delta,
                "score_delta": score_delta,
                "global_x_pos": global_x,
                "global_max_x_pos": self._max_global_x,
                "native_reward": float(native_reward),
                "shaped_reward": reward,
            }
        )

        self._score = state["score"]
        self._lives = state["lives"]
        self._level = state["level"]
        return observation, reward, terminated, truncated, info

    def _shape_reward(
        self,
        native_reward,
        *,
        progress_delta,
        score_delta,
        completed,
        lost_life,
        done,
    ):
        config = self.reward_config
        mode = str(config.get("reward_mode", "native"))
        reward_scale = float(config.get("reward_scale", 10.0)) or 1.0
        terminal_reward = float(config.get("terminal_reward", 50.0))
        progress_cap = float(config.get("progress_reward_cap", 30.0))
        capped_progress = min(float(progress_delta), progress_cap)
        if mode == "native":
            shaped = native_reward
        elif mode == "bounded":
            raw = (
                terminal_reward
                if completed
                else -terminal_reward
                if lost_life
                else capped_progress
            )
            shaped = raw / reward_scale
        elif mode == "baseline":
            raw = native_reward + float(score_delta) / 40.0
            if completed:
                raw += terminal_reward
            elif done:
                raw -= terminal_reward
            shaped = raw / reward_scale
        else:
            native_component = (
                native_reward if config.get("use_native_reward", False) else 0.0
            )
            if mode == "score":
                progress = (
                    capped_progress
                    if config.get("score_progress_clipped", False)
                    else float(progress_delta)
                )
                score_component = 0.01 * float(score_delta)
            else:
                progress = float(progress_delta)
                score_component = 0.0
            shaped = (
                native_component
                + float(config.get("progress_reward_scale", 1.0)) * progress
                + score_component
                + (float(config.get("completion_reward", 0.0)) if completed else 0.0)
                - (float(config.get("death_penalty", 25.0)) if lost_life else 0.0)
            )
        shaped -= float(config.get("time_penalty", 0.0))
        if config.get("clip_rewards", False):
            shaped = 1.0 if shaped > 0 else -1.0 if shaped < 0 else 0.0
        return float(shaped)
