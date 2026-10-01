"""PPO adapters and risk-constrained training for :mod:`threat_aware_qkd`."""

from __future__ import annotations

from collections import deque
from dataclasses import replace
from pathlib import Path

import numpy as np

from threat_aware_qkd import AvailabilityEnv, SimConfig, evaluate_policy


OBS_MODES = ("qber_only", "metadata", "metadata_pool")
DEFAULT_PENALTIES = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0)


def _base_features(obs: dict, cfg: SimConfig, obs_mode: str) -> np.ndarray:
    max_count = max(float(np.max(cfg.demand_rates)), 1.0)
    metadata = np.log1p(max(0.0, obs["metadata_count"])) / np.log1p(max_count * 2.0)
    pool = obs["pool_fraction"] if obs_mode == "metadata_pool" else 0.0
    if obs_mode == "qber_only":
        metadata = 0.0
    return np.asarray([
        np.clip(metadata, 0.0, 1.5),
        np.clip(obs["qber_hat"] / 0.10, 0.0, 1.5),
        np.clip(obs["cusum_frac"] / 2.0, 0.0, 2.0),
        np.clip(obs["budget_frac"], 0.0, 1.0),
        np.clip(pool, 0.0, 1.0) if pool >= 0 else 0.0,
        np.clip(obs["time_fraction"], 0.0, 1.0),
    ], dtype=np.float32)


def _append_frame(history: deque, features: np.ndarray, frame_stack: int, reset=False):
    if reset:
        history.clear()
        for _ in range(frame_stack):
            history.append(features.copy())
    else:
        history.append(features.copy())
    return np.concatenate(tuple(history)).astype(np.float32)


def make_gym_env_class():
    """Load Gymnasium lazily so non-RL simulator use has no Gym import cost."""
    import gymnasium as gym
    from gymnasium import spaces

    class GymAvailabilityEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self, cfg: SimConfig, obs_mode="metadata", frame_stack=5,
                     detection_penalty=0.5, reward_scale=1000.0):
            super().__init__()
            if obs_mode not in OBS_MODES:
                raise ValueError(f"obs_mode must be one of {OBS_MODES}")
            self.cfg, self.obs_mode = cfg, obs_mode
            self.frame_stack = frame_stack
            self.detection_penalty = float(detection_penalty)
            self.reward_scale = float(reward_scale)
            self.core = AvailabilityEnv(cfg)
            self.history = deque(maxlen=frame_stack)
            self.action_space = spaces.Discrete(len(cfg.attack_levels))
            self.observation_space = spaces.Box(
                low=0.0, high=2.0, shape=(6 * frame_stack,), dtype=np.float32
            )

        def _transform(self, obs, reset=False):
            features = _base_features(obs, self.cfg, self.obs_mode)
            return _append_frame(self.history, features, self.frame_stack, reset)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            obs = self.core.reset(
                seed=seed,
                metadata_visible=self.obs_mode in {"metadata", "metadata_pool"},
                pool_visible=self.obs_mode == "metadata_pool",
            )
            return self._transform(obs, reset=True), {}

        def step(self, action):
            attack = self.cfg.attack_levels[int(action)]
            obs, row, done = self.core.step(attack)
            reward = row["unmet_demand"] / self.reward_scale
            if row["new_detection"]:
                reward -= self.detection_penalty
            terminated = bool(done and self.core.detected and self.cfg.stop_on_detection)
            truncated = bool(done and not terminated)
            if obs is None:
                obs = {
                    "metadata_count": 0, "qber_hat": row["qber_hat"],
                    "cusum_frac": row["cusum"] / max(self.cfg.cusum_threshold, 1e-12),
                    "budget_frac": self.core.budget_remaining / max(self.cfg.attack_budget, 1e-12),
                    "pool_fraction": self.core.pool / self.cfg.pool_capacity,
                    "time_fraction": self.core.t / self.cfg.horizon,
                }
            return self._transform(obs), float(reward), terminated, truncated, row

    return GymAvailabilityEnv


def make_vec_env(cfg: SimConfig, obs_mode: str, seed: int, n_envs: int = 4,
                 frame_stack: int = 5, detection_penalty: float = 0.5):
    from stable_baselines3.common.vec_env import DummyVecEnv

    GymAvailabilityEnv = make_gym_env_class()
    constructors = []
    for rank in range(n_envs):
        env_cfg = replace(cfg, seed=seed + rank)
        constructors.append(lambda env_cfg=env_cfg: GymAvailabilityEnv(
            env_cfg, obs_mode=obs_mode, frame_stack=frame_stack,
            detection_penalty=detection_penalty,
        ))
    return DummyVecEnv(constructors)


def make_model(vec_env, seed: int, recurrent: bool = False, learning_rate=3e-4,
               n_steps=256, batch_size=128, ent_coef=0.01):
    if recurrent:
        from sb3_contrib import RecurrentPPO
        cls = RecurrentPPO
        policy = "MlpLstmPolicy"
    else:
        from stable_baselines3 import PPO
        cls = PPO
        policy = "MlpPolicy"
    return cls(
        policy, vec_env, learning_rate=learning_rate, n_steps=n_steps,
        batch_size=batch_size, n_epochs=10, gamma=0.99, gae_lambda=0.95,
        ent_coef=ent_coef, vf_coef=0.5, max_grad_norm=0.5,
        policy_kwargs={"net_arch": [64, 64]}, verbose=0, seed=seed, device="cpu",
    )


class PPOAttackPolicy:
    """Convert the simulator's dict observation to the trained frame-stacked input."""

    def __init__(self, cfg: SimConfig, model, obs_mode: str, frame_stack=5, recurrent=False):
        self.cfg, self.model, self.obs_mode = cfg, model, obs_mode
        self.frame_stack, self.recurrent = frame_stack, recurrent
        self.history = deque(maxlen=frame_stack)
        self.reset()

    def reset(self):
        self.history = deque(maxlen=self.frame_stack)
        self.lstm_state = None
        self.episode_start = True

    def act(self, obs: dict, t: int) -> float:
        features = _base_features(obs, self.cfg, self.obs_mode)
        state = _append_frame(self.history, features, self.frame_stack, reset=(t == 0))
        if self.recurrent:
            action, self.lstm_state = self.model.predict(
                state.reshape(1, -1), state=self.lstm_state,
                episode_start=np.asarray([self.episode_start]), deterministic=True,
            )
            self.episode_start = False
        else:
            action, _ = self.model.predict(state.reshape(1, -1), deterministic=True)
        return float(self.cfg.attack_levels[int(np.asarray(action).reshape(-1)[0])])


def train_penalty_sweep(
    cfg: SimConfig,
    obs_mode: str,
    seed: int,
    detection_limit: float,
    total_timesteps: int = 100_000,
    n_envs: int = 4,
    penalty_multipliers=DEFAULT_PENALTIES,
    validation_episodes: int = 100,
    test_episodes: int = 200,
    validation_seed0: int = 10_000,
    test_seed0: int = 100_000,
    recurrent: bool = False,
    frame_stack: int = 5,
    model_dir: str | Path = "models/threat_aware",
):
    """Train penalty-weighted PPO variants; select on validation risk, test once."""
    import pandas as pd
    from stable_baselines3.common.utils import set_random_seed

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    trained = []
    for multiplier in penalty_multipliers:
        penalty = float(multiplier)
        set_random_seed(seed)
        vec_env = make_vec_env(cfg, obs_mode, seed, n_envs, frame_stack, penalty)
        model = make_model(vec_env, seed, recurrent=recurrent)
        model.learn(total_timesteps=total_timesteps)
        path = model_dir / f"{obs_mode}_seed{seed}_lambda{penalty:g}{'_lstm' if recurrent else ''}"
        model.save(str(path))
        vec_env.close()

        factory = lambda i, model=model: PPOAttackPolicy(
            cfg, model, obs_mode, frame_stack=frame_stack, recurrent=recurrent
        )
        val = evaluate_policy(
            cfg, factory, validation_episodes, validation_seed0,
            metadata_visible=obs_mode in {"metadata", "metadata_pool"},
            pool_visible=obs_mode == "metadata_pool",
        )
        trained.append({
            "lambda": penalty, "model": model,
            "validation_damage": float(val.damage.mean()),
            "validation_detection": float(val.detected.mean()),
            "validation_budget": float(val.attack_budget_used.mean()),
            "model_path": str(path),
        })

    feasible = [x for x in trained if x["validation_detection"] <= detection_limit]
    selected = max(feasible, key=lambda x: x["validation_damage"]) if feasible else min(
        trained, key=lambda x: x["validation_detection"]
    )
    model = selected["model"]
    factory = lambda i: PPOAttackPolicy(
        cfg, model, obs_mode, frame_stack=frame_stack, recurrent=recurrent
    )
    test = evaluate_policy(
        cfg, factory, test_episodes, test_seed0,
        metadata_visible=obs_mode in {"metadata", "metadata_pool"},
        pool_visible=obs_mode == "metadata_pool",
    )
    test_summary = {
        "obs_mode": obs_mode, "seed": seed, "recurrent": recurrent,
        "selected_lambda": selected["lambda"],
        "validation_damage": selected["validation_damage"],
        "validation_detection": selected["validation_detection"],
        "validation_feasible": selected["validation_detection"] <= detection_limit,
        "test_damage": float(test.damage.mean()),
        "test_detection": float(test.detected.mean()),
        "test_budget_used": float(test.attack_budget_used.mean()),
        "test_pool_fraction": float(test.mean_pool_fraction.mean()),
        "test_overhead": float(test.metadata_overhead.mean()),
        "model_path": selected["model_path"],
    }
    sweep = pd.DataFrame([{k: v for k, v in x.items() if k != "model"} for x in trained])
    return model, test_summary, sweep


def evaluate_model_mismatch(train_cfg: SimConfig, test_cfg: SimConfig, model,
                            obs_mode="metadata", episodes=200, seed0=300_000,
                            recurrent=False, frame_stack=5):
    """Evaluate a fixed trained attacker after changing the true environment."""
    factory = lambda i: PPOAttackPolicy(
        train_cfg, model, obs_mode, frame_stack=frame_stack, recurrent=recurrent
    )
    return evaluate_policy(
        test_cfg, factory, episodes, seed0,
        metadata_visible=obs_mode in {"metadata", "metadata_pool"},
        pool_visible=obs_mode == "metadata_pool",
    )
