"""Controlled simulator for stealth-constrained QKD availability attacks.

This is a research simulator, not a QKD security implementation. It models
finite-sample QBER monitoring, an attack-limited BB84 link, a capped key pool,
and hidden Markov application demand with separately controlled metadata
coupling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


STATE_NAMES = ("idle", "normal", "burst")
STATE_PRIOR = np.array([0.20, 0.60, 0.20], dtype=float)
STATE_DEMAND_RATIO = np.array([0.15, 1.0, 4.0], dtype=float)


def binary_entropy(q: float) -> float:
    q = float(np.clip(q, 1e-12, 1.0 - 1e-12))
    return float(-q * np.log2(q) - (1.0 - q) * np.log2(1.0 - q))


def bb84_key_fraction(qber: float) -> float:
    """Simplified asymptotic BB84 rate fraction, max(0, 1 - 2 h2(Q))."""
    return max(0.0, 1.0 - 2.0 * binary_entropy(qber))


def markov_transition(persistence: float, prior=STATE_PRIOR) -> np.ndarray:
    """Three-state refresh chain with the requested stationary distribution."""
    p = float(np.clip(persistence, 0.0, 0.999999))
    pi = np.asarray(prior, dtype=float)
    pi = pi / pi.sum()
    return p * np.eye(3) + (1.0 - p) * np.tile(pi, (3, 1))


@dataclass
class SimConfig:
    horizon: int = 240
    qber_baseline: float = 0.01
    qber_sample_size: int = 500
    cusum_reference: float = 0.003
    cusum_threshold: float | None = None
    false_alarm_target: float = 0.01
    raw_key_bits_per_round: float = 1000.0
    pool_capacity: float = 5000.0
    initial_pool_fraction: float = 1.0
    load: float = 1.0
    metadata_coupling: float = 0.5
    metadata_noise: float = 0.0
    persistence: float = 0.90
    attack_levels: tuple[float, ...] = (0.0, 0.01, 0.03, 0.05)
    attack_budget: float = 3.0  # sum_t f_t; one full-strength round costs 0.05
    observe_pool: bool = False
    stop_on_detection: bool = True
    seed: int = 0
    demand_state_ratio: tuple[float, ...] = tuple(STATE_DEMAND_RATIO)
    state_prior: tuple[float, ...] = tuple(STATE_PRIOR)

    @property
    def prior(self) -> np.ndarray:
        p = np.asarray(self.state_prior, dtype=float)
        return p / p.sum()

    @property
    def transition(self) -> np.ndarray:
        return markov_transition(self.persistence, self.prior)

    @property
    def demand_rates(self) -> np.ndarray:
        ratios = np.asarray(self.demand_state_ratio, dtype=float)
        baseline_supply = self.raw_key_bits_per_round * bb84_key_fraction(self.qber_baseline)
        return ratios * (self.load * baseline_supply / float(self.prior @ ratios))


def calibrate_cusum_threshold(
    qber_baseline: float,
    sample_size: int,
    horizon: int,
    reference: float,
    false_alarm_target: float = 0.01,
    episodes: int = 6000,
    seed: int = 9127,
) -> float:
    """Calibrate h from attack-free finite-sample QBER paths."""
    rng = np.random.default_rng(seed)
    s = np.zeros(episodes, dtype=float)
    peaks = np.zeros(episodes, dtype=float)
    for _ in range(horizon):
        qhat = rng.binomial(sample_size, qber_baseline, size=episodes) / sample_size
        s = np.maximum(0.0, s + qhat - qber_baseline - reference)
        peaks = np.maximum(peaks, s)
    return float(np.quantile(peaks, 1.0 - false_alarm_target, method="higher"))


class AvailabilityEnv:
    """Finite-horizon simulator; observation is a dict for transparent policies."""

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        if cfg.cusum_threshold is None:
            cfg.cusum_threshold = calibrate_cusum_threshold(
                cfg.qber_baseline, cfg.qber_sample_size, cfg.horizon,
                cfg.cusum_reference, cfg.false_alarm_target,
            )
        self.transition = cfg.transition
        self.prior = cfg.prior
        self.demand_rates = cfg.demand_rates

    def _metadata_count(self, state: int) -> int:
        # With probability rho metadata is generated from the real state.
        # Otherwise it is generated from an independent stationary state.
        # Consequently rho=0 is a true null control for metadata.
        rho = float(np.clip(self.cfg.metadata_coupling, 0.0, 1.0))
        source_state = state if self.rng.random() < rho else int(self.rng.choice(3, p=self.prior))
        count = int(self.rng.poisson(self.demand_rates[source_state]))
        noise = self.cfg.metadata_noise
        if noise > 0:
            count = max(0, int(round(count + rng.normal(0.0, noise * np.sqrt(max(count, 1))))))
        return count

    def _observation(self) -> dict:
        return {
            "metadata_count": self.current_metadata if self.metadata_visible else -1,
            "qber_hat": self.last_qber,
            "cusum": self.cusum,
            "cusum_frac": self.cusum / max(self.cfg.cusum_threshold, 1e-12),
            "budget_remaining": self.budget_remaining,
            "budget_frac": self.budget_remaining / max(self.cfg.attack_budget, 1e-12),
            "pool_fraction": self.pool / self.cfg.pool_capacity if self.pool_visible else -1.0,
            "time_fraction": self.t / self.cfg.horizon,
        }

    def reset(self, seed: int | None = None, metadata_visible: bool = True, pool_visible: bool | None = None):
        self.rng = np.random.default_rng(self.cfg.seed if seed is None else seed)
        self.metadata_visible = bool(metadata_visible)
        self.pool_visible = self.cfg.observe_pool if pool_visible is None else bool(pool_visible)
        self.state = int(self.rng.choice(3, p=self.prior))
        self.t = 0
        self.pool = self.cfg.initial_pool_fraction * self.cfg.pool_capacity
        self.budget_remaining = self.cfg.attack_budget
        self.cusum = 0.0
        self.last_qber = self.cfg.qber_baseline
        self.detected = False
        self.done = False
        self.current_metadata = self._metadata_count(self.state)
        self.rows = []
        return self._observation()

    def step(self, requested_attack: float):
        if self.done:
            raise RuntimeError("step() called after episode completion")
        c = self.cfg
        f = float(np.clip(requested_attack, 0.0, max(c.attack_levels)))
        f = min(f, self.budget_remaining)
        self.budget_remaining = max(0.0, self.budget_remaining - f)

        demand = float(self.rng.poisson(self.demand_rates[self.state]))
        true_qber = min(1.0, c.qber_baseline + 0.25 * f)
        errors = int(self.rng.binomial(c.qber_sample_size, true_qber))
        qhat = errors / c.qber_sample_size
        self.cusum = max(0.0, self.cusum + qhat - c.qber_baseline - c.cusum_reference)
        self.last_qber = qhat

        generated = c.raw_key_bits_per_round * bb84_key_fraction(true_qber)
        available = min(c.pool_capacity, self.pool + generated)
        overflow = max(0.0, self.pool + generated - c.pool_capacity)
        unmet = max(0.0, demand - available)
        self.pool = max(0.0, available - demand)
        newly_detected = self.cusum >= c.cusum_threshold and not self.detected
        if newly_detected:
            self.detected = True

        row = {
            "round": self.t, "state": self.state, "metadata_count": self.current_metadata,
            "demand": demand, "attack_fraction": f, "true_qber": true_qber,
            "qber_hat": qhat, "cusum": self.cusum, "generated_bits": generated,
            "pool": self.pool, "unmet_demand": unmet, "pool_overflow": overflow,
            "detected": bool(self.detected), "new_detection": bool(newly_detected),
        }
        self.rows.append(row)
        self.t += 1
        self.done = self.t >= c.horizon or (newly_detected and c.stop_on_detection)
        if not self.done:
            self.state = int(self.rng.choice(3, p=self.transition[self.state]))
            self.current_metadata = self._metadata_count(self.state)
        obs = self._observation() if not self.done else None
        return obs, row, self.done


def poisson_pmf(k: int, rate: float) -> float:
    if k < 0:
        return 0.0
    # Stable enough for the moderate rates used by the controlled simulator.
    from math import exp, lgamma, log
    if rate == 0:
        return 1.0 if k == 0 else 0.0
    return float(exp(-rate + k * log(rate) - lgamma(k + 1)))


class BeliefPolicy:
    """HMM filtering policy with a detector-state throttle and an optional pool signal."""

    def __init__(self, cfg: SimConfig, use_metadata: bool, pool_aware: bool = False,
                 state_threshold: float = 0.0, cusum_limit: float = 1.0,
                 attack_level: float | None = None):
        self.cfg = cfg
        self.use_metadata = use_metadata
        self.pool_aware = pool_aware
        self.state_threshold = state_threshold
        self.cusum_limit = cusum_limit
        self.attack_level = max(cfg.attack_levels) if attack_level is None else attack_level
        self.reset()

    def reset(self):
        self.belief = self.cfg.prior.copy()
        self.first = True

    def _update_belief(self, count: int):
        if not self.first:
            self.belief = self.belief @ self.cfg.transition
        self.first = False
        if self.use_metadata and count >= 0:
            rho = float(np.clip(self.cfg.metadata_coupling, 0.0, 1.0))
            marginal = float(self.cfg.prior @ np.array([
                poisson_pmf(count, x) for x in self.cfg.demand_rates
            ]))
            likelihood = np.array([
                rho * poisson_pmf(count, self.cfg.demand_rates[s]) + (1.0 - rho) * marginal
                for s in range(3)
            ])
            self.belief *= likelihood
            z = self.belief.sum()
            self.belief = self.belief / z if z > 0 else self.cfg.prior.copy()

    def act(self, obs: dict, t: int) -> float:
        self._update_belief(int(obs["metadata_count"]))
        shortage_signal = float(self.belief[2]) >= self.state_threshold
        if self.pool_aware and obs["pool_fraction"] >= 0:
            shortage_signal = shortage_signal or obs["pool_fraction"] < 0.35
        if obs["cusum_frac"] >= self.cusum_limit or not shortage_signal:
            return 0.0
        return self.attack_level


class RandomPolicy:
    def __init__(self, cfg: SimConfig, seed: int = 0):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
    def reset(self):
        pass
    def act(self, obs: dict, t: int) -> float:
        return float(self.rng.choice(self.cfg.attack_levels))


class PeriodicPolicy:
    def __init__(self, cfg: SimConfig, period: int = 8, phase: int = 0):
        self.cfg, self.period, self.phase = cfg, period, phase
    def reset(self):
        pass
    def act(self, obs: dict, t: int) -> float:
        return float(max(self.cfg.attack_levels) if (t - self.phase) % self.period == 0 else 0.0)


class DetectorAwarePolicy:
    def __init__(self, cfg: SimConfig, cusum_limit: float = 0.6):
        self.cfg, self.cusum_limit = cfg, cusum_limit
    def reset(self):
        pass
    def act(self, obs: dict, t: int) -> float:
        return float(max(self.cfg.attack_levels) if obs["cusum_frac"] < self.cusum_limit else 0.0)


class StateOraclePolicy:
    """Upper bound that observes the current hidden demand state."""
    def __init__(self, cfg: SimConfig, cusum_limit: float = 1.0):
        self.cfg, self.cusum_limit = cfg, cusum_limit
    def reset(self):
        pass
    def bind(self, env: AvailabilityEnv):
        self.env = env
    def act(self, obs: dict, t: int) -> float:
        if obs["cusum_frac"] >= self.cusum_limit:
            return 0.0
        return float(max(self.cfg.attack_levels) if self.env.state == 2 else 0.0)


def run_episode(cfg: SimConfig, policy, seed: int, metadata_visible: bool = True,
                pool_visible: bool | None = None) -> tuple[dict, list[dict]]:
    env = AvailabilityEnv(cfg)
    obs = env.reset(seed, metadata_visible=metadata_visible, pool_visible=pool_visible)
    if hasattr(policy, "reset"):
        policy.reset()
    if hasattr(policy, "bind"):
        policy.bind(env)
    while not env.done:
        attack = policy.act(obs, env.t) if hasattr(policy, "act") else policy(obs, env.t)
        obs, _, done = env.step(attack)
        if done:
            break
    log = env.rows
    return ({
        "damage": float(sum(r["unmet_demand"] for r in log)),
        "damage_rounds": int(sum(r["unmet_demand"] > 0 for r in log)),
        "detected": bool(env.detected),
        "episode_length": len(log),
        "attack_budget_used": float(sum(r["attack_fraction"] for r in log)),
        "mean_pool_fraction": float(np.mean([r["pool"] for r in log]) / cfg.pool_capacity),
        "pool_overflow": float(sum(r["pool_overflow"] for r in log)),
        "mean_attack": float(np.mean([r["attack_fraction"] for r in log])),
    }, log)


def evaluate_policy(cfg: SimConfig, policy_factory: Callable[[int], object], episodes: int = 200,
                    seed0: int = 100_000, metadata_visible: bool = True,
                    pool_visible: bool | None = None):
    import pandas as pd
    rows = []
    for i in range(episodes):
        policy = policy_factory(i)
        metrics, _ = run_episode(cfg, policy, seed0 + i, metadata_visible, pool_visible)
        rows.append(metrics)
    return pd.DataFrame(rows)


def select_under_detection_constraint(
    cfg: SimConfig,
    policy_factory: Callable[[float, float], object],
    detection_limit: float,
    threshold_grid=(0.25, 0.5, 0.75, 1.0, 1.25, 1.5),
    episodes: int = 120,
    seed0: int = 10_000,
    metadata_visible: bool = True,
    pool_visible: bool | None = None,
) -> tuple[dict, object]:
    """Validation-only threshold selection; caller should test on disjoint seeds."""
    candidates = []
    for state_threshold in threshold_grid:
        for cusum_limit in threshold_grid:
            factory = lambda i, st=state_threshold, cl=cusum_limit: policy_factory(st, cl)
            df = evaluate_policy(cfg, factory, episodes, seed0, metadata_visible, pool_visible)
            row = {
                "state_threshold": state_threshold, "cusum_limit": cusum_limit,
                "damage": float(df.damage.mean()), "detection_rate": float(df.detected.mean()),
                "budget_used": float(df.attack_budget_used.mean()),
            }
            candidates.append((row, state_threshold, cusum_limit))
    feasible = [x for x in candidates if x[0]["detection_rate"] <= detection_limit]
    if not feasible:
        chosen = min(candidates, key=lambda x: x[0]["detection_rate"])
    else:
        chosen = max(feasible, key=lambda x: x[0]["damage"])
    row, st, cl = chosen
    row["feasible"] = row["detection_rate"] <= detection_limit
    return row, policy_factory(st, cl)


def metadata_predictive_logloss(cfg: SimConfig, episodes: int = 200, seed0: int = 200_000):
    """Evaluate one-step hidden-state prediction with/without metadata filtering."""
    losses_prior, losses_meta = [], []
    for ep in range(episodes):
        env = AvailabilityEnv(cfg)
        env.reset(seed0 + ep, metadata_visible=True)
        belief = cfg.prior.copy()
        for t in range(cfg.horizon):
            if t:
                belief = belief @ cfg.transition
            state = env.state
            losses_prior.append(-np.log(max(belief[state], 1e-12)))
            count = env.current_metadata
            policy = BeliefPolicy(cfg, use_metadata=True)
            policy.belief = belief.copy(); policy.first = True
            policy._update_belief(count)
            losses_meta.append(-np.log(max(policy.belief[state], 1e-12)))
            if t + 1 < cfg.horizon:
                env.state = int(env.rng.choice(3, p=cfg.transition[env.state]))
                env.current_metadata = env._metadata_count(env.state)
    return {
        "logloss_no_metadata": float(np.mean(losses_prior)),
        "logloss_metadata": float(np.mean(losses_meta)),
        "delta_logloss": float(np.mean(losses_prior) - np.mean(losses_meta)),
    }
