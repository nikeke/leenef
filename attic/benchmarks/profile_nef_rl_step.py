"""Profile NEF-FQI agent vs gym env stepping at production base config.

Runs a short gym training run (50 episodes) at the current base config and
attributes wall time to:
  - env.reset / env.step  (gym Box2D)
  - agent.select_action
  - agent.update          (incl. internal solve trigger)

Prints a per-component breakdown so we can decide whether envpool's faster
env stepping could ever pay off for this workload.
"""

from __future__ import annotations

import json
import time

import gymnasium as gym

from benchmarks.rl_repro import BASE_CONFIG_PATH, reset_env, training_seed, warmup_seed
from leenef.rl import NEFFQIAgent

CFG_PATH = BASE_CONFIG_PATH
ENV_NAME = "LunarLander-v3"
N_EPISODES = 50
SEED = 0


def main() -> None:
    cfg = json.loads(CFG_PATH.read_text())
    env = gym.make(ENV_NAME)
    obs, _ = reset_env(env, seed=SEED)
    d_obs = obs.shape[0]
    n_actions = env.action_space.n  # type: ignore[attr-defined]

    agent_kwargs = dict(
        d_obs=d_obs,
        n_actions=n_actions,
        n_neurons=cfg["n_neurons"],
        gamma=cfg["gamma"],
        alpha=cfg["alpha"],
        epsilon_start=cfg["epsilon_start"],
        epsilon_end=cfg["epsilon_end"],
        epsilon_decay=cfg["epsilon_decay"],
        solve_every=cfg["solve_every"],
        target_mode=cfg["target_mode"],
        n_step=cfg["n_step"],
        forget_factor=cfg["forget_factor"],
        fqi_iters=cfg["fqi_iters"],
        tau=cfg["tau"],
        broadness=cfg["broadness"],
        recenter_interval=cfg["recenter_interval"],
        seed=SEED,
    )
    agent = NEFFQIAgent(**agent_kwargs)

    # Warmup just to set normalization (tiny; not measured).
    warmup_env = gym.make(ENV_NAME)
    agent.warmup(env=warmup_env, n_episodes=20, seed=warmup_seed(SEED))
    warmup_env.close()

    t_select = 0.0
    t_step = 0.0
    t_update = 0.0
    t_endep = 0.0
    t_reset = 0.0
    n_steps = 0
    n_episodes_done = 0

    obs, _ = reset_env(env, seed=SEED + 100)
    t_total_start = time.perf_counter()

    while n_episodes_done < N_EPISODES:
        t0 = time.perf_counter()
        a = agent.select_action(obs)
        t1 = time.perf_counter()
        next_obs, r, term, trunc, _ = env.step(a)
        t2 = time.perf_counter()
        agent.update(obs, int(a), float(r), next_obs, bool(term or trunc))
        t3 = time.perf_counter()

        t_select += t1 - t0
        t_step += t2 - t1
        t_update += t3 - t2
        n_steps += 1

        if term or trunc:
            te0 = time.perf_counter()
            agent.end_episode()
            te1 = time.perf_counter()
            obs, _ = reset_env(env, seed=training_seed(SEED + 100, n_episodes_done + 1))
            te2 = time.perf_counter()
            t_endep += te1 - te0
            t_reset += te2 - te1
            n_episodes_done += 1
        else:
            obs = next_obs

    t_total = time.perf_counter() - t_total_start
    accounted = t_select + t_step + t_update + t_endep + t_reset

    print(f"Config: {cfg}")
    print(f"\nProfile over {N_EPISODES} episodes / {n_steps} steps")
    print(f"Wall clock total:   {t_total:8.2f} s")
    print(f"Accounted:          {accounted:8.2f} s ({100 * accounted / t_total:5.1f}%)")
    print("\nBreakdown (per-step / total / share-of-accounted):")
    for name, t in [
        ("env.step       ", t_step),
        ("agent.select   ", t_select),
        ("agent.update   ", t_update),
        ("agent.end_ep   ", t_endep),
        ("env.reset      ", t_reset),
    ]:
        per = t / max(n_steps, 1) * 1000
        share = 100 * t / accounted
        print(f"  {name}  {per:7.2f} ms/step   {t:8.2f} s   {share:5.1f}%")

    env_share = t_step / accounted
    agent_share = (t_select + t_update + t_endep) / accounted
    print(f"\nEnv stepping share:    {100 * env_share:5.1f}%")
    print(f"Agent compute share:   {100 * agent_share:5.1f}%")
    print(
        f"\nUpper bound on envpool benefit (if env.step → 0): "
        f"{1 / (1 - env_share):.2f}× speedup per process"
    )


if __name__ == "__main__":
    main()
