"""LunarLander ensemble exploration benchmark.

Tests NEFFQIEnsemble with Thompson sampling and UCB exploration
against single-agent baseline on LunarLander.

Configs:
- Single agent (RLS β=0.999, MC, 8000 neurons) — baseline
- Thompson (5 members × 8000 neurons)
- UCB (5 members × 8000 neurons, coeff=1.0)
"""

import json
import sys
import time

import gymnasium as gym
import numpy as np

from leenef.rl import NEFFQIAgent, NEFFQIEnsemble

sys.stdout.reconfigure(line_buffering=True)

ENV_NAME = "LunarLander-v3"
env = gym.make(ENV_NAME)
D_OBS = env.observation_space.shape[0]
N_ACTIONS = env.action_space.n
env.close()
print(f"LunarLander: d_obs={D_OBS}, n_actions={N_ACTIONS}")

EPISODES = 1000
EVAL_EVERY = 50
OUT_PATH = "results/rl/ensemble_lunar.json"


def evaluate_single(agent, n=20, seed=42):
    env = gym.make(ENV_NAME)
    returns = []
    for i in range(n):
        obs, _ = env.reset(seed=seed + i)
        done, total = False, 0.0
        while not done:
            q = agent.q_values(obs)
            action = int(q.argmax().item())
            obs, r, t, tr, _ = env.step(action)
            done = t or tr
            total += r
        returns.append(total)
    env.close()
    return float(np.mean(returns))


def evaluate_ensemble(ensemble, n=20, seed=42):
    env = gym.make(ENV_NAME)
    returns = []
    for i in range(n):
        obs, _ = env.reset(seed=seed + i)
        done, total = False, 0.0
        while not done:
            q = ensemble.q_values(obs)
            action = int(q.argmax().item())
            obs, r, t, tr, _ = env.step(action)
            done = t or tr
            total += r
        returns.append(total)
    env.close()
    return float(np.mean(returns))


def train_single(label, agent_kwargs):
    print(f"\n>>> {label}")
    agent = NEFFQIAgent(d_obs=D_OBS, n_actions=N_ACTIONS, **agent_kwargs)
    warmup_env = gym.make(ENV_NAME)
    agent.warmup(env=warmup_env, n_episodes=20)
    warmup_env.close()

    env = gym.make(ENV_NAME)
    t0 = time.time()
    evals = []
    for ep in range(EPISODES):
        obs, _ = env.reset(seed=ep)
        done = False
        while not done:
            action = agent.select_action(obs)
            next_obs, r, t, tr, _ = env.step(action)
            done = t or tr
            agent.update(obs, action, r, next_obs, done)
            obs = next_obs
        agent.end_episode()

        if (ep + 1) % EVAL_EVERY == 0:
            ev = evaluate_single(agent, n=20, seed=10000)
            evals.append({"episode": ep + 1, "eval": ev})
            elapsed = time.time() - t0
            print(f"  ep {ep + 1:4d}  eval={ev:7.1f}  t={elapsed:.0f}s")

    env.close()
    elapsed = time.time() - t0
    final = evaluate_single(agent, n=20, seed=99999)
    best = max(e["eval"] for e in evals) if evals else final
    print(f"  FINAL={final:.1f}  BEST={best:.1f}  TIME={elapsed:.1f}s")
    return {
        "label": label,
        "final": final,
        "best": best,
        "time": round(elapsed, 1),
        "evals": evals,
    }


def train_ensemble(label, ensemble_kwargs, agent_kwargs):
    print(f"\n>>> {label}")
    ensemble = NEFFQIEnsemble(
        d_obs=D_OBS,
        n_actions=N_ACTIONS,
        **ensemble_kwargs,
        **agent_kwargs,
    )
    warmup_env = gym.make(ENV_NAME)
    ensemble.warmup(env=warmup_env, n_episodes=20)
    warmup_env.close()

    env = gym.make(ENV_NAME)
    t0 = time.time()
    evals = []
    for ep in range(EPISODES):
        obs, _ = env.reset(seed=ep)
        done = False
        while not done:
            action = ensemble.select_action(obs)
            next_obs, r, t, tr, _ = env.step(action)
            done = t or tr
            ensemble.update(obs, action, r, next_obs, done)
            obs = next_obs
        ensemble.end_episode()

        if (ep + 1) % EVAL_EVERY == 0:
            ev = evaluate_ensemble(ensemble, n=20, seed=10000)
            evals.append({"episode": ep + 1, "eval": ev})
            elapsed = time.time() - t0
            print(f"  ep {ep + 1:4d}  eval={ev:7.1f}  t={elapsed:.0f}s")

    env.close()
    elapsed = time.time() - t0
    final = evaluate_ensemble(ensemble, n=20, seed=99999)
    best = max(e["eval"] for e in evals) if evals else final
    print(f"  FINAL={final:.1f}  BEST={best:.1f}  TIME={elapsed:.1f}s")
    return {
        "label": label,
        "n_members": ensemble_kwargs.get("n_members", 5),
        "strategy": ensemble_kwargs.get("explore_strategy", "thompson"),
        "final": final,
        "best": best,
        "time": round(elapsed, 1),
        "evals": evals,
    }


def flush_results(results):
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  [flushed {len(results)} results to {OUT_PATH}]")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--neurons", type=int, default=4000, help="Neurons per member")
    parser.add_argument("--members", type=int, default=3, help="Ensemble members")
    parser.add_argument("--episodes", type=int, default=EPISODES)
    args = parser.parse_args()

    n = args.neurons
    m = args.members

    base_agent = dict(
        n_neurons=n,
        gamma=0.99,
        epsilon_decay=300,
        solve_every=10,
        target_mode="mc",
        forget_factor=0.999,
    )

    results = []

    # 1. Single-agent baseline (same neuron count as each member)
    r = train_single(f"Single n={n}", dict(base_agent, seed=0))
    results.append(r)
    flush_results(results)

    # 2. Thompson sampling
    r = train_ensemble(
        f"Thompson ({m}×{n // 1000}k)",
        dict(n_members=m, explore_strategy="thompson", base_seed=0),
        base_agent,
    )
    results.append(r)
    flush_results(results)

    # 3. UCB
    r = train_ensemble(
        f"UCB ({m}×{n // 1000}k)",
        dict(n_members=m, explore_strategy="ucb", ucb_coeff=1.0, base_seed=0),
        base_agent,
    )
    results.append(r)
    flush_results(results)

    print()
    print("Summary:")
    print(f"  {'Config':<25s}  {'Best':>8s}  {'Final':>8s}  {'Time':>7s}")
    for r in results:
        print(f"  {r['label']:<25s}  {r['best']:8.1f}  {r['final']:8.1f}  {r['time']:6.0f}s")
