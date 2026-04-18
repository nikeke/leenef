"""LunarLander neuron-scaling sweep with ablations.

Tests 8000/16000/32000 neurons × {baseline, RLS-only, recenter-only, RLS+recenter}.
Results are flushed after every completed config.
"""

import json
import sys
import time

import gymnasium as gym
import numpy as np

from leenef.rl import NEFFQIAgent

sys.stdout.reconfigure(line_buffering=True)

ENV_NAME = "LunarLander-v3"
env = gym.make(ENV_NAME)
D_OBS = env.observation_space.shape[0]
N_ACTIONS = env.action_space.n
env.close()
print(f"LunarLander: d_obs={D_OBS}, n_actions={N_ACTIONS}")

EPISODES = 1000
EVAL_EVERY = 50
OUT_PATH = "results/rl/lunar_scaling.json"


def evaluate(agent, n=20, seed=42):
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


def train(label, agent_kwargs, episodes=EPISODES, eval_every=EVAL_EVERY):
    print(f"\n>>> {label}")
    agent = NEFFQIAgent(d_obs=D_OBS, n_actions=N_ACTIONS, **agent_kwargs)
    warmup_env = gym.make(ENV_NAME)
    agent.warmup(env=warmup_env, n_episodes=20)
    warmup_env.close()

    env = gym.make(ENV_NAME)
    t0 = time.time()
    evals = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=ep)
        done = False
        while not done:
            action = agent.select_action(obs)
            next_obs, r, t, tr, _ = env.step(action)
            done = t or tr
            agent.update(obs, action, r, next_obs, done)
            obs = next_obs
        agent.end_episode()

        if (ep + 1) % eval_every == 0:
            ev = evaluate(agent, n=20, seed=10000)
            evals.append({"episode": ep + 1, "eval": ev})
            elapsed = time.time() - t0
            recentered = getattr(agent, "_recentered_total", 0)
            print(f"  ep {ep + 1:4d}  eval={ev:7.1f}  recentered={recentered}  t={elapsed:.0f}s")

    env.close()
    elapsed = time.time() - t0
    final = evaluate(agent, n=20, seed=99999)
    best = max(e["eval"] for e in evals) if evals else final
    print(f"  FINAL={final:.1f}  BEST={best:.1f}  TIME={elapsed:.1f}s")
    return {
        "label": label,
        "n_neurons": agent_kwargs["n_neurons"],
        "final": final,
        "best": best,
        "time": round(elapsed, 1),
        "evals": evals,
    }


def make_base_kwargs(n_neurons, solve_every=10):
    return dict(
        n_neurons=n_neurons,
        gamma=0.99,
        epsilon_decay=300,
        solve_every=solve_every,
        target_mode="mc",
        seed=0,
    )


def flush_results(results):
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  [flushed {len(results)} results to {OUT_PATH}]")


# Build config list from CLI args or defaults
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--neurons",
        type=int,
        nargs="+",
        default=[8000, 16000],
        help="Neuron counts to test",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["all"],
        choices=["all", "combined", "baseline", "rls", "recenter"],
        help="Which configs to run",
    )
    parser.add_argument(
        "--solve-every",
        type=int,
        default=10,
        help="Solve frequency",
    )
    args = parser.parse_args()

    configs_to_run = set(args.configs)
    if "all" in configs_to_run:
        configs_to_run = {"combined", "baseline", "rls", "recenter"}

    CONFIGS = []
    for n in args.neurons:
        se = args.solve_every
        if "baseline" in configs_to_run:
            kw = make_base_kwargs(n, se)
            CONFIGS.append((f"Baseline n={n}", kw))
        if "rls" in configs_to_run:
            kw = make_base_kwargs(n, se)
            kw["forget_factor"] = 0.999
            CONFIGS.append((f"RLS-only n={n}", kw))
        if "recenter" in configs_to_run:
            kw = make_base_kwargs(n, se)
            kw["recenter_interval"] = 100
            kw["recenter_percentile"] = 5.0
            CONFIGS.append((f"Recenter-only n={n}", kw))
        if "combined" in configs_to_run:
            kw = make_base_kwargs(n, se)
            kw["forget_factor"] = 0.999
            kw["recenter_interval"] = 100
            kw["recenter_percentile"] = 5.0
            CONFIGS.append((f"RLS+recenter n={n}", kw))

    print(f"\n{'=' * 70}")
    print(f"Running {len(CONFIGS)} configs: {[c[0] for c in CONFIGS]}")
    print(f"{'=' * 70}")

    results = []
    for label, kwargs in CONFIGS:
        r = train(label, kwargs)
        results.append(r)
        flush_results(results)

    print()
    print("=" * 70)
    header = f"{'Config':<30} {'Neurons':>8} {'Final':>8} {'Best':>8} {'Time':>9}"
    print(header)
    print("-" * 67)
    for r in results:
        line = (
            f"{r['label']:<30} {r['n_neurons']:>8}"
            f" {r['final']:8.1f} {r['best']:8.1f} {r['time']:8.1f}s"
        )
        print(line)
    print("(Prior 8k RLS+recenter β=0.999: best=109.4, final=102.2)")
    print("(Baseline 8k: best=40.8 / DQN: best=214.0)")
