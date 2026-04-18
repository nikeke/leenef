"""LunarLander benchmark for RLS forgetting + dead neuron recentering."""

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


def train(label, agent_kwargs, episodes=1000, eval_every=50):
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
        "final": final,
        "best": best,
        "time": round(elapsed, 1),
        "evals": evals,
    }


CONFIGS = [
    (
        "RLS b=0.995 + recenter/100",
        dict(
            n_neurons=8000,
            gamma=0.99,
            epsilon_decay=300,
            solve_every=10,
            target_mode="mc",
            forget_factor=0.995,
            recenter_interval=100,
            recenter_percentile=5.0,
            seed=0,
        ),
    ),
    (
        "RLS b=0.999 + recenter/100",
        dict(
            n_neurons=8000,
            gamma=0.99,
            epsilon_decay=300,
            solve_every=10,
            target_mode="mc",
            forget_factor=0.999,
            recenter_interval=100,
            recenter_percentile=5.0,
            seed=0,
        ),
    ),
]

results = []
for label, kwargs in CONFIGS:
    r = train(label, kwargs)
    results.append(r)

print()
print("=" * 70)
header = f"{'Config':<35} {'Final':>8} {'Best':>8} {'Time':>8}"
print(header)
print("-" * 63)
for r in results:
    line = f"{r['label']:<35} {r['final']:8.1f} {r['best']:8.1f} {r['time']:7.1f}s"
    print(line)
print("(Baseline NEF-FQI: best=40.8 / DQN: best=214.0)")

out_path = "results/rl/rls_lunarlander.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved to {out_path}")
