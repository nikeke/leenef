"""LunarLander TD(λ) sweep.

Tests λ-returns against pure MC on LunarLander with RLS forgetting.
Configs: λ ∈ {1.0 (MC), 0.95, 0.9, 0.8} × RLS β=0.999 × 8000 neurons.
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
OUT_PATH = "results/rl/td_lambda.json"


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
            print(f"  ep {ep + 1:4d}  eval={ev:7.1f}  t={elapsed:.0f}s")

    env.close()
    elapsed = time.time() - t0
    final = evaluate(agent, n=20, seed=99999)
    best = max(e["eval"] for e in evals) if evals else final
    print(f"  FINAL={final:.1f}  BEST={best:.1f}  TIME={elapsed:.1f}s")
    return {
        "label": label,
        "td_lambda": agent_kwargs.get("td_lambda"),
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
    parser.add_argument(
        "--lambdas",
        type=float,
        nargs="+",
        default=[1.0, 0.95, 0.9, 0.8],
        help="Lambda values to test (1.0 = pure MC)",
    )
    parser.add_argument("--neurons", type=int, default=8000)
    parser.add_argument("--beta", type=float, default=0.999)
    parser.add_argument("--episodes", type=int, default=EPISODES)
    parser.add_argument("--solve-every", type=int, default=10)
    args = parser.parse_args()

    base = dict(
        n_neurons=args.neurons,
        gamma=0.99,
        epsilon_decay=300,
        solve_every=args.solve_every,
        target_mode="mc",
        forget_factor=args.beta,
        seed=0,
    )

    CONFIGS = []
    for lam in args.lambdas:
        kw = dict(base)
        label = f"λ={lam}" if lam < 1.0 else "MC (λ=1.0)"
        if lam < 1.0:
            kw["td_lambda"] = lam
        CONFIGS.append((label, kw))

    print(f"\n{'=' * 70}")
    print(f"Running {len(CONFIGS)} configs: {[c[0] for c in CONFIGS]}")
    print(f"β={args.beta}, n={args.neurons}, episodes={args.episodes}")
    print(f"{'=' * 70}")

    results = []
    for label, kwargs in CONFIGS:
        r = train(label, kwargs, episodes=args.episodes)
        results.append(r)
        flush_results(results)

    print()
    print("Summary:")
    print(f"  {'Config':<20s}  {'Best':>8s}  {'Final':>8s}  {'Time':>7s}")
    for r in results:
        print(f"  {r['label']:<20s}  {r['best']:8.1f}  {r['final']:8.1f}  {r['time']:6.0f}s")
