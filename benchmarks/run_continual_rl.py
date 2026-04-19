"""Continual RL benchmark: sequential LunarLander task variants.

Trains NEF-FQI on a sequence of LunarLander environments with different
dynamics (gravity, wind) and evaluates backward/forward transfer.

The key claim: NEF-FQI's additive sufficient statistics (AᵀA, AᵀY) retain
knowledge from prior tasks, enabling continual RL without catastrophic
forgetting.

Tasks:
  A: Default LunarLander (gravity=-10, no wind)
  B: Low gravity (gravity=-5, no wind)
  C: Wind (gravity=-10, wind, wind_power=15)

Protocols:
  separate  — independent agent per task (upper bound)
  continual — one agent trained sequentially on A → B → C
"""

import argparse
import json
import sys
import time

import gymnasium as gym
import numpy as np

from leenef.rl import NEFFQIAgent

sys.stdout.reconfigure(line_buffering=True)

TASKS = {
    "A_default": dict(gravity=-10.0, enable_wind=False),
    "B_lowgrav": dict(gravity=-5.0, enable_wind=False),
    "C_wind": dict(gravity=-10.0, enable_wind=True, wind_power=15.0, turbulence_power=1.5),
}

TASK_ORDER = ["A_default", "B_lowgrav", "C_wind"]


def make_env(task_name: str) -> gym.Env:
    return gym.make("LunarLander-v3", **TASKS[task_name])


def evaluate(agent: NEFFQIAgent, task_name: str, n: int = 20, seed: int = 42) -> float:
    """Evaluate greedy policy on a specific task variant."""
    env = make_env(task_name)
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


def evaluate_all(agent: NEFFQIAgent, n: int = 20, seed: int = 42) -> dict[str, float]:
    """Evaluate on all tasks, return {task_name: mean_return}."""
    return {task: evaluate(agent, task, n=n, seed=seed) for task in TASK_ORDER}


def train_phase(
    agent: NEFFQIAgent,
    task_name: str,
    n_episodes: int,
    eval_every: int = 50,
    eval_n: int = 20,
) -> list[dict]:
    """Train agent on a specific task for n_episodes.

    Returns list of eval checkpoints [{episode, task_evals}].
    """
    env = make_env(task_name)
    checkpoints = []

    for ep in range(1, n_episodes + 1):
        obs, _ = env.reset()
        done = False
        while not done:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.update(obs, action, reward, next_obs, done)
            obs = next_obs
        agent.end_episode()

        if ep % eval_every == 0:
            evals = evaluate_all(agent, n=eval_n)
            checkpoints.append({"episode": ep, "evals": evals})
            current_task_score = evals[task_name]
            print(
                f"    ep {ep:4d}  {task_name}={current_task_score:7.1f}  "
                f"[A={evals['A_default']:7.1f} B={evals['B_lowgrav']:7.1f} C={evals['C_wind']:7.1f}]"
            )

    env.close()
    return checkpoints


def make_agent(
    d_obs: int, n_actions: int, n_neurons: int, beta: float, seed: int = 0
) -> NEFFQIAgent:
    return NEFFQIAgent(
        d_obs=d_obs,
        n_actions=n_actions,
        n_neurons=n_neurons,
        alpha=1e-2,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=300,
        solve_every=10,
        target_mode="mc",
        forget_factor=beta,
        recenter_interval=100,
        recenter_percentile=5.0,
        seed=seed,
    )


def run_separate(d_obs, n_actions, n_neurons, beta, episodes_per_task, args):
    """Train independent agents per task (upper bound)."""
    print("\n=== SEPARATE (independent agent per task) ===")
    results = {}
    for task in TASK_ORDER:
        print(f"\n  Task {task}:")
        agent = make_agent(d_obs, n_actions, n_neurons, beta, seed=args.seed)
        # Warmup on this task
        env = make_env(task)
        agent.warmup(env=env, n_episodes=20)
        env.close()

        checkpoints = train_phase(agent, task, episodes_per_task)
        final = evaluate_all(agent)
        results[task] = {
            "checkpoints": checkpoints,
            "final": final,
        }
        print(
            f"  Final: A={final['A_default']:.1f} B={final['B_lowgrav']:.1f} C={final['C_wind']:.1f}"
        )

    return results


def run_continual(d_obs, n_actions, n_neurons, beta, episodes_per_task, args, label="continual"):
    """Train one agent sequentially on A → B → C."""
    print(f"\n=== {label.upper()} (β={beta}) ===")
    agent = make_agent(d_obs, n_actions, n_neurons, beta, seed=args.seed)

    # Warmup on task A
    env = make_env(TASK_ORDER[0])
    agent.warmup(env=env, n_episodes=20)
    env.close()

    all_checkpoints = {}
    for task in TASK_ORDER:
        print(f"\n  Phase: {task}")
        checkpoints = train_phase(agent, task, episodes_per_task)
        all_checkpoints[task] = checkpoints

    final = evaluate_all(agent)
    print(
        f"\n  Final (after all tasks): A={final['A_default']:.1f} B={final['B_lowgrav']:.1f} C={final['C_wind']:.1f}"
    )

    return {
        "checkpoints": all_checkpoints,
        "final": final,
        "beta": beta,
    }


def main():
    parser = argparse.ArgumentParser(description="Continual RL benchmark")
    parser.add_argument("--neurons", type=int, default=4000)
    parser.add_argument("--episodes", type=int, default=500, help="Episodes per task")
    parser.add_argument("--beta", type=float, default=0.999, help="RLS forget factor")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="results/rl/continual_rl.json")
    args = parser.parse_args()

    # Get env dimensions
    env = gym.make("LunarLander-v3")
    d_obs = env.observation_space.shape[0]
    n_actions = env.action_space.n
    env.close()
    print(f"LunarLander continual RL: d_obs={d_obs}, n_actions={n_actions}")
    print(f"Tasks: {' → '.join(TASK_ORDER)}")
    print(f"Neurons={args.neurons}, β={args.beta}, episodes/task={args.episodes}")

    results = {"config": vars(args), "tasks": dict(TASKS)}

    t0 = time.time()

    # 1. Separate (independent per task)
    results["separate"] = run_separate(
        d_obs, n_actions, args.neurons, args.beta, args.episodes, args
    )

    t1 = time.time()
    print(f"\n  Separate total time: {t1 - t0:.0f}s")

    # 2. Continual with β=0.999 (mild forgetting)
    results["continual"] = run_continual(
        d_obs,
        n_actions,
        args.neurons,
        args.beta,
        args.episodes,
        args,
        label=f"continual (β={args.beta})",
    )

    t2 = time.time()
    print(f"\n  Continual total time: {t2 - t1:.0f}s")

    # 3. Continual with β=1.0 (pure accumulation, no forgetting)
    results["continual_nf"] = run_continual(
        d_obs,
        n_actions,
        args.neurons,
        1.0,
        args.episodes,
        args,
        label="continual (β=1.0, no forgetting)",
    )

    t3 = time.time()
    print(f"\n  Continual-NF total time: {t3 - t2:.0f}s")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Separate results (per-task best)
    sep = results["separate"]
    print("\nSeparate (upper bound per task):")
    for task in TASK_ORDER:
        score = sep[task]["final"][task]
        print(f"  {task}: {score:.1f}")

    # Continual with forgetting
    cont = results["continual"]
    print(f"\nContinual (β={args.beta}) — final eval on all tasks:")
    for task in TASK_ORDER:
        score = cont["final"][task]
        sep_score = sep[task]["final"][task]
        delta = score - sep_score
        print(f"  {task}: {score:.1f} (vs separate {sep_score:.1f}, Δ={delta:+.1f})")

    # Continual no forgetting
    cont_nf = results["continual_nf"]
    print("\nContinual (β=1.0, no forgetting) — final eval on all tasks:")
    for task in TASK_ORDER:
        score = cont_nf["final"][task]
        sep_score = sep[task]["final"][task]
        delta = score - sep_score
        print(f"  {task}: {score:.1f} (vs separate {sep_score:.1f}, Δ={delta:+.1f})")

    print(f"\nTotal time: {t3 - t0:.0f}s")

    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
