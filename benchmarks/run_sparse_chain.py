"""Toy sparse-chain benchmark — closed-form Q* regression for NEF agents.

The :class:`leenef.rl_diag.SparseChainEnv` is a deterministic K-state
chain with a single terminal +1 reward.  Its ``optimal_q()`` method
returns the exact ground-truth Q-table by value iteration — this gives
us a unit-style sparse-reward environment where every agent can be
graded without seed noise or eval-policy artefacts.

This driver runs four agent variants on a small grid of chain lengths
and writes a JSON summary with three metrics per (variant, n_states,
seed):

- **first_success_episode** — first training episode in which the
  agent reached the terminal state (``None`` if never).  Leading
  indicator of exploration.
- **q_max_abs_err** — ``max |Q̂(s, a) - Q*(s, a)|`` over reachable
  non-terminal states and both actions.  Direct grade of the
  analytical solve.
- **policy_correct** — ``True`` iff ``argmax_a Q̂(s, a) == 1`` for all
  non-terminal ``s`` (the optimal policy is always-RIGHT).

Variants:

- ``nef_fqi`` — :class:`leenef.rl.NEFFQIAgent`, MC targets.  Plain
  reward FQI; sees the +1 only when episodes terminate at the goal.
  Sanity baseline.
- ``nef_sf`` — :class:`leenef.rl_sf.NEFSFAgent`.  Successor decoder
  ``Ψ_a`` plus reward decoder ``w_a`` solved analytically; ``Q = Ψ·w``.
- ``nef_sf_novelty`` — same as above with leverage novelty bonus
  ``β · sqrt(φᵀ(AᵀA + αI)⁻¹φ)`` added to extrinsic reward.  Tests
  whether D2 helps when no goal handle is given.

For each ``nef_sf*`` row we also report a derived metric
**eval_goal_policy_correct**: the greedy policy implied by the
``ψ(s, a) · u(φ(g*))`` projection with ``g* = obs(terminal)``,
*using the very same Ψ that was solved for the reward Q*.  This is
the headline sparse-reward mechanism applied where Q* is known
exactly: no hindsight buffer, no extra solve, just a linear
projection of the existing successor decoder onto the task goal
direction.  On a single-goal chain it should match the optimal
greedy policy without ever consulting the reward channel.

The full :class:`leenef.rl_sf.NEFHindsightSFAgent` is intentionally
omitted: the chain has only one rewarded state, so the hindsight
buffer collapses to a single-goal distribution and the
goal-relabeling mechanism becomes a no-op compared to the bare-Ψ
projection above.  Hindsight's value is on multi-goal /
trajectory-anchored sparse tasks (see MountainCar / Acrobot
results); on the chain we already test the *underlying linear
projection* directly.

Usage::

    python -u benchmarks/run_sparse_chain.py \\
        --n-states 5 10 20 --seeds 0 1 2 \\
        --output results/rl_sparse/sparse_chain.json

Runs are short by design (the chain is tiny) — the whole sweep
finishes on CPU in well under a minute for the default grid.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from leenef.rl import NEFFQIAgent
from leenef.rl_diag import SparseChainEnv
from leenef.rl_sf import NEFSFAgent

VARIANTS = (
    "nef_fqi",
    "nef_sf",
    "nef_sf_novelty",
)


def _make_agent(variant: str, *, n_neurons: int, gamma: float, seed: int):
    common = dict(
        d_obs=1,
        n_actions=2,
        n_neurons=n_neurons,
        gamma=gamma,
        alpha=1e-3,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=200,
        seed=seed,
    )
    if variant == "nef_fqi":
        return NEFFQIAgent(
            target_mode="mc",
            solve_every=10,
            **common,
        )
    if variant == "nef_sf":
        return NEFSFAgent(
            n_step=None,
            solve_every=10,
            **common,
        )
    if variant == "nef_sf_novelty":
        return NEFSFAgent(
            n_step=None,
            solve_every=10,
            novelty_beta=0.3,
            **common,
        )
    raise ValueError(f"unknown variant {variant!r}")


def _train_episode(agent, env, *, max_steps: int) -> tuple[float, bool, int]:
    obs, _ = env.reset(seed=int(agent.rng.integers(2**31 - 1)))
    total = 0.0
    steps = 0
    terminated = False
    for _ in range(max_steps):
        action = agent.select_action(obs)
        next_obs, reward, term, trunc, _ = env.step(action)
        agent.update(obs, action, float(reward), next_obs, bool(term or trunc))
        total += float(reward)
        steps += 1
        obs = next_obs
        if term:
            terminated = True
            break
        if trunc:
            break
    if hasattr(agent, "end_episode"):
        agent.end_episode()
    return total, terminated, steps


def _q_table(agent, env: SparseChainEnv) -> np.ndarray:
    """Predicted Q on every non-terminal state, shape (n_states, 2)."""
    n = env.n_states
    Q = np.zeros((n, 2), dtype=np.float64)
    for s in range(n):
        obs = np.array([s / (n - 1)], dtype=np.float32)
        with torch.no_grad():
            q = agent.q_values(obs).cpu().numpy().astype(np.float64)
        Q[s] = q
    return Q


def _grade(
    Q_pred: np.ndarray,
    Q_star: np.ndarray,
) -> tuple[float, bool]:
    """Return (max_abs_err over non-terminal states, all-greedy-correct)."""
    n = Q_star.shape[0]
    err = np.abs(Q_pred[: n - 1] - Q_star[: n - 1]).max()
    greedy = Q_pred[: n - 1].argmax(axis=1)
    return float(err), bool((greedy == 1).all())


def _eval_goal_greedy_correct(agent, env: SparseChainEnv) -> bool | None:
    """Greedy policy under the eval_goal projection ``ψ(s,a)·u(φ(g*))``.

    Returns ``None`` for agents without a successor decoder (e.g. plain
    NEFFQIAgent), and ``True`` iff RIGHT is greedy at every non-terminal
    state for SF agents.  ``g*`` is the terminal-state observation.
    """
    if not hasattr(agent, "successor"):
        return None
    n = env.n_states
    g_star = np.array([1.0], dtype=np.float32)
    with torch.no_grad():
        phi_g = agent.successor.features.encode(torch.from_numpy(g_star)).double()
        norm = float(torch.linalg.vector_norm(phi_g).item())
        if norm < 1e-8:
            return None
        u = (phi_g / norm).to(torch.float32)
        Psi = agent.successor.Psi  # (n_actions, n_neurons, n_neurons)
        ok_all = True
        for s in range(n - 1):
            obs = np.array([s / (n - 1)], dtype=np.float32)
            phi = agent.successor.features.encode(torch.from_numpy(obs))
            psi = torch.einsum("n,anm->am", phi, Psi)  # (n_actions, n_neurons)
            q = psi @ u  # (n_actions,)
            if int(q.argmax().item()) != 1:
                ok_all = False
                break
    return ok_all


def run_one(
    *,
    variant: str,
    n_states: int,
    seed: int,
    n_episodes: int,
    n_neurons: int,
    gamma: float,
) -> dict:
    env = SparseChainEnv(n_states=n_states, gamma=gamma, max_steps=4 * n_states, seed=seed)
    torch.manual_seed(seed)
    agent = _make_agent(variant, n_neurons=n_neurons, gamma=gamma, seed=seed)
    Q_star = env.optimal_q()

    first_success = None
    returns: list[float] = []
    t0 = time.perf_counter()
    for ep in range(n_episodes):
        ret, term, _ = _train_episode(agent, env, max_steps=4 * n_states)
        returns.append(ret)
        if term and first_success is None:
            first_success = ep
    elapsed = time.perf_counter() - t0

    # Final solve (some agents only solve every K episodes).
    if hasattr(agent, "successor"):
        agent.successor.solve()
    elif hasattr(agent, "_solve"):
        agent._solve()

    Q_pred = _q_table(agent, env)
    err, ok = _grade(Q_pred, Q_star)
    eval_goal_ok = _eval_goal_greedy_correct(agent, env)
    return dict(
        variant=variant,
        n_states=n_states,
        seed=seed,
        n_episodes=n_episodes,
        first_success_episode=first_success,
        q_max_abs_err=err,
        policy_correct=ok,
        eval_goal_policy_correct=eval_goal_ok,
        elapsed_s=round(elapsed, 3),
        terminal_count=int(sum(1 for r in returns if r > 0.0)),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-states", type=int, nargs="+", default=[5, 10, 20])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--n-neurons", type=int, default=128)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--variants", type=str, nargs="+", default=list(VARIANTS))
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("results/rl_sparse/sparse_chain.json"),
    )
    args = ap.parse_args()

    rows = []
    for variant in args.variants:
        if variant not in VARIANTS:
            raise SystemExit(f"unknown variant {variant!r}; allowed: {VARIANTS}")
        for n_states in args.n_states:
            for seed in args.seeds:
                row = run_one(
                    variant=variant,
                    n_states=n_states,
                    seed=seed,
                    n_episodes=args.episodes,
                    n_neurons=args.n_neurons,
                    gamma=args.gamma,
                )
                rows.append(row)
                print(
                    f"[{variant:>16s}] K={n_states:>2d} seed={seed} "
                    f"first_success={row['first_success_episode']!s:>5s} "
                    f"q_err={row['q_max_abs_err']:.4f} "
                    f"policy_ok={row['policy_correct']!s:>5s} "
                    f"eval_goal_ok={row['eval_goal_policy_correct']!s:>5s} "
                    f"({row['elapsed_s']:.2f}s)",
                    flush=True,
                )

    # Aggregate per (variant, n_states): mean q_err, fraction policy_correct,
    # fraction reaching goal at all, mean first_success_episode (over seeds
    # that succeeded).
    agg: dict[tuple[str, int], dict] = {}
    for r in rows:
        key = (r["variant"], r["n_states"])
        a = agg.setdefault(
            key,
            dict(
                seeds=0,
                q_err_sum=0.0,
                policy_ok=0,
                eval_goal_ok=0,
                eval_goal_n=0,
                goal_reached=0,
                first_success_sum=0,
                first_success_n=0,
            ),
        )
        a["seeds"] += 1
        a["q_err_sum"] += r["q_max_abs_err"]
        a["policy_ok"] += int(r["policy_correct"])
        if r["eval_goal_policy_correct"] is not None:
            a["eval_goal_n"] += 1
            a["eval_goal_ok"] += int(r["eval_goal_policy_correct"])
        a["goal_reached"] += int(r["terminal_count"] > 0)
        if r["first_success_episode"] is not None:
            a["first_success_sum"] += r["first_success_episode"]
            a["first_success_n"] += 1

    summary = []
    for (variant, n_states), a in sorted(agg.items()):
        first_success_mean = (
            a["first_success_sum"] / a["first_success_n"] if a["first_success_n"] > 0 else None
        )
        summary.append(
            dict(
                variant=variant,
                n_states=n_states,
                seeds=a["seeds"],
                q_err_mean=a["q_err_sum"] / a["seeds"],
                policy_ok_frac=a["policy_ok"] / a["seeds"],
                eval_goal_ok_frac=(
                    None if a["eval_goal_n"] == 0 else a["eval_goal_ok"] / a["eval_goal_n"]
                ),
                goal_reached_frac=a["goal_reached"] / a["seeds"],
                first_success_mean=(
                    None if first_success_mean is None else round(first_success_mean, 1)
                ),
            )
        )

    out = dict(
        config=dict(
            n_states=args.n_states,
            seeds=args.seeds,
            episodes=args.episodes,
            n_neurons=args.n_neurons,
            gamma=args.gamma,
            variants=args.variants,
        ),
        rows=rows,
        summary=summary,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {len(rows)} runs and {len(summary)} summary entries to {args.output}")

    print(
        "\nSummary (variant, n_states): q_err_mean | policy_ok | eval_goal_ok | goal_reached | mean_first_success"
    )
    for s in summary:
        first = "—" if s["first_success_mean"] is None else f"{s['first_success_mean']:.1f}"
        eg = "—" if s["eval_goal_ok_frac"] is None else f"{s['eval_goal_ok_frac']:.2f}"
        print(
            f"  {s['variant']:>16s}  K={s['n_states']:>2d}  "
            f"q_err={s['q_err_mean']:.4f}  "
            f"policy_ok={s['policy_ok_frac']:.2f}  "
            f"eval_goal_ok={eg}  "
            f"goal={s['goal_reached_frac']:.2f}  "
            f"first_success={first}"
        )


if __name__ == "__main__":
    main()
