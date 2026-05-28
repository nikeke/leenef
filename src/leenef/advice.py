"""Method-selection helpers for leenef users.

The public API is intentionally small: :func:`recommend_method` returns a
plain dictionary that can be printed, logged, or used by a higher-level
assistant skill.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class MethodRecommendation:
    """A practical starting point for configuring leenef."""

    method: str
    when_to_use: str
    starting_config: dict[str, object]
    notes: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-friendly representation."""
        return {
            "method": self.method,
            "when_to_use": self.when_to_use,
            "starting_config": dict(self.starting_config),
            "notes": list(self.notes),
        }


def recommend_method(
    task: str,
    *,
    input_kind: str = "tabular",
    online: bool = False,
    continual: bool = False,
    gpu: bool = False,
) -> MethodRecommendation:
    """Recommend a leenef method family and conservative starting config.

    Parameters
    ----------
    task:
        One of ``"supervised"``, ``"temporal"``, ``"continual"``, or ``"rl"``.
    input_kind:
        ``"tabular"``, ``"image"``, or ``"sequence"``.  This is a hint for
        encoder and architecture choice, not a strict type.
    online:
        Prefer accumulated or Woodbury-style updates over one batch solve.
    continual:
        Prefer additive sufficient-statistic workflows.
    gpu:
        Prefer accumulate-and-solve paths that avoid maintaining a dense
        Woodbury inverse on-device.
    """

    task_key = task.lower().replace("_", "-")
    input_key = input_kind.lower().replace("_", "-")
    if task_key not in {"supervised", "temporal", "continual", "rl"}:
        raise ValueError("task must be one of: supervised, temporal, continual, rl")
    if input_key not in {"tabular", "image", "sequence"}:
        raise ValueError("input_kind must be one of: tabular, image, sequence")

    if task_key == "temporal" or input_key == "sequence":
        config = {
            "class": "StreamingNEFClassifier",
            "n_neurons": 8000,
            "window_size": 10,
            "activation": "abs",
            "encoder_strategy": "hypersphere",
            "fit_path": "accumulate + solve" if gpu or online else "fit",
        }
        return MethodRecommendation(
            method="Stream-NEF",
            when_to_use="Sequence classification where short local temporal context is enough.",
            starting_config=config,
            notes=(
                "Use centers=training_sequences for data-driven biases.",
                "Use accumulate(..., chunk_stats=True) on large GPU runs.",
                "Mean pooling is a known limitation for tasks that require long-range order.",
            ),
        )

    if task_key == "continual" or continual:
        return MethodRecommendation(
            method="NEFLayer accumulated sufficient statistics",
            when_to_use="Task streams with stable output semantics, especially class-incremental streams.",
            starting_config={
                "class": "NEFLayer",
                "n_neurons": 5000,
                "activation": "abs",
                "encoder_strategy": "receptive_field" if input_key == "image" else "hypersphere",
                "fit_path": "partial_fit per task, then solve_accumulated",
                "alpha": (1e-2, "trace"),
            },
            notes=(
                "Sequential partial_fit matches the joint fixed-feature solution.",
                "For visual streams, a ConvNEF front end can raise the fixed-feature ceiling.",
                "This removes readout forgetting, not representation limits.",
            ),
        )

    if task_key == "rl":
        return MethodRecommendation(
            method="NEF action-value or successor-feature regression",
            when_to_use="Discrete-action control experiments where target construction is explicit.",
            starting_config={
                "dense_control": "NEFFQIAgent(target_mode='nstep', n_step=100)",
                "sparse_control": "NEFSFAgent with task-matched successor readout",
                "n_neurons": 2000,
                "device": "cuda" if gpu else "cpu",
            },
            notes=(
                "Prefer non-bootstrapped or otherwise stable targets first.",
                "Sparse rewards need goal, hindsight, or successor target structure.",
                "Treat this as a research component, not a drop-in replacement for tuned RL stacks.",
            ),
        )

    return MethodRecommendation(
        method="NEFLayer",
        when_to_use="Static supervised regression or classification with a known target matrix.",
        starting_config={
            "class": "NEFLayer",
            "n_neurons": 5000,
            "activation": "abs",
            "encoder_strategy": "receptive_field" if input_key == "image" else "hypersphere",
            "fit_path": "accumulate + solve" if gpu or online else "fit",
            "alpha": (1e-2, "trace"),
        },
        notes=(
            "Use centers=x_train for data-driven biases.",
            "Use receptive_field encoders for image inputs when spatial locality matters.",
            "If accuracy is feature-limited, improve the front end rather than only adding width.",
        ),
    )


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for ``leenef-recommend``."""

    parser = argparse.ArgumentParser(description="Recommend a leenef method family.")
    parser.add_argument("task", choices=["supervised", "temporal", "continual", "rl"])
    parser.add_argument(
        "--input-kind", choices=["tabular", "image", "sequence"], default="tabular"
    )
    parser.add_argument(
        "--online", action="store_true", help="prefer streaming/accumulated updates"
    )
    parser.add_argument("--continual", action="store_true", help="prefer continual-learning paths")
    parser.add_argument("--gpu", action="store_true", help="prefer GPU-friendly paths")
    args = parser.parse_args(argv)

    rec = recommend_method(
        args.task,
        input_kind=args.input_kind,
        online=args.online,
        continual=args.continual,
        gpu=args.gpu,
    )
    print(f"Method: {rec.method}")
    print(f"When to use: {rec.when_to_use}")
    print("Starting config:")
    for key, value in rec.starting_config.items():
        print(f"  {key}: {value!r}")
    print("Notes:")
    for note in rec.notes:
        print(f"  - {note}")


if __name__ == "__main__":
    main()
