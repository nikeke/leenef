#!/usr/bin/env python3
"""Generate clean RL sweep figures under docs/figures."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    from benchmarks.plot_rl_sweeps import main as plot_main

    plot_main()


if __name__ == "__main__":
    main()
