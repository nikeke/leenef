"""Continual-learning clean sweeps (Paper II infrastructure).

Mirrors ``benchmarks.sl_clean`` in structure and discipline.  The
operational contract (code-identity, dataset hashing, runner
idempotency, atomic JSON writes, drivers don't change mid-run) is
inherited verbatim — see ``docs/cl_clean_sweeps.md`` and AGENTS.md.
"""
