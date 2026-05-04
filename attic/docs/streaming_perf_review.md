# GPT-5.5 Streaming Perf Review (2026-04-30T23:05Z)

Verdict: biggest opportunity is **sweep-level reuse**, not inner kernels.

## P1 (do BEFORE final-panel reruns)

1. **Group sweep by (n, seed, encoder), share AᵀA/AᵀY across α and alpha_scale.**
   Currently re-encode 10× for each (n, seed). Massive sweep speedup.
   `_streaming_exp.py:68-117`, `sl_hp_driver.sh:130-146`.
3. **Default `chunk_stats=True` in accumulate** — at n=8k, current default
   materialises a 1.9 GB pooled-A matrix.  `streaming.py:281-286`.
4. **Bound `max_tokens` by activity elements not raw tokens** — `max_tokens *
   n_neurons` should respect a memory budget.  `streaming.py:132-146`.
5. **Float32 accumulate option.** Currently `_accumulate_encoded` upcasts
   AᵀA to float64 even on GPU path.  Measure first.  `streaming.py:246-247`.
6. **GPU support in runner.** No `.to(device)` anywhere. Big at n≥8k.
   `_streaming_exp.py:95-175`.

## P2

2. **Trace pilot accumulation discarded for accumulate mode.** Save the
   pilot (AᵀA, AᵀY), reuse for solve.  `_streaming_exp.py:126-145`.
10. **`refresh_inverse()` may mutate `_ata` in place** when already float64.
   `streaming.py:324-326`, `layers.py:364-366`.  Add `.clone()` before α.

## P3

7. encoder.T transposed view (probably a wash on modern BLAS).
8. delay features unfold (already optimised).
9. dataset cache per-worker (only matters if we add a Pool).

## Action items added to todos

- `b2-perf-grouped-sweep`  — apply finding 1 in next sweep design.
- `b2-perf-chunk-stats-default`  — set default + add memory budget knob.
- `b2-perf-float32-test`  — small experiment per finding 5.
- `b2-perf-refresh-inverse-clone`  — fix finding 10 (correctness).
