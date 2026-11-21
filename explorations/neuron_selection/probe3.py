"""Throwaway probe 3 (single seed, not paper evidence).

H: residual-directed proposals (principal-Hessian directions of the
   residual-weighted input second moment) + OBS selection, k=1000.
T: multiplier-free encoders: sparse ternary {-1,0,+1}, unit gain,
   a = |e.x - e.c| (integer adds + abs only).
"""

import sys
import time

import torch

WHAT = sys.argv[2]
sys.argv = [sys.argv[0], sys.argv[1], ""]
sys.path.insert(0, "/home/amnesia/.copilot/session-state/16826c17-3a80-4bb2-9fd2-f41d72e7a4b0/files")
import probe2 as p  # noqa: E402

Xtr, Ytr, Xva, Xte, d_in = p.Xtr, p.Ytr, p.Xva, p.Xte, p.d_in
mu = Xtr.mean(0)


def phd_bases(P, D, q=40):
    """Per-class top-|lambda| eigenpairs of sum_x r_c(x) (x-mu)(x-mu)^T."""
    Xc = Xtr - mu
    res = torch.cat(
        [(Ytr[i : i + 10000] - p.enc(Xtr[i : i + 10000], P).double() @ D).float() for i in range(0, Xtr.shape[0], 10000)]
    )
    bases = []
    for c in range(res.shape[1]):
        H = (Xc * res[:, c : c + 1]).T @ Xc
        w, V = torch.linalg.eigh(H.double())
        top = torch.argsort(w.abs(), descending=True)[:q]
        bases.append((V[:, top] * w[top].abs().sqrt()).float())
    return bases


def make_phd(n, seed, bases):
    g = torch.Generator().manual_seed(seed)
    cls = torch.randint(len(bases), (n,), generator=g)
    E = torch.empty(n, d_in)
    for c, B in enumerate(bases):
        m = cls == c
        E[m] = torch.randn(int(m.sum()), B.shape[1], generator=g) @ B.T
    E = E / E.norm(dim=1, keepdim=True)
    gain = torch.rand(n, generator=g) * 1.5 + 0.5
    C = Xtr[torch.randint(Xtr.shape[0], (n,), generator=g)]
    return E, gain, -gain * (C * E).sum(1)


if "H" in WHAT:
    for k in (250, 1000):
        for mode in ("phd", "mix"):
            t0 = time.time()
            P = p.make(k, 1)
            G, R = p.stats(P)
            v, t, a, D = p.best(G, R, p.enc(Xva, P), p.enc(Xte, P))
            out = [f"{t:.2f}"]
            for gen in range(10):
                B = phd_bases(P, D)
                if mode == "phd":
                    Q = make_phd(k, 1000 + gen, B)
                else:
                    Q = p.cat(make_phd(k // 2, 1000 + gen, B), p.make(k - k // 2, 2000 + gen))
                PQ = p.cat(P, Q)
                G, R = p.stats(PQ)
                keep = p.obs_prune(G, R, k)
                surv = (keep < k).sum().item()
                P = p.take(PQ, keep)
                G, R = G[keep][:, keep], R[keep]
                v, t, a, D = p.best(G, R, p.enc(Xva, P), p.enc(Xte, P))
                out.append(f"{t:.2f}({surv})")
            print(f"[H] k={k} {mode}: " + " ".join(out) + f"  ({time.time() - t0:.0f}s)", flush=True)

if "T" in WHAT:
    live = (Xtr.std(0) > 0).nonzero().squeeze(1)

    def make_tern(n, nnz, seed, live_only):
        g = torch.Generator().manual_seed(seed)
        E = torch.zeros(n, d_in)
        pool = live if live_only else torch.arange(d_in)
        if nnz >= len(pool):
            cols = pool.expand(n, -1)
        else:
            cols = pool[torch.rand(n, len(pool), generator=g).argsort(1)[:, :nnz]]
        sg = (torch.randint(0, 2, cols.shape, generator=g) * 2 - 1).float()
        E.scatter_(1, cols, sg)
        gain = torch.ones(n)
        C = Xtr[torch.randint(Xtr.shape[0], (n,), generator=g)]
        return E, gain, -(C * E).sum(1)

    for n in (1000, 4000):
        P = p.make(n, 1)
        G, R = p.stats(P)
        v, t, a, D = p.best(G, R, p.enc(Xva, P), p.enc(Xte, P))
        out = [f"dense-float {t:.2f}"]
        for nnz, lo in ((784, False), (64, True), (24, True), (8, True), (4, True)):
            P = make_tern(n, nnz, 1, lo)
            G, R = p.stats(P)
            v, t, a, D = p.best(G, R, p.enc(Xva, P), p.enc(Xte, P))
            out.append(f"tern-nnz{nnz} {t:.2f}")
        print(f"[T] n={n}: " + "  ".join(out), flush=True)
