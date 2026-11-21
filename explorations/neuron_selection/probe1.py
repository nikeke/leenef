"""Throwaway feasibility probe (single seed, not paper evidence).

P-A: exact OBS pruning from sufficient statistics vs random / low-activity /
     magnitude pruning, pool 4000 -> k.
P-B: AGOP-shaped encoders (analytic solve -> Jacobian outer product ->
     resample encoders) vs random hypersphere encoders at equal width.
"""

import sys
import time

import torch

sys.path.insert(0, "/home/amnesia/Persistent/src/leenef")
from benchmarks.sl_clean.datasets import load_dataset  # noqa: E402

torch.set_num_threads(10)
DS = sys.argv[1] if len(sys.argv) > 1 else "mnist"
b = load_dataset(DS)
names = [n for n in dir(b) if not n.startswith("_")]
print(names, flush=True)
Xtr, ytr, Xte, yte = b.x_train, b.y_train, b.x_test, b.y_test
g0 = torch.Generator().manual_seed(0)
perm = torch.randperm(Xtr.shape[0], generator=g0)
va, tr = perm[:10000], perm[10000:]
Xva, yva, Xtr, ytr = Xtr[va], ytr[va], Xtr[tr], ytr[tr]
d_in = Xtr.shape[1]
Ytr = torch.nn.functional.one_hot(ytr, 10).double()
ALPHAS = (1e-5, 1e-4, 1e-3, 1e-2)


def make(n, seed, M_half=None):
    g = torch.Generator().manual_seed(seed)
    E = torch.randn(n, d_in, generator=g)
    if M_half is not None:
        E = E @ M_half
    E = E / E.norm(dim=1, keepdim=True)
    gain = torch.rand(n, generator=g) * 1.5 + 0.5
    C = Xtr[torch.randint(Xtr.shape[0], (n,), generator=g)]
    bias = -gain * (C * E).sum(1)
    return E, gain, bias


def enc(X, E, gain, bias):
    return ((X @ E.T) * gain + bias).abs_()


def stats(E, gain, bias):
    G = torch.zeros(E.shape[0], E.shape[0], dtype=torch.float64)
    R = torch.zeros(E.shape[0], 10, dtype=torch.float64)
    for i in range(0, Xtr.shape[0], 10000):
        A = enc(Xtr[i : i + 10000], E, gain, bias)
        G += (A.T @ A).double()
        R += A.T.double() @ Ytr[i : i + 10000]
    return G, R


def solve(G, R, alpha):
    n = G.shape[0]
    lam = max(alpha * torch.trace(G).item() / n, alpha)
    M = G + lam * torch.eye(n, dtype=torch.float64)
    return torch.linalg.solve(M, R), M


def acc(A, D, y):
    return ((A.double() @ D).argmax(1) == y).float().mean().item() * 100


def best(G, R, Ava, Ate):
    """alpha chosen on validation; report (val, test, alpha)."""
    r = []
    for a in ALPHAS:
        D, _ = solve(G, R, a)
        r.append((acc(Ava, D, yva), acc(Ate, D, yte), a, D))
    return max(r, key=lambda t: t[0])


def sub(G, R, idx):
    return G[idx][:, idx], R[idx]


# ---------------- P-A: pruning ----------------
if "A" in (sys.argv[2] if len(sys.argv) > 2 else "AB"):
    t0 = time.time()
    N = 4000
    E, gain, bias = make(N, 1)
    G, R = stats(E, gain, bias)
    Ava, Ate = enc(Xva, E, gain, bias), enc(Xte, E, gain, bias)
    v, t, a, D = best(G, R, Ava, Ate)
    print(f"[A] pool N={N}: val {v:.2f} test {t:.2f} alpha {a}  ({time.time() - t0:.0f}s)", flush=True)
    targets = [2000, 1000, 500, 250]
    act = torch.diagonal(G).sqrt()
    mag = D.norm(dim=1) * act
    gr = torch.Generator().manual_seed(7)
    rnd = torch.randperm(N, generator=gr)
    # OBS greedy backward elimination, 10% per step, exact refit each step
    keep = torch.arange(N)
    obs_sets = {}
    for k in targets:
        while len(keep) > k:
            Gs, Rs = sub(G, R, keep)
            Ds, Ms = solve(Gs, Rs, a)
            inv_diag = torch.diagonal(torch.linalg.inv(Ms))
            s = (Ds**2).sum(1) / inv_diag
            n_drop = min(max(len(keep) // 10, 1), len(keep) - k)
            order = torch.argsort(s)
            keep = keep[order[n_drop:].sort().values]
        obs_sets[k] = keep.clone()
    for k in targets:
        row = {}
        for name, idx in (
            ("random", rnd[:k]),
            ("high-activity", torch.argsort(act, descending=True)[:k]),
            ("magnitude", torch.argsort(mag, descending=True)[:k]),
            ("OBS-exact", obs_sets[k]),
        ):
            Gs, Rs = sub(G, R, idx)
            v, t, a2, _ = best(Gs, Rs, Ava[:, idx], Ate[:, idx])
            row[name] = t
        print(f"[A] k={k}: " + "  ".join(f"{n} {x:.2f}" for n, x in row.items()), flush=True)
    print(f"[A] done {time.time() - t0:.0f}s", flush=True)

# ---------------- P-B: AGOP encoders ----------------
if "B" in (sys.argv[2] if len(sys.argv) > 2 else "AB"):
    gs = torch.Generator().manual_seed(3)
    Xs = Xtr[torch.randperm(Xtr.shape[0], generator=gs)[:10000]]

    def agop(E, gain, bias, D):
        S = torch.sign((Xs @ E.T) * gain + bias) * gain
        M = torch.zeros(d_in, d_in, dtype=torch.float64)
        for c in range(10):
            J = (S * D[:, c].float()) @ E
            M += (J.T @ J).double()
        return M / Xs.shape[0]

    def mpow(M, p):
        w, V = torch.linalg.eigh(M)
        w = w.clamp(min=0)
        w = w / w.max()
        return ((V * w**p) @ V.T).float()

    for n in (250, 500, 1000, 2000, 4000):
        t0 = time.time()
        E, gain, bias = make(n, 1)
        G, R = stats(E, gain, bias)
        v, t, a, D = best(G, R, enc(Xva, E, gain, bias), enc(Xte, E, gain, bias))
        out = [f"random {t:.2f}"]
        for p in (0.5, 1.0):
            Ei, gi, bi, Di = E, gain, bias, D
            for it in range(3):
                Mh = mpow(agop(Ei, gi, bi, Di), p)
                Ei, gi, bi = make(n, 100 + it, Mh)
                Gi, Ri = stats(Ei, gi, bi)
                v, t, a, Di = best(Gi, Ri, enc(Xva, Ei, gi, bi), enc(Xte, Ei, gi, bi))
                out.append(f"p{p}-it{it + 1} {t:.2f}")
        print(f"[B] n={n}: " + "  ".join(out) + f"  ({time.time() - t0:.0f}s)", flush=True)
