"""Throwaway probe 2 (single seed, not paper evidence).

G: generational grow-and-prune at fixed budget k (propose k fresh neurons,
   accumulate 2k Gram, exact-OBS prune back to k), random vs AGOP proposals.
D: replication/distillation: NEF student regressing CNN-teacher logits vs
   one-hot labels, with and without teacher-labelled shift augmentation.
"""

import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/home/amnesia/Persistent/src/leenef")
from benchmarks.sl_clean.datasets import load_dataset  # noqa: E402

torch.set_num_threads(10)
DS = sys.argv[1]
WHAT = sys.argv[2]
b = load_dataset(DS)
Xtr, ytr, Xte, yte = b.x_train, b.y_train, b.x_test, b.y_test
g0 = torch.Generator().manual_seed(0)
perm = torch.randperm(Xtr.shape[0], generator=g0)
va, tr = perm[:10000], perm[10000:]
Xva, yva, Xtr, ytr = Xtr[va], ytr[va], Xtr[tr], ytr[tr]
d_in = Xtr.shape[1]
Ytr = F.one_hot(ytr, 10).double()
ALPHAS = (1e-5, 1e-4, 1e-3, 1e-2)


def make(n, seed, M_half=None):
    g = torch.Generator().manual_seed(seed)
    E = torch.randn(n, d_in, generator=g)
    if M_half is not None:
        E = E @ M_half
    E = E / E.norm(dim=1, keepdim=True)
    gain = torch.rand(n, generator=g) * 1.5 + 0.5
    C = Xtr[torch.randint(Xtr.shape[0], (n,), generator=g)]
    return E, gain, -gain * (C * E).sum(1)


def enc(X, P):
    E, gain, bias = P
    return ((X @ E.T) * gain + bias).abs_()


def stats(P, X=None, Y=None, G=None, R=None):
    X = Xtr if X is None else X
    Y = Ytr if Y is None else Y
    n = P[0].shape[0]
    G = torch.zeros(n, n, dtype=torch.float64) if G is None else G
    R = torch.zeros(n, Y.shape[1], dtype=torch.float64) if R is None else R
    for i in range(0, X.shape[0], 10000):
        A = enc(X[i : i + 10000], P)
        G += (A.T @ A).double()
        R += A.T.double() @ Y[i : i + 10000].double()
    return G, R


def solve(G, R, alpha):
    n = G.shape[0]
    lam = max(alpha * torch.trace(G).item() / n, alpha)
    M = G + lam * torch.eye(n, dtype=torch.float64)
    return torch.linalg.solve(M, R), M


def acc(A, D, y):
    return ((A.double() @ D).argmax(1) == y).float().mean().item() * 100


def best(G, R, Ava, Ate):
    r = []
    for a in ALPHAS:
        D, _ = solve(G, R, a)
        r.append((acc(Ava, D, yva), acc(Ate, D, yte), a, D))
    return max(r, key=lambda t: t[0])


def obs_prune(G, R, k, alpha=1e-3, frac=0.1):
    keep = torch.arange(G.shape[0])
    while len(keep) > k:
        Gs, Rs = G[keep][:, keep], R[keep]
        Ds, Ms = solve(Gs, Rs, alpha)
        s = (Ds**2).sum(1) / torch.diagonal(torch.linalg.inv(Ms))
        n_drop = min(max(int(len(keep) * frac), 1), len(keep) - k)
        keep = keep[torch.argsort(s)[n_drop:].sort().values]
    return keep


def cat(P, Q):
    return tuple(torch.cat([p, q]) for p, q in zip(P, Q))


def take(P, idx):
    return tuple(p[idx] for p in P)


gs = torch.Generator().manual_seed(3)
Xs = Xtr[torch.randperm(Xtr.shape[0], generator=gs)[:10000]]


def agop_half(P, D):
    E, gain, bias = P
    S = torch.sign((Xs @ E.T) * gain + bias) * gain
    M = torch.zeros(d_in, d_in, dtype=torch.float64)
    for c in range(D.shape[1]):
        J = (S * D[:, c].float()) @ E
        M += (J.T @ J).double()
    w, V = torch.linalg.eigh(M)
    w = w.clamp(min=0)
    w = w / w.max()
    return ((V * w**0.5) @ V.T).float()


if "G" in WHAT:
    for k in (500, 1000):
        for mode in ("random", "agop"):
            t0 = time.time()
            P = make(k, 1)
            G, R = stats(P)
            v, t, a, D = best(G, R, enc(Xva, P), enc(Xte, P))
            out = [f"{t:.2f}"]
            for gen in range(12):
                Mh = agop_half(P, D) if mode == "agop" else None
                Q = make(k, 1000 + gen, Mh)
                PQ = cat(P, Q)
                G, R = stats(PQ)
                keep = obs_prune(G, R, k)
                surv = (keep < k).sum().item()
                P = take(PQ, keep)
                G, R = G[keep][:, keep], R[keep]
                v, t, a, D = best(G, R, enc(Xva, P), enc(Xte, P))
                out.append(f"{t:.2f}({surv})")
            print(f"[G] k={k} {mode}: " + " ".join(out) + f"  ({time.time() - t0:.0f}s)", flush=True)
    # one-shot big pool reference
    t0 = time.time()
    P = make(8000, 1)
    G, R = stats(P)
    for k in (1000, 500):
        keep = obs_prune(G, R, k)
        Pk = take(P, keep)
        v, t, a, D = best(G[keep][:, keep], R[keep], enc(Xva, Pk), enc(Xte, Pk))
        print(f"[G] one-shot pool 8000 -> {k}: {t:.2f}  ({time.time() - t0:.0f}s)", flush=True)

if "D" in WHAT:
    torch.manual_seed(0)
    side = int(d_in**0.5)
    net = nn.Sequential(
        nn.Unflatten(1, (1, side, side)),
        nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(64 * (side // 4) ** 2, 128), nn.ReLU(), nn.Linear(128, 10),
    )  # fmt: skip
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    t0 = time.time()
    for ep in range(3):
        p = torch.randperm(Xtr.shape[0])
        for i in range(0, len(p), 128):
            j = p[i : i + 128]
            opt.zero_grad()
            F.cross_entropy(net(Xtr[j]), ytr[j]).backward()
            opt.step()
    net.eval()

    @torch.no_grad()
    def teach(X):
        return torch.cat([net(X[i : i + 5000]) for i in range(0, X.shape[0], 5000)])

    Tte = teach(Xte)
    print(f"[D] teacher test acc {(Tte.argmax(1) == yte).float().mean() * 100:.2f}  ({time.time() - t0:.0f}s)", flush=True)

    def shift(X, seed):
        g = torch.Generator().manual_seed(seed)
        Xi = X.view(-1, side, side)
        out = torch.empty_like(Xi)
        s = torch.randint(-2, 3, (X.shape[0], 2), generator=g)
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                m = (s[:, 0] == dx) & (s[:, 1] == dy)
                out[m] = torch.roll(Xi[m], (dx, dy), (1, 2))
        return out.reshape(X.shape[0], -1)

    for n in (1000, 4000):
        P = make(n, 1)
        Ava, Ate = enc(Xva, P), enc(Xte, P)
        res = {}
        for name, tgt, n_aug in (
            ("onehot", "label", 0),
            ("logits", "teacher", 0),
            ("onehot+aug4", "label", 4),
            ("logits+aug4", "teacher", 4),
            ("logits+aug12", "teacher", 12),
        ):
            Y = Ytr if tgt == "label" else teach(Xtr).double()
            G, R = stats(P, Xtr, Y)
            for s in range(n_aug):
                Xa = shift(Xtr, 50 + s)
                Ya = Ytr if tgt == "label" else teach(Xa).double()
                G, R = stats(P, Xa, Ya, G, R)
            v, t, a, D = best(G, R, Ava, Ate)
            fid = ((Ate.double() @ D).argmax(1) == Tte.argmax(1)).float().mean().item() * 100
            res[name] = f"{t:.2f} (fid {fid:.1f})"
        print(f"[D] student n={n}: " + "  ".join(f"{k} {x}" for k, x in res.items()), flush=True)
