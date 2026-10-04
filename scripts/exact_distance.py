"""Точное расстояние от выборки точек одной модели до поверхности другой (point→triangle),
с использованием KD-дерева по вершинам (поиск треугольников-кандидатов)."""
from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree


def closest_point_on_tri(p, a, b, c):
    """Векторно, алгоритм Ericson. p: (n,3); a,b,c: (n,3) или (n,k,3)."""
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = np.einsum("...i,...i->...", ab, ap)
    d2 = np.einsum("...i,...i->...", ac, ap)
    out = a.copy()
    # 1) область A
    cond = (d1 <= 0) & (d2 <= 0)
    # 2) область B
    bp = p - b
    d3 = np.einsum("...i,...i->...", ab, bp)
    d4 = np.einsum("...i,...i->...", ac, bp)
    cond |= (d3 >= 0) & (d4 <= d3)
    out = np.where(cond[..., None], b, out)
    # 3) ребро AB
    vc = d1 * d4 - d3 * d2
    cond = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    v = np.where(np.abs(d1 - d3) > 1e-30, d1 / np.where(d1 - d3 == 0, 1, d1 - d3), 0.0)
    cand = a + v[..., None] * ab
    out = np.where(cond[..., None], cand, out)
    # 4) область C
    cp = p - c
    d5 = np.einsum("...i,...i->...", ab, cp)
    d6 = np.einsum("...i,...i->...", ac, cp)
    cond = (d6 >= 0) & (d5 <= d6)
    out = np.where(cond[..., None], c, out)
    # 5) ребро AC
    vb = d5 * d2 - d1 * d6
    cond = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    w = np.where(np.abs(d2 - d6) > 1e-30, d2 / np.where(d2 - d6 == 0, 1, d2 - d6), 0.0)
    cand = a + w[..., None] * ac
    out = np.where(cond[..., None], cand, out)
    # 6) ребро BC
    va = d3 * d6 - d5 * d4
    cond = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    den = (d4 - d3) + (d5 - d6)
    w = np.where(np.abs(den) > 1e-30, (d4 - d3) / np.where(den == 0, 1, den), 0.0)
    cand = b + w[..., None] * (c - b)
    out = np.where(cond[..., None], cand, out)
    # 7) внутренняя область
    denom = va + vb + vc
    vv = np.where(np.abs(denom) > 1e-30, vb / np.where(denom == 0, 1, denom), 0.0)
    ww = np.where(np.abs(denom) > 1e-30, vc / np.where(denom == 0, 1, denom), 0.0)
    cand = a + vv[..., None] * ab + ww[..., None] * ac
    cond = np.abs(denom) > 1e-30
    out = np.where(cond[..., None], cand, out)
    return out


def build_mesh_index(V, tol=1e-4):
    """Возвращает вершины (uniq), треугольники в индексах uniq, и список треугольников на вершину."""
    flat = V.reshape(-1, 3)
    q = np.round(flat / tol).astype(np.int64)
    uniq, inv = np.unique(q, axis=0, return_inverse=True)
    uniq = uniq * tol
    tri = inv.reshape(-1, 3)
    inc = [[] for _ in range(len(uniq))]
    for t, (i, j, k) in enumerate(tri):
        inc[i].append(t)
        inc[j].append(t)
        inc[k].append(t)
    return uniq, tri, inc


def surface_distance(P, V, k=16, chunk=20_000, verbose=True):
    uniq, tri, inc = build_mesh_index(V)
    tree = cKDTree(uniq)
    A = V[:, 0][tri]
    B = V[:, 1][tri]
    C = V[:, 2][tri]
    k = min(k, len(uniq))
    out = np.empty(len(P))
    for s in range(0, len(P), chunk):
        p = P[s:s + chunk]
        _, nb = tree.query(p, k=k)
        if nb.ndim == 1:
            nb = nb[:, None]
        for i in range(len(p)):
            ts = set()
            for v in nb[i]:
                ts.update(inc[v])
            t = np.fromiter(ts, dtype=np.int64)
            q = closest_point_on_tri(p[i], A[t], B[t], C[t])
            out[s + i] = np.linalg.norm(q - p[i], axis=1).min()
        if verbose and s % (chunk * 10) == 0:
            print(f"    {s + len(p)}/{len(P)}", flush=True)
    return out


def main():
    A = np.load("/tmp/kwikset_canon.npy")
    B = np.load("/tmp/hall_canon.npy")
    # треугольники для точного расстояния
    import stl_tools
    sys.path.insert(0, "scripts")
    from stl_tools import load_ascii_stl, load_binary_stl, is_binary_stl
    from align import canonicalize

    VT = {}
    for name, path in (("A", "kwikset-41332-1791104343.stl"), ("B", "малый зал.stl")):
        if is_binary_stl(path):
            v, n, a, h = load_binary_stl(path)
        else:
            v, n, h, t = load_ascii_stl(path)
        Vc, M, c = canonicalize(v, name)
        VT[name] = Vc.reshape(-1, 3, 3)

    rng = np.random.default_rng(1)
    nq = 150_000
    PA = A[rng.choice(len(A), nq, replace=False)]
    PB = B[rng.choice(len(B), nq, replace=False)]

    print("A→B (kwikset → малый зал)")
    dA = surface_distance(PA, VT["B"])
    print("B→A (малый зал → kwikset)")
    dB = surface_distance(PB, VT["A"])

    for nm, d in (("A→B", dA), ("B→A", dB)):
        print(f"  {nm}: медиана {np.median(d):.3f} мм, 90% {np.percentile(d,90):.3f}, "
              f"95% {np.percentile(d,95):.3f}, макс {d.max():.3f}; >0.3мм {100*(d>0.3).mean():.1f}%, "
              f">0.5мм {100*(d>0.5).mean():.1f}%, >1мм {100*(d>1).mean():.1f}%")

    fig, axs = plt.subplots(2, 1, figsize=(16, 11), dpi=110)
    for ax, P, d, nm in ((axs[0], PA, dA, "kwikset → малый зал"),
                         (axs[1], PB, dB, "малый зал → kwikset")):
        sc = ax.scatter(P[:, 0], P[:, 1], c=np.clip(d, 0, 2), s=3, cmap="turbo", vmin=0, vmax=2)
        ax.set_aspect("equal")
        ax.set_title(f"Точное расстояние до поверхности второй модели ({nm}), мм")
        ax.set_ylabel("y, мм")
        ax.grid(alpha=.2)
        plt.colorbar(sc, ax=ax, shrink=.85)
    axs[1].set_xlabel("x от кончика жала, мм")
    plt.tight_layout()
    plt.savefig("renders/exact_distance.png")
    print("saved renders/exact_distance.png")
    np.save("/tmp/dA.npy", dA)
    np.save("/tmp/dB.npy", dB)
    np.save("/tmp/PA.npy", PA)
    np.save("/tmp/PB.npy", PB)


if __name__ == "__main__":
    main()
