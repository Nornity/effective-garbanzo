"""Структура мешей: оболочки, отверстия, сечения ключа."""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "scripts")
from stl_tools import (
    is_binary_stl, load_ascii_stl, load_binary_stl, dedup_vertices, tri_areas, tri_normals,
)


class DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def shells(V, tol=1e-4):
    _, idx = dedup_vertices(V, tol)
    n = idx.max() + 1
    d = DSU(n)
    for f in idx:
        d.union(int(f[0]), int(f[1]))
        d.union(int(f[1]), int(f[2]))
    groups = {}
    for i in range(len(V)):
        groups.setdefault(d.find(int(idx[i][0])), []).append(i)
    return idx, list(groups.values())


def report(path):
    binary = is_binary_stl(path)
    if binary:
        V, N, A, H = load_binary_stl(path)
    else:
        V, N, H, _ = load_ascii_stl(path)
        A = None
    print("=" * 70)
    print(path)
    idx, sh = shells(V)
    areas = tri_areas(V)
    print(f"  оболочек: {len(sh)}")
    for i, s in enumerate(sorted(sh, key=len, reverse=True)):
        P = V[s].reshape(-1, 3, 3)
        print(f"    #{i}: граней {len(s):5d}  ({100*len(s)/len(V):4.1f} %)  "
              f"габариты {np.round(P.reshape(-1,3).max(0)-P.reshape(-1,3).min(0),2)}  "
              f"объём(по тетр.) {abs(np.einsum('ij,ij->i', P[:,0], np.cross(P[:,1],P[:,2])).sum()/6):.1f}")
    # невыпуклость: отверстия в головке — считаем рёбра, образующие замкнутые контуры
    print(f"  всего граней: {len(V)}, площадь {areas.sum():.1f} мм^2")
    # размеры граней
    print(f"  площадь граней: медиана {np.median(areas):.4f}, макс {areas.max():.3f} мм^2")
    if A is not None:
        print(f"  COLOR-атрибут: уникальных значений {len(np.unique(A))}, значение {np.unique(A)}")
    return V, idx, sh


if __name__ == "__main__":
    for p in ("kwikset-41332-1791104343.stl", "малый зал.stl"):
        report(p)
