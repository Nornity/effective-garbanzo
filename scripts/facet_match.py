"""Поиск общих граней между мешами по длинам сторон треугольников."""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "scripts")
from stl_tools import is_binary_stl, load_ascii_stl, load_binary_stl


def load(path):
    if is_binary_stl(path):
        v, n, a, h = load_binary_stl(path)
    else:
        v, n, h, t = load_ascii_stl(path)
    return v


def sides(V):
    e = np.stack([
        np.linalg.norm(V[:, 1] - V[:, 0], axis=1),
        np.linalg.norm(V[:, 2] - V[:, 1], axis=1),
        np.linalg.norm(V[:, 0] - V[:, 2], axis=1),
    ], axis=1)
    return np.sort(e, axis=1)


def main():
    A = load("kwikset-41332-1791104343.stl")
    B = load("малый зал.stl")
    sa, sb = sides(A), sides(B)
    qa = np.round(sa, 3)
    qb = np.round(sb, 3)
    ka = set(map(tuple, qa))
    kb = set(map(tuple, qb))
    common = ka & kb
    print(f"граней: kwikset {len(A)}, малый зал {len(B)}")
    print(f"уникальных форм треугольников: {len(ka)} и {len(kb)}; общих: {len(common)}")
    # сколько граней каждого меша имеют форму, встречающуюся у другого
    ma = np.array([tuple(x) in kb for x in qa])
    mb = np.array([tuple(x) in ka for x in qb])
    print(f"  у kwikset {ma.sum()} из {len(A)} ({100*ma.mean():.1f}%) имеют пару в 'малый зал'")
    print(f"  у 'малый зал' {mb.sum()} из {len(B)} ({100*mb.mean():.1f}%) имеют пару в kwikset")

    # статистика по несопоставленным граням: площадь и расположение
    def area(V):
        return 0.5 * np.linalg.norm(np.cross(V[:, 1] - V[:, 0], V[:, 2] - V[:, 0]), axis=1)

    aa = area(A)
    print(f"\nплощадь уникальных граней kwikset (нет пары): {aa[~ma].sum():.1f} мм^2 "
          f"({100*aa[~ma].sum()/aa.sum():.1f}% площади)")
    print(f"площадь уникальных граней 'малый зал': {area(B)[~mb].sum():.1f} мм^2 "
          f"({100*area(B)[~mb].sum()/area(B).sum():.1f}% площади)")
    # центры несопоставленных граней kwikset
    cen = A[~ma].mean(axis=1)
    print("центры несопоставленных граней kwikset: границы x/y/z:",
          np.round(cen.min(0), 2), np.round(cen.max(0), 2))
    cenb = B[~mb].mean(axis=1)
    print("центры несопоставленных граней 'малый зал':",
          np.round(cenb.min(0), 2), np.round(cenb.max(0), 2))


if __name__ == "__main__":
    main()
