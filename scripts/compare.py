"""Выравнивание двух ключей по оси полотна и сравнение профилей."""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "scripts")
from stl_tools import is_binary_stl, load_ascii_stl, load_binary_stl


def load(path):
    if is_binary_stl(path):
        v, n, a, h = load_binary_stl(path)
        return v, a
    v, n, h, t = load_ascii_stl(path)
    return v, None


def pca_axes(V: np.ndarray):
    P = V.reshape(-1, 3)
    c = P.mean(0)
    X = P - c
    cov = X.T @ X / len(X)
    w, R = np.linalg.eigh(cov)  # по возрастанию
    return c, R


def order_axes(V):
    """Возвращает V, повёрнутый так, что:
    x — длинная ось полотна, y — 'высота' (нарезка), z — толщина."""
    P = V.reshape(-1, 3)
    c = P.mean(0)
    Q = P - c
    cov = Q.T @ Q / len(Q)
    w, R = np.linalg.eigh(cov)
    # R[:, i] — собственный вектор для w[i]; берём порядок: толщина, высота, длина
    # сортируем: наименьший разброс = толщина (z), наибольший = длина (x)
    idx = np.argsort(w)  # [thin, mid, long]
    Rz, Ry, Rx = R[:, idx[0]], R[:, idx[1]], R[:, idx[2]]
    # ортонормируем
    Rx = Rx / np.linalg.norm(Rx)
    Ry = Ry - Rx * (Ry @ Rx)
    Ry /= np.linalg.norm(Ry)
    Rz = np.cross(Rx, Ry)
    M = np.stack([Rx, Ry, Rz], axis=0)  # строки — новые базисные векторы
    Vr = V.reshape(-1, 3) @ M.T
    return Vr.reshape(V.shape), M, c


def blade_analysis(Vr):
    """По длинной оси x считаем поперечные габариты; находим полотно и профиль."""
    P = Vr.reshape(-1, 3)
    x = P[:, 0]
    x0, x1 = x.min(), x.max()
    nbin = 400
    edges = np.linspace(x0, x1, nbin + 1)
    # индексы вершин в бине
    idx = np.clip(np.digitize(x, edges) - 1, 0, nbin - 1)
    height = np.full(nbin, np.nan)
    thick = np.full(nbin, np.nan)
    ymin = np.full(nbin, np.nan)
    ymax = np.full(nbin, np.nan)
    for i in range(nbin):
        m = idx == i
        if m.sum() < 1:
            continue
        y = P[m, 1]
        z = P[m, 2]
        height[i] = y.max() - y.min()
        thick[i] = z.max() - z.min()
        ymin[i] = y.min()
        ymax[i] = y.max()
    centers = (edges[:-1] + edges[1:]) / 2
    # полотно: там где толщина мала (< 3 мм) и высота < 15 мм
    blade = (thick < 3.0) & (height < 15.0)
    return dict(centers=centers, height=height, thick=thick, ymin=ymin, ymax=ymax,
                blade=blade, x0=x0, x1=x1)


def main():
    a, _ = load("kwikset-41332-1791104343.stl")
    b, _ = load("малый зал.stl")
    for name, V in (("kwikset", a), ("малый зал", b)):
        Vr, M, c = order_axes(V)
        P = Vr.reshape(-1, 3)
        print(f"--- {name}: extents (x=длина, y=высота, z=толщина) =",
              np.round(P.max(0) - P.min(0), 3))
        ba = blade_analysis(Vr)
        bl = ba["blade"]
        print("   полотно: x от", round(ba["centers"][bl].min(), 2), "до",
              round(ba["centers"][bl].max(), 2),
              "| ширина полотна", round(np.nanmedian(ba["height"][bl]), 2),
              "| толщина", round(np.nanmedian(ba["thick"][bl]), 3))
        np.save(f"/tmp/{name.replace(' ','_')}_Vr.npy", Vr)


if __name__ == "__main__":
    main()
