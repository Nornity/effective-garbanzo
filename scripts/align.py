"""Каноничное выравнивание ключей: жало -> x=0, головка -> +x, плоскость полотна = +y."""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "scripts")


def pca_frame(V):
    P = V.reshape(-1, 3)
    c = P.mean(0)
    Q = P - c
    cov = Q.T @ Q / len(Q)
    w, R = np.linalg.eigh(cov)
    idx = np.argsort(w)          # [толщина, высота, длина]
    Ez, Ey, Ex = R[:, idx[0]], R[:, idx[1]], R[:, idx[2]]
    Ex = Ex / np.linalg.norm(Ex)
    Ey = Ey - Ex * (Ey @ Ex)
    Ey /= np.linalg.norm(Ey)
    Ez = np.cross(Ex, Ey)
    return np.stack([Ex, Ey, Ez]), c


def profile(P, nbin=800):
    x = P[:, 0]
    x0, x1 = x.min(), x.max()
    edges = np.linspace(x0, x1, nbin + 1)
    idx = np.clip(np.digitize(x, edges) - 1, 0, nbin - 1)
    ymin = np.full(nbin, np.inf)
    ymax = np.full(nbin, -np.inf)
    np.minimum.at(ymin, idx, P[:, 1])
    np.maximum.at(ymax, idx, P[:, 1])
    cx = (edges[:-1] + edges[1:]) / 2
    ok = np.isfinite(ymin)
    return cx[ok], ymin[ok], ymax[ok]


def canonicalize(V, tag=""):
    M, c = pca_frame(V)
    P = (V.reshape(-1, 3) - c) @ M.T

    # головка = область с большой высотой; полотно = тонкая часть
    cx, ymn, ymx = profile(P)
    h = ymx - ymn
    head_frac = (h > 0.75 * h.max()).mean()
    # направление: головка должна быть в +x
    mid = (cx.min() + cx.max()) / 2
    h_left = h[cx < mid].max()
    h_right = h[cx > mid].max()
    if h_left > h_right:                     # головка слева -> разворачиваем
        P[:, 0] *= -1
        cx, ymn, ymx = profile(P)
        h = ymx - ymn
    # начало координат по x = кончик жала (тонкий край)
    x_tip = cx.min()
    P[:, 0] -= x_tip

    # ориентация по y: сторона без нарезки (более прямая граница) -> вниз? Возьмём
    # более ровную границу и направим её в +y (спинка полотна сверху).
    blade = h < 0.5 * h.max()
    top_jag = np.abs(np.diff(ymx[blade])).mean()
    bot_jag = np.abs(np.diff(ymn[blade])).mean()
    if bot_jag < top_jag:                    # сейчас ровная граница снизу -> переворачиваем
        P[:, 1] *= -1
    # толщина: центр по z в 0
    P[:, 2] -= (P[:, 2].min() + P[:, 2].max()) / 2
    return P, M, c


def main():
    for name, path in (("kwikset", "kwikset-41332-1791104343.stl"),
                       ("hall", "малый зал.stl")):
        P = np.load(f"/tmp/{name}_pts.npy")
        Pc, M, c = canonicalize(P, name)
        np.save(f"/tmp/{name}_canon.npy", Pc)
        print(name, "габариты (x, y, z) =", np.round(Pc.max(0) - Pc.min(0), 3))
        cx, ymn, ymx = profile(Pc)
        h = ymx - ymn
        print("   высота полотна медиана:", round(float(np.median(h[h < 12])), 3),
              " макс:", round(float(h.max()), 2))


if __name__ == "__main__":
    main()
