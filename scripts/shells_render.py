"""Рендер оболочек разными цветами + общий вид двух ключей."""
from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection

sys.path.insert(0, "scripts")
from stl_tools import is_binary_stl, load_ascii_stl, load_binary_stl, dedup_vertices, tri_areas
from mesh_struct import shells
from align import pca_frame


def load(path):
    if is_binary_stl(path):
        v, n, a, h = load_binary_stl(path)
    else:
        v, n, h, t = load_ascii_stl(path)
    return v


def draw(ax, polys, colors, elev=0, azim=-90, title=""):
    V = polys
    P = V.reshape(-1, 3).copy()
    c = (P.min(0) + P.max(0)) / 2
    V = V - c
    e, a = np.radians(elev), np.radians(azim)
    Rx = np.array([[1, 0, 0], [0, np.cos(e), -np.sin(e)], [0, np.sin(e), np.cos(e)]])
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    P = V.reshape(-1, 3) @ (Rx @ Rz).T
    P = P.reshape(-1, 3, 3)
    depth = P[:, :, 1].mean(1)
    order = np.argsort(depth)
    ax.add_collection(PolyCollection(P[order][:, :, [0, 2]], facecolors=np.array(colors)[order],
                                     edgecolors="none"))
    ax.set_xlim(P[:, :, 0].min() - 1, P[:, :, 0].max() + 1)
    ax.set_ylim(P[:, :, 2].min() - 1, P[:, :, 2].max() + 1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=10)


def main():
    names = [("kwikset-41332 (ASCII)", "kwikset-41332-1791104343.stl"),
             ("малый зал (binary)", "малый зал.stl")]
    fig, axs = plt.subplots(2, 2, figsize=(16, 9), dpi=110)
    palette = ["#3b6fb6", "#d9534f", "#4caf50", "#ffcc00"]
    for row, (label, path) in enumerate(names):
        V = load(path)
        M, c = pca_frame(V)
        Vr = ((V.reshape(-1, 3) - c) @ M.T).reshape(-1, 3, 3)
        # ось Y = высота -> положим её вверх на картинке: отрисуем вид сверху на плоскость X-Y
        idx, sh = shells(Vr)
        sh = sorted(sh, key=len, reverse=True)
        cols = np.zeros((len(Vr), 4))
        for i, s in enumerate(sh):
            cols[s] = matplotlib.colors.to_rgba(palette[i % len(palette)])
        # вид на плоскость X-Y (нормаль Z)
        ax = axs[row, 0]
        depth = Vr[:, :, 2].mean(1)
        order = np.argsort(depth)
        ax.add_collection(PolyCollection(Vr[order][:, :, [0, 1]], facecolors=cols[order],
                                         edgecolors="#333", linewidths=0.15))
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"{label} — вид на плоскость полотна, цвета = оболочки", fontsize=10)
        # вид сбоку (нормаль Y) — виден профиль
        ax = axs[row, 1]
        depth = Vr[:, :, 1].mean(1)
        order = np.argsort(depth)
        ax.add_collection(PolyCollection(Vr[order][:, :, [0, 2]], facecolors=cols[order],
                                         edgecolors="#333", linewidths=0.15))
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"{label} — вид сбоку (профиль нарезки)", fontsize=10)
    plt.tight_layout()
    plt.savefig("renders/shells.png")
    print("saved renders/shells.png")


if __name__ == "__main__":
    main()
