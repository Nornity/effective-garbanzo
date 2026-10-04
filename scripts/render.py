"""Простой рендер STL (painter's algorithm) средствами numpy+matplotlib."""
from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "scripts")
from stl_tools import load_ascii_stl, load_binary_stl, is_binary_stl, tri_normals, tri_areas


def load(path):
    if is_binary_stl(path):
        v, _, _, _ = load_binary_stl(path)
        return v
    v, _, _, _ = load_ascii_stl(path)
    return v


def rotation(elev_deg, azim_deg):
    e, a = np.radians(elev_deg), np.radians(azim_deg)
    Rx = np.array([[1, 0, 0], [0, np.cos(e), -np.sin(e)], [0, np.sin(e), np.cos(e)]])
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    return Rx @ Rz


def render(ax, verts, elev, azim, title="", up="z"):
    V = verts.copy().astype(float)
    c = (V.reshape(-1, 3).min(0) + V.reshape(-1, 3).max(0)) / 2
    V -= c
    R = rotation(elev, azim)
    P = V @ R.T
    # камера смотрит вдоль -Y после поворота: используем X-Z экран (z вверх)
    n = tri_normals(V)
    n_rot = n @ R.T
    areas = tri_areas(V)

    light = np.array([0.35, -0.8, 0.5])
    light = light / np.linalg.norm(light)
    shade = np.clip(n_rot @ light, 0, 1) * 0.85 + 0.15

    depth = P[:, :, 1].mean(axis=1)
    order = np.argsort(depth)  # дальние (меньший y) рисуем первыми
    from matplotlib.collections import PolyCollection

    polys = P[:, :, [0, 2]][order]
    cols = np.zeros((len(order), 4))
    cols[:, 0] = 0.55 * shade[order] + 0.25
    cols[:, 1] = 0.65 * shade[order] + 0.25
    cols[:, 2] = 0.80 * shade[order] + 0.20
    cols[:, 3] = 1.0
    pc = PolyCollection(polys, facecolors=cols, edgecolors="none", linewidths=0)
    ax.add_collection(pc)
    ax.set_xlim(P[:, :, 0].min() - 1, P[:, :, 0].max() + 1)
    ax.set_ylim(P[:, :, 2].min() - 1, P[:, :, 2].max() + 1)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def main():
    files = [
        ("kwikset-41332-1791104343.stl", "kwikset-41332 (ASCII)", "z"),
        ("малый зал.stl", "малый зал (binary)", "y"),
    ]
    views = [(25, -60), (25, 30), (25, 120), (88, -90), (0, -90), (0, 0)]
    fig, axes = plt.subplots(2, len(views), figsize=(3.2 * len(views), 7), dpi=110)
    for row, (path, label, _up) in enumerate(files):
        V = load(path)
        for col, (e, a) in enumerate(views):
            render(axes[row, col], V, e, a, title=label if col == 0 else "")
        # отдельный крупный план с разных сторон
    plt.tight_layout()
    plt.savefig("renders/overview.png")
    print("saved renders/overview.png")


if __name__ == "__main__":
    main()
