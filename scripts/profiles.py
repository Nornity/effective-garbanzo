"""Силуэты и профили обоих ключей в выровненных координатах + картинки."""
from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def prof(P, nbin=600):
    x = P[:, 0]
    x0, x1 = x.min(), x.max()
    edges = np.linspace(x0, x1, nbin + 1)
    idx = np.clip(np.digitize(x, edges) - 1, 0, nbin - 1)
    ymin = np.full(nbin, np.inf)
    ymax = np.full(nbin, -np.inf)
    zmin = np.full(nbin, np.inf)
    zmax = np.full(nbin, -np.inf)
    np.minimum.at(ymin, idx, P[:, 1])
    np.maximum.at(ymax, idx, P[:, 1])
    np.minimum.at(zmin, idx, P[:, 2])
    np.maximum.at(zmax, idx, P[:, 2])
    cx = (edges[:-1] + edges[1:]) / 2
    ok = np.isfinite(ymin)
    return cx[ok], ymin[ok], ymax[ok], zmin[ok], zmax[ok]


def main():
    out = {}
    for name in ("kwikset", "hall"):
        P = np.load(f"/tmp/{name}_pts.npy")
        cx, ymin, ymax, zmin, zmax = prof(P)
        out[name] = (cx, ymin, ymax, zmin, zmax)
        h = ymax - ymin
        t = zmax - zmin
        print(f"{name}: длина полотна+головки {cx.max()-cx.min():.2f}; "
              f"макс высота {h.max():.2f}; толщина медиана {np.median(t[ h<12 ]):.3f}")

    fig, axs = plt.subplots(4, 1, figsize=(14, 12), dpi=100)
    for name, col in (("kwikset", "C0"), ("hall", "C3")):
        cx, ymin, ymax, zmin, zmax = out[name]
        c0 = cx.min()
        axs[0].plot(cx - c0, ymax, col, lw=1, label=name + " верх")
        axs[0].plot(cx - c0, ymin, col, lw=1, ls="--", label=name + " низ")
        axs[1].plot(cx - c0, ymax - ymin, col, lw=1, label=name)
        axs[2].plot(cx - c0, zmax - zmin, col, lw=1, label=name)
        axs[3].plot(cx - c0, (ymax + ymin) / 2, col, lw=1, label=name)
    axs[0].set_title("силуэт: верхняя/нижняя граница по длине ключа")
    axs[1].set_title("высота сечения")
    axs[2].set_title("толщина сечения")
    axs[3].set_title("центр по высоте")
    for a in axs:
        a.grid(alpha=.3)
        a.legend(fontsize=8)
        a.set_xlabel("расстояние от левого края, мм")
    plt.tight_layout()
    plt.savefig("renders/profiles.png")
    print("saved renders/profiles.png")


if __name__ == "__main__":
    main()
