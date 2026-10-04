"""Карта расхождения поверхностей: расстояние от точек A до поверхности B."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree


def main():
    A = np.load("/tmp/kwikset_canon.npy")
    B = np.load("/tmp/hall_canon.npy")

    # сдвиг 0.9 мм по x найден ранее при выравнивании масок по лучшему IoU;
    # вычтем из B для честного сравнения (сдвиг = допуск совмещения)
    rng = np.random.default_rng(0)
    BUD = 1_200_000
    if len(A) > BUD:
        A = A[rng.choice(len(A), BUD, replace=False)]
    if len(B) > BUD:
        B = B[rng.choice(len(B), BUD, replace=False)]

    for dx in (0.0, 0.9):
        Bt = B.copy()
        Bt[:, 0] -= dx
        tree = cKDTree(Bt)
        d = np.empty(len(A))
        step = 200_000
        for i in range(0, len(A), step):
            dd, _ = tree.query(A[i:i+step], k=1)
            d[i:i+step] = dd
        print(f"сдвиг B по x = {dx:.1f} мм: расстояние A->B: "
              f"медиана {np.median(d):.3f}, среднее {d.mean():.3f}, "
              f"95-й перц {np.percentile(d,95):.3f}, макс {d.max():.3f} мм; "
              f"доля точек с d>0.5мм: {100*(d>0.5).mean():.1f}%")

    dx = 0.9
    Bt = B.copy()
    Bt[:, 0] -= dx
    tree = cKDTree(Bt)
    dA, _ = tree.query(A, k=1)
    treeA = cKDTree(A)
    dB = np.empty(len(Bt))
    for i in range(0, len(Bt), 200_000):
        dd, _ = treeA.query(Bt[i:i+200_000], k=1)
        dB[i:i+200_000] = dd

    fig, axs = plt.subplots(2, 1, figsize=(16, 10), dpi=110, sharex=True)
    for ax, P, d, name in ((axs[0], A, dA, "kwikset → малый зал"),
                           (axs[1], Bt, dB, "малый зал → kwikset")):
        sc = ax.scatter(P[:, 0], P[:, 1], c=np.clip(d, 0, 3), s=1.4, cmap="turbo",
                        vmin=0, vmax=3)
        ax.set_aspect("equal")
        ax.set_title(f"Расстояние от поверхности ({name}), мм (0 = поверхности совпадают)")
        ax.set_ylabel("y (высота), мм")
        ax.grid(alpha=.2)
        plt.colorbar(sc, ax=ax, shrink=.8)
    axs[1].set_xlabel("x (от кончика), мм")
    plt.tight_layout()
    plt.savefig("renders/distance_map.png")
    print("saved renders/distance_map.png")

    # статистика по зонам
    def zone_stats(P, d, name):
        zones = {
            "полотно (x<28)": P[:, 0] < 28,
            "головка (x>=28)": P[:, 0] >= 28,
        }
        for zn, m in zones.items():
            print(f"  {name} {zn}: медиана {np.median(d[m]):.3f} мм, "
                  f"доля >0.5 мм {100*(d[m]>0.5).mean():.1f}%")

    print("\nПо зонам:")
    zone_stats(A, dA, "A")
    zone_stats(Bt, dB, "B")


if __name__ == "__main__":
    main()
