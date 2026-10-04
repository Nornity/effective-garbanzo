"""Извлечение зубцов (реза) каждого ключа: позиции и глубины."""
from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks, savgol_filter

RES = 0.05


def main():
    MA = np.load("/tmp/MAf.npy")
    MB = np.load("/tmp/MBs.npy")
    x0, x1, y0, y1, res = np.load("/tmp/geom.npy")
    xs = x0 + np.arange(MA.shape[1]) * res

    bot = lambda M: y0 + np.where(M.any(0), M.argmax(0), np.nan) * res
    top = lambda M: y0 + (M.shape[0] - 1 - np.where(M.any(0), M[::-1].argmax(0), -1)) * res
    bA, bB = bot(MA), bot(MB)
    tA, tB = top(MA), top(MB)

    blade = (xs > 3.0) & (xs < 27.2)

    def cuts(botline, baseline, name):
        y = botline.copy()
        # глубина реза от базовой линии (спинки)
        d = baseline - y
        d_s = savgol_filter(np.nan_to_num(d, nan=0.0), 21, 2)
        pk, props = find_peaks(d_s, prominence=0.4, distance=10)
        pk = [p for p in pk if blade[p]]
        print(f"\n{name}: зубцов найдено {len(pk)}")
        rows = []
        for k, p in enumerate(pk):
            rows.append((xs[p], d_s[p], props["prominences"][k]))
        for i, (x, dep, prom) in enumerate(rows, 1):
            print(f"   {i}: x = {x:5.2f} мм от кончика, глубина реза = {dep:4.2f} мм, "
                  f"выступ = {prom:4.2f}")
        if len(rows) > 1:
            xi = np.array([r[0] for r in rows])
            print("   шаг между зубцами:", np.round(np.diff(xi), 2), "мм")
        return rows

    baseA = np.nanmedian(tA[blade])
    baseB = np.nanmedian(tB[blade])
    ra = cuts(bA, baseA, "kwikset (база — спинка %.2f мм)" % baseA)
    rb = cuts(bB, baseB, "малый зал (база — спинка %.2f мм)" % baseB)

    # сопоставление зубцов по позиции (в системе упора головки, сдвиг hall = -0.94 мм)
    print("\nСопоставление зубцов (позиции приведены к системе kwikset, мм):")
    print(f"{'kwikset x':>10} {'глубина':>8} | {'малый зал x':>12} {'глубина':>8} | "
          f"{'Δx':>6} {'Δглуб':>6}")
    for x, dep, _p in ra:
        cand = [(abs((x - 0.94) - xb), xb, db) for xb, db, _ in rb]
        if cand:
            _, xb, db = min(cand)
            print(f"{x:10.2f} {dep:8.2f} | {(xb+0.94):12.2f} {db:8.2f} | "
                  f"{(x - xb - 0.94):6.2f} {(dep - db):6.2f}")

    # метрики KW-подобного ключа: ширина полотна, глубина реза min/max
    for nm, b, t in (("kwikset", bA, tA), ("малый зал", bB, tB)):
        m = blade
        h = t[m] - b[m]
        print(f"\n{nm}: полотно — высота {np.nanmedian(h):.2f} мм "
              f"(min {np.nanmin(h):.2f}, max {np.nanmax(h):.2f})")

    # картинка
    fig, axs = plt.subplots(2, 1, figsize=(15, 8), dpi=110, sharex=True)
    for ax, (nm, b, t, base, col, rows) in zip(axs, (
            ("kwikset", bA, tA, baseA, "C0", ra),
            ("малый зал", bB, tB, baseB, "C3", rb))):
        ax.plot(xs[blade], base - b[blade], col, lw=1.5)
        for x, dep, prom in rows:
            ax.plot(x, dep, "v", color="k", ms=6)
            ax.annotate(f"{x:.2f}", (x, dep), textcoords="offset points", xytext=(0, -14),
                        ha="center", fontsize=8)
        ax.set_title(f"{nm}: глубина реза от спинки (▼ — найденные зубцы)")
        ax.set_ylabel("глубина, мм")
        ax.grid(alpha=.3)
    axs[1].set_xlabel("x от кончика жала, мм")
    plt.tight_layout()
    plt.savefig("renders/cuts.png")
    print("\nsaved renders/cuts.png")


if __name__ == "__main__":
    main()
