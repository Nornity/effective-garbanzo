"""Головка (отверстия), поперечные сечения полотна, форма жала."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

RES = 0.02


def raster_raw(P, x0, x1, y0, y1, res=RES):
    nx = int(round((x1 - x0) / res)) + 1
    ny = int(round((y1 - y0) / res)) + 1
    M = np.zeros((ny, nx), bool)
    ix = np.clip(((P[:, 0] - x0) / res).round().astype(int), 0, nx - 1)
    iy = np.clip(((P[:, 1] - y0) / res).round().astype(int), 0, ny - 1)
    M[iy, ix] = True
    M = ndimage.binary_closing(M, np.ones((5, 5)))
    return M


def main():
    A = np.load("/tmp/kwikset_canon.npy")
    B = np.load("/tmp/hall_canon.npy")
    x0 = min(A[:, 0].min(), B[:, 0].min()) - 1
    x1 = max(A[:, 0].max(), B[:, 0].max()) + 1
    y0 = min(A[:, 1].min(), B[:, 1].min()) - 1
    y1 = max(A[:, 1].max(), B[:, 1].min()) + 1
    y1 = max(A[:, 1].max(), B[:, 1].max()) + 1

    RA = raster_raw(A, x0, x1, y0, y1)
    RB = raster_raw(B, x0, x1, y0, y1)

    def holes_of(M, nm):
        solid = ndimage.binary_fill_holes(M)
        holes = solid & ~M
        lab, n = ndimage.label(holes)
        objs = ndimage.find_objects(lab)
        sizes = ndimage.sum(holes, lab, range(1, n + 1)) * RES * RES
        res = []
        for i, (s, o) in enumerate(zip(sizes, objs)):
            if s < 0.5:
                continue
            ys, xs_ = o
            cx = x0 + (xs_.start + xs_.stop) / 2 * RES
            cy = y0 + (ys.start + ys.stop) / 2 * RES
            w = (xs_.stop - xs_.start) * RES
            h = (ys.stop - ys.start) * RES
            res.append((s, cx, cy, w, h))
        print(f"  {nm}: отверстий {len(res)}")
        for s, cx, cy, w, h in sorted(res, reverse=True):
            print(f"     S={s:6.2f} мм^2, центр ({cx:6.2f}, {cy:6.2f}), размеры {w:5.2f} x {h:5.2f}")
        return res

    print("Отверстия в головке (в канонич. координатах, x от кончика):")
    hA = holes_of(RA, "kwikset")
    hB = holes_of(RB, "малый зал")

    # сравнение: сопоставим отверстия по порядку и посчитаем разницу положений
    if len(hA) == len(hB):
        for (sa, cxa, cya, wa, ha_), (sb, cxb, cyb, wb, hb_) in zip(sorted(hA, reverse=True),
                                                                   sorted(hB, reverse=True)):
            print(f"     Δцентр ({cxa-cxb:+.2f}, {cya-cyb:+.2f}) мм, ΔS {sa-sb:+.2f} мм^2")

    # габариты головки (только x > упор - 2 мм)
    for nm, P in (("kwikset", A), ("малый зал", B)):
        sh = 28.33 if nm == "kwikset" else 27.39
        m = P[:, 0] > sh - 2
        Q = P[m]
        print(f"  {nm}: головка x {Q[:,0].min():.2f}..{Q[:,0].max():.2f} "
              f"(ширина {Q[:,0].max()-Q[:,0].min():.2f}), "
              f"y {Q[:,1].min():.2f}..{Q[:,1].max():.2f} (высота {Q[:,1].max()-Q[:,1].min():.2f})")

    # поперечные сечения полотна
    fig, axs = plt.subplots(2, 6, figsize=(18, 6), dpi=110)
    xs_slice = [2, 6, 10, 14, 18, 24]
    for k, xc in enumerate(xs_slice):
        for row, (nm, P) in enumerate((("kwikset", A), ("малый зал", B))):
            m = np.abs(P[:, 0] - xc) < 0.08
            Q = P[m]
            ax = axs[row, k]
            if len(Q):
                ax.scatter(Q[:, 1], Q[:, 2], s=1, c="C0" if row == 0 else "C3")
                ax.set_aspect("equal")
            ax.set_title(f"{nm}\nx={xc} мм", fontsize=8)
            ax.grid(alpha=.3)
            ax.tick_params(labelsize=6)
    plt.tight_layout()
    plt.savefig("renders/sections.png")
    print("saved renders/sections.png")

    # разница сечений: наложение
    fig, axs = plt.subplots(1, 6, figsize=(18, 3.4), dpi=110, sharey=True)
    for k, xc in enumerate(xs_slice):
        ax = axs[k]
        for nm, P, col in (("kwikset", A, "C0"), ("малый зал", B, "C3")):
            m = np.abs(P[:, 0] - xc) < 0.08
            Q = P[m]
            if len(Q):
                QC = Q.copy()
                QC[:, 1] -= 0.287  # компенсация систематического смещения по спине
                ax.scatter(QC[:, 1], QC[:, 2], s=1.2, c=col, alpha=.7)
        ax.set_title(f"x={xc}", fontsize=9)
        ax.set_aspect("equal")
        ax.grid(alpha=.3)
        ax.tick_params(labelsize=6)
    plt.tight_layout()
    plt.savefig("renders/sections_overlay.png")
    print("saved renders/sections_overlay.png")


if __name__ == "__main__":
    main()
