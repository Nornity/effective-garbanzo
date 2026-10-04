"""Сравнение силуэтов (вид сбоку) по маскам: IoU, разностная карта, списки зубцов."""
from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RES = 0.05  # мм/пиксель


def rasterize(P, x0, x1, y0, y1):
    nx = int(round((x1 - x0) / RES)) + 1
    ny = int(round((y1 - y0) / RES)) + 1
    M = np.zeros((ny, nx), dtype=bool)
    ix = np.clip(((P[:, 0] - x0) / RES).round().astype(int), 0, nx - 1)
    iy = np.clip(((P[:, 1] - y0) / RES).round().astype(int), 0, ny - 1)
    M[iy, ix] = True
    # морфологическое закрытие: заливка дырок в маске (3x3)
    M = close(M)
    return M


def close(M, it=1):
    for _ in range(it):
        M = M | np.roll(M, 1, 0) | np.roll(M, -1, 0) | np.roll(M, 1, 1) | np.roll(M, -1, 1)
    for _ in range(it):
        M = M & np.roll(M, 1, 0) & np.roll(M, -1, 0) & np.roll(M, 1, 1) & np.roll(M, -1, 1)
    return M


def fill_holes(M):
    """Заливка замкнутых пустот (например отверстия в головке) — через flood fill снаружи."""
    ny, nx = M.shape
    free = ~M
    seen = np.zeros_like(free)
    stack = [(0, y) for y in range(ny)] + [(nx - 1, y) for y in range(ny)] + \
            [(x, 0) for x in range(nx)] + [(x, ny - 1) for x in range(nx)]
    from collections import deque
    dq = deque()
    for x, y in stack:
        if free[y, x] and not seen[y, x]:
            seen[y, x] = True
            dq.append((x, y))
    while dq:
        x, y = dq.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a, b = x + dx, y + dy
            if 0 <= a < nx and 0 <= b < ny and free[b, a] and not seen[b, a]:
                seen[b, a] = True
                dq.append((a, b))
    holes = free & ~seen
    return M | holes


def main():
    A = np.load("/tmp/kwikset_canon.npy")
    B = np.load("/tmp/hall_canon.npy")
    x0 = min(A[:, 0].min(), B[:, 0].min()) - 1
    x1 = max(A[:, 0].max(), B[:, 0].max()) + 1
    y0 = min(A[:, 1].min(), B[:, 1].min()) - 1
    y1 = max(A[:, 1].max(), B[:, 1].max()) + 1

    MA = rasterize(A, x0, x1, y0, y1)
    MB = rasterize(B, x0, x1, y0, y1)
    MAf, MBf = fill_holes(MA), fill_holes(MB)

    inter = (MAf & MBf).sum()
    union = (MAf | MBf).sum()
    print(f"Силуэт (вид сбоку), маски {MAf.shape}, шаг {RES} мм")
    print(f"  площадь kwikset {MAf.sum()*RES*RES:.1f} мм^2, hall {MBf.sum()*RES*RES:.1f} мм^2")
    print(f"  IoU = {inter/union:.4f}; только kwikset {(MAf&~MBf).sum()*RES*RES:.1f} мм^2; "
          f"только hall {(MBf&~MAf).sum()*RES*RES:.1f} мм^2")

    # сдвиг для наилучшего совмещения (грубый поиск)
    best = (0, 0, inter / union)
    from itertools import product
    for dx, dy in product(range(-40, 41, 2), range(-40, 41, 2)):
        M2 = np.roll(np.roll(MBf, dy, 0), dx, 1)
        i = (MAf & M2).sum()
        u = (MAf | M2).sum()
        if i / u > best[2]:
            best = (dx, dy, i / u)
    dx, dy, iou = best
    print(f"  лучший сдвиг: dx={dx*RES:.2f} мм dy={dy*RES:.2f} мм -> IoU={iou:.4f}")
    MBs = np.roll(np.roll(MBf, dy, 0), dx, 1)

    # картинки
    fig, axs = plt.subplots(2, 2, figsize=(15, 9), dpi=100)
    ext = [x0, x1, y0, y1]

    axs[0, 0].imshow(MAf, origin="lower", extent=ext, cmap="Blues", alpha=.75)
    axs[0, 0].imshow(MBs, origin="lower", extent=ext, cmap="Reds", alpha=.45)
    axs[0, 0].set_title("Силуэты: kwikset (синий) vs малый зал (красный), совмещено")

    diff = np.zeros(MAf.shape + (3,))
    diff[..., 0] = (MBs & ~MAf)
    diff[..., 1] = (MAf & MBs) * 0.45 + 0.15
    diff[..., 2] = (MAf & ~MBs)
    axs[0, 1].imshow(diff, origin="lower", extent=ext)
    axs[0, 1].set_title("Разностная карта: синий — только kwikset, красный — только малый зал")
    axs[0, 1].legend()

    # профили высоты
    def prof_from_mask(M):
        xs = np.arange(M.shape[1])
        rows = np.where(M.any(0), M.argmax(0), -1)
        cols = np.where(M.any(0), M.shape[0] - 1 - M[::-1].argmax(0), -1)
        gx = x0 + xs * RES
        return gx, y0 + rows * RES, y0 + cols * RES

    gx, lo_a, hi_a = prof_from_mask(MAf)
    gx2, lo_b, hi_b = prof_from_mask(MBs)
    m = (hi_a > -100) & (hi_b > -100)
    axs[1, 0].plot(gx[m], hi_a[m], "C0", lw=1, label="kwikset верх")
    axs[1, 0].plot(gx[m], lo_a[m], "C0", lw=1, ls=":")
    axs[1, 0].plot(gx2[m], hi_b[m], "C3", lw=1, label="малый зал верх")
    axs[1, 0].plot(gx2[m], lo_b[m], "C3", lw=1, ls=":")
    axs[1, 0].set_title("Профили (после совмещения)")
    axs[1, 0].set_xlabel("x, мм")
    axs[1, 0].legend()
    axs[1, 0].grid(alpha=.3)

    # профиль только полотна (x<30)
    axs[1, 1].set_title("Разница профилей: (верх kwikset−верх hall) и (низ …)")
    d_top = hi_a[m] - hi_b[m]
    d_bot = lo_a[m] - lo_b[m]
    axs[1, 1].plot(gx[m], d_top, "C0", lw=1, label="Δ верх")
    axs[1, 1].plot(gx[m], d_bot, "C3", lw=1, label="Δ низ")
    axs[1, 1].axhline(0, color="k", lw=.5)
    axs[1, 1].set_xlabel("x, мм")
    axs[1, 1].grid(alpha=.3)
    axs[1, 1].legend()
    plt.tight_layout()
    plt.savefig("renders/mask_compare.png")
    print("saved renders/mask_compare.png")

    np.save("/tmp/MAf.npy", MAf)
    np.save("/tmp/MBs.npy", MBs)
    np.save("/tmp/geom.npy", np.array([x0, x1, y0, y1, RES]))


if __name__ == "__main__":
    main()
