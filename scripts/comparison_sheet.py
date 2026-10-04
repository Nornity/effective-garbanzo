"""Итоговый сравнительный лист по двум STL."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RES = 0.05


def main():
    MA = np.load("/tmp/MAf.npy")
    MB = np.load("/tmp/MBs.npy")
    x0, x1, y0, y1, res = np.load("/tmp/geom.npy")
    xs = x0 + np.arange(MA.shape[1]) * res
    ext = [x0, x1, y0, y1]

    top = lambda M: np.where(M.any(0), M.shape[0] - 1 - M[::-1].argmax(0), np.nan)
    bot = lambda M: np.where(M.any(0), M.argmax(0), np.nan)
    tA, bA, tB, bB = (y0 + top(MA)*res, y0 + bot(MA)*res,
                      y0 + top(MB)*res, y0 + bot(MB)*res)

    fig = plt.figure(figsize=(18, 13), dpi=110)
    gs = fig.add_gridspec(3, 3, height_ratios=[1.15, 1.15, 1.0], hspace=.32, wspace=.22)

    # A: наложение силуэтов
    ax = fig.add_subplot(gs[0, :2])
    rgb = np.ones(MA.shape + (3,))
    both = MA & MB
    rgb[MB & ~MA] = [0.85, 0.2, 0.2]
    rgb[MA & ~MB] = [0.15, 0.35, 0.8]
    rgb[both] = [0.55, 0.6, 0.68]
    ax.imshow(rgb, origin="lower", extent=ext)
    ax.set_title("A. Силуэты (вид на плоскость полотна), совмещены по упору головки\n"
                 "серый — совпадает, синий — только kwikset, красный — только «малый зал»", fontsize=10)
    ax.set_xlabel("x, мм")
    ax.set_ylabel("y, мм")
    ax.grid(alpha=.15)

    # B: зум головки
    ax = fig.add_subplot(gs[0, 2])
    m = xs > 26
    ax.imshow(rgb[:, m], origin="lower", extent=[xs[m][0], xs[m][-1], y0, y1])
    ax.set_title("B. Головка: совпадение ~99%", fontsize=10)
    ax.grid(alpha=.15)
    ax.set_xlabel("x, мм")

    # C: профиль полотна
    ax = fig.add_subplot(gs[1, :2])
    m = (xs > 0.6) & (xs < 28)
    ax.plot(xs[m], tA[m], "C0", lw=1.1)
    ax.plot(xs[m], bA[m], "C0", lw=1.1)
    ax.plot(xs[m], tB[m], "C3", lw=1.1)
    ax.plot(xs[m], bB[m], "C3", lw=1.1)
    ax.plot([], [], "C0", label="kwikset (спинка/бородка)")
    ax.plot([], [], "C3", label="малый зал (спинка/бородка)")
    ax.fill_between(xs[m], bA[m], bB[m], color="grey", alpha=.3, label="разница бородки")
    ax.set_title("C. Профиль полотна после совмещения по упору головки", fontsize=10)
    ax.set_xlabel("x от кончика жала, мм")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=.3)

    # D: разница уровня реза
    ax = fig.add_subplot(gs[1, 2])
    d = bA[m] - bB[m]
    ax.plot(xs[m], d, "k", lw=1)
    ax.axhline(0, color="grey", lw=.7)
    ax.axhspan(-0.25, 0.25, color="green", alpha=.12)
    ax.set_title("D. Разница бородки (kwikset − «малый зал»), мм\n"
                 "зелёная зона = в пределах ±0.25 мм", fontsize=9)
    ax.set_xlabel("x от кончика, мм")
    ax.grid(alpha=.3)

    # E: поперечное сечение
    A = np.load("/tmp/kwikset_canon.npy")
    B = np.load("/tmp/hall_canon.npy")
    ax = fig.add_subplot(gs[2, 0])
    for P, col, nm in ((A, "C0", "kwikset"), (B, "C3", "малый зал")):
        q = P[np.abs(P[:, 0] - 14) < 0.06]
        ax.scatter(q[:, 1], q[:, 2], s=1.5, c=col, alpha=.8, label=nm)
    ax.set_aspect("equal")
    ax.set_title("E. Сечение полотна, x = 14 мм", fontsize=10)
    ax.set_xlabel("y, мм")
    ax.set_ylabel("толщина z, мм")
    ax.legend(fontsize=8)
    ax.grid(alpha=.3)

    # F: числовая сводка
    ax = fig.add_subplot(gs[2, 1:])
    ax.axis("off")
    med = np.nanmedian(bA[m] - bB[m])
    frac25 = 100 * np.nanmean(np.abs(d) > 0.25)
    frac50 = 100 * np.nanmean(np.abs(d) > 0.5)
    txt = (
        "F. Сводка\n\n"
        "Формат:            kwikset — ASCII STL      | «малый зал» — binary STL (STLB ATF 15.8.0.0)\n"
        "Размер файла:      1 004 489 Б (6,3×)       | 158 584 Б\n"
        "Граней:            3 111                    | 3 170\n"
        "Уникальных вершин: 1 504                    | 1 583\n"
        "Габарит Д×В×Т:     52,34 × 21,4 × 2,0 мм   | 51,07 × 21,4 × 2,0 мм\n"
        "Полотно (кончик→упор): 28,4 мм             | 27,4 мм  (Δ 0,94 мм)\n"
        "Толщина полотна:   2,0 мм                  | 2,0 мм  (совпадает)\n\n"
        "Совпадение головки (IoU):   98,8 %   — одинаковая заготовка\n"
        "Совпадение полотна (IoU):   86,3 %   — разная нарезка\n"
        "Общее совпадение (IoU):     94,8 %\n\n"
        f"Разница уровня реза: медиана {med:+.2f} мм; >0,25 мм на {frac25:.0f} % длины, "
        f">0,5 мм на {frac50:.0f} % длины полотна\n\n"
        "Качество сетки:    101 вырожденная грань, 269 нулевых нормалей, 52 не-манифолд ребра,\n"
        "                   «мусорная» вершина в (52,31; 15,02; 1,11)  |  чистая: 0 / 0 / 0"
    )
    ax.text(0, 1, txt, va="top", ha="left", fontsize=10, family="DejaVu Sans Mono")

    plt.savefig("renders/comparison_sheet.png", bbox_inches="tight")
    print("saved renders/comparison_sheet.png")


if __name__ == "__main__":
    main()
