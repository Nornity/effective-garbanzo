"""Шаблон 1:1 (PDF A4) с контурами обоих ключей, сечениями и таблицей контрольных размеров."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy import ndimage

MM = 1 / 25.4


def raster(P, x0, x1, y0, y1, res, res_y=None):
    nx = int(round((x1 - x0) / res)) + 1
    ny = int(round((y1 - y0) / res)) + 1
    M = np.zeros((ny, nx), bool)
    ix = np.clip(((P[:, 0] - x0) / res).round().astype(int), 0, nx - 1)
    iy = np.clip(((P[:, 1] - y0) / res).round().astype(int), 0, ny - 1)
    M[iy, ix] = True
    return M


def poly_from_mask(mask, x0, y0, res, max_height_mm=None):
    """Полигон силуэта по верхней/нижней границе (маска должна быть заполнена).
    max_height_mm — отбросить колонки-«иглы» выше указанной высоты (артефакты меша)."""
    has = mask.any(0)
    if max_height_mm is not None:
        h = mask.sum(0) * res
        has = has & (h < max_height_mm)
    xs = np.where(has)[0]
    if len(xs) == 0:
        return None
    top = mask.shape[0] - 1 - mask[:, xs][::-1].argmax(0)
    bot = mask[:, xs].argmax(0)
    gx = x0 + xs * res
    top_xy = np.c_[gx, y0 + top * res]
    bot_xy = np.c_[gx[::-1], y0 + bot[::-1] * res]
    return np.vstack([top_xy, bot_xy])


def section_outline(P, xc, tol=0.06, res=0.02):
    """Сечение полотна в плоскости y-z -> полигон."""
    q = P[np.abs(P[:, 0] - xc) < tol]
    if len(q) < 10:
        return None
    y, z = q[:, 1], q[:, 2]
    ny = int((y.max() - y.min()) / res) + 3
    nz = int((z.max() - z.min()) / res) + 3
    M = np.zeros((nz, ny), bool)
    iy = np.clip(((y - y.min()) / res).round().astype(int), 0, ny - 1)
    iz = np.clip(((z - z.min()) / res).round().astype(int), 0, nz - 1)
    M[iz, iy] = True
    M = ndimage.binary_fill_holes(ndimage.binary_closing(M, np.ones((5, 5))))
    poly = poly_from_mask(M, y.min(), z.min(), res)
    return poly


def main():
    MA = np.load("/tmp/MAf.npy")
    MB = np.load("/tmp/MBs.npy")
    x0, x1, y0, y1, res = np.load("/tmp/geom.npy")
    A = np.load("/tmp/kwikset_canon.npy")
    B = np.load("/tmp/hall_canon.npy")

    # отсечь артефакты сэмплирования от «мусорной» вершины ASCII-файла
    Ma = A[(A[:, 1] < 11.0) & (A[:, 0] < 51.6)]
    MA = ndimage.binary_fill_holes(
        ndimage.binary_closing(raster(Ma, x0, x1, y0, y1, res), np.ones((3, 3))))

    shift_px = int(round(0.94 / res))
    MBs = np.roll(MB, -shift_px, axis=1)
    MBs_y0 = y0  # та же вертикаль (разница спинок 0.15 мм — не важна для шаблона)

    pa = poly_from_mask(MA, x0, y0, res, max_height_mm=23.0)
    pb = poly_from_mask(MBs, x0 + shift_px * res, MBs_y0, res)

    with PdfPages("renders/key_template_1to1.pdf") as pdf:
        fig = plt.figure(figsize=(210 * MM, 297 * MM))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, 210)
        ax.set_ylim(0, 297)
        ax.set_aspect("equal")
        ax.axis("off")

        # калибровочная линейка
        for k in range(0, 101):
            h = 4 if k % 10 == 0 else (2.4 if k % 5 == 0 else 1.4)
            ax.plot([15 + k, 15 + k], [283, 283 + h], "k", lw=.6 if k % 10 else .9)
        ax.text(15, 289, "калибровка: 0–100 мм; проверьте линейкой перед использованием",
                fontsize=6, va="bottom")

        ax.text(15, 271, "Шаблон 1:1 — приложите ключ и сравните контур",
                fontsize=9.5, va="bottom", weight="bold")
        ax.text(15, 265.5,
                "Верхний контур — kwikset-41332: полотно 28,4 мм, 4 зубца   |   "
                "нижний — «малый зал»: полотно 27,4 мм, 3 зубца",
                fontsize=6.5, va="bottom")

        # ключи, выровненные по упору головки
        for poly, dy, color, label in ((pa, 232, "#1f4e9c", "kwikset-41332"),
                                       (pb, 196, "#b03030", "«малый зал»")):
            p = poly.copy()
            p[:, 0] -= p[:, 0].min() - 15
            p[:, 1] += dy
            ax.add_patch(plt.Polygon(p, closed=True, fill=False, edgecolor=color, lw=.8))
            ax.text(15, dy - 13, label, fontsize=6.5, color=color, va="top")
            # линия упора головки
            sh = p[:, 0].min() + 28.36 if color == "#1f4e9c" else p[:, 0].min() + 27.42
            ax.plot([sh, sh], [dy - 12, dy + 13], color=color, lw=.4, ls=(0, (4, 2)))
            ax.text(sh + .5, dy + 13.5, "упор", fontsize=4.8, color=color)

        ax.annotate("", xy=(15, 252), xytext=(15 + 28.36, 252),
                    arrowprops=dict(arrowstyle="<->", lw=.5, color="#1f4e9c"))
        ax.text(15 + 14.2, 253, "28,4 мм", fontsize=5.5, color="#1f4e9c", ha="center")
        ax.annotate("", xy=(15, 164), xytext=(15 + 27.42, 164),
                    arrowprops=dict(arrowstyle="<->", lw=.5, color="#b03030"))
        ax.text(15 + 13.7, 165, "27,4 мм", fontsize=5.5, color="#b03030", ha="center")

        # сечения полотна
        ax.text(15, 150, "Поперечное сечение полотна на x = 14 мм от кончика, 1:1 "
                         "(ширина × толщина)", fontsize=7, va="top", weight="bold")
        for P, ox, oy, color, label in ((A, 15, 120, "#1f4e9c", "kwikset"),
                                        (B, 50, 120, "#b03030", "«малый зал»")):
            poly = section_outline(P, 14)
            if poly is None:
                continue
            p = poly.copy()
            p[:, 0] -= p[:, 0].min()
            p[:, 0] += ox
            p[:, 1] += oy
            ax.add_patch(plt.Polygon(p, closed=True, fill=True, facecolor=color, alpha=.25,
                                     edgecolor=color, lw=.6))
            ax.text(ox, oy + 12.5, f"{label}: 10,2 × 2,0 мм", fontsize=5.8, color=color)

        # таблица контрольных размеров
        ax.text(15, 100, "Контрольные размеры (штангенциркуль/линейка):", fontsize=7.5,
                va="top", weight="bold")
        rows = [
            ("полотно: от кончика жала до упора головки", "28,4 мм", "27,4 мм"),
            ("высота полотна (спинка → бородка), максимум", "9,05 мм", "7,90 мм"),
            ("высота полотна, типичная", "7,6–7,7 мм", "6,9 мм"),
            ("толщина полотна", "2,0 мм", "2,0 мм"),
            ("ширина полотна (по сечению)", "10,2 мм", "10,2 мм"),
            ("число зубцов на кромке", "4", "3"),
            ("расстояние кончик → 1-й зубец", "5,2 мм", "5,8 мм"),
            ("глубина реза в 1-м зубце (от спинки)", "8,7 мм", "8,2 мм"),
            ("головка: высота / ширина", "21,4 / 25,9 мм", "21,4 / 23,7 мм"),
        ]
        y = 95
        ax.text(15, y, "параметр", fontsize=6, weight="bold")
        ax.text(105, y, "kwikset", fontsize=6, weight="bold", color="#1f4e9c")
        ax.text(135, y, "малый зал", fontsize=6, weight="bold", color="#b03030")
        y -= 3.6
        for r in rows:
            ax.text(15, y, r[0], fontsize=5.8)
            ax.text(105, y, r[1], fontsize=5.8, color="#1f4e9c")
            ax.text(135, y, r[2], fontsize=5.8, color="#b03030")
            y -= 3.6

        ax.text(15, y - 3,
                "Как пользоваться: 1) печать без масштабирования (100 %); 2) проверьте калибровочный отрезок — "
                "ровно 100 мм;\n3) приложите ключ полотном к верхнему, затем к нижнему контуру; 4) сравните: длину "
                "полотна, число зубцов,\nвысоту полотна и толщину; 5) если совпал один из контуров — печатайте этот "
                "файл (для kwikset см. п. 5 отчёта).",
                fontsize=6, va="top")
        pdf.savefig(fig)
        fig.savefig("renders/key_template_1to1.png", dpi=250)
        plt.close(fig)
    print("saved renders/key_template_1to1.pdf, .png")


if __name__ == "__main__":
    main()
