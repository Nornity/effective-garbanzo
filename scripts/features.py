"""Структурное сравнение ключей: привязка к упору головки, пазы, головка, поперечные сечения."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

RES = 0.02  # мм/пиксель


def raster(P, x0, x1, y0, y1, res=RES):
    nx = int(round((x1 - x0) / res)) + 1
    ny = int(round((y1 - y0) / res)) + 1
    M = np.zeros((ny, nx), bool)
    ix = np.clip(((P[:, 0] - x0) / res).round().astype(int), 0, nx - 1)
    iy = np.clip(((P[:, 1] - y0) / res).round().astype(int), 0, ny - 1)
    M[iy, ix] = True
    M = ndimage.binary_closing(M, np.ones((3, 3)))
    M = ndimage.binary_fill_holes(M)   # силуэт вместе с отверстиями
    return M


def boundaries(M, x0, y0, res=RES):
    has = M.any(0)
    ys = np.arange(M.shape[0])
    top = np.where(has, M.shape[0] - 1 - M[::-1].argmax(0), np.nan)
    bot = np.where(has, M.argmax(0), np.nan)
    gx = x0 + np.arange(M.shape[1]) * res
    return gx, y0 + top * res, y0 + bot * res, has


def main():
    A = np.load("/tmp/kwikset_canon.npy")
    B = np.load("/tmp/hall_canon.npy")
    x0 = min(A[:, 0].min(), B[:, 0].min()) - 1
    x1 = max(A[:, 0].max(), B[:, 0].max()) + 1
    y0 = min(A[:, 1].min(), B[:, 1].min()) - 1
    y1 = max(A[:, 1].max(), B[:, 1].max()) + 1

    MA = raster(A, x0, x1, y0, y1)
    MB = raster(B, x0, x1, y0, y1)
    pass  # отверстия залиты на этапе raster

    gxA, topA, botA, hasA = boundaries(MA, x0, y0)
    gxB, topB, botB, hasB = boundaries(MB, x0, y0)

    # упор головки: первая колонка, где высота > 12 мм
    def shoulder(gx, top, bot):
        h = top - bot
        idx = np.where(np.nan_to_num(h) > 12)[0]
        return gx[idx[0]] if len(idx) else np.nan

    shA, shB = shoulder(gxA, topA, botA), shoulder(gxB, topB, botB)
    tipA = gxA[hasA][0]
    tipB = gxB[hasB][0]
    print(f"Упор головки: kwikset x={shA:.2f}, малый зал x={shB:.2f}")
    print(f"Кончик: kwikset x={tipA:.2f}, малый зал x={tipB:.2f}")
    print(f"  => длина от кончика до упора: kwikset {shA-tipA:.2f} мм, "
          f"малый зал {shB-tipB:.2f} мм")

    # смещение B относительно A по упору
    dx = shB - shA
    print(f"\nСдвиг B относительно A по упору: {dx:+.2f} мм")
    gxB_al = gxB - dx

    # профиль бородки в системе координат упора
    def interp(profile_axis, g, x):
        return np.interp(x, profile_axis, g, left=np.nan, right=np.nan)

    # общий диапазон по x (от кончика A)
    xs = np.arange(max(tipA, tipB - dx) + 0.2, min(shA, shB - dx), 0.02)
    topA_i = interp(gxA, topA, xs)
    botA_i = interp(gxA, botA, xs)
    topB_i = interp(gxB_al, topB, xs)
    botB_i = interp(gxB_al, botB, xs)
    print(f"Общих сечений полотна: {len(xs)}  (x={xs[0]:.1f}..{xs[-1]:.1f} от кончика), "
          f"суммарная длина {xs[-1]-xs[0]:.1f} мм")

    mA = np.isfinite(topA_i) & np.isfinite(botA_i) & np.isfinite(topB_i) & np.isfinite(botB_i)
    # вычтем систематический сдвиг по спине (средняя разница спинки)
    d_spine = np.nanmean(topA_i[mA] - topB_i[mA])
    print(f"Систематическое смещение по спине (A-B): {d_spine:+.3f} мм -> компенсируем")
    topB_i = topB_i + d_spine
    botB_i = botB_i + d_spine

    hA, hB = topA_i - botA_i, topB_i - botB_i
    depthA, depthB = topA_i - botA_i, topB_i - botB_i  # глубина реза от спинки
    d_depth = depthA - depthB

    print(f"\nВысота полотна (спинка−бородка): A медиана {np.nanmedian(hA):.2f}, "
          f"B медиана {np.nanmedian(hB):.2f} мм")
    print(f"Разница глубины реза (A−B): средняя {np.nanmean(d_depth):+.3f} мм, "
          f"медиана {np.nanmedian(d_depth):+.3f}, 90-й перцентиль по модулю "
          f"{np.nanpercentile(np.abs(d_depth),90):.3f}, макс {np.nanmax(np.abs(d_depth)):.3f} мм")
    frac = np.nanmean(np.abs(d_depth) > 0.25)
    print(f"Доля длины полотна, где глубина реза различается >0.25 мм: {100*frac:.1f}%")
    frac5 = np.nanmean(np.abs(d_depth) > 0.5)
    print(f"  ... >0.5 мм: {100*frac5:.1f}%;  >1 мм: {100*np.nanmean(np.abs(d_depth)>1):.1f}%")

    # ---- уровни реза (плато) и позиции уступов
    def plateaus(x, y, min_len=0.4):
        """Разбиение профиля на плато: (x_start, x_end, y_mean) для почти горизонтальных участков."""
        seg = []
        s = 0
        for i in range(1, len(y)):
            if not np.isfinite(y[i]) or abs(y[i] - y[s]) > 0.15:
                if x[i - 1] - x[s] >= min_len:
                    seg.append((x[s], x[i - 1], np.nanmean(y[s:i]), x[i - 1] - x[s]))
                s = i
        return seg

    print("\nПолка (уровни реза) бородки, от спинки, мм:")
    for nm, x_, y_ in (("kwikset", xs, depthA), ("малый зал", xs, depthB)):
        pl = plateaus(x_, y_)
        pl = [p for p in pl if p[3] > 0.5]
        print(f"  {nm}: {len(pl)} площадок")
        for p in pl:
            print(f"     x {p[0]:6.2f}..{p[1]:6.2f} (len {p[3]:4.2f})  глубина {p[2]:5.2f} мм")

    # ---- головка
    print("\nГоловка:")
    for nm, gx, has, M in (("kwikset", gxA, hasA, MA), ("малый зал", gxB, hasB, MB)):
        xh = gx[has][0]
        head = gx > (shA if nm == "kwikset" else shB) - 1
        xs_h = gx[head]
        hs = (topA if nm == "kwikset" else topB)[head]
        hs2 = (botA if nm == "kwikset" else botB)[head]
        m = np.isfinite(hs) & np.isfinite(hs2)
        hh = np.nanmax(hs[m]) - np.nanmin(hs2[m])
        # отверстия
        holes = ndimage.binary_fill_holes(M) & ~M
        lab, n = ndimage.label(holes)
        sizes = ndimage.sum(holes, lab, range(1, n + 1)) * RES * RES
        coms = ndimage.center_of_mass(holes, lab, range(1, n + 1))
        big = [(s, c) for s, c in zip(sizes, coms) if s > 1.0]
        print(f"  {nm}: ширина головки {xs_h.max()-xs_h.min():.2f} мм, высота {hh:.2f} мм, "
              f"отверстий >1мм^2: {len(big)}")
        for s, c in sorted(big, reverse=True):
            cx = x0 + c[1] * RES
            cy = y0 + c[0] * RES
            print(f"     отверстие S={s:6.2f} мм^2 центр ({cx:6.2f}, {cy:6.2f}) "
                  f"относительно упора ({cx-(shA if nm=='kwikset' else shB):+6.2f})")

    # ---- картинка
    fig = plt.figure(figsize=(17, 12), dpi=100)
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1.4, 1.2])
    ax = fig.add_subplot(gs[0, :])
    ax.plot(xs, topA_i, "C0", lw=1.2, label="kwikset спинка")
    ax.plot(xs, botA_i, "C0", lw=1.2, ls="--", label="kwikset бородка")
    ax.plot(xs, topB_i, "C3", lw=1.2, label="малый зал спинка")
    ax.plot(xs, botB_i, "C3", lw=1.2, ls="--", label="малый зал бородка")
    ax.set_title("Профили, выровненные по упору головки (вертикальное смещение скомпенсировано)")
    ax.set_xlabel("x от кончика жала, мм")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=.3)

    ax = fig.add_subplot(gs[1, :])
    ax.plot(xs, depthA, "C0", lw=1.6, label="kwikset")
    ax.plot(xs, depthB, "C3", lw=1.6, label="малый зал")
    ax.fill_between(xs, depthA, depthB, where=np.isfinite(d_depth),
                    color="grey", alpha=.25, label="разница")
    ax.set_ylabel("глубина реза от спинки, мм")
    ax.set_xlabel("x от кончика жала, мм")
    ax.set_title("Профиль нарезки: сравнение по уровням")
    ax.legend(fontsize=8)
    ax.grid(alpha=.3)

    ax = fig.add_subplot(gs[2, :])
    ax.plot(xs, d_depth, "k", lw=1)
    ax.axhline(0, color="grey", lw=.7)
    for lv in (0.25, 0.5, 1.0):
        ax.axhline(lv, color="r", lw=.5, ls=":")
        ax.axhline(-lv, color="r", lw=.5, ls=":")
    ax.set_ylabel("Δ глубины (kwikset − малый зал), мм")
    ax.set_xlabel("x от кончика жала, мм")
    ax.set_title("Разница профилей нарезки")
    ax.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig("renders/features.png")
    print("\nsaved renders/features.png")

    np.save("/tmp/xs.npy", xs)
    np.save("/tmp/prof.npy", np.array([topA_i, botA_i, topB_i, botB_i]))
    for nm, M in (("A", MA), ("B", MB)):
        np.save(f"/tmp/mask_{nm}.npy", M)
    np.save("/tmp/geom2.npy", np.array([x0, x1, y0, y1, RES, shA, shB, tipA, tipB]))


if __name__ == "__main__":
    main()
