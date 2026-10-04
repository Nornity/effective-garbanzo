"""Анализ нарезки (бородки) ключей: положение и глубина зубцов."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RES = 0.05


def profiles(M, x0, y0):
    xs = np.arange(M.shape[1])
    has = M.any(0)
    top = np.where(has, M.shape[0] - 1 - M[::-1].argmax(0), np.nan)
    bot = np.where(has, M.argmax(0), np.nan)
    gx = x0 + xs * RES
    return gx, y0 + top * RES, y0 + bot * RES


def extrema(x, y, kind="min", order=4):
    """Простые локальные экстремумы с окном."""
    idx = []
    for i in range(order, len(y) - order):
        w = y[i - order:i + order + 1]
        if kind == "min" and y[i] == w.min() and np.isfinite(y[i]):
            idx.append(i)
        if kind == "max" and y[i] == w.max() and np.isfinite(y[i]):
            idx.append(i)
    # прореживание близких
    out = []
    for i in idx:
        if not out or i - out[-1] > 8:
            out.append(i)
    return out


def main():
    MAf = np.load("/tmp/MAf.npy")
    MBs = np.load("/tmp/MBs.npy")
    x0, x1, y0, y1, res = np.load("/tmp/geom.npy")

    gx, topA, botA = profiles(MAf, x0, y0)
    _, topB, botB = profiles(MBs, x0, y0)

    # полотно: до начала головки — берём x <= 28 мм
    blade = (gx >= 0.5) & (gx <= 28.0)
    hA, hB = topA - botA, topB - botB

    print("Спинка полотна (верх), медиана: kwikset %.2f мм, hall %.2f мм"
          % (np.nanmedian(topA[blade]), np.nanmedian(topB[blade])))
    print("Низ (бородка) медиана: kwikset %.2f мм, hall %.2f мм"
          % (np.nanmedian(botA[blade]), np.nanmedian(botB[blade])))

    # глубина пазов: считаем от спинки
    baseA = np.nanmedian(topA[blade])
    baseB = np.nanmedian(topB[blade])
    depthA = baseA - botA
    depthB = baseB - botB

    print("\nПазы (минимумы глубины реза) на полотне — положение от кончика, мм:")
    for name, dep in (("kwikset", depthA), ("малый зал", depthB)):
        idx = [i for i in extrema(gx, dep, "min") if blade[i]]
        print(f"  {name}:")
        for i in idx:
            print(f"     x={gx[i]:6.2f}  глубина реза={dep[i]:5.2f}")

    # корреляции профиля низа полотна
    m = blade & np.isfinite(botA) & np.isfinite(botB)
    a, b = botA[m], botB[m]
    print("\nКорреляция низа полотна (бородка):")
    print("  прямая            r = %.4f" % np.corrcoef(a, b)[0, 1])
    print("  hall зеркально    r = %.4f" % np.corrcoef(a, b[::-1])[0, 1])
    print("  средняя |Δ| = %.3f мм, макс |Δ| = %.3f мм"
          % (np.abs(a - b).mean(), np.abs(a - b).max()))

    # ступени реза
    print("\nУникальные уровни глубины реза (округление 0.25 мм):")
    for name, dep in (("kwikset", depthA), ("hall", depthB)):
        d = dep[blade]
        hist, edges = np.histogram(d, bins=np.arange(np.nanmin(d), np.nanmax(d) + 0.25, 0.25))
        top = sorted(zip(hist, edges[:-1]), reverse=True)[:6]
        print(f"  {name}: " + ", ".join(f"{e:.2f}мм: {h}" for h, e in top))

    # ---- картинка: полотно в деталях
    fig, axs = plt.subplots(3, 1, figsize=(15, 10), dpi=100, sharex=True)
    axs[0].plot(gx, topA, "C0", lw=1.2, label="kwikset спинка")
    axs[0].plot(gx, botA, "C0", lw=1.2, ls="--", label="kwikset бородка")
    axs[0].plot(gx, topB, "C3", lw=1.2, label="малый зал спинка")
    axs[0].plot(gx, botB, "C3", lw=1.2, ls="--", label="малый зал бородка")
    axs[0].set_title("Полотно: спинка и бородка (совмещённые координаты)")
    axs[0].legend(fontsize=8)
    axs[0].grid(alpha=.3)
    axs[0].set_xlim(-1, 33)

    axs[1].plot(gx, hA, "C0", lw=1.2, label="kwikset")
    axs[1].plot(gx, hB, "C3", lw=1.2, label="малый зал")
    axs[1].set_ylabel("высота полотна, мм")
    axs[1].set_title("Высота полотна (спинка − бородка)")
    axs[1].legend(fontsize=8)
    axs[1].grid(alpha=.3)

    axs[2].plot(gx, depthA, "C0", lw=1.4, label="kwikset")
    axs[2].plot(gx, depthB, "C3", lw=1.4, label="малый зал")
    axs[2].set_ylabel("глубина реза от спинки, мм")
    axs[2].set_xlabel("положение от кончика, мм")
    axs[2].set_title("Профиль нарезки (бородки)")
    axs[2].legend(fontsize=8)
    axs[2].grid(alpha=.3)
    plt.tight_layout()
    plt.savefig("renders/bitting.png")
    print("\nsaved renders/bitting.png")

    np.save("/tmp/bitting.npy", np.array([gx, topA, botA, topB, botB]))


if __name__ == "__main__":
    main()
