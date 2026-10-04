"""Проверка гипотез: одинаковый ли рез у ключей (корреляция профилей при сдвиге)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

xs = np.load("/tmp/xs.npy")
topA, botA, topB, botB = np.load("/tmp/prof.npy")

# профиль бородки относительно спинки (устойчив к вертикальному смещению)
pA = topA - botA
pB = topB - botB
step = xs[1] - xs[0]

print("Шаг дискретизации", round(step, 4), "мм, точек", len(xs))


def score(shift_mm):
    """Сдвигаем профиль B вдоль x и считаем остаточную разницу с A."""
    k = int(round(shift_mm / step))
    a = pA
    if k >= 0:
        b = np.concatenate([np.full(k, np.nan), pB[:len(pB) - k]]) if k < len(pB) else np.full_like(pB, np.nan)
        xb = xs - shift_mm
    else:
        b = np.concatenate([pB[-k:], np.full(-k, np.nan)])
        xb = xs - shift_mm
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 50:
        return np.nan, np.nan, m.sum()
    d = a[m] - b[m]
    return np.sqrt((d ** 2).mean()), np.abs(d).mean(), m.sum()


shifts = np.arange(-3.0, 3.001, 0.02)
rms = np.array([score(s)[0] for s in shifts])
best_i = np.nanargmin(rms)
print(f"Лучший сдвиг B относительно A: {shifts[best_i]:+.2f} мм, "
      f"RMS разницы высоты полотна = {rms[best_i]:.3f} мм")
rms0 = score(0.0)[0]
print(f"Без сдвига (по упору головки): RMS = {rms0:.3f} мм")
print(f"Медианная |разница| без сдвига: {score(0.0)[1]:.3f} мм; "
      f"при лучшем сдвиге: {score(shifts[best_i])[1]:.3f} мм")

# насколько сдвиг объясняет разницу: сравним с чисто случайным
print(f"\nУлучшение от подбора сдвига: {100*(1 - rms[best_i]/rms0):.1f}%")

# корреляция
mA = np.isfinite(pA)
peak = np.nanmax(pB)
plt.figure(figsize=(14, 5), dpi=110)
plt.plot(shifts, rms, "k")
plt.axvline(shifts[best_i], color="C3", ls="--", label=f"лучший сдвиг {shifts[best_i]:+.2f} мм")
plt.axvline(0, color="C0", ls=":", label="выравнивание по упору")
plt.xlabel("сдвиг профиля 'малый зал' вдоль полотна, мм")
plt.ylabel("RMS разницы высоты полотна, мм")
plt.title("Подбор сдвига нарезки: минимум RMS")
plt.grid(alpha=.3)
plt.legend()
plt.tight_layout()
plt.savefig("renders/crosscorr.png")

# Таблица разницы в контрольных точках (в системе упора)
print("\nВысота полотна (спинка−бородка) по секциям, мм:")
print(f"{'x, мм':>7} {'kwikset':>9} {'малый зал':>10} {'Δ':>7}")
for x in np.arange(2.0, 28.0, 1.0):
    i = np.argmin(np.abs(xs - x))
    if np.isfinite(pA[i]) and np.isfinite(pB[i]):
        print(f"{x:7.1f} {pA[i]:9.2f} {pB[i]:10.2f} {pA[i]-pB[i]:+7.2f}")
