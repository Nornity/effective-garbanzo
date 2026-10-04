"""Плотная дискретизация треугольников + извлечение силуэта и профиля нарезки."""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "scripts")
from stl_tools import is_binary_stl, load_ascii_stl, load_binary_stl
from compare import order_axes


def load(path):
    if is_binary_stl(path):
        v, n, a, h = load_binary_stl(path)
        return v
    v, n, h, t = load_ascii_stl(path)
    return v


def sample_surface(V: np.ndarray, step: float = 0.12) -> np.ndarray:
    """Равномерная выборка точек по поверхности (барицентрическая сетка)."""
    a = V[:, 1] - V[:, 0]
    b = V[:, 2] - V[:, 0]
    na = np.linalg.norm(np.cross(a, b), axis=1)
    out = []
    maxn = max(2, int(np.ceil(np.sqrt(na.max()) / step)) + 1)
    for k in range(maxn + 1):
        for m in range(maxn + 1 - k):
            u = k / maxn
            w = m / maxn
            pts = V[:, 0] + u * a + w * b
            out.append(pts)
    return np.concatenate(out, axis=0)


def resample_profile(cx, ymin, ymax, npts=1200):
    x0, x1 = cx.min(), cx.max()
    gx = np.linspace(x0, x1, npts)
    return gx, np.interp(gx, cx, ymin), np.interp(gx, cx, ymax)


def main():
    keys = {}
    for name, path in (("kwikset", "kwikset-41332-1791104343.stl"), ("hall", "малый зал.stl")):
        V = load(path)
        Vr, M, c = order_axes(V)
        Vr = Vr - Vr.reshape(-1, 3).mean(0)
        P = sample_surface(Vr, step=0.15)
        keys[name] = (Vr, P)
        print(name, "выборка точек:", len(P), "extents", np.round(P.max(0) - P.min(0), 3))
        np.save(f"/tmp/{name}_pts.npy", P)
        np.save(f"/tmp/{name}_Vr.npy", Vr)


if __name__ == "__main__":
    main()
