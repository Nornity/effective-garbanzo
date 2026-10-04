"""Минимальные утилиты для чтения/анализа STL (ASCII и binary) без внешних зависимостей
кроме numpy."""
from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------- загрузка ---

def is_binary_stl(path: str) -> bool:
    with open(path, "rb") as f:
        head = f.read(84)
        if len(head) < 84:
            return False
        n = struct.unpack("<I", head[80:84])[0]
        import os
        size = os.path.getsize(path)
        if 84 + n * 50 == size:
            return True
        if head[:5].lower() == b"solid" and b"facet" in head.lower():
            return False
        # эвристика: если в начале есть "solid", но размеры не сходятся -> считаем ASCII
        return False


def load_binary_stl(path: str):
    with open(path, "rb") as f:
        data = f.read()
    header = data[:80]
    n = struct.unpack("<I", data[80:84])[0]
    rec = np.frombuffer(
        data[84:84 + n * 50],
        dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]),
    )
    verts = rec["v"].astype(np.float64)
    normals = rec["n"].astype(np.float64)
    attrs = rec["a"].copy()
    return verts, normals, attrs, header


def load_ascii_stl(path: str, encoding="utf-8"):
    with open(path, "rb") as f:
        raw = f.read()
    try:
        txt = raw.decode(encoding)
    except UnicodeDecodeError:
        txt = raw.decode("latin-1")
    header = txt.splitlines()[0] if txt else ""

    # быстрый векторный разбор: все строки vertex
    v_lines = re.findall(
        r"vertex\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)", txt
    )
    verts = np.array(v_lines, dtype=np.float64).reshape(-1, 3, 3)
    if len(verts) == 0:  # fallback
        verts = np.zeros((0, 3, 3))
    n_lines = re.findall(
        r"facet\s+normal\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)", txt
    )
    normals = np.array(n_lines, dtype=np.float64).reshape(-1, 3) if n_lines else np.zeros((len(verts), 3))
    return verts, normals, header, txt


# --------------------------------------------------------------- метрики ---

def tri_areas(verts: np.ndarray) -> np.ndarray:
    a = verts[:, 1] - verts[:, 0]
    b = verts[:, 2] - verts[:, 0]
    return 0.5 * np.linalg.norm(np.cross(a, b), axis=1)


def tri_normals(verts: np.ndarray) -> np.ndarray:
    a = verts[:, 1] - verts[:, 0]
    b = verts[:, 2] - verts[:, 0]
    n = np.cross(a, b)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    ln[ln == 0] = 1.0
    return n / ln


def signed_volume(verts: np.ndarray) -> float:
    """Объём по теореме о дивергенции (сумма тетраэдров от начала координат)."""
    v0, v1, v2 = verts[:, 0], verts[:, 1], verts[:, 2]
    return float(np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum() / 6.0)


def dedup_vertices(verts: np.ndarray, tol_abs: float):
    """Склеивает вершины, совпадающие с точностью tol_abs. Возвращает (uniq, index)."""
    flat = verts.reshape(-1, 3)
    q = np.round(flat / tol_abs).astype(np.int64)
    uniq, inverse = np.unique(q, axis=0, return_inverse=True)
    return uniq * tol_abs, inverse.reshape(verts.shape[0], 3)


class _DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        p = self.p
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


@dataclass
class StlStats:
    path: str
    fmt: str
    size_bytes: int
    header: str
    n_facets: int
    n_vertices_raw: int
    n_vertices_uniq: int
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    dims: np.ndarray
    area: float
    volume_signed: float
    degenerate: int
    components: int
    boundary_edges: int
    nonmanifold_edges: int
    flipped_normals: int
    zero_normals: int
    euler_char: int
    genus: float
    shell_volumes: list = field(default_factory=list)
    colors: Optional[np.ndarray] = None

    def as_dict(self):
        d = dict(self.__dict__)
        for k in ("bbox_min", "bbox_max", "dims"):
            d[k] = [round(float(x), 4) for x in d[k]]
        for k in ("area", "volume_signed"):
            d[k] = round(float(d[k]), 4)
        d.pop("colors", None)
        return d


def analyze(
    path: str,
    name: str = "",
    tol_abs: float = 1e-4,
    max_open_edges_list: float = 5e6,
) -> StlStats:
    import os

    size = os.path.getsize(path)
    binary = is_binary_stl(path)
    attrs = None
    if binary:
        verts, normals, attrs, raw_header = load_binary_stl(path)
        header = raw_header.decode("latin-1").rstrip("\x00 ").strip()
        fmt = "binary"
    else:
        verts, normals, header, _ = load_ascii_stl(path)
        fmt = "ascii"

    n = len(verts)
    areas = tri_areas(verts)
    flat = verts.reshape(-1, 3)
    bmin, bmax = flat.min(0), flat.max(0)
    dims = bmax - bmin

    scale = max(float(dims.max()), 1e-9)
    tol = max(tol_abs, scale * 1e-7) if tol_abs <= 0 else tol_abs

    uniq, idx = dedup_vertices(verts, tol)
    n_uniq = len(uniq)

    # топология: рёбра
    e = np.concatenate([idx[:, [0, 1]], idx[:, [1, 2]], idx[:, [2, 0]]], axis=0)
    e = np.sort(e, axis=1)
    uniq_e, counts = np.unique(e, axis=0, return_counts=True)
    boundary = int((counts == 1).sum())
    nonmanifold = int((counts > 2).sum())

    # компоненты связности по общим рёбрам
    n_e = len(uniq_e)
    if n_e < max_open_edges_list:
        dsu = _DSU(n_uniq)
        for a, b in uniq_e:
            dsu.union(int(a), int(b))
        roots = {}
        for i in range(n_uniq):
            roots.setdefault(dsu.find(i), 0)
        components = len(roots)
    else:
        components = -1

    # объёмы оболочек (по знаку тетраэдров от точки внутри bbox)
    center = (bmin + bmax) / 2.0
    v0, v1, v2 = verts[:, 0] - center, verts[:, 1] - center, verts[:, 2] - center
    tet = np.einsum("ij,ij->i", v0, np.cross(v1, v2)) / 6.0

    # ориентация нормалей: сравнение записанной нормали с вычисленной
    cn = tri_normals(verts)
    valid = areas > 0
    dot = np.einsum("ij,ij->i", cn, normals) if len(normals) == n else np.ones(n)
    flipped = int(((dot[valid] < -0.5)).sum())
    zero_n = int(((np.linalg.norm(normals, axis=1) < 1e-12)).sum()) if len(normals) == n else 0

    V = n_uniq - n_e + n  # V - E + F
    genus = (2 * components - V) / 2.0 if components > 0 else float("nan")

    colors = None
    if attrs is not None:
        colors = attrs.copy()

    return StlStats(
        path=path,
        fmt=fmt,
        size_bytes=size,
        header=header[:120],
        n_facets=n,
        n_vertices_raw=n * 3,
        n_vertices_uniq=n_uniq,
        bbox_min=bmin,
        bbox_max=bmax,
        dims=dims,
        area=float(areas.sum()),
        volume_signed=float(tet.sum()),
        degenerate=int((areas <= (scale ** 2) * 1e-12).sum()),
        components=components,
        boundary_edges=boundary,
        nonmanifold_edges=nonmanifold,
        flipped_normals=flipped,
        zero_normals=zero_n,
        euler_char=V,
        genus=genus,
        colors=colors,
    )
