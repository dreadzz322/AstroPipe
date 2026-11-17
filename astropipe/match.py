"""Сопоставление двух наборов точек с неизвестным сдвигом, поворотом, масштабом и отражением.

Метод треугольников: отношения сторон треугольника не меняются при подобии,
поэтому одинаковые тройки звёзд находятся в обоих списках независимо от ориентации.
Используется и для привязки к каталогу, и для совмещения кадров между собой.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree


@dataclass
class MatchResult:
    transform: np.ndarray  # 2x3: dst = A @ src + t
    src_idx: np.ndarray
    dst_idx: np.ndarray
    rms: float  # в единицах dst
    scale: float  # единиц dst на единицу src

    def apply(self, pts: np.ndarray) -> np.ndarray:
        return apply_transform(self.transform, pts)


def apply_transform(transform: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=float)
    return pts @ transform[:, :2].T + transform[:, 2]


def _triangles(pts: np.ndarray):
    """Все треугольники из точек: вершины в каноническом порядке, инварианты, длинная сторона."""
    idx = np.array(list(combinations(range(len(pts)), 3)), dtype=np.intp)
    if idx.size == 0:
        return idx.reshape(0, 3), np.zeros((0, 2)), np.zeros(0)
    p = pts[idx]
    sides = np.stack(  # сторона i лежит напротив вершины i
        [
            np.linalg.norm(p[:, 1] - p[:, 2], axis=1),
            np.linalg.norm(p[:, 0] - p[:, 2], axis=1),
            np.linalg.norm(p[:, 0] - p[:, 1], axis=1),
        ],
        axis=1,
    )
    order = np.argsort(sides, axis=1)
    s = np.take_along_axis(sides, order, axis=1)
    verts = np.take_along_axis(idx, order, axis=1)
    short, mid, long = s[:, 0], s[:, 1], s[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        # Отбрасываем вырожденные и почти равнобедренные: у них порядок вершин неустойчив.
        good = (short > 0) & (short / long > 0.25) & (long / mid > 1.03) & (mid / short > 1.03)
    inv = np.stack([mid[good] / long[good], short[good] / long[good]], axis=1)
    return verts[good], inv, long[good]


def _affine_exact(p: np.ndarray, q: np.ndarray) -> np.ndarray | None:
    m = np.column_stack([p, np.ones(3)])
    try:
        return np.linalg.solve(m, q).T
    except np.linalg.LinAlgError:
        return None


def _affine_lstsq(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    m = np.column_stack([p, np.ones(len(p))])
    return np.linalg.lstsq(m, q, rcond=None)[0].T


def _scale_excess(log_scale: np.ndarray, offset: float, width: float = 0.015):
    """Гистограмма масштабов: номер ячейки для каждой пары и превышение над локальным уровнем."""
    bins = np.floor(log_scale / width + offset).astype(np.int64)
    bins -= bins.min()
    counts = np.bincount(bins).astype(float)
    return bins, counts - ndimage.median_filter(counts, size=9, mode="reflect")


def _one_to_one(dist: np.ndarray, idx: np.ndarray, n_dst: int):
    """Из ответа KD-дерева оставляет взаимно-однозначные пары (ближайший выигрывает)."""
    ok = np.flatnonzero(np.isfinite(dist) & (idx < n_dst))
    ok = ok[np.argsort(dist[ok])]
    _, first = np.unique(idx[ok], return_index=True)
    src = np.sort(ok[first])
    return src, idx[src]


def match_points(
    src: np.ndarray,
    dst: np.ndarray,
    tol: float = 2.0,
    n_src: int = 25,
    n_dst: int = 40,
    inv_tol: float = 0.008,
    scale_range: tuple[float, float] | None = None,
    max_trials: int = 5000,
    min_inliers: int = 6,
    max_anisotropy: float = 1.15,
    n_verify: int = 150,
    per_bin: int = 400,
    dst_skip: int = 0,
) -> MatchResult | None:
    """Ищет преобразование src → dst. Оба списка должны идти от ярких к слабым.

    tol — допуск совпадения в единицах src (пикселях). Возвращает None, если надёжного
    решения нет.
    """
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    if len(src) < 3 or len(dst) < 3:
        return None
    tri_s, inv_s, len_s = _triangles(src[:n_src])
    # dst_skip: пропустить самые яркие точки dst при построении треугольников (на пластинках
    # ярчайшие звёзды — насыщенные пятна, которых нет среди найденных источников)
    tri_d, inv_d, len_d = _triangles(dst[dst_skip : dst_skip + n_dst])
    tri_d = tri_d + dst_skip
    if len(tri_s) == 0 or len(tri_d) == 0:
        return None

    pairs = cKDTree(inv_s).sparse_distance_matrix(cKDTree(inv_d), inv_tol, output_type="coo_matrix")
    a, b = pairs.row, pairs.col
    if len(a) == 0:
        return None
    log_scale = np.log(len_d[b] / len_s[a])
    if scale_range is not None:
        keep = (log_scale >= math.log(scale_range[0])) & (log_scale <= math.log(scale_range[1]))
        a, b, log_scale = a[keep], b[keep], log_scale[keep]
        if len(a) == 0:
            return None

    # Истинные совпадения дают один и тот же масштаб — сначала проверяем пик гистограммы.
    # Случайные пары размазаны по масштабам широким «холмом», истинные — узкий выброс над ним,
    # поэтому ранжируем по превышению над локальным уровнем, а не по самому числу пар.
    # Две гистограммы со сдвигом на полячейки — на случай, если выброс лёг на границу ячеек.
    bins1, excess1 = _scale_excess(log_scale, 0.0)
    bins2, excess2 = _scale_excess(log_scale, 0.5)
    second = excess2[bins2] > excess1[bins1]
    score = np.where(second, excess2[bins2], excess1[bins1])
    group = np.where(second, bins2 + len(excess1), bins1)
    # Из каждой ячейки берём ограниченную случайную выборку: если ячейка верная, этого хватит,
    # а если ложная — не тратим на неё все попытки.
    shuffled = np.random.default_rng(0).permutation(len(group))
    by_group = shuffled[np.argsort(group[shuffled], kind="stable")]
    _, first, sizes = np.unique(group[by_group], return_index=True, return_counts=True)
    rank = np.arange(len(group)) - np.repeat(first, sizes)
    sample = by_group[rank < per_bin]
    order = sample[np.argsort(-score[sample], kind="stable")][:max_trials]

    dst_tree = cKDTree(dst)
    verify = src[:n_verify]
    early = max(10, int(0.35 * len(verify)))
    best_n, best_t = 0, None
    for k in order:
        t = _affine_exact(src[tri_s[a[k]]], dst[tri_d[b[k]]])
        if t is None:
            continue
        sv = np.linalg.svd(t[:, :2], compute_uv=False)
        if sv[1] <= 0 or sv[0] / sv[1] > max_anisotropy:
            continue
        scale = math.sqrt(sv[0] * sv[1])
        d, _ = dst_tree.query(apply_transform(t, verify), distance_upper_bound=tol * scale)
        n = int(np.isfinite(d).sum())
        if n > best_n:
            best_n, best_t = n, t
            if n >= early:
                break
    if best_t is None or best_n < 3:
        return None

    t = best_t
    si = di = np.zeros(0, dtype=np.intp)
    for _ in range(5):  # уточнение по всем звёздам, а не только по ярким
        scale = math.sqrt(abs(np.linalg.det(t[:, :2])))
        d, j = dst_tree.query(apply_transform(t, src), distance_upper_bound=tol * scale)
        si, di = _one_to_one(d, j, len(dst))
        if len(si) < 3:
            return None
        t = _affine_lstsq(src[si], dst[di])
    scale = math.sqrt(abs(np.linalg.det(t[:, :2])))
    resid = apply_transform(t, src[si]) - dst[di]
    rms = float(np.sqrt((resid**2).sum(axis=1).mean()))

    # Защита от ложных решений: совпадений должно быть заметно больше, чем даёт случай.
    span = np.ptp(dst, axis=0)
    area = max(float(span[0] * span[1]), 1e-30)
    chance = len(src) * len(dst) * math.pi * (tol * scale) ** 2 / area
    if len(si) < max(min_inliers, 3 * chance + 3):
        return None
    return MatchResult(t, si, di, rms, scale)
