"""Поиск движущихся объектов (астероидов, комет) в серии совмещённых кадров."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class Tracklet:
    frames: list[int]  # номера кадров, где объект найден
    indices: list[int]  # индексы детекций в этих кадрах
    positions: np.ndarray  # (n, 2) координаты в системе опорного кадра
    velocity: np.ndarray  # (vx, vy), пикселей в секунду
    rms: float  # отклонение от прямолинейного движения, пиксели

    @property
    def speed(self) -> float:
        return float(np.hypot(*self.velocity))


def find_moving(
    detections: list[np.ndarray],
    times: np.ndarray,
    tol: float = 1.5,
    stationary_radius: float = 2.0,
    max_speed: float | None = None,
    min_detections: int = 3,
    fluxes: list[np.ndarray] | None = None,
    max_flux_ratio: float = 3.0,
) -> list[Tracklet]:
    """Ищет объекты, движущиеся равномерно и прямолинейно.

    detections — списки (N, 2) координат в общей системе; times — секунды.
    fluxes — потоки детекций (необязательно): яркость объекта в цепочке не должна
    различаться больше чем в max_flux_ratio раз, это отсекает случайные цепочки из шума.
    Сначала выбрасываем всё неподвижное (звёзды), потом среди остатка ищем цепочки,
    лежащие на одной прямой с постоянной скоростью.
    """
    n = len(detections)
    times = np.asarray(times, dtype=float)
    if n < 3:
        raise ValueError("Нужно минимум 3 кадра")
    min_detections = min(min_detections, n)
    dets = [np.asarray(d, dtype=float).reshape(-1, 2) for d in detections]
    trees = [cKDTree(d) if len(d) else None for d in dets]

    # 1. Неподвижное: на том же месте есть детекция хотя бы в одном другом кадре
    #    (слабые звёзды на пороге обнаружения видны не на всех кадрах).
    need = 1
    cand_idx = []
    for i, d in enumerate(dets):
        seen = np.zeros(len(d), dtype=int)
        for j in range(n):
            if j != i and trees[j] is not None and len(d):
                dist, _ = trees[j].query(d, distance_upper_bound=stationary_radius)
                seen += np.isfinite(dist)
        cand_idx.append(np.flatnonzero(seen < need))
    cands = [d[idx] for d, idx in zip(dets, cand_idx)]
    ctrees = [cKDTree(c) if len(c) else None for c in cands]

    span = times.max() - times.min()
    if max_speed is None:  # по умолчанию — не больше 200 пикселей за всю серию
        max_speed = 200.0 / max(span, 1e-9)

    # 2. Каждая пара кадров задаёт гипотезу скорости; проверяем её по остальным кадрам.
    found: dict[tuple, Tracklet] = {}
    for i in range(n):
        for j in range(i + 1, n):
            if ctrees[i] is None or ctrees[j] is None:
                continue
            dt = times[j] - times[i]
            if dt <= 0:
                continue
            pairs = ctrees[i].query_ball_tree(ctrees[j], max_speed * dt)
            for ia, partners in enumerate(pairs):
                for ib in partners:
                    v = (cands[j][ib] - cands[i][ia]) / dt
                    if np.hypot(*v) * span < 2 * stationary_radius:
                        continue  # почти не сдвинулся — неотличимо от звезды
                    chain = {i: ia, j: ib}
                    for k in range(n):
                        if k in chain or ctrees[k] is None:
                            continue
                        pred = cands[i][ia] + v * (times[k] - times[i])
                        dist, ik = ctrees[k].query(pred, distance_upper_bound=2 * tol)
                        if np.isfinite(dist):
                            chain[k] = int(ik)
                    if len(chain) < min_detections:
                        continue
                    fr = sorted(chain)
                    pos = np.array([cands[k][chain[k]] for k in fr])
                    t = times[fr]
                    design = np.column_stack([np.ones(len(t)), t - t.mean()])
                    coef, *_ = np.linalg.lstsq(design, pos, rcond=None)
                    rms = float(np.sqrt(((design @ coef - pos) ** 2).sum(axis=1).mean()))
                    if rms > tol:
                        continue
                    if fluxes is not None:
                        f = np.array([fluxes[k][cand_idx[k][chain[k]]] for k in fr], dtype=float)
                        if f.min() <= 0 or f.max() / f.min() > max_flux_ratio:
                            continue
                    key = tuple((k, int(cand_idx[k][chain[k]])) for k in fr)
                    found[key] = Tracklet(fr, [idx for _, idx in key], pos, coef[1], rms)

    # 3. Убираем цепочки, целиком входящие в более длинные.
    keys = sorted(found, key=len, reverse=True)
    kept: list[tuple] = []
    for key in keys:
        if not any(set(key) <= set(other) for other in kept):
            kept.append(key)
    return [found[k] for k in kept]
