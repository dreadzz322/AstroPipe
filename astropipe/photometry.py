"""Апертурная фотометрия и привязка звёздных величин к каталогу."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def aperture_photometry(
    data: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    r: float = 4.0,
    r_in: float | None = None,
    r_out: float | None = None,
    gain: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Поток в круглой апертуре за вычетом локального фона из кольца. Возвращает (flux, err)."""
    r_in = r_in if r_in is not None else max(r + 2.0, 1.5 * r)
    r_out = r_out if r_out is not None else r_in + max(4.0, r)
    ny, nx = data.shape
    half = int(math.ceil(r_out)) + 1
    flux = np.full(len(x), np.nan)
    err = np.full(len(x), np.nan)
    for k, (cx, cy) in enumerate(zip(np.asarray(x, float), np.asarray(y, float))):
        ix, iy = int(round(cx)), int(round(cy))
        if ix - half < 0 or iy - half < 0 or ix + half >= nx or iy + half >= ny:
            continue  # апертура выходит за кадр
        cut = data[iy - half : iy + half + 1, ix - half : ix + half + 1]
        yy, xx = np.mgrid[iy - half : iy + half + 1, ix - half : ix + half + 1]
        d2 = (xx - cx) ** 2 + (yy - cy) ** 2
        ap = d2 <= r * r
        ring = cut[(d2 >= r_in * r_in) & (d2 <= r_out * r_out)]
        ring = ring[np.isfinite(ring)]
        if ring.size < 8:
            continue
        sky = np.median(ring)
        sig = 1.4826 * np.median(np.abs(ring - sky))
        keep = np.abs(ring - sky) <= 3 * max(sig, 1e-9)  # соседние звёзды в кольце не считаем
        if keep.sum() >= 8:
            ring = ring[keep]
            sky = np.median(ring)
        sky_std = float(ring.std())
        n_ap = int(ap.sum())
        f = float(cut[ap].sum() - sky * n_ap)
        flux[k] = f
        err[k] = math.sqrt(max(f, 0.0) / gain + n_ap * sky_std**2 * (1 + n_ap / ring.size))
    return flux, err


@dataclass
class ZeroPoint:
    coeffs: np.ndarray  # степень 0: [zp]; иначе коэффициенты полинома от инструментальной величины
    rms: float
    n_used: int

    def apply(self, inst_mag: np.ndarray) -> np.ndarray:
        inst_mag = np.asarray(inst_mag, dtype=float)
        if len(self.coeffs) == 1:
            return inst_mag + self.coeffs[0]
        return np.polyval(self.coeffs, inst_mag)


def fit_zeropoint(
    inst_mag: np.ndarray, cat_mag: np.ndarray, degree: int = 0, clip: float = 2.5, iters: int = 5
) -> ZeroPoint | None:
    """Связь инструментальных величин с каталожными.

    degree=0 — линейный приёмник (ПЗС): каталог = инструмент + нуль-пункт.
    degree≥1 — нелинейный отклик (фотопластинки, плохая оцифровка): полином.
    """
    inst_mag = np.asarray(inst_mag, dtype=float)
    cat_mag = np.asarray(cat_mag, dtype=float)
    good = np.isfinite(inst_mag) & np.isfinite(cat_mag)
    if good.sum() < max(3, degree + 2):
        return None
    coeffs = np.array([0.0])
    rms = np.inf
    for _ in range(iters):
        if degree == 0:
            coeffs = np.array([np.median(cat_mag[good] - inst_mag[good])])
            model = inst_mag + coeffs[0]
        else:
            coeffs = np.polyfit(inst_mag[good], cat_mag[good], degree)
            model = np.polyval(coeffs, inst_mag)
        resid = cat_mag - model
        sig = 1.4826 * np.median(np.abs(resid[good] - np.median(resid[good])))
        rms = float(np.sqrt(np.mean(resid[good] ** 2)))
        new_good = good & (np.abs(resid) <= clip * max(sig, 1e-4))
        if new_good.sum() < max(3, degree + 2) or new_good.sum() == good.sum():
            break
        good = new_good
    return ZeroPoint(coeffs, rms, int(good.sum()))


@dataclass
class CalibrationCurve:
    """Калибровочная кривая пластинки: каталожная величина как функция инструментальной."""

    inst: np.ndarray  # узлы, по возрастанию
    mag: np.ndarray
    rms: float
    n_used: int

    @property
    def coeffs(self) -> np.ndarray:  # для отчёта: узлы кривой парами
        return np.column_stack([self.inst, self.mag]).ravel()

    def apply(self, inst_mag: np.ndarray) -> np.ndarray:
        x = np.asarray(inst_mag, dtype=float)
        out = np.interp(x, self.inst, self.mag)
        if len(self.inst) >= 2:  # за краями — линейное продолжение, а не «полка»
            lo = (self.mag[1] - self.mag[0]) / (self.inst[1] - self.inst[0])
            hi = (self.mag[-1] - self.mag[-2]) / (self.inst[-1] - self.inst[-2])
            out = np.where(x < self.inst[0], self.mag[0] + lo * (x - self.inst[0]), out)
            out = np.where(x > self.inst[-1], self.mag[-1] + hi * (x - self.inst[-1]), out)
        return np.where(np.isfinite(x), out, np.nan)


def fit_curve(inst_mag: np.ndarray, cat_mag: np.ndarray, clip: float = 3.0) -> CalibrationCurve | None:
    """Строит кривую по медианам в группах звёзд, от ярких к слабым.

    Полином здесь не годится: слабых звёзд в сотни раз больше, и он подгоняется под них,
    а на ярком (насыщенном) конце врёт. Медианы по группам устойчивы и к ошибочным
    отождествлениям. Группы у яркого конца маленькие, дальше растут.
    """
    inst_mag = np.asarray(inst_mag, dtype=float)
    cat_mag = np.asarray(cat_mag, dtype=float)
    good = np.flatnonzero(np.isfinite(inst_mag) & np.isfinite(cat_mag))
    if len(good) < 12:
        return None
    curve = None
    for _ in range(3):
        order = good[np.argsort(inst_mag[good])]
        cap = max(6, len(order) // 15)
        knots_x, knots_y, start, size = [], [], 0, 6.0
        while start < len(order):
            stop = min(len(order), start + int(size))
            if len(order) - stop < 4:
                stop = len(order)
            chunk = order[start:stop]
            knots_x.append(np.median(inst_mag[chunk]))
            knots_y.append(np.median(cat_mag[chunk]))
            start, size = stop, min(cap, size * 1.5)
        x, y = np.array(knots_x), np.maximum.accumulate(np.array(knots_y))  # кривая монотонна
        keep = np.concatenate([[True], np.diff(x) > 1e-6])
        x, y = x[keep], y[keep]
        if len(x) < 2:
            return None
        curve = CalibrationCurve(x, y, np.nan, len(good))
        resid = cat_mag[good] - curve.apply(inst_mag[good])
        sig = 1.4826 * np.median(np.abs(resid - np.median(resid)))
        curve.rms = float(sig)
        inliers = np.abs(resid) <= clip * max(sig, 0.02)
        if inliers.all():
            break
        good = good[inliers]
        curve.n_used = len(good)
    return curve


def instrumental_mag(flux: np.ndarray) -> np.ndarray:
    flux = np.asarray(flux, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(flux > 0, -2.5 * np.log10(flux), np.nan)
