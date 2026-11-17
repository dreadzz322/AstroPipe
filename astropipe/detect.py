"""Оценка фона и поиск звёзд на снимке."""

from __future__ import annotations

import math

import numpy as np
from astropy.table import Table
from scipy import ndimage

FLAG_SATURATED = 1
FLAG_EDGE = 2


def _upsample(mesh: np.ndarray, ny: int, nx: int, by: int, bx: int) -> np.ndarray:
    """Билинейная интерполяция сетки фона на полный размер кадра."""
    yc = (np.arange(mesh.shape[0]) + 0.5) * by - 0.5
    xc = (np.arange(mesh.shape[1]) + 0.5) * bx - 0.5
    y, x = np.arange(ny), np.arange(nx)
    tmp = np.empty((ny, mesh.shape[1]), dtype=np.float64)
    for j in range(mesh.shape[1]):
        tmp[:, j] = np.interp(y, yc, mesh[:, j])
    out = np.empty((ny, nx), dtype=np.float32)
    for i in range(ny):
        out[i] = np.interp(x, xc, tmp[i])
    return out


def estimate_background(
    data: np.ndarray, box: int = 64, clip: float = 3.0, iters: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """Карта фона и карта шума: медиана с сигма-отсечением по ячейкам сетки.

    Справляется с виньетированием, градиентами засветки и неравномерной подсветкой сканера.
    """
    data = np.asarray(data, dtype=np.float32)
    ny, nx = data.shape
    nby, nbx = max(1, ny // box), max(1, nx // box)
    by, bx = ny // nby, nx // nbx
    med = np.empty((nby, nbx))
    std = np.empty((nby, nbx))
    for r in range(nby):  # по строкам сетки, чтобы не держать копию всего кадра
        strip = data[r * by : (r + 1) * by, : nbx * bx]
        t = strip.reshape(by, nbx, bx).swapaxes(0, 1).reshape(nbx, by * bx).astype(np.float32)
        for _ in range(iters):
            m = np.nanmedian(t, axis=1, keepdims=True)
            s = 1.4826 * np.nanmedian(np.abs(t - m), axis=1, keepdims=True)
            t[np.abs(t - m) > clip * np.maximum(s, 1e-9)] = np.nan
        med[r] = np.nanmedian(t, axis=1)
        std[r] = np.nanstd(t, axis=1)
    for mesh in (med, std):
        bad = ~np.isfinite(mesh)
        if bad.any():
            mesh[bad] = np.nanmedian(mesh) if (~bad).any() else 0.0
    if min(nby, nbx) >= 3:  # ячейки, испорченные яркой звездой или туманностью
        med = ndimage.median_filter(med, size=3, mode="nearest")
        std = ndimage.median_filter(std, size=3, mode="nearest")
    # Пол шума: у 8-битных сканов фон бывает идеально ровным, а деление на 0 недопустимо.
    sample = data[::7, ::7]
    integer_like = bool(np.all(sample == np.round(sample)))
    floor = 0.29 if integer_like else 1e-6 * max(float(np.ptp(sample)), 1e-6)
    std = np.maximum(std, floor)
    return _upsample(med, ny, nx, by, bx), _upsample(std, ny, nx, by, bx)


def _refine_centroid(sub: np.ndarray, x: float, y: float, sigma: float) -> tuple[float, float]:
    """Уточнение центра итерациями с гауссовым окном (точность до сотых пикселя)."""
    ny, nx = sub.shape
    r = int(math.ceil(3 * sigma)) + 1
    x0, y0 = x, y
    for _ in range(8):
        ix, iy = int(round(x)), int(round(y))
        xa, xb = max(0, ix - r), min(nx, ix + r + 1)
        ya, yb = max(0, iy - r), min(ny, iy + r + 1)
        cut = sub[ya:yb, xa:xb]
        yy, xx = np.mgrid[ya:yb, xa:xb]
        w = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma**2)) * np.clip(cut, 0, None)
        s = w.sum()
        if s <= 0:
            return x0, y0
        dx = 2 * ((w * xx).sum() / s - x)
        dy = 2 * ((w * yy).sum() / s - y)
        x, y = x + dx, y + dy
        if abs(x - x0) > 2 * r or abs(y - y0) > 2 * r:
            return x0, y0
        if dx * dx + dy * dy < 1e-6:
            break
    return x, y


def detect_sources(
    data: np.ndarray,
    nsigma: float = 4.0,
    fwhm: float = 3.0,
    minarea: int = 5,
    box: int = 64,
    saturation: float | None = None,
    edge: int = 3,
    background: tuple[np.ndarray, np.ndarray] | None = None,
) -> Table:
    """Находит точечные источники. Возвращает таблицу, отсортированную по убыванию потока.

    Колонки: x, y (пиксели, от 0), flux, peak, npix, fwhm, ellip, snr, flag.
    """
    data = np.asarray(data, dtype=np.float32)
    bkg, rms = background if background is not None else estimate_background(data, box)
    sub = data - bkg
    sigma = max(fwhm / 2.3548, 0.5)
    conv = ndimage.gaussian_filter(sub, sigma, mode="nearest")
    noise_factor = min(1.0, 1.0 / (2 * sigma * math.sqrt(math.pi)))  # шум после сглаживания
    labels, n = ndimage.label(conv > nsigma * rms * noise_factor, structure=np.ones((3, 3)))
    ny, nx = data.shape
    rows = []
    for i, sl in enumerate(ndimage.find_objects(labels), start=1):
        m = labels[sl] == i
        npix = int(m.sum())
        if npix < minarea:
            continue
        v = np.clip(sub[sl], 0, None) * m
        flux = float(v.sum())
        if flux <= 0:
            continue
        yy, xx = np.mgrid[sl]
        x = float((v * xx).sum() / flux)
        y = float((v * yy).sum() / flux)
        mxx = max(float((v * (xx - x) ** 2).sum() / flux), 1 / 12)
        myy = max(float((v * (yy - y) ** 2).sum() / flux), 1 / 12)
        mxy = float((v * (xx - x) * (yy - y)).sum() / flux)
        half = (mxx + myy) / 2
        root = math.sqrt(((mxx - myy) / 2) ** 2 + mxy**2)
        a2, b2 = half + root, max(half - root, 1e-6)
        peak = float(data[sl][m].max())
        x, y = _refine_centroid(sub, x, y, sigma)
        flag = 0
        if saturation is not None and peak >= saturation:
            flag |= FLAG_SATURATED
        if x < edge or y < edge or x > nx - 1 - edge or y > ny - 1 - edge:
            flag |= FLAG_EDGE
        local_rms = float(rms[min(ny - 1, max(0, int(round(y)))), min(nx - 1, max(0, int(round(x))))])
        rows.append(
            (
                x,
                y,
                flux,
                peak,
                npix,
                2.3548 * math.sqrt(half),
                1 - math.sqrt(b2 / a2),
                flux / (local_rms * math.sqrt(npix)),
                flag,
            )
        )
    names = ("x", "y", "flux", "peak", "npix", "fwhm", "ellip", "snr", "flag")
    dtypes = ("f8", "f8", "f8", "f8", "i4", "f8", "f8", "f8", "i4")
    table = Table(rows=rows, names=names, dtype=dtypes)
    table.sort("flux", reverse=True)
    return table


def star_like(sources: Table, max_ellip: float = 0.5) -> np.ndarray:
    """Маска «похоже на звезду»: не вытянутое, не на краю, размер близок к типичному."""
    if len(sources) == 0:
        return np.zeros(0, bool)
    med = float(np.median(sources["fwhm"]))
    ok = (sources["ellip"] < max_ellip) & ((sources["flag"] & FLAG_EDGE) == 0)
    ok &= (sources["fwhm"] > 0.4 * med) & (sources["fwhm"] < 3.0 * med)
    return np.asarray(ok)
