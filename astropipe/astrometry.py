"""Астрометрическая привязка (plate solving): пиксели → небесные координаты."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from astropy.coordinates import Angle, SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from scipy.spatial import cKDTree

from .match import match_points


@dataclass
class Solution:
    wcs: WCS
    n_matched: int
    rms_arcsec: float
    scale_arcsec: float  # угловых секунд на пиксель
    rotation_deg: float  # позиционный угол оси Y кадра (от севера к востоку)
    mirrored: bool
    src_idx: np.ndarray  # индексы в таблице источников
    cat_idx: np.ndarray  # индексы в каталоге
    center: tuple[float, float]  # RA, Dec центра кадра, градусы

    def summary(self) -> dict:
        return {
            "matched_stars": self.n_matched,
            "rms_arcsec": round(self.rms_arcsec, 3),
            "scale_arcsec_per_px": round(self.scale_arcsec, 4),
            "rotation_deg": round(self.rotation_deg, 2),
            "mirrored": self.mirrored,
            "center_ra_deg": round(self.center[0], 6),
            "center_dec_deg": round(self.center[1], 6),
        }


def tan_project(ra, dec, ra0: float, dec0: float):
    """Гномоническая проекция на касательную плоскость. Всё в градусах."""
    ra, dec = np.radians(ra), np.radians(dec)
    ra0, dec0 = math.radians(ra0), math.radians(dec0)
    cosc = math.sin(dec0) * np.sin(dec) + math.cos(dec0) * np.cos(dec) * np.cos(ra - ra0)
    xi = np.cos(dec) * np.sin(ra - ra0) / cosc
    eta = (math.cos(dec0) * np.sin(dec) - math.sin(dec0) * np.cos(dec) * np.cos(ra - ra0)) / cosc
    return np.degrees(xi), np.degrees(eta)


def pointing_from_header(header: fits.Header) -> tuple[float, float] | None:
    """Приблизительное наведение из заголовка FITS (RA/DEC, OBJCTRA/OBJCTDEC, CRVAL)."""
    for kra, kdec in (("RA", "DEC"), ("OBJCTRA", "OBJCTDEC"), ("TELRA", "TELDEC"), ("CRVAL1", "CRVAL2")):
        if kra in header and kdec in header:
            try:
                return parse_ra(header[kra]), parse_dec(header[kdec])
            except Exception:
                continue
    return None


def parse_ra(value) -> float:
    """RA в градусы: число — уже градусы, строка «чч мм сс» — часы."""
    if isinstance(value, (int, float)):
        return float(value) % 360
    s = str(value).strip()
    try:
        return float(s) % 360
    except ValueError:
        return float(Angle(s, unit="hourangle").degree) % 360


def parse_dec(value) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    try:
        return float(s)
    except ValueError:
        return float(Angle(s, unit="deg").degree)


def scale_from_header(header: fits.Header) -> float | None:
    """Масштаб (″/пиксель) из заголовка, если его можно вычислить."""
    for key in ("PIXSCALE", "SECPIX", "SCALE"):
        if key in header:
            try:
                v = float(header[key])
                if v > 0:
                    return v
            except (TypeError, ValueError):
                pass
    try:
        pix_um = float(header["XPIXSZ"])
        focal_mm = float(header["FOCALLEN"])
        if pix_um > 0 and focal_mm > 0:
            return 206.265 * pix_um / focal_mm
    except (KeyError, TypeError, ValueError):
        pass
    return None


def _linear_wcs(transform: np.ndarray, ra0: float, dec0: float) -> WCS:
    """WCS из аффинного преобразования пиксели → касательная плоскость."""
    cd = transform[:, :2]
    crpix = -np.linalg.solve(cd, transform[:, 2]) + 1.0  # FITS считает пиксели с 1
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [ra0, dec0]
    w.wcs.crpix = crpix
    w.wcs.cd = cd
    w.wcs.cunit = ["deg", "deg"]
    return w


def _residuals_arcsec(wcs: WCS, xy: np.ndarray, sky: SkyCoord) -> np.ndarray:
    ra, dec = wcs.all_pix2world(xy[:, 0], xy[:, 1], 0)
    return SkyCoord(ra, dec, unit="deg").separation(sky).arcsec


def solve_field(
    sources: Table,
    shape: tuple[int, int],
    catalog: Table,
    ra: float,
    dec: float,
    scale: float | None = None,
    scale_tol: float = 0.25,
    pix_tol: float = 3.0,
    sip_degree: int | None = None,
) -> Solution | None:
    """Привязывает кадр к каталогу.

    sources — результат detect_sources (от ярких к слабым); catalog — ra, dec, mag;
    ra, dec — приблизительный центр кадра; scale — приблизительный масштаб (″/пикс), если известен.
    """
    if len(sources) < 4 or len(catalog) < 4:
        return None
    ny, nx = shape
    xy = np.column_stack([sources["x"], sources["y"]]).astype(float)
    order = np.argsort(np.asarray(catalog["mag"]))
    cat_ra = np.asarray(catalog["ra"], dtype=float)[order]
    cat_dec = np.asarray(catalog["dec"], dtype=float)[order]
    xi, eta = tan_project(cat_ra, cat_dec, ra, dec)
    dst = np.column_stack([xi, eta])

    n_src = min(25, len(xy))
    scale_range = None
    n_dst = 60
    if scale:
        deg = scale / 3600.0
        scale_range = (deg / (1 + scale_tol), deg * (1 + scale_tol))
        span = np.ptp(dst, axis=0)
        ratio = max(1.0, float(span[0] * span[1]) / (nx * ny * deg * deg))
        n_dst = int(min(120, max(40, 1.5 * n_src * ratio)))
    res = None
    for skip in (0, 30, 80, 160, 320):  # если ярчайших звёзд каталога нет на снимке
        if skip and skip + 10 > len(dst):
            break
        res = match_points(xy, dst, tol=pix_tol, n_src=n_src, n_dst=n_dst,
                           scale_range=scale_range, dst_skip=skip)
        if res is not None:
            break
    if res is None:
        return None

    wcs = _linear_wcs(res.transform, ra, dec)
    cat_sky = SkyCoord(cat_ra, cat_dec, unit="deg")
    src_idx, cat_idx = res.src_idx, res.dst_idx
    src_tree = cKDTree(xy)

    def fit(si, ci, degree):
        from astropy.wcs.utils import fit_wcs_from_points

        return fit_wcs_from_points(
            (xy[si, 0], xy[si, 1]), cat_sky[ci], projection="TAN", sip_degree=degree
        )

    for it in range(6 if sip_degree else 3):  # уточняем: подгонка по совпавшим звёздам → новое сопоставление
        degree = sip_degree if (sip_degree and it > 0 and len(src_idx) >= 10 * sip_degree**2) else None
        try:
            cand = fit(src_idx, cat_idx, degree)
            old = np.sqrt(np.mean(_residuals_arcsec(wcs, xy[src_idx], cat_sky[cat_idx]) ** 2))
            new = np.sqrt(np.mean(_residuals_arcsec(cand, xy[src_idx], cat_sky[cat_idx]) ** 2))
            if np.isfinite(new) and new <= old * 1.05:
                wcs = cand
        except Exception:
            pass  # остаёмся на линейном решении
        px, py = wcs.all_world2pix(cat_ra, cat_dec, 0, quiet=True)
        ok = np.isfinite(px) & np.isfinite(py)
        d, j = src_tree.query(np.column_stack([px[ok], py[ok]]), distance_upper_bound=pix_tol)
        hit = np.flatnonzero(np.isfinite(d))
        hit = hit[np.argsort(d[hit])]
        _, first = np.unique(j[hit], return_index=True)  # один источник — одна звезда каталога
        hit = hit[first]
        if len(hit) < 4:
            break
        cat_idx = np.flatnonzero(ok)[hit]
        src_idx = j[hit]

    # Отбрасываем грубые промахи (слипшиеся пары, случайные совпадения) и подгоняем начисто.
    resid = _residuals_arcsec(wcs, xy[src_idx], cat_sky[cat_idx])
    keep = resid <= max(3 * 1.4826 * float(np.median(resid)), 1e-3)
    if 4 <= keep.sum() < len(resid):
        src_idx, cat_idx = src_idx[keep], cat_idx[keep]
        try:
            degree = sip_degree if (sip_degree and len(src_idx) >= 10 * sip_degree**2) else None
            cand = fit(src_idx, cat_idx, degree)
            new = _residuals_arcsec(cand, xy[src_idx], cat_sky[cat_idx])
            if np.isfinite(new).all() and np.mean(new**2) <= np.mean(resid[keep] ** 2):
                wcs = cand
        except Exception:
            pass
        resid = _residuals_arcsec(wcs, xy[src_idx], cat_sky[cat_idx])
    cd = wcs.pixel_scale_matrix
    det = float(np.linalg.det(cd))
    cra, cdec = wcs.all_pix2world((nx - 1) / 2, (ny - 1) / 2, 0)
    return Solution(
        wcs=wcs,
        n_matched=len(src_idx),
        rms_arcsec=float(np.sqrt(np.mean(resid**2))),
        scale_arcsec=math.sqrt(abs(det)) * 3600.0,
        rotation_deg=math.degrees(math.atan2(cd[0, 1], cd[1, 1])),
        mirrored=det > 0,  # обычное небо (восток слева) даёт det < 0
        src_idx=np.asarray(src_idx),
        cat_idx=order[cat_idx],
        center=(float(cra), float(cdec)),
    )
