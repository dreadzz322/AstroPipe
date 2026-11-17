"""Генератор синтетических наблюдений — чтобы проверять конвейер без телескопа.

Делает серию кадров с известным «небом», калибровочные кадры, движущийся объект
и вариант «плохого скана» (негатив, нелинейность, полосы, 8 бит).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time, TimeDelta
from astropy.wcs import WCS

from .io import save_fits


def make_wcs(shape, ra, dec, scale, rot=0.0, flip=False, shift=(0.0, 0.0)) -> WCS:
    """TAN-проекция: scale в ″/пикс, rot в градусах, shift — сдвиг центра в пикселях."""
    ny, nx = shape
    s = scale / 3600.0
    c, sn = math.cos(math.radians(rot)), math.sin(math.radians(rot))
    cd = s * np.array([[-c, sn], [sn, c]])
    if flip:
        cd[:, 0] *= -1
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [ra, dec]
    w.wcs.crpix = [nx / 2 + 0.5 + shift[0], ny / 2 + 0.5 + shift[1]]
    w.wcs.cd = cd
    return w


def random_catalog(ra, dec, radius, n, rng, mag_bright=8.0, mag_faint=17.0) -> Table:
    """Случайное звёздное поле: слабых звёзд больше, чем ярких (как на настоящем небе)."""
    d = dec + rng.uniform(-radius, radius, n)
    r = ra + rng.uniform(-radius, radius, n) / math.cos(math.radians(dec))
    mag = np.clip(mag_faint + np.log10(rng.uniform(1e-9, 1, n)) / 0.3, mag_bright, mag_faint)
    t = Table({"ra": r, "dec": d, "mag": mag})
    t.sort("mag")
    return t


def mag_to_flux(mag, exptime=60.0):
    return 1e5 * 10 ** (-0.4 * (np.asarray(mag) - 10.0)) * exptime / 60.0


def render_stars(shape, x, y, flux, fwhm=3.0) -> np.ndarray:
    ny, nx = shape
    img = np.zeros(shape, dtype=np.float64)
    sigma = fwhm / 2.3548
    r = int(math.ceil(5 * sigma))
    for cx, cy, f in zip(x, y, flux):
        ix, iy = int(round(cx)), int(round(cy))
        xa, xb = max(0, ix - r), min(nx, ix + r + 1)
        ya, yb = max(0, iy - r), min(ny, iy + r + 1)
        if xa >= xb or ya >= yb:
            continue
        yy, xx = np.mgrid[ya:yb, xa:xb]
        img[ya:yb, xa:xb] += f * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2)) / (
            2 * math.pi * sigma**2
        )
    return img


def vignette(shape, strength=0.3) -> np.ndarray:
    ny, nx = shape
    yy, xx = np.mgrid[:ny, :nx]
    r2 = ((xx - nx / 2) ** 2 + (yy - ny / 2) ** 2) / ((nx / 2) ** 2 + (ny / 2) ** 2)
    return 1.0 - strength * r2


def sky_image(shape, wcs, catalog, exptime=60.0, fwhm=3.0, sky=200.0, extra=()) -> np.ndarray:
    """Идеальное изображение неба без шума. extra — добавочные объекты (x, y, поток)."""
    x, y = wcs.all_world2pix(catalog["ra"], catalog["dec"], 0)
    flux = mag_to_flux(catalog["mag"], exptime)
    ny, nx = shape
    inside = (x > -10) & (x < nx + 10) & (y > -10) & (y < ny + 10)
    x, y, flux = list(x[inside]), list(y[inside]), list(flux[inside])
    for ex, ey, ef in extra:
        x.append(ex), y.append(ey), flux.append(ef)
    yy, xx = np.mgrid[:ny, :nx]
    gradient = sky * (1 + 0.15 * xx / nx)  # засветка с одного края
    return render_stars(shape, x, y, flux, fwhm) + gradient


def to_scan(clean: np.ndarray, rng) -> np.ndarray:
    """Имитация плохой оцифровки пластинки: нелинейность, негатив, полосы, шум, 8 бит."""
    lo = np.median(clean)
    norm = np.clip((clean - lo) / (60 * math.sqrt(lo)), 0, 1)
    neg = 225.0 - 200.0 * norm**0.6
    neg += rng.normal(0, 4.0, (clean.shape[0], 1))  # полосы сканера
    neg += rng.normal(0, 1.5, clean.shape)
    return np.clip(np.round(neg), 0, 255).astype(np.uint8)


def make_plate(path, seed=1, shape=(1500, 2000), scale=4.0, rot=-12.0) -> dict:
    """Имитация скана фотопластинки: 16-битный TIFF-негатив без заголовка.

    Широкое поле с радиальной дисторсией, сканер с разным масштабом по осям, насыщение
    эмульсии (яркие звёзды — растущие «блины»), зерно, царапины, пылинки, надпись на краю.
    Рядом пишется каталог <имя>.catalog.csv. Возвращает истинные параметры.
    """
    from PIL import Image

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    ny, nx = shape
    ra0, dec0 = 290.0, 41.0
    catalog = random_catalog(ra0, dec0, 1.7, 7000, rng, mag_bright=5.5, mag_faint=15.0)
    catalog.write(path.with_suffix(".catalog.csv"), overwrite=True)

    wcs = make_wcs(shape, ra0, dec0, scale, rot)
    x, y = wcs.all_world2pix(catalog["ra"], catalog["dec"], 0)
    dx, dy = x - nx / 2, y - ny / 2
    k = 3e-9  # дисторсия: ~6 пикселей в углах
    r2 = dx * dx + dy * dy
    x = nx / 2 + dx * (1 + k * r2) * 1.004  # сканер тянет по одной оси
    y = ny / 2 + dy * (1 + k * r2)
    inside = (x > -30) & (x < nx + 30) & (y > -30) & (y < ny + 30)
    flux = 3e4 * 10 ** (-0.4 * (catalog["mag"] - 10.0))
    sky = 40.0
    exposure = render_wide(shape, x[inside], y[inside], np.asarray(flux[inside]), fwhm=3.5) + sky
    exposure *= vignette(shape, 0.25)

    # характеристическая кривая эмульсии: плотность насыщается
    gamma, e_half, d_max = 1.3, 20 * sky, 2.6
    ratio = (exposure / e_half) ** gamma
    density = d_max * ratio / (1 + ratio) + rng.normal(0, 0.012, shape)  # зерно

    yy, xx = np.mgrid[:ny, :nx]
    for _ in range(4):  # царапины: эмульсия содрана
        x1, y1, ang = rng.uniform(0, nx), rng.uniform(0, ny), rng.uniform(0, math.pi)
        dist = np.abs((xx - x1) * math.sin(ang) - (yy - y1) * math.cos(ang))
        along = (xx - x1) * math.cos(ang) + (yy - y1) * math.sin(ang)
        density[(dist < 0.8) & (np.abs(along) < rng.uniform(100, 500))] = 0.02
    for _ in range(120):  # пылинки и дефекты эмульсии: тёмные пятнышки неправильной формы
        cx, cy = rng.uniform(5, nx - 5), rng.uniform(5, ny - 5)
        a, b, ang = rng.uniform(1, 4), rng.uniform(0.6, 1.5), rng.uniform(0, math.pi)
        u = (xx - cx) * math.cos(ang) + (yy - cy) * math.sin(ang)
        v = -(xx - cx) * math.sin(ang) + (yy - cy) * math.cos(ang)
        density[(u / a) ** 2 + (v / b) ** 2 < 1] = d_max
    density[30:70, 60:420][rng.uniform(size=(40, 360)) < 0.5] = d_max  # подпись тушью на краю

    transmission = 60000.0 * 10 ** (-np.clip(density, 0, None))
    transmission += rng.normal(0, 60, shape)  # шум сканера
    Image.fromarray(np.clip(transmission, 0, 65535).astype(np.uint16)).save(path)
    truth = {"ra": ra0, "dec": dec0, "scale": scale * (1 + 1 / 1.004) / 2, "rot": rot,
             "n_stars_in_field": int(inside.sum())}
    path.with_suffix(".truth.json").write_text(json.dumps(truth, indent=2), encoding="utf-8")
    return truth


def render_wide(shape, x, y, flux, fwhm=3.5) -> np.ndarray:
    """Как render_stars, но с протяжёнными крыльями: яркая звезда засвечивает большой круг."""
    ny, nx = shape
    img = np.zeros(shape, dtype=np.float64)
    sigma = fwhm / 2.3548
    for cx, cy, f in zip(x, y, flux):
        r = int(min(60, max(8, 4 * sigma * (1 + math.log10(max(f, 10)) / 2))))
        ix, iy = int(round(cx)), int(round(cy))
        xa, xb = max(0, ix - r), min(nx, ix + r + 1)
        ya, yb = max(0, iy - r), min(ny, iy + r + 1)
        if xa >= xb or ya >= yb:
            continue
        yy, xx = np.mgrid[ya:yb, xa:xb]
        d2 = (xx - cx) ** 2 + (yy - cy) ** 2
        core = np.exp(-d2 / (2 * sigma**2)) / (2 * math.pi * sigma**2)
        wings = 0.08 / (math.pi * (3 * sigma) ** 2) * (1 + d2 / (3 * sigma) ** 2) ** -2  # Моффат
        img[ya:yb, xa:xb] += f * (0.92 * core + wings)
    return img


def write_dataset(outdir, n_frames=4, seed=1, shape=(600, 800), scale=2.0, rot=23.0) -> dict:
    """Пишет в outdir полный набор: light_*.fits, bias/dark/flat, catalog.csv, scan.png, truth.json."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    ra0, dec0, exptime, fwhm = 83.8, 22.0, 60.0, 3.0
    bias_level, read_noise, dark_rate = 300.0, 5.0, 0.3
    catalog = random_catalog(ra0, dec0, 0.4, 1500, rng)
    catalog.write(outdir / "catalog.csv", overwrite=True)

    flat = vignette(shape)
    hot = (rng.integers(0, shape[0], 60), rng.integers(0, shape[1], 60))
    dark = np.full(shape, dark_rate)
    dark[hot] = rng.uniform(20, 80, 60)

    def readout(signal):
        return rng.poisson(signal).astype(np.float64) + bias_level + rng.normal(0, read_noise, shape)

    for i in range(5):
        save_fits(outdir / f"bias_{i}.fits", readout(np.zeros(shape)), _hdr("bias", 0))
        save_fits(outdir / f"dark_{i}.fits", readout(dark * exptime), _hdr("dark", exptime))
        save_fits(outdir / f"flat_{i}.fits", readout(20000 * flat), _hdr("flat", 2))

    t0 = Time("2026-03-01T20:00:00", scale="utc")
    interval = 300.0
    wcs0 = make_wcs(shape, ra0, dec0, scale, rot)
    mover_start, mover_v = np.array([300.0, 250.0]), np.array([0.02, 0.012])  # пикс и пикс/с
    truth = {"ra": ra0, "dec": dec0, "scale": scale, "rot": rot, "interval": interval,
             "mover_velocity": mover_v.tolist(), "mover_ref_xy": [], "frames": []}
    for i in range(n_frames):
        shift = (0.0, 0.0) if i == 0 else tuple(rng.uniform(-6, 6, 2))
        wcs = make_wcs(shape, ra0, dec0, scale, rot + (0 if i == 0 else rng.uniform(-0.05, 0.05)),
                       shift=shift)
        ref_xy = mover_start + mover_v * interval * i
        mra, mdec = wcs0.all_pix2world(ref_xy[0], ref_xy[1], 0)
        mx, my = wcs.all_world2pix(mra, mdec, 0)
        clean = sky_image(shape, wcs, catalog, exptime, fwhm,
                          extra=[(float(mx), float(my), float(mag_to_flux(13.5, exptime)))])
        data = readout(clean * flat + dark * exptime)
        hdr = _hdr("light", exptime)
        hdr["DATE-OBS"] = (t0 + TimeDelta(interval * i, format="sec")).isot
        hdr["RA"] = "05 35 24"  # наведение «с ошибкой», как в реальных старых заголовках
        hdr["DEC"] = "+22 03 00"
        name = f"light_{i}.fits"
        save_fits(outdir / name, data, hdr)
        truth["mover_ref_xy"].append(ref_xy.tolist())
        truth["frames"].append(name)
        if i == 0:
            from PIL import Image

            Image.fromarray(to_scan(clean, rng)[::-1]).save(outdir / "scan.png")
    (outdir / "truth.json").write_text(json.dumps(truth, indent=2), encoding="utf-8")
    return truth


def _hdr(kind: str, exptime: float) -> fits.Header:
    h = fits.Header()
    h["IMAGETYP"] = kind
    h["EXPTIME"] = exptime
    return h
