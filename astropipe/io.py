"""Загрузка и сохранение снимков: FITS и растровые сканы (TIFF/PNG/JPEG)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.time import Time

FITS_SUFFIXES = {".fits", ".fit", ".fts", ".fz", ".gz"}


@dataclass
class Frame:
    """Один кадр: пиксели (float32, 2D) и заголовок."""

    data: np.ndarray
    header: fits.Header = field(default_factory=fits.Header)
    path: Path | None = None
    is_scan: bool = False  # растровый скан, а не «родной» FITS с прибора

    @property
    def shape(self) -> tuple[int, int]:
        return self.data.shape

    @property
    def exptime(self) -> float | None:
        for key in ("EXPTIME", "EXPOSURE", "EXP_TIME"):
            if key in self.header:
                try:
                    return float(self.header[key])
                except (TypeError, ValueError):
                    pass
        return None

    @property
    def time(self) -> Time | None:
        """Момент наблюдения. Старые заголовки бывают кривыми — пробуем несколько вариантов."""
        h = self.header
        try:
            if "DATE-OBS" in h:
                s = str(h["DATE-OBS"]).strip()
                if "T" not in s:
                    for key in ("TIME-OBS", "UT", "UTC", "UTSTART"):
                        if key in h and ":" in str(h[key]):
                            s = f"{s}T{str(h[key]).strip()}"
                            break
                return Time(s, scale="utc")
            if "JD" in h:
                return Time(float(h["JD"]), format="jd", scale="utc")
            if "MJD-OBS" in h:
                return Time(float(h["MJD-OBS"]), format="mjd", scale="utc")
        except Exception:
            return None
        return None


def is_fits(path: Path) -> bool:
    return path.suffix.lower() in FITS_SUFFIXES


def load_image(path: str | Path) -> Frame:
    """Читает FITS или растровое изображение и приводит к 2D float32."""
    path = Path(path)
    if is_fits(path):
        with fits.open(path, memmap=False) as hdul:
            for hdu in hdul:
                if hdu.data is not None and hdu.data.ndim >= 2:
                    data = np.asarray(hdu.data, dtype=np.float32)
                    header = hdu.header.copy()
                    break
            else:
                raise ValueError(f"В файле {path} нет изображения")
        while data.ndim > 2:  # цветной куб или стопка — усредняем
            data = data.mean(axis=0)
        return Frame(data, header, path, is_scan=False)

    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None  # сканы пластинок бывают очень большими
    with Image.open(path) as img:
        if img.mode not in ("L", "I", "F", "I;16", "I;16B", "I;16L", "RGB"):
            img = img.convert("RGB")
        arr = np.asarray(img)
    data = arr.astype(np.float32)
    if data.ndim == 3:
        data = data[..., :3].mean(axis=2)
    header = fits.Header()
    header["ORIGFILE"] = path.name[:60]
    return Frame(np.ascontiguousarray(data), header, path, is_scan=True)


def bin_image(data: np.ndarray, n: int) -> np.ndarray:
    """Уменьшает кадр в n раз усреднением блоков n×n (для огромных сканов)."""
    ny, nx = data.shape[0] // n * n, data.shape[1] // n * n
    return data[:ny, :nx].reshape(ny // n, n, nx // n, n).mean(axis=(1, 3)).astype(np.float32)


def save_fits(path: str | Path, data: np.ndarray, header: fits.Header | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.PrimaryHDU(np.asarray(data, dtype=np.float32), header=header).writeto(path, overwrite=True)


def save_preview(path: str | Path, data: np.ndarray, marks=None, max_size: int = 2400) -> None:
    """PNG-превью с растяжкой контраста. marks — список (x, y, радиус, цвет)."""
    from PIL import Image, ImageDraw

    finite = data[np.isfinite(data)]
    med = float(np.median(finite))
    sig = float(1.4826 * np.median(np.abs(finite - med))) or float(finite.std()) or 1.0
    lo, hi = med - sig, med + 40 * sig
    norm = np.clip((np.nan_to_num(data, nan=med) - lo) / (hi - lo), 0, 1)
    img8 = (255 * np.arcsinh(10 * norm) / np.arcsinh(10)).astype(np.uint8)
    img = Image.fromarray(img8[::-1])  # FITS: начало координат снизу
    ny, nx = data.shape
    k = min(1.0, max_size / max(ny, nx))
    if k < 1:
        img = img.resize((max(1, round(nx * k)), max(1, round(ny * k))), Image.BILINEAR)
    img = img.convert("RGB")
    if marks:
        draw = ImageDraw.Draw(img)
        for x, y, r, color in marks:
            cx, cy, rr = x * k, (ny - 1 - y) * k, max(3.0, r * k)
            draw.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], outline=color)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
