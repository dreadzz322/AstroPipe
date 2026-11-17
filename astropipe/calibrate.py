"""Калибровка кадров и чистка плохой оцифровки.

Для ПЗС-кадров: мастер-bias / dark / flat и их применение.
Для сканов и старых камер: определение негатива, битые пиксели, полосы считывания.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import ndimage

from .io import Frame, load_image


def mad_std(a: np.ndarray) -> float:
    """Робастная оценка СКО через медианное абсолютное отклонение."""
    a = a[np.isfinite(a)]
    if a.size == 0:
        return 0.0
    med = np.median(a)
    return float(1.4826 * np.median(np.abs(a - med)))


def combine(stack: np.ndarray, clip: float | None = 3.0) -> np.ndarray:
    """Сложение стопки кадров (ось 0): медиана либо среднее с сигма-отсечением."""
    stack = np.asarray(stack, dtype=np.float32)
    if clip is None or stack.shape[0] < 4:
        return np.nanmedian(stack, axis=0).astype(np.float32)
    med = np.nanmedian(stack, axis=0)
    sig = 1.4826 * np.nanmedian(np.abs(stack - med), axis=0)
    good = np.abs(stack - med) <= clip * np.maximum(sig, 1e-6)
    good &= np.isfinite(stack)
    total = np.where(good, stack, 0).sum(axis=0)
    count = good.sum(axis=0)
    return np.where(count > 0, total / np.maximum(count, 1), med).astype(np.float32)


def _load_stack(paths: Iterable[str | Path]) -> tuple[np.ndarray, list[Frame]]:
    frames = [load_image(p) for p in paths]
    if not frames:
        raise ValueError("Пустой список калибровочных кадров")
    return np.stack([f.data for f in frames]), frames


def master_bias(paths: Iterable[str | Path]) -> np.ndarray:
    stack, _ = _load_stack(paths)
    return combine(stack)


def master_dark(paths: Iterable[str | Path], bias: np.ndarray | None = None) -> np.ndarray:
    """Мастер-dark, приведённый к 1 секунде выдержки (если выдержка известна)."""
    stack, frames = _load_stack(paths)
    if bias is not None:
        stack = stack - bias
    for i, f in enumerate(frames):
        if f.exptime:
            stack[i] /= f.exptime
    return combine(stack)


def master_flat(
    paths: Iterable[str | Path],
    bias: np.ndarray | None = None,
    dark: np.ndarray | None = None,
) -> np.ndarray:
    """Мастер-flat, нормированный на медиану."""
    stack, frames = _load_stack(paths)
    for i, f in enumerate(frames):
        if bias is not None:
            stack[i] -= bias
        if dark is not None and f.exptime:
            stack[i] -= dark * f.exptime
        stack[i] /= max(float(np.median(stack[i])), 1e-6)
    flat = combine(stack)
    return (flat / max(float(np.median(flat)), 1e-6)).astype(np.float32)


def calibrate(
    frame: Frame,
    bias: np.ndarray | None = None,
    dark: np.ndarray | None = None,
    flat: np.ndarray | None = None,
) -> Frame:
    """(кадр − bias − dark·t) / flat."""
    data = frame.data.astype(np.float32, copy=True)
    if bias is not None:
        data -= bias
    if dark is not None:
        data -= dark * (frame.exptime or 1.0)
    if flat is not None:
        data /= np.where(flat > 0.05, flat, np.nan)  # сильно завиньетированные углы не доверяем
        data = np.nan_to_num(data, nan=float(np.nanmedian(data)))
    header = frame.header.copy()
    header["CALSTAT"] = "".join(c for c, m in zip("BDF", (bias, dark, flat)) if m is not None)
    return Frame(data, header, frame.path, frame.is_scan)


# ---------------------------------------------------------------- плохая оцифровка


def is_negative(data: np.ndarray) -> bool:
    """Негатив (тёмные звёзды на светлом фоне)? Сравниваем хвосты распределения."""
    sample = data[:: max(1, data.shape[0] // 1500), :: max(1, data.shape[1] // 1500)]
    med = np.median(sample)
    sig = mad_std(sample) or float(sample.std()) or 1.0
    bright = np.count_nonzero(sample > med + 5 * sig)
    dark = np.count_nonzero(sample < med - 5 * sig)
    return dark > 2 * bright and dark > 10


def fix_bad_pixels(data: np.ndarray, nsigma: float = 8.0) -> tuple[np.ndarray, int]:
    """Заменяет одиночные горячие и мёртвые пиксели медианой окрестности.

    Звезду от горячего пикселя отличаем по соседям: у звезды они тоже подняты,
    у горячего пикселя — на уровне фона.
    """
    data = np.asarray(data, dtype=np.float32)
    bg = ndimage.median_filter(data, size=5, mode="nearest")
    diff = data - bg
    sig = mad_std(diff[::3, ::3]) or 1e-6
    ring = np.ones((3, 3), bool)
    ring[1, 1] = False
    nmax = ndimage.maximum_filter(data, footprint=ring, mode="nearest") - bg
    nmin = ndimage.minimum_filter(data, footprint=ring, mode="nearest") - bg
    # соседи «спокойны»: в пределах шума либо много слабее самого выброса
    hot = (diff > nsigma * sig) & (nmax < np.maximum(0.25 * diff, 3.5 * sig))
    dead = (diff < -nsigma * sig) & (nmin > np.minimum(0.25 * diff, -3.5 * sig))
    bad = hot | dead
    out = data.copy()
    out[bad] = bg[bad]
    return out, int(bad.sum())


def remove_banding(data: np.ndarray) -> np.ndarray:
    """Убирает построчные и постолбцовые полосы (дефект сканеров и старой электроники)."""
    data = np.asarray(data, dtype=np.float32)
    level = np.median(data)
    out = data - (np.median(data, axis=1, keepdims=True) - level)
    out -= np.median(out, axis=0, keepdims=True) - level
    return out


def clean_scan(frame: Frame, invert: bool | None = None, banding: bool = True) -> Frame:
    """Подготовка скана: инверсия негатива, полосы, битые пиксели."""
    data = frame.data.astype(np.float32, copy=True)
    header = frame.header.copy()
    if invert is None:
        invert = is_negative(data)
    if invert:
        data = float(data.max()) - data
        header["NEGATIVE"] = (True, "source was a negative, inverted")
    if banding:
        data = remove_banding(data)
    data, nbad = fix_bad_pixels(data)
    header["NBADPIX"] = (nbad, "isolated bad pixels fixed")  # FITS допускает только ASCII
    return Frame(data, header, frame.path, True)
