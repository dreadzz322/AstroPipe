"""Совмещение кадров по звёздам и сложение."""

from __future__ import annotations

import numpy as np
from astropy.table import Table
from scipy import ndimage

from .calibrate import combine
from .match import MatchResult, match_points


def register(sources: Table, reference: Table, tol: float = 2.0) -> MatchResult | None:
    """Преобразование координат кадра в координаты опорного кадра."""
    src = np.column_stack([sources["x"], sources["y"]])
    dst = np.column_stack([reference["x"], reference["y"]])
    return match_points(src, dst, tol=tol, n_src=30, n_dst=40, scale_range=(0.8, 1.25))


def warp(data: np.ndarray, transform: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Перекладывает кадр на сетку опорного. transform: опорный = A·кадр + t (в x, y)."""
    a_inv = np.linalg.inv(transform[:, :2])
    offset_xy = -a_inv @ transform[:, 2]
    swap = np.array([[0.0, 1.0], [1.0, 0.0]])  # scipy работает в порядке (строка, столбец)
    return ndimage.affine_transform(
        data.astype(np.float32),
        swap @ a_inv @ swap,
        offset=offset_xy[::-1],
        output_shape=shape,
        order=3,
        mode="constant",
        cval=np.nan,
    )


def stack_frames(
    images: list[np.ndarray], transforms: list[np.ndarray], clip: float | None = 3.0
) -> np.ndarray:
    """Складывает кадры после совмещения. Фон каждого кадра выравнивается по медиане."""
    shape = images[0].shape
    warped = []
    for img, t in zip(images, transforms):
        w = warp(img, t, shape)
        warped.append(w - np.nanmedian(w))
    return combine(np.stack(warped), clip=clip)
