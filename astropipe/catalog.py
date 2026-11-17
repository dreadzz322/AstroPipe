"""Опорный звёздный каталог: Gaia DR3 через VizieR (с кэшем на диске) или локальный CSV."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.table import Table

GAIA_EPOCH = 2016.0
VIZIER_SERVERS = ("vizier.cds.unistra.fr", "vizier.cfa.harvard.edu")


def load_local(path: str | Path) -> Table:
    """Локальный каталог: CSV/ECSV с колонками ra, dec (градусы) и mag."""
    t = Table.read(path)
    lower = {c.lower(): c for c in t.colnames}
    out = Table()
    for want, options in (("ra", ("ra", "ra_icrs", "raj2000")), ("dec", ("dec", "de_icrs", "dej2000")),
                          ("mag", ("mag", "gmag", "vmag", "phot_g_mean_mag"))):
        for o in options:
            if o in lower:
                out[want] = np.asarray(t[lower[o]], dtype=float)
                break
        else:
            raise ValueError(f"В каталоге {path} нет колонки {want}")
    out.sort("mag")
    return out


def query_gaia(
    ra: float,
    dec: float,
    radius: float,
    maglim: float = 16.0,
    epoch: float | None = None,
    cache_dir: str | Path | None = None,
) -> Table:
    """Звёзды Gaia DR3 в круге radius (градусы) ярче maglim.

    epoch — год наблюдения: положения переносятся на него по собственным движениям.
    Для старых пластинок (десятки лет) это заметные доли угловой секунды и больше.
    """
    cache = None
    if cache_dir is not None:
        cache = Path(cache_dir) / f"gaia_{ra:.4f}_{dec:+.4f}_r{radius:.3f}_m{maglim:.1f}.ecsv"
        if cache.exists():
            raw = Table.read(cache)
            return _to_epoch(raw, epoch)

    try:
        import astropy.units as u
        from astropy.coordinates import SkyCoord
        from astroquery.vizier import Vizier
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Для загрузки Gaia нужен пакет astroquery (pip install astroquery) "
            "или передайте локальный каталог через --catalog"
        ) from exc

    result, last_error = None, None
    for server in VIZIER_SERVERS:  # основной сервер CDS бывает недоступен — идём по зеркалам
        vizier = Vizier(
            columns=["RA_ICRS", "DE_ICRS", "Gmag", "pmRA", "pmDE"],
            column_filters={"Gmag": f"<{maglim}"},
            row_limit=-1,
            vizier_server=server,
        )
        try:
            result = vizier.query_region(
                SkyCoord(ra, dec, unit="deg"), radius=radius * u.deg, catalog="I/355/gaiadr3"
            )
            break
        except Exception as exc:
            last_error = exc
    if result is None:
        raise RuntimeError(f"ни один сервер VizieR не ответил: {last_error}")
    if len(result) == 0:
        raise RuntimeError("VizieR не вернул звёзд — проверьте координаты и радиус")
    t = result[0]
    raw = Table()
    raw["ra"] = np.asarray(t["RA_ICRS"], dtype=float)
    raw["dec"] = np.asarray(t["DE_ICRS"], dtype=float)
    raw["mag"] = np.asarray(t["Gmag"], dtype=float)
    raw["pmra"] = np.nan_to_num(np.ma.filled(t["pmRA"].astype(float), 0.0))
    raw["pmdec"] = np.nan_to_num(np.ma.filled(t["pmDE"].astype(float), 0.0))
    raw.sort("mag")
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        raw.write(cache, overwrite=True)
    return _to_epoch(raw, epoch)


def _to_epoch(raw: Table, epoch: float | None) -> Table:
    out = raw["ra", "dec", "mag"].copy()
    if epoch is not None and "pmra" in raw.colnames:
        dt = epoch - GAIA_EPOCH
        out["dec"] = raw["dec"] + raw["pmdec"] * dt / 3.6e6
        out["ra"] = raw["ra"] + raw["pmra"] * dt / 3.6e6 / np.cos(np.radians(raw["dec"]))
    return out
