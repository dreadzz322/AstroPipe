"""Сквозные проверки конвейера на синтетических данных с известным ответом."""

import json

import numpy as np
import pytest
from astropy.table import Table

from astropipe import calibrate as cal
from astropipe import cli
from astropipe.astrometry import solve_field
from astropipe.detect import detect_sources, star_like
from astropipe.io import load_image
from astropipe.match import apply_transform, match_points
from astropipe.moving import find_moving
from astropipe.photometry import aperture_photometry, fit_zeropoint, instrumental_mag
from astropipe.simulate import make_wcs, random_catalog, render_stars, sky_image, write_dataset


@pytest.fixture(scope="session")
def dataset(tmp_path_factory):
    path = tmp_path_factory.mktemp("sim")
    truth = write_dataset(path, n_frames=4, seed=3)
    return path, truth


def test_detect_positions():
    rng = np.random.default_rng(0)
    x, y = rng.uniform(20, 380, 40), rng.uniform(20, 280, 40)
    img = render_stars((300, 400), x, y, np.full(40, 20000.0)) + 100
    img = rng.poisson(img).astype(float)
    src = detect_sources(img)
    assert 35 <= len(src) <= 42
    for sx, sy, flux in zip(src["x"], src["y"], src["flux"]):
        d = np.hypot(x - sx, y - sy).min()
        assert d < 0.1 or flux > 30000  # либо точно на звезде, либо слипшаяся пара (двойной поток)


@pytest.mark.parametrize("flip", [False, True])
def test_match_points_similarity(flip):
    rng = np.random.default_rng(1)
    src = rng.uniform(0, 1000, (60, 2))
    ang = np.radians(37)
    a = 0.7 * np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    if flip:
        a = a @ np.diag([-1.0, 1.0])
    dst = src @ a.T + [50, -20] + rng.normal(0, 0.05, src.shape)
    dst = np.vstack([dst[:50], rng.uniform(-500, 800, (30, 2))])  # часть потеряна, есть мусор
    res = match_points(src, dst[rng.permutation(len(dst))][:], tol=1.0, n_src=20, n_dst=80)
    assert res is not None and len(res.src_idx) >= 40
    assert abs(res.scale - 0.7) < 0.005
    assert np.allclose(res.transform[:, :2], a, atol=0.01)


def test_match_rejects_unrelated():
    rng = np.random.default_rng(2)
    assert match_points(rng.uniform(0, 1000, (40, 2)), rng.uniform(0, 1000, (40, 2)), tol=1.0) is None


@pytest.mark.parametrize("flip,known_scale", [(False, True), (True, False)])
def test_plate_solve(flip, known_scale):
    rng = np.random.default_rng(5)
    shape = (600, 800)
    catalog = random_catalog(150.0, -30.0, 0.4, 1500, rng)
    wcs = make_wcs(shape, 150.0, -30.0, 2.0, rot=-64.0, flip=flip)
    img = rng.poisson(sky_image(shape, wcs, catalog)).astype(float)
    src = detect_sources(img)
    src = src[star_like(src)]
    # наведение известно с ошибкой ~4′
    sol = solve_field(src, shape, catalog, 150.05, -30.04, scale=2.1 if known_scale else None)
    assert sol is not None
    assert sol.n_matched > 80
    assert sol.rms_arcsec < 0.7  # ≈0.35 пикс: предел точности центров слабых звёзд
    assert abs(sol.scale_arcsec - 2.0) < 0.01
    assert sol.mirrored == flip
    ra, dec = sol.wcs.all_pix2world(400.0, 300.0, 0)
    tra, tdec = wcs.all_pix2world(400.0, 300.0, 0)
    assert abs(ra - tra) * np.cos(np.radians(30)) * 3600 < 0.5 and abs(dec - tdec) * 3600 < 0.5


def test_calibration_flattens_field(dataset):
    path, _ = dataset
    bias = cal.master_bias(sorted(path.glob("bias_*.fits")))
    dark = cal.master_dark(sorted(path.glob("dark_*.fits")), bias)
    flat = cal.master_flat(sorted(path.glob("flat_*.fits")), bias, dark)
    raw = load_image(path / "light_0.fits")
    done = cal.calibrate(raw, bias, dark, flat)

    def corner_to_centre(d):
        return np.median(d[:60, :60]) / np.median(d[270:330, 370:430])

    assert corner_to_centre(raw.data - 300) < 0.85  # виньетирование видно
    assert 0.9 < corner_to_centre(done.data) < 1.0  # осталась только небольшая засветка неба
    assert abs(np.median(dark) - 0.3) < 0.1


def test_bad_pixels_keep_stars():
    rng = np.random.default_rng(4)
    img = render_stars((200, 200), [50.0, 120.3], [60.0, 90.7], [5e4, 2e3], fwhm=2.5) + 100
    img = rng.poisson(img).astype(np.float32)
    star_peak = img[60, 50]
    img[150, 150] = 5000  # горячий
    img[30, 170] = 0  # мёртвый
    fixed, n = cal.fix_bad_pixels(img)
    assert abs(fixed[150, 150] - 100) < 40 and abs(fixed[30, 170] - 100) < 40
    assert fixed[60, 50] == star_peak
    assert n < 10


def test_photometry_zeropoint():
    rng = np.random.default_rng(6)
    mags = rng.uniform(10, 14, 60)
    x, y = rng.uniform(30, 570, 60), rng.uniform(30, 570, 60)
    flux = 1e5 * 10 ** (-0.4 * (mags - 10))
    img = rng.poisson(render_stars((600, 600), x, y, flux) + 150).astype(float)
    f, e = aperture_photometry(img, x, y, r=5)
    zp = fit_zeropoint(instrumental_mag(f), mags)
    assert zp is not None and zp.rms < 0.05
    assert abs(zp.coeffs[0] - 22.5) < 0.05  # 10ᵐ ↔ 1e5 отсчётов
    assert np.all(e[np.isfinite(e)] > 0)


def test_find_moving_synthetic():
    rng = np.random.default_rng(7)
    stars = rng.uniform(0, 1000, (300, 2))
    times = np.array([0.0, 300, 600, 900])
    dets = []
    for t in times:
        mover = np.array([[400 + 0.03 * t, 500 - 0.01 * t]])
        junk = rng.uniform(0, 1000, (15, 2))  # космические лучи и шум
        dets.append(np.vstack([stars + rng.normal(0, 0.1, stars.shape), mover, junk]))
    tracks = find_moving(dets, times)
    assert len(tracks) == 1
    assert len(tracks[0].frames) == 4
    assert np.allclose(tracks[0].velocity, [0.03, -0.01], atol=1e-3)


def _args(extra):
    return cli.build_parser().parse_args(extra)


def test_cli_end_to_end(dataset, tmp_path):
    path, truth = dataset
    out = tmp_path / "out"
    masters = tmp_path / "masters"
    assert cli.main(["masters", "--out", str(masters),
                     "--bias-frames", *map(str, sorted(path.glob("bias_*.fits"))),
                     "--dark-frames", *map(str, sorted(path.glob("dark_*.fits"))),
                     "--flat-frames", *map(str, sorted(path.glob("flat_*.fits")))]) == 0
    common = ["--bias", str(masters / "master_bias.fits"), "--dark", str(masters / "master_dark.fits"),
              "--flat", str(masters / "master_flat.fits"), "--catalog", str(path / "catalog.csv"),
              "--out", str(out)]
    lights = [str(path / n) for n in truth["frames"]]

    # одиночный кадр: координаты берутся из заголовка, масштаб неизвестен
    assert cli.main(["process", lights[0], *common]) == 0
    rep = json.loads((out / "light_0.report.json").read_text(encoding="utf-8"))
    assert rep["astrometry"]["solved"]
    assert rep["astrometry"]["rms_arcsec"] < 0.7
    assert abs(rep["astrometry"]["scale_arcsec_per_px"] - truth["scale"]) < 0.01
    assert rep["photometry"]["rms_mag"] < 0.15
    table = Table.read(out / "light_0.sources.csv")
    assert {"ra", "dec", "mag"} <= set(table.colnames)
    assert (out / "light_0.preview.png").exists()

    # серия: движущийся объект
    assert cli.main(["moving", *lights, *common]) == 0
    found = json.loads((out / "moving.json").read_text(encoding="utf-8"))
    assert len(found) == 1
    assert np.allclose(found[0]["xy_ref"], truth["mover_ref_xy"], atol=0.5)
    expected_rate = np.hypot(*truth["mover_velocity"]) * 60 * truth["scale"]
    assert abs(found[0]["rate_arcsec_per_min"] - expected_rate) < 0.1

    # сложение
    stacked = tmp_path / "stack.fits"
    assert cli.main(["stack", *lights, *common[:6], "--out", str(stacked)]) == 0
    deep = detect_sources(load_image(stacked).data)
    single = detect_sources(cal.calibrate(load_image(lights[0])).data)
    assert len(deep) > len(single)


def test_bad_scan_negative(dataset, tmp_path):
    """8-битный негатив с полосами и нелинейностью: должен распознаться и привязаться."""
    path, truth = dataset
    assert cal.is_negative(load_image(path / "scan.png").data)
    out = tmp_path / "scan_out"
    assert cli.main(["process", str(path / "scan.png"), "--catalog", str(path / "catalog.csv"),
                     "--ra", "83.85", "--dec", "22.03", "--out", str(out)]) == 0
    rep = json.loads((out / "scan.report.json").read_text(encoding="utf-8"))
    assert rep["negative_inverted"]
    assert rep["astrometry"]["solved"]
    assert rep["astrometry"]["mirrored"]  # скан сохранён перевёрнутым
    assert rep["astrometry"]["rms_arcsec"] < 1.0
    assert abs(rep["astrometry"]["scale_arcsec_per_px"] - truth["scale"]) < 0.02


def test_photographic_plate(tmp_path):
    """Скан пластинки: негатив без заголовка, дисторсия, насыщение, царапины, пыль, подпись."""
    from astropipe.simulate import make_plate

    truth = make_plate(tmp_path / "plate.tif", seed=11)
    out = tmp_path / "out"
    # масштаб задаётся через паспорт телескопа и разрешение сканера: 206265/3438 мм · 25.4/381 dpi ≈ 4″
    assert cli.main(["process", str(tmp_path / "plate.tif"), "--catalog", str(tmp_path / "plate.catalog.csv"),
                     "--ra", "19 21 00", "--dec", "+41 10 00", "--epoch", "1965.5",
                     "--focal", "3438", "--dpi", "381", "--out", str(out)]) == 0
    rep = json.loads((out / "plate.report.json").read_text(encoding="utf-8"))
    assert rep["negative_inverted"]
    ast = rep["astrometry"]
    assert ast["solved"] and ast["matched_stars"] > 0.6 * truth["n_stars_in_field"]
    assert ast["rms_arcsec"] < 1.2  # 0.3 пикселя при дисторсии до 6 пикселей в углах
    assert abs(ast["scale_arcsec_per_px"] - truth["scale"]) < 0.01
    assert abs(ast["center_ra_deg"] - truth["ra"]) < 3e-4 and abs(ast["center_dec_deg"] - truth["dec"]) < 3e-4
    assert rep["photometry"]["model"] == "curve" and rep["photometry"]["rms_mag"] < 0.15

    table = Table.read(out / "plate.sources.csv")
    diff = np.asarray(table["mag"] - table["cat_mag"], dtype=float)
    cat = np.asarray(table["cat_mag"], dtype=float)
    for lo, hi in ((8, 10), (10, 12), (12, 15)):  # насыщенные, средние и слабые звёзды
        sel = (cat >= lo) & (cat < hi) & np.isfinite(diff)
        assert sel.sum() > 20
        assert abs(np.median(diff[sel])) < 0.1

    # без учёта дисторсии точность заметно хуже — значит, SIP действительно работает
    assert cli.main(["process", str(tmp_path / "plate.tif"), "--catalog", str(tmp_path / "plate.catalog.csv"),
                     "--ra", "290.2", "--dec", "41.2", "--sip", "0", "--out", str(tmp_path / "nosip")]) == 0
    flat = json.loads((tmp_path / "nosip" / "plate.report.json").read_text(encoding="utf-8"))
    assert flat["astrometry"]["rms_arcsec"] > 2 * ast["rms_arcsec"]


def test_binning_keeps_sky_scale(tmp_path):
    from astropipe.simulate import make_plate

    truth = make_plate(tmp_path / "plate.tif", seed=12)
    assert cli.main(["process", str(tmp_path / "plate.tif"), "--catalog", str(tmp_path / "plate.catalog.csv"),
                     "--ra", "290.0", "--dec", "41.0", "--scale", "4.0", "--bin", "2",
                     "--fwhm", "2", "--out", str(tmp_path / "out")]) == 0
    rep = json.loads((tmp_path / "out" / "plate.report.json").read_text(encoding="utf-8"))
    assert rep["shape"] == [750, 1000]
    assert rep["astrometry"]["solved"]
    assert abs(rep["astrometry"]["scale_arcsec_per_px"] - 2 * truth["scale"]) < 0.03
