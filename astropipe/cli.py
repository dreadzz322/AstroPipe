"""Командная строка: python -m astropipe <команда> ..."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table

from . import calibrate as cal
from .astrometry import (Solution, parse_dec, parse_ra, pointing_from_header, scale_from_header,
                         solve_field)
from .catalog import load_local, query_gaia
from .detect import FLAG_EDGE, detect_sources, star_like
from .io import Frame, bin_image, load_image, save_fits, save_preview
from .match import apply_transform
from .moving import find_moving
from .photometry import aperture_photometry, fit_curve, fit_zeropoint, instrumental_mag
from .stack import register, stack_frames


def prepare(path: Path, args) -> Frame:
    """Загрузка + калибровка + чистка дефектов оцифровки."""
    frame = load_image(path)
    if args.bin and args.bin > 1:
        frame.data = bin_image(frame.data, args.bin)
        frame.header["BINNING"] = args.bin
    masters = [load_image(p).data if p else None for p in (args.bias, args.dark, args.flat)]
    if any(m is not None for m in masters):
        frame = cal.calibrate(frame, *masters)
    if frame.is_scan or args.scan:
        frame = cal.clean_scan(frame, invert=args.invert)
    else:
        frame.data, nbad = cal.fix_bad_pixels(frame.data)
        frame.header["NBADPIX"] = nbad
    return frame


def detect(frame: Frame, args) -> Table:
    return detect_sources(frame.data, nsigma=args.nsigma, fwhm=args.fwhm, saturation=args.saturation)


NAMES_RU = {  # частые русские названия → понятные справочнику SIMBAD
    "плеяды": "M45", "вега": "Vega", "сириус": "Sirius", "полярная": "Polaris",
    "денеб": "Deneb", "альтаир": "Altair", "капелла": "Capella", "арктур": "Arcturus",
    "бетельгейзе": "Betelgeuse", "ригель": "Rigel", "антарес": "Antares", "альдебаран": "Aldebaran",
    "туманность андромеды": "M31", "андромеда": "M31", "туманность ориона": "M42",
    "гиады": "Hyades", "ясли": "M44", "крабовидная туманность": "M1",
}


def resolve_target(text: str) -> tuple[float, float]:
    """Название объекта или координаты одной строкой → (RA, Dec) в градусах."""
    import re

    import astropy.units as u

    s = " ".join(text.replace(",", " ").split())
    parts = s.split()
    if len(parts) == 2 and all(re.fullmatch(r"[+-]?\d+(\.\d+)?", p) for p in parts):
        return float(parts[0]) % 360, float(parts[1])  # два числа — градусы
    if re.match(r"^\d", s) and re.search(r"[+-]\s*\d", s[1:]):
        c = SkyCoord(s, unit=(u.hourangle, u.deg))  # «чч мм сс ±гг мм сс»
        return float(c.ra.deg), float(c.dec.deg)
    name = NAMES_RU.get(s.lower(), s)
    c = SkyCoord.from_name(name)  # справочник SIMBAD/NED, нужен интернет
    return float(c.ra.deg), float(c.dec.deg)


def pixel_scale(frame: Frame, args) -> float | None:
    """Масштаб в ″ на пиксель обрабатываемого (возможно, уменьшенного) кадра."""
    scale = args.scale
    if not scale and args.dpi:
        plate = args.plate_scale or (206264.8 / args.focal if args.focal else None)  # ″/мм
        if plate:
            scale = plate * 25.4 / args.dpi
    if not scale:
        scale = scale_from_header(frame.header)
    return scale * (args.bin or 1) if scale else None


def get_catalog(frame: Frame, ra: float, dec: float, scale: float | None, args, outdir: Path) -> Table:
    if args.catalog:
        return load_local(args.catalog)
    ny, nx = frame.shape
    if args.radius:
        radius = args.radius
    elif scale:
        radius = 0.65 * math.hypot(nx, ny) * scale / 3600 + 0.1  # полдиагонали + ошибка наведения
    else:
        radius = 1.0
    maglim = args.maglim or (16.0 if radius <= 1.0 else 14.0)
    epoch = args.epoch or (frame.time.decimalyear if frame.time is not None else None)
    return query_gaia(ra, dec, radius, maglim, epoch, cache_dir=outdir / "catalog_cache")


def solve(frame: Frame, sources: Table, args, outdir: Path) -> tuple[Solution | None, Table | None, str]:
    """Астрометрия одного кадра. Возвращает (решение, каталог, пояснение)."""
    if args.target:
        try:
            ra, dec = resolve_target(args.target)
        except Exception as exc:
            return None, None, f"не удалось понять «{args.target}»: {exc}"
        print(f"  центр поля: RA={ra:.4f}° Dec={dec:+.4f}° («{args.target}»)")
    elif args.ra is not None and args.dec is not None:
        ra, dec = parse_ra(args.ra), parse_dec(args.dec)
    else:
        pointing = pointing_from_header(frame.header)
        if pointing is None:
            return None, None, "нет приблизительных координат: укажите --ra и --dec"
        ra, dec = pointing
    scale = pixel_scale(frame, args)
    try:
        catalog = get_catalog(frame, ra, dec, scale, args, outdir)
    except Exception as exc:
        return None, None, f"каталог недоступен: {exc}"
    stars = sources[star_like(sources)]
    sip = args.sip if args.sip is not None else (3 if frame.is_scan else 0)  # у пластинок — дисторсия
    sol = solve_field(stars, frame.shape, catalog, ra, dec, scale=scale, pix_tol=args.match_tol,
                      sip_degree=sip or None)
    if sol is None:
        return None, catalog, "звёзды не удалось сопоставить с каталогом (проверьте координаты/масштаб)"
    # индексы решения относятся к отфильтрованной таблице — переводим в исходную
    sol.src_idx = np.flatnonzero(star_like(sources))[sol.src_idx]
    return sol, catalog, "ok"


def crossmatch(sources: Table, catalog: Table, wcs, tol: float) -> np.ndarray:
    """Каталожная величина для каждого источника (NaN, если звезды рядом нет).

    В отличие от астрометрии, здесь участвуют и большие насыщенные звёзды: допуск растёт
    с размером изображения, а из нескольких кандидатов берётся самая яркая звезда каталога.
    """
    from scipy.spatial import cKDTree

    px, py = wcs.all_world2pix(catalog["ra"], catalog["dec"], 0, quiet=True)
    ok = np.isfinite(px) & np.isfinite(py)
    mags = np.asarray(catalog["mag"], dtype=float)[ok]
    tree = cKDTree(np.column_stack([px[ok], py[ok]]))
    xy = np.column_stack([sources["x"], sources["y"]])
    fwhm = np.asarray(sources["fwhm"], dtype=float)
    big = fwhm > 1.8 * np.median(fwhm)  # насыщенные «блины»
    out = np.full(len(sources), np.nan)
    dist, nearest = tree.query(xy, distance_upper_bound=tol)
    hit = np.isfinite(dist) & ~big
    out[hit] = mags[nearest[hit]]
    for i in np.flatnonzero(big):
        near = tree.query_ball_point(xy[i], max(tol, 0.5 * fwhm[i]))
        if near:
            out[i] = mags[near].min()
    return out


def process_one(path: Path, args, outdir: Path) -> dict:
    frame = prepare(path, args)
    sources = detect(frame, args)
    stem = path.stem
    report: dict = {"file": str(path), "shape": list(frame.shape), "n_sources": len(sources),
                    "negative_inverted": bool(frame.header.get("NEGATIVE", False)),
                    "bad_pixels_fixed": int(frame.header.get("NBADPIX", 0))}
    print(f"[{path.name}] источников: {len(sources)}")
    if len(sources) == 0:
        report["note"] = "источники не найдены"
        return report
    fwhm = float(np.median(sources["fwhm"][star_like(sources)])) if star_like(sources).any() else args.fwhm
    report["median_fwhm_px"] = round(fwhm, 2)

    r_ap = max(2.5, 1.2 * fwhm)
    flux, err = aperture_photometry(frame.data, sources["x"], sources["y"], r=r_ap)
    sources["ap_flux"], sources["ap_err"] = flux, err
    inst = instrumental_mag(flux)

    marks = [(s["x"], s["y"], 2 * fwhm, "yellow") for s in sources[:3000]]
    sol, catalog, why = solve(frame, sources, args, outdir)
    if sol is None:
        report["astrometry"] = {"solved": False, "reason": why}
        print(f"  астрометрия: не решено — {why}")
    else:
        ra, dec = sol.wcs.all_pix2world(sources["x"], sources["y"], 0)
        sources["ra"], sources["dec"] = ra, dec
        cat_mag = crossmatch(sources, catalog, sol.wcs, args.match_tol)
        sources["cat_mag"] = cat_mag
        report["astrometry"] = {"solved": True, **sol.summary()}
        print(f"  астрометрия: {sol.n_matched} звёзд, RMS {sol.rms_arcsec:.2f}″, "
              f"масштаб {sol.scale_arcsec:.3f}″/пикс, центр RA={sol.center[0]:.5f}° Dec={sol.center[1]:+.5f}°")
        degree = args.phot_degree if args.phot_degree is not None else (-1 if frame.is_scan else 0)
        if frame.is_scan:
            # На пластинке яркие звёзды насыщены и растут вширь: апертура фиксированного радиуса
            # их «не видит», поэтому берём поток по всей площади изображения звезды.
            flux = np.asarray(sources["flux"], dtype=float)
            inst = instrumental_mag(flux)
        zp = fit_curve(inst, cat_mag) if degree < 0 else fit_zeropoint(inst, cat_mag, degree=degree)
        if zp is not None:
            sources["mag"] = zp.apply(inst)
            with np.errstate(divide="ignore", invalid="ignore"):
                sources["mag_err"] = zp.rms if frame.is_scan else 1.0857 * err / flux
            good = np.isfinite(sources["mag"])
            limit = float(np.percentile(sources["mag"][good], 95)) if good.any() else None
            report["photometry"] = {"model": "curve" if degree < 0 else f"poly{degree}",
                                    "coeffs": [round(float(c), 4) for c in zp.coeffs],
                                    "rms_mag": round(zp.rms, 3), "stars_used": zp.n_used,
                                    "faint_limit_mag": None if limit is None else round(limit, 2)}
            print(f"  фотометрия: RMS {zp.rms:.3f}ᵐ по {zp.n_used} звёздам, предел ≈ {limit:.1f}ᵐ")
        for key in list(frame.header):
            if key.startswith(("CD1_", "CD2_", "PC1_", "PC2_", "CDELT", "CROTA", "A_", "B_", "AP_", "BP_")):
                del frame.header[key]
        frame.header.update(sol.wcs.to_header(relax=True))
        matched = set(sol.src_idx.tolist())
        marks = [(s["x"], s["y"], 2 * fwhm, "lime" if i in matched else "yellow")
                 for i, s in enumerate(sources[:3000])]

    outdir.mkdir(parents=True, exist_ok=True)
    sources.write(outdir / f"{stem}.sources.csv", format="ascii.csv", overwrite=True)
    save_fits(outdir / f"{stem}.proc.fits", frame.data, frame.header)
    save_preview(outdir / f"{stem}.preview.png", frame.data, marks)
    (outdir / f"{stem}.report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                                encoding="utf-8")
    return report


# ------------------------------------------------------------------ команды


def cmd_simulate(args) -> int:
    from .simulate import write_dataset

    write_dataset(args.out, n_frames=args.frames, seed=args.seed)
    from .simulate import make_plate

    make_plate(Path(args.out) / "plate.tif", seed=args.seed)
    print(f"Тестовый набор записан в {args.out}")
    return 0


def cmd_masters(args) -> int:
    out = Path(args.out)
    bias = dark = None
    if args.bias_frames:
        bias = cal.master_bias(args.bias_frames)
        save_fits(out / "master_bias.fits", bias)
        print(f"master_bias.fits: {len(args.bias_frames)} кадров")
    if args.dark_frames:
        dark = cal.master_dark(args.dark_frames, bias)
        save_fits(out / "master_dark.fits", dark)
        print(f"master_dark.fits: {len(args.dark_frames)} кадров (на 1 с выдержки)")
    if args.flat_frames:
        save_fits(out / "master_flat.fits", cal.master_flat(args.flat_frames, bias, dark))
        print(f"master_flat.fits: {len(args.flat_frames)} кадров")
    return 0


def cmd_process(args) -> int:
    out = Path(args.out)
    failed = 0
    for p in args.images:
        try:
            process_one(Path(p), args, out)
        except Exception as exc:
            failed += 1
            print(f"[{Path(p).name}] ошибка: {exc}")
    print(f"Результаты: {out}")
    return 1 if failed else 0


def _register_series(args):
    frames = [prepare(Path(p), args) for p in args.images]
    tables = [detect(f, args) for f in frames]
    ref = tables[0][star_like(tables[0])]
    transforms = [np.array([[1.0, 0, 0], [0, 1.0, 0]])]
    for path, table in zip(args.images[1:], tables[1:]):
        res = register(table[star_like(table)], ref, tol=args.match_tol)
        if res is None:
            raise RuntimeError(f"не удалось совместить {path} с первым кадром")
        transforms.append(res.transform)
    return frames, tables, transforms


def cmd_stack(args) -> int:
    frames, _, transforms = _register_series(args)
    result = stack_frames([f.data for f in frames], transforms)
    header = frames[0].header.copy()
    header["NCOMBINE"] = len(frames)
    save_fits(args.out, result, header)
    save_preview(Path(args.out).with_suffix(".preview.png"), np.nan_to_num(result))
    print(f"Сложено кадров: {len(frames)} → {args.out}")
    return 0


def cmd_moving(args) -> int:
    out = Path(args.out)
    frames, tables, transforms = _register_series(args)
    if args.interval:
        times = np.arange(len(frames)) * float(args.interval)
    elif all(f.time is not None for f in frames):
        times = np.array([(f.time - frames[0].time).sec for f in frames])
    else:
        print("В заголовках нет времени наблюдения — укажите --interval (секунды между кадрами)")
        return 2
    dets, fluxes = [], []
    for table, t in zip(tables, transforms):
        ok = (table["ellip"] < 0.7) & ((table["flag"] & FLAG_EDGE) == 0)
        dets.append(apply_transform(t, np.column_stack([table["x"][ok], table["y"][ok]])))
        fluxes.append(np.asarray(table["flux"][ok]))
    tracks = find_moving(dets, times, tol=args.tol, max_speed=args.max_speed, fluxes=fluxes)

    sol, _, why = solve(frames[0], tables[0], args, out)
    if sol is None:
        print(f"Астрометрия опорного кадра не решена ({why}) — результаты только в пикселях")
    results = []
    for n, tr in enumerate(tracks, 1):
        item = {"id": n, "frames": [Path(args.images[k]).name for k in tr.frames],
                "xy_ref": np.round(tr.positions, 2).tolist(),
                "speed_px_per_min": round(tr.speed * 60, 3), "rms_px": round(tr.rms, 3)}
        line = f"  объект {n}: {len(tr.frames)} кадров, {tr.speed * 60:.2f} пикс/мин"
        if sol is not None:
            ra, dec = sol.wcs.all_pix2world(tr.positions[:, 0], tr.positions[:, 1], 0)
            sky = SkyCoord(ra, dec, unit="deg")
            dt = times[tr.frames[-1]] - times[tr.frames[0]]
            rate = sky[0].separation(sky[-1]).arcsec / dt * 60
            pa = sky[0].position_angle(sky[-1]).deg
            item.update(ra_deg=np.round(ra, 6).tolist(), dec_deg=np.round(dec, 6).tolist(),
                        rate_arcsec_per_min=round(float(rate), 3), position_angle_deg=round(float(pa), 1))
            line += f", {rate:.2f}″/мин, PA {pa:.0f}°, RA={ra[0]:.5f}° Dec={dec[0]:+.5f}°"
        results.append(item)
        print(line)
    print(f"Движущихся объектов: {len(tracks)}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "moving.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    marks = [(x, y, 8, "red") for tr in tracks for x, y in tr.positions]
    save_preview(out / "moving.preview.png", frames[0].data, marks)
    return 0


SCAN_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


def _ask(question: str, example: str = "") -> str:
    print(question)
    if example:
        print(f"   например: {example}")
    try:
        return input("   Ответ: ").replace("﻿", "").strip()
    except EOFError:
        return ""


def _ask_file(prompt: str) -> Path | None:
    while True:
        try:
            text = input(prompt).replace("﻿", "").strip().strip('"').strip("'").strip()
        except EOFError:
            return None
        if not text:
            return None
        path = Path(text)
        if path.is_file():
            return path
        if path.is_dir():
            print("   Это папка. Перетащите сам файл снимка.")
        else:
            print(f"   Файл не найден: {text}")


def cmd_wizard(args) -> int:
    """Пошаговый режим для run.bat: вопросы простыми словами, затем обработка."""
    path = Path(args.file.strip('"')) if args.file else None
    if path is None:
        print("Перетащите файл снимка или скана мышкой в это окно и нажмите Enter.")
        print("Подходят FITS, TIFF, PNG, JPEG.\n")
        path = _ask_file("Файл: ")
    while path is not None:
        print("\n" + "=" * 60)
        print(f"Файл: {path.name}")
        print("Ответьте на вопросы. Если ответа не знаете — просто нажмите Enter.")
        print("-" * 60 + "\n")
        params = build_parser().parse_args(["process", str(path)])
        params.target = _ask("1. Что снято? Название объекта или координаты центра снимка.",
                             "M45, Плеяды, Вега, NGC 7000 или 19 21 00 +41 10 00") or None
        print()
        while True:
            epoch = _ask("2. Год съёмки.", "1965 или 1965.5")
            try:
                params.epoch = float(epoch.replace(",", ".")) if epoch else None
                break
            except ValueError:
                print("   Нужно число, например 1965.\n")
        print()
        params.scale = _number(_ask("3. Масштаб снимка: угловых секунд на пиксель (если известен).", "1.6"))
        if not params.scale:
            print()
            params.focal = _number(_ask("4. Фокусное расстояние телескопа, мм.", "2000"))
            if path.suffix.lower() in SCAN_SUFFIXES:
                print()
                params.dpi = _number(_ask("5. С каким разрешением сканировали, dpi.", "1200"))
        params.out = str(path.parent / "astropipe_out")
        print("\n" + "-" * 60)
        print("Обрабатываю... (обычно до минуты, большие сканы — дольше)\n")
        try:
            process_one(path, params, Path(params.out))
        except Exception as exc:
            print(f"Ошибка: {exc}")
        print("\n" + "-" * 60)
        print(f"Результаты в папке: {params.out}")
        if os.name == "nt":
            try:
                os.startfile(params.out)  # открыть папку в проводнике
            except OSError:
                pass
        print("\nЧтобы обработать ещё один файл, перетащите его в это окно и нажмите Enter.")
        print("Чтобы закончить, нажмите Enter или закройте окно.")
        path = _ask_file("Файл: ")
    return 0


def _number(text: str) -> float | None:
    if not text:
        return None
    try:
        value = float(text.replace(",", "."))
        return value if value > 0 else None
    except ValueError:
        print("   Не похоже на число — пропускаю этот вопрос.")
        return None


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    g = common.add_argument_group("подготовка кадра")
    g.add_argument("--bias", help="мастер-bias (FITS)")
    g.add_argument("--dark", help="мастер-dark на 1 с (FITS)")
    g.add_argument("--flat", help="мастер-flat (FITS)")
    g.add_argument("--scan", action="store_true", help="обрабатывать как скан (полосы, негатив)")
    g.add_argument("--bin", type=int, help="уменьшить кадр в N раз перед обработкой (огромные сканы)")
    g.add_argument("--invert", action=argparse.BooleanOptionalAction, default=None,
                   help="негатив: принудительно инвертировать или нет (по умолчанию — автоопределение)")
    g = common.add_argument_group("поиск звёзд")
    g.add_argument("--nsigma", type=float, default=4.0, help="порог обнаружения в сигмах шума")
    g.add_argument("--fwhm", type=float, default=3.0, help="ожидаемый размер звезды, пикселей")
    g.add_argument("--saturation", type=float, help="уровень насыщения")
    g = common.add_argument_group("астрометрия")
    g.add_argument("--target", help="что снято: название объекта (M45, Вега, NGC 7000) "
                   "или координаты центра одной строкой («19 21 00 +41 10 00» или «290.25 41.17»)")
    g.add_argument("--ra", help="приблизительное RA центра: градусы или «чч мм сс»")
    g.add_argument("--dec", help="приблизительное Dec центра: градусы или «±гг мм сс»")
    g.add_argument("--scale", type=float, help="приблизительный масштаб, ″/пиксель исходного файла")
    g.add_argument("--dpi", type=float, help="разрешение сканирования (вместе с --focal или --plate-scale)")
    g.add_argument("--focal", type=float, help="фокусное расстояние телескопа, мм")
    g.add_argument("--plate-scale", type=float, help="масштаб пластинки, ″/мм")
    g.add_argument("--epoch", type=float, help="год съёмки (например 1962.3) — для собственных движений")
    g.add_argument("--radius", type=float, help="радиус запроса каталога, градусы")
    g.add_argument("--maglim", type=float, help="предельная величина каталога")
    g.add_argument("--catalog", help="локальный каталог CSV (ra, dec, mag) вместо Gaia")
    g.add_argument("--match-tol", type=float, default=3.0, help="допуск сопоставления, пикселей")
    g.add_argument("--sip", type=int, help="степень полинома дисторсии SIP; 0 — выключить "
                   "(по умолчанию 3 для сканов, 0 для FITS)")
    g.add_argument("--phot-degree", type=int, help="фотометрическая модель: 0 — нуль-пункт (ПЗС), "
                   "1–3 — полином, -1 — калибровочная кривая (по умолчанию для сканов)")

    p = argparse.ArgumentParser(prog="astropipe", description="Конвейер обработки астроснимков")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("simulate", help="создать тестовый набор синтетических кадров")
    s.add_argument("out")
    s.add_argument("--frames", type=int, default=4)
    s.add_argument("--seed", type=int, default=1)
    s.set_defaults(func=cmd_simulate)

    s = sub.add_parser("wizard", help="пошаговый режим с вопросами (его запускает run.bat)")
    s.add_argument("file", nargs="?")
    s.set_defaults(func=cmd_wizard)

    s = sub.add_parser("masters", help="собрать мастер-кадры bias/dark/flat")
    s.add_argument("--bias-frames", nargs="+")
    s.add_argument("--dark-frames", nargs="+")
    s.add_argument("--flat-frames", nargs="+")
    s.add_argument("--out", default="masters")
    s.set_defaults(func=cmd_masters)

    s = sub.add_parser("process", parents=[common], help="полная обработка: звёзды, координаты, величины")
    s.add_argument("images", nargs="+")
    s.add_argument("--out", default="out")
    s.set_defaults(func=cmd_process)

    s = sub.add_parser("stack", parents=[common], help="совместить и сложить серию кадров")
    s.add_argument("images", nargs="+")
    s.add_argument("--out", default="stack.fits")
    s.set_defaults(func=cmd_stack)

    s = sub.add_parser("moving", parents=[common], help="найти движущиеся объекты в серии кадров")
    s.add_argument("images", nargs="+")
    s.add_argument("--out", default="out")
    s.add_argument("--interval", type=float, help="секунд между кадрами, если времени нет в заголовках")
    s.add_argument("--tol", type=float, default=1.5, help="допуск отклонения от прямой, пикселей")
    s.add_argument("--max-speed", type=float, help="максимальная скорость, пикселей в секунду")
    s.set_defaults(func=cmd_moving)
    return p


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    args = build_parser().parse_args(argv)
    import glob

    for name in ("images", "bias_frames", "dark_frames", "flat_frames"):
        if getattr(args, name, None):  # PowerShell и cmd не раскрывают шаблоны вида *.fits
            expanded = []
            for item in getattr(args, name):
                hits = sorted(glob.glob(item)) if any(c in item for c in "*?[") else [item]
                expanded.extend(hits or [item])
            setattr(args, name, expanded)
    try:
        return args.func(args)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Ошибка: {exc}")
        return 1
