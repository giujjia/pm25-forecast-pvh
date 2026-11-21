"""Ingestion of GOES-16/19 ABI-L2-AODF (AOD, full disk) cropped to the ERA5 area.

Author's decision (16/11/2026): keep the 6 images of each daytime hour (every 10 min), and
leave the choice between the top-of-the-hour image and the hour mean to the modelling.

Source: public AWS buckets, no account. GOES-16 until 04/04/2025 14:59 UTC and GOES-19 from
04/04/2025 15:00 UTC, when GOES-19 became the operational GOES-East.
Hours: 10 to 22 UTC (06 to 18 h in Porto Velho), the hours with sun over the area.

Partial read: the remote netCDF4 is opened with h5py over byte ranges
(faixa.py) with a block cache; only the AOD and DQF crop is read. The rows and columns
of the crop come from the file's projection (GOES-R PUG formulas) and are checked against the
x/y coordinates of each file.

Output: one file per day, data/raw/goes/{year}/goes_aod_{YYYYMMDD}.nc, with AOD (float32, NaN
where the file has no value) and DQF (uint8, 255 where it has none) of each image, the scan
start time, the satellite, the source file name and the latitude/longitude of each
pixel. 2026 AOD goes to data/raw/goes/selado_2026/ (blind test). Manifest in
data/raw/goes/manifesto_goes.csv. Days already written are skipped. No value is printed.

Usage:
    python src/ingest/goes.py teste 2023-08-15 15 [block_kb]   # one hour, measures bytes
    python src/ingest/goes.py baixar 2020-01-01 2026-10-31 [workers]
"""
import csv
import hashlib
import math
import multiprocessing as mp
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import h5py
import numpy as np
import requests
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from faixa import ArquivoRemoto, sessao_com_retentativas  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from teste_goes import Blocos, latlon_para_xy  # noqa: E402

RAW = Path(__file__).resolve().parents[2] / "data" / "raw" / "goes"
MANIFESTO = RAW / "manifesto_goes.csv"
AREA = (-8.0, -65.0, -10.5, -62.5)  # N, W, S, E (same as ERA5)
HORAS = range(10, 23)
TROCA = datetime(2025, 4, 4, 15, tzinfo=timezone.utc)
TESTE_CEGO = 2026
BUCKET = "https://noaa-{sat}.s3.amazonaws.com"
_trava = threading.Lock()


def satelite(dt):
    return "goes16" if dt < TROCA else "goes19"


def listar(s, sat, d, hora):
    url = BUCKET.format(sat=sat) + f"/?list-type=2&prefix=ABI-L2-AODF/{d.year}/{d.timetuple().tm_yday:03d}/{hora:02d}/"
    t = s.get(url, timeout=60)
    t.raise_for_status()
    return [k.split("</Key>")[0] for k in t.text.split("<Key>")[1:]]


def inicio(chave):
    s = chave.split("_s")[1][:13]  # sYYYYDDDHHMMSS (tenth of a second discarded)
    return datetime.strptime(s, "%Y%j%H%M%S").replace(tzinfo=timezone.utc)


def escala(ds, fatia=()):
    v = ds[fatia].astype("f8") if fatia else ds[()].astype("f8")
    return v * ds.attrs.get("scale_factor", 1.0) + ds.attrs.get("add_offset", 0.0)


def janela(f):
    """Rows and columns of the crop and the lat/lon of each pixel, from the projection."""
    proj = {k: float(np.ravel(v)[0]) for k, v in f["goes_imager_projection"].attrs.items()
            if isinstance(v, np.ndarray) and v.dtype.kind == "f"}
    x, y = escala(f["x"]), escala(f["y"])
    cantos = [latlon_para_xy(la, lo, proj) for la in (AREA[0], AREA[2]) for lo in (AREA[1], AREA[3])]
    xs, ys = [c[0] for c in cantos], [c[1] for c in cantos]
    c0, c1 = int(np.searchsorted(x, min(xs))), int(np.searchsorted(x, max(xs)))
    yd = y[::-1]
    l1 = len(y) - int(np.searchsorted(yd, min(ys)))
    l0 = len(y) - int(np.searchsorted(yd, max(ys)))
    lat, lon = xy_para_latlon(*np.meshgrid(x[c0:c1], y[l0:l1]), proj)
    return (l0, l1, c0, c1), x[c0:c1], y[l0:l1], lat, lon


def xy_para_latlon(x, y, p):
    """Inverse of the GOES-R PUG (vol. 4, §4.2.8.1)."""
    req, rpol = p["semi_major_axis"], p["semi_minor_axis"]
    H = p["perspective_point_height"] + req
    l0 = math.radians(p["longitude_of_projection_origin"])
    a = np.sin(x) ** 2 + np.cos(x) ** 2 * (np.cos(y) ** 2 + (req ** 2 / rpol ** 2) * np.sin(y) ** 2)
    b = -2 * H * np.cos(x) * np.cos(y)
    c = H ** 2 - req ** 2
    rs = (-b - np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
    sx, sy, sz = rs * np.cos(x) * np.cos(y), -rs * np.sin(x), rs * np.cos(x) * np.sin(y)
    lat = np.degrees(np.arctan((req ** 2 / rpol ** 2) * sz / np.sqrt((H - sx) ** 2 + sy ** 2)))
    lon = np.degrees(l0 - np.arctan(sy / (H - sx)))
    return lat.astype("f4"), lon.astype("f4")


class Janelas:
    """Keeps each satellite's window; each file checks the crop coordinates."""

    def __init__(self):
        self.ref = {}

    def ler(self, f, sat):
        if sat not in self.ref:
            with _trava:
                if sat not in self.ref:
                    self.ref[sat] = janela(f)
        (l0, l1, c0, c1), xr_, yr_, lat, lon = self.ref[sat]
        if not (np.allclose(escala(f["x"], slice(c0, c1)), xr_) and
                np.allclose(escala(f["y"], slice(l0, l1)), yr_)):
            raise RuntimeError("coordenadas x/y do recorte mudaram")
        return (l0, l1, c0, c1), lat, lon


def sem_sinal(ds, fatia):
    """Reads the slice and the _FillValue honoring the _Unsigned attribute of the netCDF convention.

    Up to 2020 (at least), AODF stores AOD as int16 and DQF as int8 with
    _Unsigned = "true" and _FillValue −1. Read as they are, AOD above ~2.5 (dense smoke)
    would turn negative. From 2021 the same fields come as uint16/uint8 (finding of
    16/11/2026).
    """
    bruto = ds[fatia]
    fill = np.asarray(ds.attrs.get("_FillValue", np.array([-1])), dtype=ds.dtype).ravel()[:1]
    uns = ds.attrs.get("_Unsigned", b"false")
    if (uns.decode() if isinstance(uns, bytes) else str(uns)).lower() == "true" and ds.dtype.kind == "i":
        sem = np.dtype(f"u{ds.dtype.itemsize}")
        bruto, fill = bruto.view(sem), fill.view(sem)
    return bruto, fill[0]


def ler_imagem(url, janelas, sat, bloco, sessao=None):
    cru = ArquivoRemoto(url, sessao)
    with h5py.File(Blocos(cru, bloco), "r") as f:
        (l0, l1, c0, c1), lat, lon = janelas.ler(f, sat)
        fatia = (slice(l0, l1), slice(c0, c1))
        d = f["AOD"]
        bruto, fill = sem_sinal(d, fatia)
        aod = np.where(bruto == fill, np.nan, bruto * d.attrs.get("scale_factor", 1.0)
                       + d.attrs.get("add_offset", 0.0)).astype("f4")
        q = f["DQF"]
        dqf_b, qfill = sem_sinal(q, fatia)
        dqf = np.where(dqf_b == qfill, 255, dqf_b).astype("u1")
    return aod, dqf, lat, lon, cru.transferidos, cru.pedidos


_janelas_proc = Janelas()
_sessao_proc = None


def _gravar_linha_manifesto(linha):
    with _trava:
        novo = not MANIFESTO.exists()
        with open(MANIFESTO, "a", newline="") as f:
            w = csv.writer(f)
            if novo:
                w.writerow(["dia", "imagens", "satelites", "horas_sem_arquivo", "bytes_transferidos",
                            "segundos", "arquivo", "sha256"])
            w.writerow(linha)


def dia(d, janelas=None, bloco=32 << 10, gravar_manifesto=True):
    global _sessao_proc
    if _sessao_proc is None:
        _sessao_proc = sessao_com_retentativas()
    s = _sessao_proc
    janelas = janelas or _janelas_proc
    t0 = time.perf_counter()
    imagens, total, horas_vazias = [], 0, []
    for h in HORAS:
        momento = datetime(d.year, d.month, d.day, h, tzinfo=timezone.utc)
        sat = satelite(momento)
        chaves = listar(s, sat, d, h)
        if not chaves:
            horas_vazias.append(h)
        for k in chaves:
            aod, dqf, lat, lon, n, _ = ler_imagem(BUCKET.format(sat=sat) + "/" + k, janelas, sat, bloco, s)
            total += n
            imagens.append((inicio(k), sat, k.split("/")[-1], aod, dqf))
    if not imagens:
        raise RuntimeError("nenhuma imagem no dia")
    imagens.sort(key=lambda x: x[0])
    sat0 = imagens[0][1]
    _, lat, lon = janelas.ref[sat0][0], janelas.ref[sat0][3], janelas.ref[sat0][4]
    tempos = np.array([i[0].replace(tzinfo=None) for i in imagens], dtype="datetime64[s]")
    ds = xr.Dataset(
        {"aod": (("imagem", "linha", "coluna"), np.stack([i[3] for i in imagens]), {"units": "1"}),
         "dqf": (("imagem", "linha", "coluna"), np.stack([i[4] for i in imagens]),
                 {"descricao": "DQF do produto (0 alta, 1 média, 2 baixa, 3 sem retorno); 255 = sem valor"}),
         "satelite": (("imagem",), np.array([i[1] for i in imagens], dtype="U6")),
         "arquivo": (("imagem",), np.array([i[2] for i in imagens], dtype="U80")),
         "lat": (("linha", "coluna"), lat), "lon": (("linha", "coluna"), lon)},
        coords={"inicio_varredura": (("imagem",), tempos)},
        attrs={"dia": d.isoformat(), "horas_utc": f"{HORAS.start}-{HORAS.stop - 1}",
               "horas_sem_arquivo": ",".join(map(str, horas_vazias)) or "nenhuma",
               "gerado_por": "src/ingest/goes.py"},
    )
    destino = alvo(d)
    destino.parent.mkdir(parents=True, exist_ok=True)
    parcial = destino.with_suffix(".nc.part")
    ds.to_netcdf(parcial, engine="h5netcdf", encoding={"aod": {"zlib": True, "complevel": 4},
                                                       "dqf": {"zlib": True, "complevel": 4}})
    parcial.rename(destino)
    sha = hashlib.sha256(destino.read_bytes()).hexdigest()
    linha = [d.isoformat(), len(imagens), " ".join(sorted({i[1] for i in imagens})),
             " ".join(map(str, horas_vazias)), total, f"{time.perf_counter() - t0:.0f}",
             str(destino.relative_to(RAW.parent)), sha]
    if gravar_manifesto:
        _gravar_linha_manifesto(linha)
        return d, len(imagens), horas_vazias, total
    return d, len(imagens), horas_vazias, total, linha


def alvo(d):
    pasta = RAW / "selado_2026" if d.year == TESTE_CEGO else RAW / f"{d.year}"
    return pasta / f"goes_aod_{d:%Y%m%d}.nc"


def teste(dia_iso, hora, bloco_kb=256):
    d = date.fromisoformat(dia_iso)
    s = requests.Session()
    s.headers["User-Agent"] = "Mozilla/5.0 (X11; Linux x86_64)"
    sat = satelite(datetime(d.year, d.month, d.day, int(hora), tzinfo=timezone.utc))
    chaves = listar(s, sat, d, int(hora))
    janelas = Janelas()
    for k in chaves:
        t0 = time.perf_counter()
        aod, dqf, lat, lon, n, p = ler_imagem(BUCKET.format(sat=sat) + "/" + k, janelas, sat, int(bloco_kb) << 10)
        fora = int(((aod < -0.06) | (aod > 5.1)).sum())  # outside the product range (valid_range)
        print(f"{k.split('/')[-1][:60]}: recorte {aod.shape}, {int(np.isfinite(aod).sum())} pixels com AOD, "
              f"{fora} fora da faixa, DQF sem valor onde não há AOD: {bool(((dqf == 255) == ~np.isfinite(aod)).all())}, "
              f"{n / 1e6:.2f} MB em {p} pedidos, {time.perf_counter() - t0:.1f} s", flush=True)
    print(f"lat {float(lat.min()):.2f} a {float(lat.max()):.2f}; lon {float(lon.min()):.2f} a {float(lon.max()):.2f}")


def baixar(ini, fim, trabalhadores, bloco=32 << 10):  # 32 KB: ~0.5 MB per image
    a, b = date.fromisoformat(ini), date.fromisoformat(fim)
    faltam = [a + timedelta(days=i) for i in range((b - a).days + 1) if not alvo(a + timedelta(days=i)).exists()]
    print(f"{len(faltam)} dias a baixar de {ini} a {fim}, {trabalhadores} processos em paralelo", flush=True)
    t0, feitos, total, erros = time.perf_counter(), 0, 0, 0
    ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else None
    with ProcessPoolExecutor(trabalhadores, mp_context=ctx) as ex:
        futuros = {ex.submit(dia, d, None, bloco, False): d for d in faltam}
        for fu in as_completed(futuros):
            d = futuros[fu]
            try:
                _, n_img, vazias, n, linha = fu.result()
            except Exception as e:
                erros += 1
                print(f"{d}: ERRO {str(e)[:300]}", flush=True)
                continue
            _gravar_linha_manifesto(linha)
            feitos += 1
            total += n
            if vazias or feitos <= 5 or feitos % 5 == 0 or feitos == len(faltam):
                print(f"{d}: {n_img} imagens{' (horas sem arquivo: ' + str(vazias) + ')' if vazias else ''} | "
                      f"{feitos}/{len(faltam)} em {(time.perf_counter() - t0) / 60:.1f} min, "
                      f"{total / 1e9:.1f} GB", flush=True)
    print(f"fim: {feitos} dias gravados, {erros} com erro (rodar de novo retoma só os que faltam)", flush=True)
    if erros:
        sys.exit(1)


if __name__ == "__main__":
    if sys.argv[1] == "teste":
        teste(*sys.argv[2:])
    else:
        baixar(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 4)
