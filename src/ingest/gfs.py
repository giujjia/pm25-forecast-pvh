"""Ingestion of the archived GFS: 0.25° forecasts cropped to the ERA5 area.

Author's decision (16/11/2026): 3 h step and one run per day, the 18 UTC one, from f000 to
f060 (21 steps). It is the run used at the 04 UTC issuance of the next day (METHODS.md, availability
table). Runs from 31/12/2019 to 31/10/2026.

Two routes, the same output:
  aws:  noaa-gfs-bdp-pds, from 01/01/2021 onwards. Reads the .idx index of each step,
        downloads only the GRIB2 messages of the fields by byte ranges, decodes with eccodes and
        keeps the crop. Before 22/03/2021 the files are outside the atmos/ folder.
  ncar: NSF NCAR GDEX d084001, for 2020 (AWS starts in 2021). NCAR's THREDDS
        crops on the server (NCSS), so only the 11 × 11 points arrive. Rain comes with
        mixed intervals and requires temporal=all.

Fields (METHODS.md §2, plus 925 hPa temperature): HPBL, 10 m wind, 2 m T and Td, accumulated
rain, 850 hPa wind and 925 hPa T. Rain comes out in two variables: tp_0a, accumulated
since the start of the run, and tp_balde, the interval GFS publishes at the step (with the start
in balde_inicio); at f003 and f006 the two are the same field.

Output: data/raw/gfs/{year}/gfs_{YYYYMMDD}18.nc, one file per run, with dimensions
(step, lat, lon); steps missing in the source file are NaN and are listed in the
passos_ausentes attribute and in the manifest data/raw/gfs/manifesto_gfs.csv. Runs already written
are skipped. No value is printed.

Usage:
    python src/ingest/gfs.py aws 2021-01-01 2026-10-31 [workers]
    python src/ingest/gfs.py ncar 2019-12-31 2020-12-31 [workers]
"""
import csv
import hashlib
import io
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import eccodes
import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from faixa import sessao_com_retentativas  # noqa: E402

RAW = Path(__file__).resolve().parents[2] / "data" / "raw" / "gfs"
MANIFESTO = RAW / "manifesto_gfs.csv"
HORA = "18"
PASSOS = list(range(0, 61, 3))
LATS = np.round(np.arange(-8.0, -10.51, -0.25), 2)  # 11 points, north to south
LONS = np.round(np.arange(295.0, 297.51, 0.25), 2)  # -65 to -62.5 in 0–360
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"}
AWS = "https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{d}/{h}/{sub}gfs.t{h}z.pgrb2.0p25.f{f:03d}"
NCSS = ("https://thredds.rda.ucar.edu/thredds/ncss/grid/files/d084001/{a}/{d}/"
        "gfs.0p25.{d}{h}.f{f:03d}.grib2")

# name in the output -> key in the AWS .idx
IDX = {
    "hpbl": "HPBL:surface",
    "u10": "UGRD:10 m above ground",
    "v10": "VGRD:10 m above ground",
    "t2m": "TMP:2 m above ground",
    "d2m": "DPT:2 m above ground",
    "u850": "UGRD:850 mb",
    "v850": "VGRD:850 mb",
    "t925": "TMP:925 mb",
}
UNIDADES = {"hpbl": "m", "u10": "m s-1", "v10": "m s-1", "t2m": "K", "d2m": "K", "u850": "m s-1",
            "v850": "m s-1", "t925": "K", "tp_0a": "kg m-2", "tp_balde": "kg m-2"}
NOMES = list(UNIDADES)

_trava = threading.Lock()


def dias(ini, fim):
    a, b = date.fromisoformat(ini), date.fromisoformat(fim)
    return [a + timedelta(days=i) for i in range((b - a).days + 1)]


def alvo(d):
    return RAW / f"{d.year}" / f"gfs_{d:%Y%m%d}{HORA}.nc"


def vazio():
    campos = {n: np.full((len(PASSOS), len(LATS), len(LONS)), np.nan, dtype="f4") for n in NOMES}
    campos["balde_inicio"] = np.full(len(PASSOS), np.nan, dtype="f4")
    return campos


def gravar(d, campos, fonte, ausentes, transferidos, segundos):
    ds = xr.Dataset(
        {n: (("passo", "lat", "lon"), campos[n], {"units": UNIDADES[n]}) for n in NOMES}
        | {"balde_inicio": (("passo",), campos["balde_inicio"], {"units": "h"})},
        coords={"passo": ("passo", np.array(PASSOS, dtype="i2"), {"units": "h"}),
                "lat": ("lat", LATS), "lon": ("lon", LONS - 360.0)},
        attrs={"rodada": f"{d:%Y-%m-%d}T{HORA}:00Z", "fonte": fonte,
               "passos_ausentes": ",".join(map(str, ausentes)) or "nenhum",
               "gerado_por": "src/ingest/gfs.py"},
    )
    destino = alvo(d)
    destino.parent.mkdir(parents=True, exist_ok=True)
    parcial = destino.with_suffix(".nc.part")
    ds.to_netcdf(parcial, engine="h5netcdf",
                 encoding={n: {"zlib": True, "complevel": 4} for n in NOMES})
    parcial.rename(destino)
    sha = hashlib.sha256(destino.read_bytes()).hexdigest()
    with _trava:
        novo = not MANIFESTO.exists()
        with open(MANIFESTO, "a", newline="") as f:
            w = csv.writer(f)
            if novo:
                w.writerow(["rodada", "fonte", "passos_ok", "passos_ausentes", "bytes_transferidos",
                            "segundos", "arquivo", "sha256"])
            w.writerow([f"{d:%Y-%m-%d}T{HORA}Z", fonte, len(PASSOS) - len(ausentes),
                        " ".join(map(str, ausentes)), transferidos, f"{segundos:.0f}",
                        str(destino.relative_to(RAW.parent)), sha])


def recorte_grib(msg):
    h = eccodes.codes_new_from_message(msg)
    try:
        g = {k: eccodes.codes_get(h, k) for k in ("Ni", "Nj", "latitudeOfFirstGridPointInDegrees",
                                                 "longitudeOfFirstGridPointInDegrees",
                                                 "iDirectionIncrementInDegrees")}
        assert (g["Ni"], g["Nj"], g["latitudeOfFirstGridPointInDegrees"],
                g["longitudeOfFirstGridPointInDegrees"], g["iDirectionIncrementInDegrees"]) == \
            (1440, 721, 90.0, 0.0, 0.25), g
        v = eccodes.codes_get_values(h).reshape(721, 1440)
    finally:
        eccodes.codes_release(h)
    li = np.rint((90.0 - LATS) / 0.25).astype(int)
    lj = np.rint(LONS / 0.25).astype(int)
    return v[np.ix_(li, lj)].astype("f4")


# ---------------------------------------------------------------- AWS

def _aws_passo(s, d, f, sub):
    url = AWS.format(d=f"{d:%Y%m%d}", h=HORA, sub=sub, f=f)
    r = s.get(url + ".idx", timeout=60)
    if r.status_code == 404:
        return None, 0
    r.raise_for_status()
    linhas = [ln.split(":") for ln in r.text.strip().splitlines()]
    out, n = {}, len(r.content)
    for i, c in enumerate(linhas):
        chave = f"{c[3]}:{c[4]}"
        nome = next((k for k, v in IDX.items() if v == chave), None)
        apcp = chave == "APCP:surface"
        if nome is None and not apcp:
            continue
        ini = int(c[1])
        faixa = f"bytes={ini}-{int(linhas[i + 1][1]) - 1}" if i + 1 < len(linhas) else f"bytes={ini}-"
        b = s.get(url, headers={"Range": faixa}, timeout=120)
        assert b.status_code == 206 and b.content[:4] == b"GRIB", (url, faixa, b.status_code)
        n += len(b.content)
        grade = recorte_grib(b.content)
        if apcp:  # "0-9 hour acc fcst" or "6-9 hour acc fcst"
            a0 = int(c[5].split("-")[0])
            if a0 == 0:
                out["tp_0a"] = grade
            if a0 != 0 or "tp_balde" not in out:
                out["tp_balde"], out["balde_inicio"] = grade, a0
        else:
            out[nome] = grade
    return out, n


def ciclo_aws(d):
    s = sessao_com_retentativas()
    t0 = time.perf_counter()
    sub = "atmos/" if s.head(AWS.format(d=f"{d:%Y%m%d}", h=HORA, sub="atmos/", f=0) + ".idx",
                             timeout=60).status_code == 200 else ""
    campos, ausentes, total = vazio(), [], 0
    for k, f in enumerate(PASSOS):
        out, n = _aws_passo(s, d, f, sub)
        total += n
        if out is None:
            ausentes.append(f)
            continue
        for nome, v in out.items():
            campos[nome][k] = v
    if len(ausentes) == len(PASSOS):
        raise RuntimeError("nenhum passo da rodada existe na AWS")
    gravar(d, campos, "aws:noaa-gfs-bdp-pds", ausentes, total, time.perf_counter() - t0)
    return d, ausentes, total


# ---------------------------------------------------------------- NCAR (NCSS)

VARS_NCSS = ["Planetary_Boundary_Layer_Height_surface", "u-component_of_wind_height_above_ground",
             "v-component_of_wind_height_above_ground", "Temperature_height_above_ground",
             "Dewpoint_temperature_height_above_ground", "Temperature_isobaric",
             "u-component_of_wind_isobaric", "v-component_of_wind_isobaric"]


def _nivel(da, valor):
    dim = next(x for x in da.dims if x.startswith(("height_above_ground", "isobaric")))
    return da.sel({dim: valor}).squeeze().values


def _ncar_passo(s, d, f):
    base = NCSS.format(a=d.year, d=f"{d:%Y%m%d}", h=HORA, f=f)
    meta = s.get(base + "/dataset.xml", timeout=300)
    if meta.status_code == 404:
        return None, 0
    meta.raise_for_status()
    chuva = [v for v in ("Total_precipitation_surface_3_Hour_Accumulation",
                         "Total_precipitation_surface_6_Hour_Accumulation",
                         "Total_precipitation_surface_Mixed_intervals_Accumulation")
             if f'name="{v}"' in meta.text]
    params = {"var": ",".join(VARS_NCSS + chuva), "north": -8, "south": -10.5, "west": -65,
              "east": -62.5, "accept": "netcdf", "temporal": "all"}
    for tentativa in range(5):
        r = s.get(base, params=params, timeout=600)
        if r.status_code == 200 and r.content[:3] == b"CDF":
            break
        time.sleep(30 * (tentativa + 1))
    else:
        raise RuntimeError(f"{base}: NCSS respondeu {r.status_code}: {r.text[:200]}")
    ds = xr.open_dataset(io.BytesIO(r.content), engine="scipy", decode_times=False)
    assert np.allclose(ds["latitude"].values, LATS) and np.allclose(ds["longitude"].values, LONS)
    out = {
        "hpbl": ds["Planetary_Boundary_Layer_Height_surface"].squeeze().values,
        "u10": _nivel(ds["u-component_of_wind_height_above_ground"], 10.0),
        "v10": _nivel(ds["v-component_of_wind_height_above_ground"], 10.0),
        "t2m": _nivel(ds["Temperature_height_above_ground"], 2.0),
        "d2m": _nivel(ds["Dewpoint_temperature_height_above_ground"], 2.0),
        "u850": _nivel(ds["u-component_of_wind_isobaric"], 85000.0),
        "v850": _nivel(ds["v-component_of_wind_isobaric"], 85000.0),
        "t925": _nivel(ds["Temperature_isobaric"], 92500.0),
    }
    if chuva:
        tp = ds[chuva[0]]
        lim = ds[tp.dims[0] + "_bounds"].values if tp.dims[0] + "_bounds" in ds else ds["time_bounds"].values
        for i, (a0, a1) in enumerate(lim):
            assert int(a1) == f, (base, lim)
            grade = tp.isel({tp.dims[0]: i}).values
            if int(a0) == 0:
                out["tp_0a"] = grade
            if int(a0) != 0 or "tp_balde" not in out:
                out["tp_balde"], out["balde_inicio"] = grade, int(a0)
    return out, len(meta.content) + len(r.content)


def ciclo_ncar(d):
    s = sessao_com_retentativas()
    t0 = time.perf_counter()
    campos, ausentes, total = vazio(), [], 0
    for k, f in enumerate(PASSOS):
        out, n = _ncar_passo(s, d, f)
        total += n
        if out is None:
            ausentes.append(f)
            continue
        for nome, v in out.items():
            campos[nome][k] = v
    if len(ausentes) == len(PASSOS):
        raise RuntimeError("nenhum passo da rodada existe no NCAR")
    gravar(d, campos, "ncar:d084001 (THREDDS NCSS)", ausentes, total, time.perf_counter() - t0)
    return d, ausentes, total


def lote(rota, ini, fim, trabalhadores):
    func = {"aws": ciclo_aws, "ncar": ciclo_ncar}[rota]
    faltam = [d for d in dias(ini, fim) if not alvo(d).exists()]
    print(f"{rota}: {len(faltam)} rodadas a baixar de {ini} a {fim}, {trabalhadores} em paralelo",
          flush=True)
    t0, feitos, total, erros = time.perf_counter(), 0, 0, 0
    with ThreadPoolExecutor(trabalhadores) as ex:
        futuros = {ex.submit(func, d): d for d in faltam}
        for fu in as_completed(futuros):
            d = futuros[fu]
            try:
                _, ausentes, n = fu.result()
            except Exception as e:
                erros += 1
                print(f"{d}: ERRO {str(e)[:300]}", flush=True)
                continue
            feitos += 1
            total += n
            if ausentes or feitos % 25 == 0 or feitos == len(faltam):
                dt = time.perf_counter() - t0
                print(f"{d}: ok{' (passos ausentes: ' + str(ausentes) + ')' if ausentes else ''} | "
                      f"{feitos}/{len(faltam)} em {dt / 60:.0f} min, {total / 1e9:.2f} GB", flush=True)
    print(f"fim: {feitos} rodadas gravadas, {erros} com erro (rodar de novo retoma só as que faltam)",
          flush=True)
    if erros:
        sys.exit(1)


if __name__ == "__main__":
    lote(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 4)
