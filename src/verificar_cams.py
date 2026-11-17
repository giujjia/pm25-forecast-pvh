"""Checks the content of the downloaded CAMS forecast files against what was requested.

For each cams_pm25_<YYYYMM-MM>.nc (blocks of src/ingest/cams.py): variables and
dimensions, runs (reference dates) × expected, run hour (12 UTC),
lead times × 0–96 h, grid (latitudes and longitudes, within the requested area),
missing values and the range of PM2.5 values in µg/m³.

With eac4, checks the reanalysis (data/raw/cams_eac4/, BLOCOS_EAC4 blocks of cams.py) by
coverage only: 3-hourly times × expected, grid and missing values. Shows no values, because
the 2024 PM2.5 is only opened after the freeze (METHODS.md §8).

Usage:
    python src/verificar_cams.py
    python src/verificar_cams.py eac4
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent / "ingest"))
from cams import BLOCOS, ANTECEDENCIAS, BLOCOS_EAC4, RAW_EAC4  # noqa: E402
from era5 import AREA  # noqa: E402  [N, W, S, E]

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "cams"


def achar(ds, *nomes):
    for n in nomes:
        if n in ds.coords or n in ds.dims:
            return n
    return None


def main():
    problemas = 0
    for tag, ini, fim in BLOCOS:
        p = RAW / f"cams_pm25_{tag}.nc"
        print(f"== {p.name}")
        if not p.exists():
            print("  FALTA o arquivo")
            problemas += 1
            continue
        ds = xr.open_dataset(p)
        print(f"  variáveis: {list(ds.data_vars)}")
        print(f"  dimensões: {dict(ds.sizes)}")
        var = next((v for v in ds.data_vars if "pm2p5" in v or "2.5" in v), None)
        if var is None:
            print("  PROBLEMA: variável de PM2.5 não encontrada")
            problemas += 1
            continue
        da = ds[var]
        print(f"  {var}: unidade={da.attrs.get('units')!r}, dims={da.dims}")

    # Runs
        frt = achar(ds, "forecast_reference_time", "time")
        rod = pd.to_datetime(ds[frt].values).sort_values()
        esperadas = pd.date_range(f"{ini} 12:00", f"{fim} 12:00", freq="D")
        faltam = esperadas.difference(rod)
        sobram = rod.difference(esperadas)
        ok = len(faltam) == 0 and len(sobram) == 0 and rod.is_unique
        problemas += not ok
        print(f"  rodadas: {len(rod)} de {len(esperadas)} ({rod.min()} a {rod.max()}), "
              f"horas UTC {sorted(set(rod.hour))}, faltam {len(faltam)}, sobram {len(sobram)}, "
              f"repetidas {int(rod.duplicated().sum())} -> {'OK' if ok else 'PROBLEMA'}")

    # Lead times
        fp = achar(ds, "forecast_period", "step", "leadtime_hour")
        lead = ds[fp].values
        if np.issubdtype(lead.dtype, np.timedelta64):
            lead = (lead / np.timedelta64(1, "h")).astype(int)
        lead = sorted(int(x) for x in lead)
        ok = lead == [int(h) for h in ANTECEDENCIAS]
        problemas += not ok
        print(f"  antecedências: {len(lead)} ({lead[0]}–{lead[-1]} h) -> {'OK' if ok else 'PROBLEMA'}")

    # Grid
        lat, lon = ds[achar(ds, "latitude", "lat")].values, ds[achar(ds, "longitude", "lon")].values
        lon = np.where(lon > 180, lon - 360, lon)
        n, w, s, e = AREA
        dentro = lat.min() >= s - 1e-6 and lat.max() <= n + 1e-6 and lon.min() >= w - 1e-6 and lon.max() <= e + 1e-6
        problemas += not dentro
        print(f"  grade: {len(lat)} lat ({lat.min()} a {lat.max()}) × {len(lon)} lon "
              f"({lon.min()} a {lon.max()}), dentro da área {AREA}: {'OK' if dentro else 'PROBLEMA'}")

    # Values
        v = da.values.astype("float64")
        nan = int(np.isnan(v).sum())
        problemas += nan > 0
        fator = 1e9 if str(da.attrs.get("units", "")).replace(" ", "") in ("kgm**-3", "kgm-3", "kg/m3") else 1.0
        ug = v * fator
        print(f"  ausentes: {nan} de {v.size} -> {'OK' if nan == 0 else 'PROBLEMA'}")
        print(f"  PM2.5 (µg/m³, fator {fator:g}): mín {np.nanmin(ug):.1f}, mediana {np.nanmedian(ug):.1f}, "
              f"p99 {np.nanpercentile(ug, 99):.1f}, máx {np.nanmax(ug):.1f}")
        ds.close()

    print(f"\nproblemas: {problemas}")
    sys.exit(1 if problemas else 0)


def eac4():
    """Coverage of the EAC4 files, without values."""
    problemas = 0
    for tag, ini, fim in BLOCOS_EAC4:
        p = RAW_EAC4 / f"cams_eac4_pm25_{tag}.nc"
        print(f"== {p.name}")
        if not p.exists():
            print("  FALTA o arquivo")
            problemas += 1
            continue
        ds = xr.open_dataset(p)
        var = next((v for v in ds.data_vars if "pm2p5" in v or "2.5" in v), None)
        print(f"  variáveis: {list(ds.data_vars)}; dimensões: {dict(ds.sizes)}")
        if var is None:
            print("  PROBLEMA: variável de PM2.5 não encontrada")
            problemas += 1
            continue
        da = ds[var]
        print(f"  {var}: unidade={da.attrs.get('units')!r}, dims={da.dims}")
        t = pd.to_datetime(ds[achar(ds, "valid_time", "time")].values).sort_values()
        esperadas = pd.date_range(f"{ini} 00:00", f"{fim} 21:00", freq="3h")
        faltam, sobram = esperadas.difference(t), t.difference(esperadas)
        ok = len(faltam) == 0 and len(sobram) == 0 and t.is_unique
        problemas += not ok
        print(f"  horários: {len(t)} de {len(esperadas)} ({t.min()} a {t.max()}), faltam {len(faltam)}, "
              f"sobram {len(sobram)}, repetidos {int(t.duplicated().sum())} -> {'OK' if ok else 'PROBLEMA'}")
        lat, lon = ds[achar(ds, "latitude", "lat")].values, ds[achar(ds, "longitude", "lon")].values
        lon = np.where(lon > 180, lon - 360, lon)
        n, w, s, e = AREA
        dentro = lat.min() >= s - 1e-6 and lat.max() <= n + 1e-6 and lon.min() >= w - 1e-6 and lon.max() <= e + 1e-6
        problemas += not dentro
        print(f"  grade: {len(lat)} lat ({lat.min()} a {lat.max()}) × {len(lon)} lon "
              f"({lon.min()} a {lon.max()}), dentro da área {AREA}: {'OK' if dentro else 'PROBLEMA'}")
        nan = int(np.isnan(da.values).sum())
        problemas += nan > 0
        print(f"  ausentes: {nan} de {da.size} -> {'OK' if nan == 0 else 'PROBLEMA'} (valores não mostrados)")
        ds.close()
    print(f"\nproblemas: {problemas}")
    sys.exit(1 if problemas else 0)


if __name__ == "__main__":
    eac4() if sys.argv[1:] == ["eac4"] else main()
