"""Checks the content of the downloaded ERA5 files against what was requested.

For each era5_<sl|pl>_<YYYYMM[-MM]>.nc: variables present, first and last hour,
number of hours × expected (days of the period × 24), repeated or missing hours,
grid (latitudes and longitudes) and missing values per variable.
Then, per set (sl, pl): coverage of the whole period, missing hours
and hours present in more than one file. Files with expver (recent months, which the
CDS delivers as ERA5T, preliminary) show the values found.

Usage:
    python src/verificar_era5.py                    # 2020-01 to 2024-12
    python src/verificar_era5.py 2020-01 2026-10    # coverage period
"""
import calendar
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "era5"


def periodo(args):
    """(start, end) of the coverage: two YYYY-MM arguments, or 2020-01 to 2024-12."""
    if len(args) != 2:
        return pd.Timestamp(2020, 1, 1), pd.Timestamp(2024, 12, 31, 23)
    (a1, m1), (a2, m2) = ((int(x) for x in a.split("-")) for a in args)
    return pd.Timestamp(a1, m1, 1), pd.Timestamp(a2, m2, calendar.monthrange(a2, m2)[1], 23)


PERIODO = periodo(sys.argv[1:])
ESPERADAS = {"sl": {"blh", "u10", "v10", "t2m", "d2m", "tp"}, "pl": {"u", "v"}}


def main():
    problemas = 0
    todas = {"sl": [], "pl": []}
    for p in sorted(RAW.glob("era5_*.nc")):
        chave, tag = p.stem.split("_")[1], p.stem.split("_")[2]
        ano, m1 = int(tag[:4]), int(tag[4:6])
        m2 = int(tag.split("-")[1]) if "-" in tag else m1
        ini = pd.Timestamp(ano, m1, 1)
        fim = pd.Timestamp(ano, m2, calendar.monthrange(ano, m2)[1], 23)
        esperadas_h = int((fim - ini) / pd.Timedelta(hours=1)) + 1

        ds = xr.open_dataset(p)
        tempo = "valid_time" if "valid_time" in ds.coords else "time"
        t = pd.to_datetime(ds[tempo].values)
        vars_ok = ESPERADAS[chave] <= set(ds.data_vars)
        faltam = sorted(ESPERADAS[chave] - set(ds.data_vars))
        horas = pd.date_range(ini, fim, freq="h")
        ausentes_h = len(horas.difference(t))
        repetidas = int(pd.Index(t).duplicated().sum())
        nan = {v: int(np.isnan(ds[v].values).sum()) for v in sorted(ESPERADAS[chave] & set(ds.data_vars))}
        expver = sorted({str(x) for x in np.unique(ds["expver"].values)}) if "expver" in ds.variables else []
        todas[chave].append(pd.Index(t))
        lat, lon = ds["latitude"].values, ds["longitude"].values
        ok = vars_ok and ausentes_h == 0 and repetidas == 0 and t.min() == ini and t.max() == fim
        problemas += 0 if ok else 1
        print(f"{p.name}: {'OK' if ok else 'PROBLEMA'} | horas {len(t)}/{esperadas_h} "
              f"({t.min()} → {t.max()}), faltando {ausentes_h}, repetidas {repetidas} | "
              f"variáveis {'ok' if vars_ok else 'faltam ' + str(faltam)} | "
              f"grade lat {lat.max():.2f}..{lat.min():.2f} ({len(lat)}), "
              f"lon {lon.min():.2f}..{lon.max():.2f} ({len(lon)}) | NaN {nan}"
              + (f" | expver {expver}" if expver else ""))
    print(f"\nRESUMO: {problemas} arquivo(s) com problema")

    horas = pd.date_range(*PERIODO, freq="h")
    for chave, idx in todas.items():
        t = idx[0].append(idx[1:]) if idx else pd.DatetimeIndex([])
        faltando = horas.difference(t)
        repetidas = t[t.duplicated()].unique()
        fora = t[(t < PERIODO[0]) | (t > PERIODO[1])].unique()
        ok = len(faltando) == 0 and len(repetidas) == 0
        problemas += 0 if ok else 1
        meses_falt = sorted(set(faltando.strftime("%Y-%m")))
        print(f"COBERTURA {chave}: {'OK' if ok else 'PROBLEMA'} | {len(idx)} arquivos | "
              f"{len(t.unique())}/{len(horas)} horas de {PERIODO[0]:%Y-%m-%d} a {PERIODO[1]:%Y-%m-%d} | "
              f"faltando {len(faltando)} {meses_falt if meses_falt else ''}| "
              f"em mais de um arquivo {len(repetidas)} | fora do período {len(fora)}")
    print(f"RESUMO FINAL: {problemas} problema(s)")


if __name__ == "__main__":
    main()
