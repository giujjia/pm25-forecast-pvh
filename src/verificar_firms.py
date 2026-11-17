"""Checks the FIRMS files: detections per satellite, days covered, crop and null FRP. Read-only.

Reads the annual files (<prefix>_<year>.parquet) and the monthly ones (<prefix>_<YYYYMM>_sp|nrt.parquet).
Blind-test year (TESTE_CEGO, METHODS.md §6): coverage only (satellites present, first and last
date, detections inside the requested crop and null FRP), with no counts of detections or days.

Usage:
    python src/verificar_firms.py firms1000   # 1000 km crop
    python src/verificar_firms.py firms       # 300 km crop
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "ingest"))
from firms import AREAS, TESTE_CEGO  # noqa: E402

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "firms"
prefixo = sys.argv[1] if len(sys.argv) > 1 else "firms1000"
oeste, sul, leste, norte = (float(x) for x in {p: a for a, p in AREAS.values()}[prefixo].split(","))
for arq in sorted(RAW.glob(f"{prefixo}_*.parquet")):
    ano = int(arq.stem.split("_")[1][:4])
    df = pd.read_parquet(arq)
    d = pd.to_datetime(df.acq_date)
    excesso = pd.concat([sul - df.latitude, df.latitude - norte,
                         oeste - df.longitude, df.longitude - leste], axis=1).max(axis=1)
    fora = int((excesso > 0).sum())
    dentro = "sim" if fora == 0 else f"NÃO ({fora} focos, até {excesso.max():.4f}° além da borda)"
    versoes = sorted(map(str, df["version"].dropna().unique())) if "version" in df else []
    if ano in TESTE_CEGO:
        print(f"{arq.name}: satélites {sorted(map(str, df.satellite.unique()))}; "
              f"datas {d.min().date()} a {d.max().date()}; dentro do recorte {'sim' if fora == 0 else 'NÃO'}; "
              f"FRP nulo {int(df.frp.isna().sum())}; versões {versoes} (ano do teste cego: contagens omitidas)")
        continue
    sat = {k: int(v) for k, v in df.satellite.value_counts().items()}
    print(f"{arq.name}: {len(df)} focos; satélites {sat}; {d.dt.date.nunique()} dias ({d.min().date()} a {d.max().date()}); "
          f"lat {df.latitude.min():.2f} a {df.latitude.max():.2f}; lon {df.longitude.min():.2f} a {df.longitude.max():.2f}; "
          f"dentro do recorte {dentro}; FRP nulo {int(df.frp.isna().sum())}; versões {versoes}")
