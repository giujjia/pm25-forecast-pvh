"""Preliminary annual means per sensor and correction (daily means from the API).

Approximation: the corrections were fitted on hourly means and are applied here
to daily means, with no daily completeness filter.

Usage: python src/analise_preliminar.py 39077 56831
"""
import sys
from pathlib import Path

import pandas as pd

from correcao import corrigir

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "purpleair"
CORRECOES = ["cf1", "barkjohn2021", "barkjohn2022", "airnow_fsm"]

for sid in sys.argv[1:]:
    bruto = pd.read_parquet(RAW / f"{sid}_diario.parquet")
    df = corrigir(bruto)
    g = df.groupby(df.time_stamp.dt.year)
    tab = g[CORRECOES].mean().round(1)
    tab["dias"] = g.size()
    for c in ("barkjohn2021", "airnow_fsm"):
        tab[f"dias>50_{c}"] = g[c].apply(lambda s: (s > 50).sum())
        tab[f"dias>=125_{c}"] = g[c].apply(lambda s: (s >= 125).sum())
    print(f"\nSensor {sid} — QC removeu {len(bruto) - len(df)} dias")
    print(tab[tab.dias > 200].to_string())
