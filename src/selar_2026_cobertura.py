"""Erases from data/raw/purpleair/cobertura.csv the daily maximum of cf_1 for the blind-test year (2026).

Day counts are kept. The erased value is not printed. Since 15/11/2026,
purpleair.py cobertura no longer computes that maximum for 2026 either.
"""
from pathlib import Path

import pandas as pd

ARQ = Path(__file__).resolve().parents[1] / "data" / "raw" / "purpleair" / "cobertura.csv"
tab = pd.read_csv(ARQ)
m = tab["ano"] >= 2026
tab["pm25_cf1_max"] = tab["pm25_cf1_max"].astype("object")
tab.loc[m, "pm25_cf1_max"] = ""
tab.to_csv(ARQ, index=False)
print(f"{int(m.sum())} linha(s) de 2026: máximo apagado; contagens de dias mantidas.")
