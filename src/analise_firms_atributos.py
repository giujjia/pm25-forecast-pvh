"""Attributes of the downloaded FIRMS VIIRS fire detections (1000 km radius), 2020–2024.

Counts, per year: product version, type, confidence × daynight and satellite;
and, per month, the split between Suomi-NPP (N) and NOAA-20. FIRMS attributes
only; does not read PurpleAir. Supports the choice of FIRMS filters (METHODS.md §4).
"""
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
RAW = RAIZ / "data" / "raw" / "firms"
OUT = RAIZ / "data" / "processed" / "firms_atributos"
OUT.mkdir(parents=True, exist_ok=True)
COLS = ["acq_date", "satellite", "confidence", "daynight", "type", "version"]

anual, mensal = [], []
for ano in range(2020, 2025):
    arq = RAW / f"firms1000_{ano}.parquet"
    df = pd.read_parquet(arq, columns=COLS)
    linha = {"ano": ano, "focos": len(df)}
    for v, n in df.version.value_counts().items():
        linha[f"versao_{v}"] = int(n)
    for t, n in df["type"].value_counts().sort_index().items():
        linha[f"type_{t}"] = int(n)
    for (c, d), n in df.groupby(["confidence", "daynight"]).size().items():
        linha[f"conf_{c}_{d}"] = int(n)
    for s, n in df.satellite.value_counts().items():
        linha[f"sat_{s}"] = int(n)
    anual.append(linha)
    m = pd.to_datetime(df.acq_date).dt.month
    g = df.groupby([m.rename("mes"), "satellite"]).size().unstack(fill_value=0)
    g.insert(0, "ano", ano)
    mensal.append(g.reset_index())

anual = pd.DataFrame(anual).fillna(0).astype(int)
mensal = pd.concat(mensal)
mensal["frac_N20"] = (mensal["N20"] / (mensal["N"] + mensal["N20"])).round(3)
anual.to_csv(OUT / "atributos_por_ano.csv", index=False)
mensal.to_csv(OUT / "satelite_por_mes.csv", index=False)
pd.set_option("display.width", 250)
print("## Por ano")
print(anual.T.to_string())
print("\n## Fração de focos do NOAA-20 por mês (meses de seca e extremos)")
print(mensal.pivot(index="mes", columns="ano", values="frac_N20").to_string())
print(f"\nSaídas em {OUT}")
