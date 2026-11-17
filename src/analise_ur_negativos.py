"""RH of 56831 × 39077 and negative values of the Barkjohn 2022 correction, 2020–2023.

Quality control only: RH range and gaps, divergence between the two training
sensors, and counts of hours in which the correction is negative
(< 0) or below −1 (where log(1 + y) does not exist). Does not print or save
PM2.5 distributions. Reads only time_stamp < 2024-01-01.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from correcao import barkjohn2022, qc_ab  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
RAW = RAIZ / "data" / "raw" / "purpleair"
OUT = RAIZ / "data" / "processed" / "ur_negativos"
OUT.mkdir(parents=True, exist_ok=True)
FIM = pd.Timestamp("2024-01-01", tz="UTC")
CORTE = pd.Timestamp("2022-10-01", tz="UTC")
SENSORES = ("39077", "56831")


def carregar(s):
    df = pd.read_parquet(RAW / f"{s}_horario.parquet")
    return df[df.time_stamp < FIM].copy()


dfs = {s: carregar(s) for s in SENSORES}

print("## 1. Faixa e falhas da UR (horas brutas, antes do QC)")
for s, df in dfs.items():
    h = df.humidity
    print(f"[{s}] horas {len(df)} | UR ausente {h.isna().sum()} | UR < 0: {(h < 0).sum()} | "
          f"UR > 100: {(h > 100).sum()} | UR mín {h.min():.0f} máx {h.max():.0f} | "
          f"UR = 0: {(h == 0).sum()} | UR = 100: {(h == 100).sum()}")

print("\n## 2. Mediana mensal da UR por sensor e diferença (56831 − 39077)")
med = pd.concat([df.set_index("time_stamp").humidity.resample("MS").median().rename(s)
                 for s, df in dfs.items()], axis=1)
n = pd.concat([df.set_index("time_stamp").humidity.resample("MS").count().rename("n_" + s)
               for s, df in dfs.items()], axis=1)
tab = med.join(n)
tab["dif"] = tab["56831"] - tab["39077"]
tab.index = tab.index.strftime("%Y-%m")
tab.round(1).to_csv(OUT / "ur_mediana_mensal.csv")
print(tab.round(1).to_string())

print("\n## 3. Horas pareadas: diferença de UR (56831 − 39077), por período")
par = (dfs["39077"].set_index("time_stamp")[["humidity"]]
       .join(dfs["56831"].set_index("time_stamp")[["humidity"]],
             lsuffix="_39077", rsuffix="_56831", how="inner").dropna())
par["dif"] = par.humidity_56831 - par.humidity_39077
par["periodo"] = np.where(par.index < CORTE, "A. até set/2022", "B. de out/2022 a dez/2023")
print(par.groupby("periodo").dif.describe(percentiles=[.05, .25, .5, .75, .95]).round(1).to_string())
for c in (10, 15, 20, 25, 30):
    pct = par.groupby("periodo").dif.apply(lambda x: round(100 * (x > c).mean(), 1))
    print(f"  horas pareadas com dif > +{c} pontos: {pct.to_dict()}")
mensal = par.groupby(par.index.to_period("M")).dif.agg(["median", "count"]).round(1)
mensal.to_csv(OUT / "ur_diferenca_mensal_pareada.csv")

print("\n## 4. Horas em que a correção Barkjohn 2022 fica < 0 e ≤ −1 (depois do QC A/B)")
linhas = []
for s, df in dfs.items():
    ok = qc_ab(df["pm2.5_cf_1_a"], df["pm2.5_cf_1_b"]) & df.humidity.notna()
    d = df[ok].copy()
    y = barkjohn2022((d["pm2.5_cf_1_a"] + d["pm2.5_cf_1_b"]) / 2, d.humidity)
    d["neg"], d["le1"] = y < 0, y <= -1
    d["ano"] = d.time_stamp.dt.year
    d["seca"] = d.time_stamp.dt.month.between(7, 10)
    d["periodo"] = np.where(d.time_stamp < CORTE, "A", "B")
    g = d.groupby("ano").agg(horas_qc=("neg", "size"), neg=("neg", "sum"), le1=("le1", "sum"))
    g["pct_neg"] = (100 * g.neg / g.horas_qc).round(2)
    g["pct_le1"] = (100 * g.le1 / g.horas_qc).round(2)
    g.insert(0, "sensor", s)
    linhas.append(g.reset_index())
    print(f"[{s}] descartadas pelo QC A/B ou UR ausente: {(~ok).sum()} de {len(df)}; válidas {len(d)}")
    if d.neg.any():
        print(f"[{s}] UR nas horas negativas: mediana {d.loc[d.neg, 'humidity'].median():.0f}%, "
              f"mín {d.loc[d.neg, 'humidity'].min():.0f}% | negativas na seca {int(d.loc[d.seca, 'neg'].sum())}, "
              f"fora da seca {int(d.loc[~d.seca, 'neg'].sum())} | antes de out/2022: {int(d.loc[d.periodo == 'A', 'neg'].sum())}, "
              f"depois: {int(d.loc[d.periodo == 'B', 'neg'].sum())}")
neg = pd.concat(linhas)
neg.to_csv(OUT / "negativos_por_sensor_ano.csv", index=False)
print(neg.to_string(index=False))

print("\n## 5. Cenário: 56831 com a UR do 39077 a partir de out/2022")
d = dfs["56831"].set_index("time_stamp").copy()
ur39 = dfs["39077"].set_index("time_stamp").humidity
m = d.index >= CORTE
d.loc[m, "humidity"] = ur39.reindex(d.index[m]).values
ok = qc_ab(d["pm2.5_cf_1_a"], d["pm2.5_cf_1_b"]) & d.humidity.notna()
y = barkjohn2022((d["pm2.5_cf_1_a"] + d["pm2.5_cf_1_b"]) / 2, d.humidity)[ok.values]
orig = dfs["56831"].set_index("time_stamp")
perdidas = int((orig.humidity.notna() & d.humidity.isna()).sum())
print(f"horas válidas {int(ok.sum())} (perde {perdidas} horas em que o 39077 não tem UR) | "
      f"negativas {(y < 0).sum()} | ≤ −1: {(y <= -1).sum()}")
print(f"\nSaídas em {OUT}")
