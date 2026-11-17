"""Gap 1, part two: who drifted? RH of the two PurpleAir sensors and of A925 against ERA5.

A previous comparison showed that, against A925, 39077 read ~8 points lower in 2023 than in 2022, and 56831
became more dispersed. A925 has a gap from Jul/2022 to Apr/2023. ERA5 (2 m RH from T2m and D2m,
0.25° cell of each sensor; src/era5_sensores.py) is continuous and has no instrument drift, so
it serves as the referee:
  - INMET − ERA5 stable across periods      -> the station did not change;
  - 39077 − ERA5 drops and 56831 − ERA5 not -> the drift is in 39077 (and vice versa).
Comparisons are by ERA5 RH band and within the same months of the year, so drift is not confused
with seasonality. Alignment: PurpleAir labelled t is the mean of [t, t+1h) and is compared
with the ERA5 mean at t and t+1h; A925 (top-of-the-hour value) is compared with ERA5 at t.

RH only. Does not read PM2.5. Only before 2024-01-01.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from correcao import qc_umidade  # noqa: E402
from verificar_inmet import colunas, ler, validos  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
OUT = RAIZ / "data" / "processed" / "ur_era5"
OUT.mkdir(parents=True, exist_ok=True)
FIM = pd.Timestamp("2024-01-01", tz="UTC")
CORTE = pd.Timestamp("2022-10-01", tz="UTC")

era = pd.read_parquet(RAIZ / "data" / "processed" / "era5_sensores" / "era5_sensores_2020_2024.parquet",
                      columns=["sensor", "time_stamp", "lat_celula", "lon_celula", "ur"])
era = era[era.time_stamp < FIM]
era["sensor"] = era.sensor.astype(str)


def era_sensor(s):
    e = era[era.sensor == s].set_index("time_stamp").ur.sort_index()
    return e


def periodo(idx):
    return np.where(idx < CORTE, "A_ate_set2022", "B_out2022_dez2023")


res, mensal = [], {}
print("## 1. UR do sensor − UR do ERA5 na célula do sensor (pontos), por período e faixa do ERA5")
for s in ("39077", "56831"):
    e = era_sensor(s)
    cel = era[era.sensor == s][["lat_celula", "lon_celula"]].iloc[0].tolist()
    e_meio = pd.concat([e, e.shift(-1, freq="h")], axis=1, sort=True).mean(axis=1, skipna=False).dropna()
    pa = pd.read_parquet(RAIZ / "data" / "raw" / "purpleair" / f"{s}_horario.parquet")
    pa = pa[pa.time_stamp < FIM].set_index("time_stamp").humidity
    pa = pa[qc_umidade(pa)]
    par = pd.concat([pa.rename("x"), e_meio.rename("era")], axis=1, join="inner").dropna()
    par["dif"] = par.x - par.era
    par["periodo"] = periodo(par.index)
    par["faixa"] = pd.cut(par.era, [0, 60, 80, 90, 100.01], right=False, labels=["<60", "60-80", "80-90", ">=90"])
    g = par.groupby(["periodo", "faixa"], observed=True).dif.agg(n="size", mediana="median")
    print(f"[{s}] célula ERA5 {cel}; horas pareadas {len(par)}")
    print(g.round(1).unstack(0).to_string())
    m = par.groupby([par.index.year.rename("ano"), par.index.month.rename("mes")]).dif.median().round(1)
    mensal[s] = m
    g = g.reset_index()
    g.insert(0, "serie", s)
    res.append(g)

print("\n## 2. A925 (INMET) − ERA5 na célula do 39077 (pontos): a estação mudou entre 2022 e 2023?")
partes = []
for p in sorted((RAIZ / "data" / "raw" / "inmet").glob("INMET_*A925*.CSV")):
    df = ler(p)
    c = colunas(df)["ur"]
    partes.append(pd.to_numeric(df[c].where(validos(df[c])).str.replace(",", "."), errors="coerce"))
inm = pd.concat(partes).sort_index()
inm.index = inm.index.tz_localize("UTC")
inm = inm[~inm.index.duplicated() & (inm.index < FIM) & inm.between(0, 100)]
e = era_sensor("39077")
par = pd.concat([inm.rename("x"), e.rename("era")], axis=1, join="inner").dropna()
par["dif"] = par.x - par.era
par["periodo"] = periodo(par.index)
par["faixa"] = pd.cut(par.era, [0, 60, 80, 90, 100.01], right=False, labels=["<60", "60-80", "80-90", ">=90"])
g = par.groupby(["periodo", "faixa"], observed=True).dif.agg(n="size", mediana="median")
print(f"horas pareadas {len(par)}")
print(g.round(1).unstack(0).to_string())
mensal["A925"] = par.groupby([par.index.year.rename("ano"), par.index.month.rename("mes")]).dif.median().round(1)
g = g.reset_index()
g.insert(0, "serie", "A925")
res.append(g)
pd.concat(res).to_csv(OUT / "dif_ur_era5_por_periodo_e_faixa.csv", index=False)

print("\n## 3. Mediana mensal da diferença para o ERA5 (pontos), mesmos meses em anos diferentes")
t = pd.DataFrame(mensal)
t.to_csv(OUT / "dif_ur_era5_mensal.csv")
print(t.unstack(0).to_string())
print(f"\nSaídas em {OUT}")
