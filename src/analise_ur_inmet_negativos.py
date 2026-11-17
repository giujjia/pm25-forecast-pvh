"""Gaps 1 and 13: did the 39077 RH drift too? And how many negatives remain under the new rule?

1. Compares the RH of PurpleAir sensors 39077 and 56831 with that of INMET station A925
   (Porto Velho), before and after Oct/2022, within INMET RH bands, so that seasonal change
   is not confused with sensor drift. PurpleAir reads below ambient (Barkjohn 2021;
   Couzo 2024): what matters is whether the difference CHANGES between periods.
   Alignment: the PurpleAir hourly record labelled t is the mean of [t, t+1h); INMET RH
   is the top-of-the-hour UTC value; the mean of INMET(t) and INMET(t+1h) is used.
2. Counts, under the rule decided on 16/11/2026, the Barkjohn 2022 hours below −4.99 µg/m³
   (invalidated, AirNow convention; Jiang 2023) and between −4.99 and 0 (kept in the data;
   the floor at zero only enters in the network transformation), for 39077 and for 56831 using
   the 39077 RH since Oct/2022.

Quality control only: does not print or save PM2.5 distributions. Reads only before 2024-01-01.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from correcao import barkjohn2022, qc_ab, qc_umidade  # noqa: E402
from verificar_inmet import colunas, ler, validos  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
RAW_PA = RAIZ / "data" / "raw" / "purpleair"
RAW_IN = RAIZ / "data" / "raw" / "inmet"
OUT = RAIZ / "data" / "processed" / "ur_inmet_negativos"
OUT.mkdir(parents=True, exist_ok=True)
FIM = pd.Timestamp("2024-01-01", tz="UTC")
CORTE = pd.Timestamp("2022-10-01", tz="UTC")
LIMIAR_INVALIDO = -4.99


def ur_inmet():
    partes = []
    for p in sorted(RAW_IN.glob("INMET_*A925*.CSV")):
        df = ler(p)
        c = colunas(df)["ur"]
        v = validos(df[c])
        s = pd.to_numeric(df[c].where(v).str.replace(",", "."), errors="coerce")
        partes.append(s)
    s = pd.concat(partes).sort_index()
    s.index = s.index.tz_localize("UTC")
    s = s[~s.index.duplicated()]
    return s[(s.index < FIM) & s.between(0, 100)]


def ur_pa(sensor):
    df = pd.read_parquet(RAW_PA / f"{sensor}_horario.parquet")
    df = df[df.time_stamp < FIM]
    s = df.set_index("time_stamp").humidity
    return s[qc_umidade(s)]


inm = ur_inmet()
inm_meio = pd.concat([inm, inm.shift(-1, freq="h")], axis=1).mean(axis=1, skipna=False).dropna()
print(f"## 0. Horas da A925 com UR válida antes de 2024: {len(inm)}; com as duas pontas da hora: {len(inm_meio)}")
print(inm.groupby(inm.index.year).size().rename("horas_validas_por_ano").to_string())

linhas = []
print("\n## 1. Diferença de UR (PurpleAir − INMET) por período e faixa de UR do INMET (pontos de UR)")
for s in ("39077", "56831"):
    pa = ur_pa(s)
    par = pd.concat([pa.rename("pa"), inm_meio.rename("inmet")], axis=1, join="inner").dropna()
    par["dif"] = par.pa - par.inmet
    par["periodo"] = np.where(par.index < CORTE, "A_ate_set2022", "B_out2022_dez2023")
    par["faixa"] = pd.cut(par.inmet, [0, 60, 80, 90, 100.01], right=False,
                          labels=["<60", "60-80", "80-90", ">=90"])
    g = par.groupby(["periodo", "faixa"], observed=True).dif.agg(
        n="size", mediana="median", p25=lambda x: x.quantile(.25), p75=lambda x: x.quantile(.75))
    g.insert(0, "sensor", s)
    linhas.append(g.reset_index())
    print(f"[{s}] horas pareadas: {len(par)} (A: {(par.periodo.str.startswith('A')).sum()}, B: {(par.periodo.str.startswith('B')).sum()})")
    print(g.round(1).to_string())
    m = par.groupby(par.index.to_period("M")).dif.agg(["median", "count"]).round(1)
    m.to_csv(OUT / f"dif_ur_mensal_{s}.csv")
    # same month of the year in both periods (Jan–Sep 2022 vs Jan–Sep 2023) controls for season
    mes = par[par.index.month <= 9]
    a = mes[mes.index.year == 2022].dif
    b = mes[mes.index.year == 2023].dif
    print(f"[{s}] jan–set: mediana da diferença em 2022 {a.median():.1f} (n {len(a)}) × em 2023 {b.median():.1f} (n {len(b)})")
pd.concat(linhas).to_csv(OUT / "dif_ur_por_periodo_e_faixa.csv", index=False)

print("\n## 2. Barkjohn 2022 com a regra nova: invalidadas (< −4,99) e negativas mantidas (−4,99 a 0)")
dfs = {s: pd.read_parquet(RAW_PA / f"{s}_horario.parquet").query("time_stamp < @FIM").set_index("time_stamp")
       for s in ("39077", "56831")}
d56 = dfs["56831"].copy()
m = d56.index >= CORTE
d56.loc[m, "humidity"] = dfs["39077"].humidity.reindex(d56.index[m]).values
cenarios = {"39077": dfs["39077"], "56831_ur_propria": dfs["56831"], "56831_ur_39077_desde_out2022": d56}
res = []
for nome, d in cenarios.items():
    ok = qc_ab(d["pm2.5_cf_1_a"], d["pm2.5_cf_1_b"]) & qc_umidade(d.humidity)
    d = d[ok]
    y = pd.Series(barkjohn2022((d["pm2.5_cf_1_a"] + d["pm2.5_cf_1_b"]) / 2, d.humidity), index=d.index)
    seca = d.index.month.isin([7, 8, 9, 10])
    for ano, yy in y.groupby(y.index.year):
        sc = seca[y.index.year == ano]
        res.append({"cenario": nome, "ano": ano, "horas_validas": len(yy),
                    "invalidadas_lt_-4.99": int((yy < LIMIAR_INVALIDO).sum()),
                    "negativas_mantidas": int(((yy >= LIMIAR_INVALIDO) & (yy < 0)).sum()),
                    "negativas_na_seca": int(((yy < 0) & sc).sum()),
                    "le_-1": int((yy <= -1).sum()),
                    "pct_piso_na_transformacao": round(100 * ((yy >= LIMIAR_INVALIDO) & (yy < 0)).mean(), 2)})
r = pd.DataFrame(res)
r.to_csv(OUT / "negativos_regra_nova.csv", index=False)
print(r.to_string(index=False))
print(f"\nmínimo da Barkjohn 2022 (fórmula, cf_1 = 0 e UR = 100): {barkjohn2022(0.0, 100.0):.2f} µg/m³")
print(f"Saídas em {OUT}")
