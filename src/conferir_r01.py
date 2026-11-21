"""Redoes, through the registrar, the preliminary table of doc 06 §3 (retroactive run R01, no log).

Only 2020–2023: 2024 stays closed until the pre-registration (T27), so the 2024 rows of doc 06
remain as R01 (no log).

(A) Replication of R01: daily means of the PurpleAir API (data/raw/purpleair/<sensor>_diario.parquet),
    A/B QC on cf_1 applied to the daily means (> 5 µg/m³ AND > 70%), days with humidity outside
    0–100% dropped, the three corrections on the daily means (Barkjohn 2021 and 2022 on cf_1,
    AirNow FSM on atm), annual mean per sensor and year. No completeness filter (as in R01).
(B) Project target (E135, data/processed/alvo/alvo_horario_2020_2023.parquet): daily mean of the
    hourly corrected values on valid days (>= 18 h, local day), then the annual mean of the valid days.

Prints both tables and the largest difference between the three corrections in each.
Output: data/processed/conferencias/r01_2020_2023.csv
Usage: .venv/bin/python src/conferir_r01.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

import correcao as c

RAIZ = Path(__file__).resolve().parents[1]
SENSORES = ["39077", "56831", "128603"]
ANOS = [2020, 2021, 2022, 2023]
SAIDA = RAIZ / "data" / "processed" / "conferencias" / "r01_2020_2023.csv"


def tabela_a():
    linhas = []
    for s in SENSORES:
        d = pd.read_parquet(RAIZ / "data" / "raw" / "purpleair" / f"{s}_diario.parquet")
        d["ts"] = pd.to_datetime(d["time_stamp"], utc=True)
        d = d[d["ts"].dt.year.isin(ANOS)]
        ok = c.qc_ab(d["pm2.5_cf_1_a"], d["pm2.5_cf_1_b"]) & c.qc_umidade(d["humidity"])
        d = d[ok]
        cf1 = (d["pm2.5_cf_1_a"] + d["pm2.5_cf_1_b"]) / 2
        atm = (d["pm2.5_atm_a"] + d["pm2.5_atm_b"]) / 2
        rh = d["humidity"].astype(float)
        d = d.assign(bruto=cf1, b21=c.barkjohn2021(cf1, rh), b22=c.barkjohn2022(cf1, rh),
                     fsm=c.airnow_fsm(atm, rh), ano=d["ts"].dt.year)
        for ano, g in d.groupby("ano"):
            linhas.append(dict(fonte="A_api_diaria", sensor=s, ano=ano, dias=len(g),
                               bruto=g["bruto"].mean(), barkjohn2021=g["b21"].mean(),
                               barkjohn2022=np.mean(g["b22"]), airnow_fsm=np.mean(g["fsm"]),
                               dias_acima_50_b22=int((g["b22"] > 50).sum()),
                               dias_ge_125_b22=int((g["b22"] >= 125).sum())))
    return pd.DataFrame(linhas)


def tabela_b():
    a = pd.read_parquet(RAIZ / "data" / "processed" / "alvo" / "alvo_horario_2020_2023.parquet")
    a = a[a["dia_valido"]]
    col = ["cf1", "barkjohn2021", "barkjohn2022", "airnow_fsm"]
    dia = a.groupby(["sensor", "dia_local"])[col].mean().reset_index()
    dia["ano"] = pd.to_datetime(dia["dia_local"]).dt.year
    linhas = []
    for (s, ano), g in dia.groupby(["sensor", "ano"]):
        linhas.append(dict(fonte="B_alvo_E135", sensor=str(s), ano=ano, dias=len(g),
                           bruto=g["cf1"].mean(), barkjohn2021=g["barkjohn2021"].mean(),
                           barkjohn2022=g["barkjohn2022"].mean(), airnow_fsm=g["airnow_fsm"].mean(),
                           dias_acima_50_b22=int((g["barkjohn2022"] > 50).sum()),
                           dias_ge_125_b22=int((g["barkjohn2022"] >= 125).sum())))
    return pd.DataFrame(linhas)


def main():
    t = pd.concat([tabela_a(), tabela_b()], ignore_index=True)
    corr = t[["barkjohn2021", "barkjohn2022", "airnow_fsm"]]
    t["maior_dif_correcoes"] = corr.max(axis=1) - corr.min(axis=1)
    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    t.round(2).to_csv(SAIDA, index=False)
    pd.set_option("display.width", 200)
    print(t.round(2).to_string(index=False))
    for f, g in t.groupby("fonte"):
        i = g["maior_dif_correcoes"].idxmax()
        print(f"{f}: maior diferença entre correções = {g.loc[i, 'maior_dif_correcoes']:.2f} µg/m³ "
              f"({g.loc[i, 'sensor']}, {g.loc[i, 'ano']})")


if __name__ == "__main__":
    main()
