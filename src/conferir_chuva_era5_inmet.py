"""Check 4 (18/11/2026): ERA5 rain against the INMET A925 rain gauge (Porto Velho), 2022–2023.

Only 2022–2023: the A925 has no rain in 2020–2021 (audit F45), and 2024 is left out to keep the
test year untouched. Rain only, no PM2.5.

ERA5 series (data/processed/era5_sensores/era5_sensores_2020_2024.parquet; tp of valid_time t is
the rain of (t − 1 h, t]):
  - cell of the training sensors 39077 and 56831 (−8.75, −64.00), the one the model reads;
  - cell nearest to the station (−8.75, −63.75), read from sensor 219595;
  - regional mean (tp_mm_reg).
INMET: "PRECIPITAÇÃO TOTAL, HORÁRIO (mm)"; -9999 and empty are missing. The hour label of the
gauge is checked by lagging ±1 h (the lag with the highest hourly correlation is reported, not chosen).

Compared on hours where both exist: hourly, 24 h accumulations (the model input chuva_24h) and
local-day totals (UTC−4, days with 24 gauge hours); whole period and dry season (Jul–Oct).
Metrics: total ratio ERA5/gauge, Pearson r, frequency of light (0.1–1 mm/h) and heavy (>= 5 mm/h)
hourly rain, and the agreement of the event "at least 1 mm in the previous 24 h" (POD, FAR), which
is the descriptive rain split of the pre-registration.
Output: data/processed/conferencias/chuva_era5_inmet_2022_2023.csv
Usage: .venv/bin/python src/conferir_chuva_era5_inmet.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
ANOS = [2022, 2023]
SAIDA = RAIZ / "data" / "processed" / "conferencias" / "chuva_era5_inmet_2022_2023.csv"


def ler_inmet():
    partes = []
    for ano in ANOS:
        f = next((RAIZ / "data" / "raw" / "inmet").glob(f"*A925*{ano}_A_*{ano}.CSV"))
        d = pd.read_csv(f, sep=";", skiprows=8, encoding="latin-1", decimal=",", dtype=str)
        col = [c for c in d.columns if c.upper().startswith("PRECIPITA")][0]
        hora = d["Hora UTC"].str.slice(0, 2)
        ts = pd.to_datetime(d["Data"] + " " + hora, format="%Y/%m/%d %H", utc=True)
        v = pd.to_numeric(d[col].str.replace(",", "."), errors="coerce").where(lambda x: x > -9990)
        partes.append(pd.Series(v.to_numpy(), index=ts, name="inmet"))
    s = pd.concat(partes)
    return s[~s.index.duplicated()].sort_index()


def ler_era5():
    e = pd.read_parquet(RAIZ / "data" / "processed" / "era5_sensores" / "era5_sensores_2020_2024.parquet",
                        columns=["sensor", "time_stamp", "tp_mm", "tp_mm_reg"])
    e["time_stamp"] = pd.to_datetime(e["time_stamp"], utc=True)
    a = e[e.sensor == 39077].set_index("time_stamp")
    b = e[e.sensor == 219595].set_index("time_stamp")
    return pd.DataFrame({"celula_treino": a["tp_mm"], "celula_estacao": b["tp_mm"],
                         "regional": a["tp_mm_reg"]})


def pod_far(obs, prev):
    ok = obs.notna() & prev.notna()
    o, p = obs[ok], prev[ok]
    a = int((o & p).sum()); b = int((~o & p).sum()); c = int((o & ~p).sum())
    return a / (a + c) if a + c else np.nan, b / (a + b) if a + b else np.nan


def main():
    g = ler_inmet()
    e = ler_era5()
    e = e[e.index.year.isin(ANOS)]
    linhas = []
    for lag in (-1, 0, 1):
        j = e.join(g.shift(lag, freq="h"), how="inner").dropna()
        linhas.append(dict(serie="defasagem_inmet_h", periodo=str(lag), metrica="r_horario_celula_treino",
                           valor=j["celula_treino"].corr(j["inmet"])))
    j = e.join(g, how="inner")
    j["inmet_24h"] = j["inmet"].rolling(24, min_periods=24).sum()
    for c in ["celula_treino", "celula_estacao", "regional"]:
        j[c + "_24h"] = j[c].rolling(24, min_periods=24).sum()
    loc = j.index.tz_convert("America/Porto_Velho")
    j["dia_local"] = loc.date
    j["seca"] = loc.month.isin([7, 8, 9, 10])
    for periodo, m in (("2022-2023", np.ones(len(j), bool)), ("seca_jul_out", j["seca"].to_numpy())):
        jj = j[m]
        h = jj.dropna(subset=["inmet"])
        dia = h.groupby("dia_local").agg(n=("inmet", "size"), inmet=("inmet", "sum"),
                                        celula_treino=("celula_treino", "sum"),
                                        celula_estacao=("celula_estacao", "sum"), regional=("regional", "sum"))
        dia = dia[dia.n == 24]
        linhas.append(dict(serie="inmet", periodo=periodo, metrica="horas_com_dado", valor=len(h)))
        linhas.append(dict(serie="inmet", periodo=periodo, metrica="dias_completos", valor=len(dia)))
        linhas.append(dict(serie="inmet", periodo=periodo, metrica="freq_fraca_0.1_1_mm_h",
                           valor=((h.inmet >= 0.1) & (h.inmet < 1)).mean()))
        linhas.append(dict(serie="inmet", periodo=periodo, metrica="freq_forte_ge5_mm_h", valor=(h.inmet >= 5).mean()))
        for c in ["celula_treino", "celula_estacao", "regional"]:
            hh = h.dropna(subset=[c])
            v24 = jj.dropna(subset=["inmet_24h", c + "_24h"])
            pod, far = pod_far(v24["inmet_24h"] >= 1, v24[c + "_24h"] >= 1)
            for met, val in (
                ("razao_total_era5_inmet", hh[c].sum() / hh["inmet"].sum()),
                ("r_horario", hh[c].corr(hh["inmet"])),
                ("r_24h", v24[c + "_24h"].corr(v24["inmet_24h"])),
                ("r_diario", dia[c].corr(dia["inmet"])),
                ("freq_fraca_0.1_1_mm_h", ((hh[c] >= 0.1) & (hh[c] < 1)).mean()),
                ("freq_forte_ge5_mm_h", (hh[c] >= 5).mean()),
                ("pod_chuva24h_ge1mm", pod), ("far_chuva24h_ge1mm", far),
            ):
                linhas.append(dict(serie=c, periodo=periodo, metrica=met, valor=val))
    t = pd.DataFrame(linhas)
    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(SAIDA, index=False)
    pd.set_option("display.width", 200)
    print(t.pivot_table(index=["periodo", "metrica"], columns="serie", values="valor", aggfunc="first").round(3).to_string())


if __name__ == "__main__":
    main()
