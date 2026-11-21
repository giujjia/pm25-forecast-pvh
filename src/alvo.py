"""Target table (T12): QC, Barkjohn 2022 and valid day applied to the hourly PurpleAir data.

Applies, in this order, the rules of doc 04 §3.3 (METHODS.md §3):
1. humidity of 56831 replaced by that of 39077 from 01/10/2022 (REGRA_UR, item 4);
2. A/B QC (> 5 µg/m³ AND > 70%) and humidity outside 0–100% (items 2 and 3);
3. the three corrections; Barkjohn 2022 is the main one. Values keep their sign and the hour is
   invalidated below −4.99 µg/m³ (item 6);
4. label by the end of the hour, t + 1 h (item 7);
5. local day (UTC−4) of each hour and valid day: >= 18 of 24 hours with a value (item 8).

The hourly grid is complete: hours without a record or discarded stay as NaN, with the reason.
The target is never interpolated (item 9).

2024 stays closed until the pre-registration (T27): by default only years <= 2023 are written,
and the report never reads 2024. --ate-ano 2024 is only used after the freeze.

Usage (from the root): .venv/bin/python src/alvo.py [--ate-ano 2023]
Outputs: data/processed/alvo/alvo_horario_<ini>_<fim>.parquet and relatorio_alvo_<ini>_<fim>.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import correcao  # noqa: E402
import features  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
RAW = RAIZ / "data" / "raw" / "purpleair"
OUT = RAIZ / "data" / "processed" / "alvo"

SENSORES = ("39077", "56831", "219595", "219981", "225327", "128603")
ANO_INICIO = 2020
HORAS_DIA_VALIDO = 18
FUSO_LOCAL_H = -4  # Porto Velho, UTC−4, no daylight saving


def ler_horario(sensor):
    df = pd.read_parquet(RAW / f"{sensor}_horario.parquet")
    df["time_stamp"] = pd.to_datetime(df["time_stamp"], utc=True)
    return df.drop_duplicates("time_stamp").sort_values("time_stamp").reset_index(drop=True)


def dia_local(t_fim):
    """Local day of the hour labelled by its end t: the interval (t − 1 h, t] shifted to UTC−4."""
    t = pd.to_datetime(t_fim, utc=True)
    return (t - pd.Timedelta(hours=1) + pd.Timedelta(hours=FUSO_LOCAL_H)).dt.tz_localize(None).dt.normalize()


def montar_sensor(df, df_doador=None, desde=None):
    """Hourly grid of one sensor (labels at the START, as downloaded), with corrections and reasons."""
    if df_doador is not None:
        df = correcao.substituir_ur(df, df_doador, desde)
    ok_ab = correcao.qc_ab(df["pm2.5_cf_1_a"], df["pm2.5_cf_1_b"]).to_numpy()
    ok_ur = correcao.qc_umidade(df["humidity"]).to_numpy()
    corr = correcao.corrigir(df)  # QC A/B + RH, signed, invalidation below −4.99
    out = df[["time_stamp", "humidity"]].copy()
    out["motivo"] = np.where(~ok_ab, "qc_ab", np.where(~ok_ur, "umidade", ""))
    cols = ["cf1", "barkjohn2021", "barkjohn2022", "airnow_fsm"]
    out = out.merge(corr[["time_stamp"] + cols], on="time_stamp", how="left")
    inval = (out["motivo"] == "") & out["barkjohn2022"].isna()
    out.loc[inval, "motivo"] = "negativo_invalidado"
    return out


def grade_completa(df, ini, fim):
    horas = pd.date_range(ini, fim, freq="h", tz="UTC", inclusive="left")
    g = pd.DataFrame({"time_stamp": horas}).merge(df, on="time_stamp", how="left")
    g["motivo"] = g["motivo"].fillna("sem_registro")
    return g


def construir(ate_ano=2023):
    if ate_ano >= 2024 and "--confirmo-pre-registro" not in sys.argv:
        raise SystemExit("2024 só é aberto depois do pré-registro (T27). Use --confirmo-pre-registro.")
    ini = pd.Timestamp(f"{ANO_INICIO}-01-01", tz="UTC")
    fim = pd.Timestamp(f"{ate_ano + 1}-01-01", tz="UTC")
    brutos = {s: ler_horario(s) for s in SENSORES}
    partes = []
    for s in SENSORES:
        df = brutos[s]
        df = df[(df["time_stamp"] >= ini) & (df["time_stamp"] < fim)]
        if df.empty:
            continue
        regra = correcao.REGRA_UR.get(s)
        if regra:
            doador = brutos[regra[0]]
            m = montar_sensor(df, doador, regra[1])
            # sensitivity (doc 04 §3.3, item 4): own humidity of the sensor
            propria = montar_sensor(df)[["time_stamp", "barkjohn2022"]].rename(columns={"barkjohn2022": "barkjohn2022_ur_propria"})
            m = m.merge(propria, on="time_stamp", how="left")
        else:
            m = montar_sensor(df)
            m["barkjohn2022_ur_propria"] = m["barkjohn2022"]
        g = grade_completa(m, df["time_stamp"].min().floor("D"), fim)
        g.insert(0, "sensor", int(s))
        partes.append(g)
    tab = pd.concat(partes, ignore_index=True)
    tab = features.rotular_purpleair_pelo_fim(tab)  # t = end of the hour
    tab["dia_local"] = dia_local(tab["time_stamp"])
    n_dia = tab.groupby(["sensor", "dia_local"])["barkjohn2022"].transform(lambda x: int(x.notna().sum()))
    tab["horas_validas_dia"] = n_dia
    tab["dia_valido"] = n_dia >= HORAS_DIA_VALIDO
    # the last label of the period (t + 1 h) may cross into the next year: keep only <= fim
    tab = tab[tab["time_stamp"] <= fim].reset_index(drop=True)
    return tab


def relatorio(tab):
    """T13: coverage and discards per sensor and year (by the local day of the hour)."""
    t = tab.copy()
    t["ano"] = t["dia_local"].dt.year
    linhas = []
    for (s, a), g in t.groupby(["sensor", "ano"]):
        v = g["barkjohn2022"]
        n_trunc, n_val, frac = correcao.resumo_truncamento(v.to_numpy(), piso=0.0)
        dias = g.drop_duplicates("dia_local")
        linhas.append({
            "sensor": s, "ano": a, "horas_grade": len(g),
            "sem_registro": int((g["motivo"] == "sem_registro").sum()),
            "descarte_qc_ab": int((g["motivo"] == "qc_ab").sum()),
            "descarte_umidade": int((g["motivo"] == "umidade").sum()),
            "invalidadas_negativo": int((g["motivo"] == "negativo_invalidado").sum()),
            "horas_validas": n_val,
            "horas_negativas_truncadas_no_log": n_trunc, "fracao_truncada": round(frac, 4),
            "dias": len(dias), "dias_validos": int(dias["dia_valido"].sum()),
        })
    return pd.DataFrame(linhas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ate-ano", type=int, default=2023)
    ap.add_argument("--confirmo-pre-registro", action="store_true")
    a = ap.parse_args()
    tab = construir(a.ate_ano)
    OUT.mkdir(parents=True, exist_ok=True)
    nome = f"{ANO_INICIO}_{a.ate_ano}"
    tab.to_parquet(OUT / f"alvo_horario_{nome}.parquet", index=False)
    rel = relatorio(tab)
    rel.to_csv(OUT / f"relatorio_alvo_{nome}.csv", index=False)
    print(rel.to_string(index=False))
    print(f"\n{len(tab)} horas na grade; sensores: {sorted(tab['sensor'].unique())}")


if __name__ == "__main__":
    main()
