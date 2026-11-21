"""Sensor × hour table (T18): target, ERA5 and FIRMS fire, all available at t.

Joins, for each sensor and hour t (labelled by the end of the hour):
- target and PM2.5 history: data/processed/alvo (src/alvo.py, T12);
- ERA5 at the sensor cell and the regional mean, with the derived ones: data/processed/era5_sensores (E057);
- FIRMS fire, computed here (doc 04 §3.4):
  * NOAA-20 only, fixed mask of static sources, 'l' confidence kept (16/11/2026);
  * rings centred on each sensor (0–50, 50–150, 150–300, 300–600, 600–1000 km) × 45° sectors
    (sector k covers the bearing [45k, 45k + 45) from the sensor to the detection, clockwise from north);
  * availability: NRT published up to 3 h after the overpass. The 24 h FRP at t is the sum over the
    24 h ending at the last published instant, (t − 27 h, t − 3 h]; the lagged ones are the two
    previous 24 h windows ((t − 51 h, t − 27 h] and (t − 75 h, t − 51 h]);
  * transported FRP: sum over sectors of the 24 h FRP of the sector (all rings up to 1000 km) times
    the positive part of the 850 hPa wind component (at the sensor cell, at t) blowing from the
    sector centre towards the sensor (MW·m/s);
  * satellite outage day (no NOAA-20 detection in the whole 1000 km box): fire inputs of the
    windows touching that UTC day are NaN, with an indicator, never zero.

Every window ends at or before t; the tests in tests/ check that no value depends on the future.
2024 stays closed until the pre-registration (same rule as src/alvo.py).

Usage (from the root): .venv/bin/python src/entradas.py [--ate-ano 2023]
Output: data/processed/entradas/entradas_horario_<ini>_<fim>.parquet
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
ALVO = RAIZ / "data" / "processed" / "alvo"
ERA5 = RAIZ / "data" / "processed" / "era5_sensores" / "era5_sensores_2020_2024.parquet"
FIRMS = RAIZ / "data" / "raw" / "firms"
INVENTARIO = RAIZ / "data" / "raw" / "purpleair" / "inventario.csv"
OUT = RAIZ / "data" / "processed" / "entradas"

ANEIS_KM = (0, 50, 150, 300, 600, 1000)
NOMES_ANEIS = ("a0_50", "a50_150", "a150_300", "a300_600", "a600_1000")
N_SETORES = 8
ATRASO_FIRMS_H = 3
JANELA_H = 24
RAIO_TERRA_KM = 6371.0088


def nome_col(anel, setor, defasagem=""):
    return f"frp24{defasagem}_{anel}_s{setor}"


def distancia_azimute(lat0, lon0, lat, lon):
    """Great-circle distance (km) and initial bearing (deg, clockwise from north) from (lat0, lon0)."""
    p0, l0 = np.radians(lat0), np.radians(lon0)
    p, l = np.radians(lat), np.radians(lon)
    dl = l - l0
    a = np.sin((p - p0) / 2) ** 2 + np.cos(p0) * np.cos(p) * np.sin(dl / 2) ** 2
    d = 2 * RAIO_TERRA_KM * np.arcsin(np.sqrt(a))
    y = np.sin(dl) * np.cos(p)
    x = np.cos(p0) * np.sin(p) - np.sin(p0) * np.cos(p) * np.cos(dl)
    az = (np.degrees(np.arctan2(y, x)) + 360) % 360
    return d, az


def hora_fim_deteccao(df):
    """Hour bin of each detection, labelled by its end: detection at τ goes to ceil(τ) (τ in (H − 1 h, H])."""
    ts = features.construir_timestamp_firms(df)
    return ts.dt.ceil("h")


def ler_firms(anos):
    partes = []
    for a in anos:
        f = FIRMS / f"firms1000_{a}.parquet"
        partes.append(pd.read_parquet(f, columns=["latitude", "longitude", "acq_date", "acq_time", "satellite", "confidence", "frp", "type"]))
    df = pd.concat(partes, ignore_index=True)
    dias = pd.date_range(f"{min(anos)}-01-01", f"{max(anos)}-12-31", freq="D")
    falhas = features.dias_falha_firms(df, dias, satelite="N20")
    df = features.filtrar_firms(df, tipos_permitidos=None, satelites_permitidos=features.SATELITES_FIRMS_DECIDIDOS,
                                confiancas_permitidas=features.CONFIANCAS_FIRMS_DECIDIDAS)
    df = features.aplicar_mascara_estatica(df, features.carregar_mascara_estatica())
    df["hora"] = hora_fim_deteccao(df)
    return df, falhas


def frp_horario_por_zona(df, lat0, lon0, horas):
    """Hourly FRP (by end-of-hour bin) per ring × sector, for one sensor; 40 columns, zero where no detection."""
    d, az = distancia_azimute(lat0, lon0, df["latitude"].to_numpy(), df["longitude"].to_numpy())
    anel = np.digitize(d, ANEIS_KM[1:-1], right=False)
    dentro = d < ANEIS_KM[-1]
    setor = np.floor(az / 45).astype(int) % N_SETORES
    z = pd.DataFrame({"hora": df["hora"].to_numpy()[dentro], "anel": anel[dentro], "setor": setor[dentro],
                      "frp": df["frp"].to_numpy()[dentro]})
    z["col"] = [nome_col(NOMES_ANEIS[a], s) for a, s in zip(z["anel"], z["setor"])]
    piv = z.pivot_table(index="hora", columns="col", values="frp", aggfunc="sum", fill_value=0.0)
    cols = [nome_col(a, s) for a in NOMES_ANEIS for s in range(N_SETORES)]
    return piv.reindex(index=horas, columns=cols, fill_value=0.0).fillna(0.0)


def janelas_fogo(horario, falhas, horas):
    """24 h FRP windows ending at t − 3 h, and the two lagged windows, with outage indicators."""
    soma24 = horario.rolling(JANELA_H, min_periods=JANELA_H).sum()
    out = {}
    for rotulo, desloc in (("", ATRASO_FIRMS_H), ("_d24_48", ATRASO_FIRMS_H + 24), ("_d48_72", ATRASO_FIRMS_H + 48)):
        w = soma24.shift(desloc)
        w.columns = [c.replace("frp24_", f"frp24{rotulo}_") for c in horario.columns]
        # outage: a window (t − desloc − 24 h, t − desloc] touching a failed UTC day becomes NaN
        dia_hora = pd.Series((horas - pd.Timedelta(seconds=1)).normalize(), index=horas)  # UTC day of the interval (H − 1 h, H]
        falha_h = dia_hora.isin(pd.to_datetime(falhas).tz_localize("UTC")).astype(float)
        toca = falha_h.rolling(JANELA_H, min_periods=1).max().shift(desloc).fillna(1.0) > 0
        nome_ind = f"falha_firms{rotulo or '_24h'}"
        w.loc[toca.to_numpy()] = np.nan
        w[nome_ind] = toca.to_numpy().astype(int)
        out[rotulo] = w
    return pd.concat(out.values(), axis=1)


def frp_transportado(fogo, u850, v850):
    """Σ_s FRP24_s × max(0, wind component from sector s towards the sensor)."""
    total = np.zeros(len(fogo))
    for s in range(N_SETORES):
        theta = np.radians(45 * s + 22.5)  # sector centre, bearing from the sensor
        # unit vector from the sector to the sensor (east, north) = −(sin θ, cos θ)
        comp = np.maximum(0.0, -(u850 * np.sin(theta) + v850 * np.cos(theta)))
        frp_s = sum(fogo[nome_col(a, s)].to_numpy() for a in NOMES_ANEIS)
        total = total + frp_s * comp
    return total


def construir(ate_ano=2023):
    if ate_ano >= 2024 and "--confirmo-pre-registro" not in sys.argv:
        raise SystemExit("2024 só é aberto depois do pré-registro (T27). Use --confirmo-pre-registro.")
    alvo = pd.read_parquet(ALVO / f"alvo_horario_2020_{ate_ano}.parquet")
    era5 = pd.read_parquet(ERA5)
    era5["time_stamp"] = pd.to_datetime(era5["time_stamp"], utc=True)
    era5 = era5.drop(columns=["lat_celula", "lon_celula"])
    inv = pd.read_csv(INVENTARIO).set_index("sensor_index")
    # FIRMS from the year before 2020 is not downloaded: the first 75 h of 2020 have no complete windows
    anos = list(range(2020, ate_ano + 1))
    firms, falhas = ler_firms(anos)
    horas = pd.date_range(f"{anos[0]}-01-01 01:00", f"{ate_ano + 1}-01-01 00:00", freq="h", tz="UTC")
    partes = []
    for s in sorted(alvo["sensor"].unique()):
        a = alvo[alvo["sensor"] == s][["sensor", "time_stamp", "dia_local", "dia_valido", "barkjohn2022",
                                        "barkjohn2022_ur_propria", "barkjohn2021", "airnow_fsm"]]
        lat0, lon0 = inv.loc[s, "latitude"], inv.loc[s, "longitude"]
        fogo = janelas_fogo(frp_horario_por_zona(firms, lat0, lon0, horas), falhas, horas)
        e = era5[era5["sensor"] == s].set_index("time_stamp").reindex(horas)
        fogo["frp_transportado"] = frp_transportado(fogo, e["u850"].to_numpy(), e["v850"].to_numpy())
        tab = e.drop(columns=["sensor"]).join(fogo)
        tab.index.name = "time_stamp"
        tab = tab.reset_index()
        tab = a.merge(tab, on="time_stamp", how="left")
        partes.append(tab)
    out = pd.concat(partes, ignore_index=True)
    return out, falhas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ate-ano", type=int, default=2023)
    ap.add_argument("--confirmo-pre-registro", action="store_true")
    a = ap.parse_args()
    tab, falhas = construir(a.ate_ano)
    OUT.mkdir(parents=True, exist_ok=True)
    tab.to_parquet(OUT / f"entradas_horario_2020_{a.ate_ano}.parquet", index=False)
    print(f"{len(tab)} linhas × {tab.shape[1]} colunas; sensores {sorted(tab['sensor'].unique().tolist())}")
    print("dias de falha do NOAA-20:", [str(d) for d in falhas])
    frp_cols = [c for c in tab.columns if c.startswith("frp24_")]
    print("horas com fogo de 24 h ausente (falha):", int(tab["falha_firms_24h"].sum()))
    print("FRP de 24 h total, média por hora e anel (MW), por sensor:")
    for anel in NOMES_ANEIS:
        cs = [c for c in frp_cols if f"_{anel}_" in c]
        print(" ", anel, tab[cs].sum(axis=1, min_count=1).groupby(tab["sensor"]).mean().round(1).to_dict())
    print("colunas ausentes por variável do ERA5:", int(tab["blh"].isna().sum()))


if __name__ == "__main__":
    main()
