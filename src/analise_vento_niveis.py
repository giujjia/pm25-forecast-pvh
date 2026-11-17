"""Compares ERA5 wind direction and speed at 10 m and 850 hPa over Porto Velho, in the dry season.

Used to decide which wind to use in the "transported FRP" (METHODS.md §4, selection rule, item 6).
Does NOT use PM2.5 or fire detections: meteorology only. So it does not interfere with input selection.

Defined before running (13/11/2026):
- Period: July to October 2020, 2021 and 2022 (the years with both levels downloaded).
- Location: grid point nearest Porto Velho (8.76° S; 63.90° W) and vector mean of the downloaded
  region (lat −10.5 to −8.0; lon −65.0 to −62.5).
- Local time = UTC − 4. Day = 07–18 h local; night = 19–06 h local.
- Meteorological direction (where the wind comes from): (270 − atan2(v, u)) mod 360.
- Calm at 10 m: speed < 0.5 m/s (same class as Liberato, 2019). In those hours the direction
  is undefined; they are left out of the direction statistics and counted separately.
- Tables:
  T1 speed by level; T2 frequency by 45° sector (where from), over all hours, day and night;
  T3 absolute direction difference between 850 hPa and 10 m (median and bands), overall, by
  local hour and by month, hourly and with the daily mean wind; T4 weight each level would give
  each sector in the transported FRP: max(0, speed · cos(direction − sector)), with the mean,
  the fraction of hours with positive weight, the share of total weight and the Spearman
  correlation between levels, hour by hour.

Usage:
    .venv/bin/python src/analise_vento_niveis.py
Outputs: data/processed/vento_niveis/*.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

RAIZ = Path(__file__).resolve().parents[1]
ERA5 = RAIZ / "data" / "raw" / "era5"
SAIDA = RAIZ / "data" / "processed" / "vento_niveis"
ANOS = [2020, 2021, 2022]
MESES = [7, 8, 9, 10]
PVH = (-8.76, -63.90)
CALMA = 0.5
SETORES = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
CENTROS = np.arange(0, 360, 45)


def carrega():
    # Open one file at a time and concatenate in time (open_mfdataset would require dask).
    sl = xr.concat([xr.open_dataset(ERA5 / f"era5_sl_{a}{m}.nc")[["u10", "v10"]].load()
                    for a in ANOS for m in ("05-08", "09-12")], dim="valid_time")
    pl = xr.concat([xr.open_dataset(ERA5 / f"era5_pl_{a}07-12.nc")[["u", "v"]].load()
                    for a in ANOS], dim="valid_time").squeeze("pressure_level", drop=True)
    ds = xr.merge([sl, pl], join="inner", compat="override")
    ds = ds.sel(valid_time=ds.valid_time.dt.month.isin(MESES))
    return ds.load()


def direcao(u, v):
    return np.mod(270.0 - np.degrees(np.arctan2(v, u)), 360.0)


def setor(d):
    return np.array(SETORES)[(np.floor(np.mod(d + 22.5, 360) / 45)).astype(int)]


def difang(a, b):
    d = np.abs(a - b) % 360
    return np.minimum(d, 360 - d)


def tabela_ponto(ds, nome):
    if nome == "ponto":
        p = ds.sel(latitude=PVH[0], longitude=PVH[1], method="nearest")
        print(f"Ponto de grade usado: lat {float(p.latitude):.2f}, lon {float(p.longitude):.2f}")
    else:
        p = ds.mean(["latitude", "longitude"])
    df = pd.DataFrame({k: p[k].values for k in ("u10", "v10", "u", "v")},
                      index=pd.DatetimeIndex(p.valid_time.values, name="utc"))
    df["hora_local"] = (df.index - pd.Timedelta(hours=4)).hour
    df["mes"] = df.index.month
    df["periodo"] = np.where(df.hora_local.between(7, 18), "dia", "noite")
    for nivel, (u, v) in {"10m": ("u10", "v10"), "850": ("u", "v")}.items():
        df[f"vel_{nivel}"] = np.hypot(df[u], df[v])
        df[f"dir_{nivel}"] = direcao(df[u], df[v])
        df[f"setor_{nivel}"] = setor(df[f"dir_{nivel}"])
    df["calma_10m"] = df.vel_10m < CALMA
    df["dif_dir"] = difang(df.dir_850, df.dir_10m)
    return df


def faixas(x):
    x = x.dropna()
    return pd.Series({"n": len(x), "mediana_graus": x.median(),
                      "pct_ate_45": 100 * (x <= 45).mean(),
                      "pct_45_90": 100 * ((x > 45) & (x <= 90)).mean(),
                      "pct_acima_90": 100 * (x > 90).mean()})


def analisa(df, nome):
    print(f"\n===== {nome.upper()} =====")
    print(f"Horas: {len(df)} ({df.index.min()} a {df.index.max()} UTC)")

    t1 = pd.DataFrame({n: {"media": df[f"vel_{n}"].mean(), "mediana": df[f"vel_{n}"].median(),
                           "pct_lt_0.5": 100 * (df[f"vel_{n}"] < 0.5).mean(),
                           "pct_lt_1": 100 * (df[f"vel_{n}"] < 1).mean(),
                           "pct_lt_2": 100 * (df[f"vel_{n}"] < 2).mean()} for n in ("10m", "850")}).T
    for per in ("dia", "noite"):
        s = df[df.periodo == per]
        t1[f"media_{per}"] = [s.vel_10m.mean(), s.vel_850.mean()]
    t1.round(2).to_csv(SAIDA / f"T1_velocidade_{nome}.csv")
    print("\nT1 velocidade (m/s; % de horas)\n", t1.round(2).to_string())

    linhas = []
    for per, s in [("todas", df), ("dia", df[df.periodo == "dia"]), ("noite", df[df.periodo == "noite"])]:
        for n in ("10m", "850"):
            ss = s[~s.calma_10m] if n == "10m" else s
            freq = ss[f"setor_{n}"].value_counts(normalize=True).reindex(SETORES, fill_value=0) * 100
            linhas.append(pd.Series(freq, name=f"{n}_{per}"))
    t2 = pd.DataFrame(linhas).round(1)
    t2.to_csv(SAIDA / f"T2_setores_{nome}.csv")
    print("\nT2 % de horas por setor de onde o vento vem (10 m sem calmarias)\n", t2.to_string())
    jas = df[df.mes.isin([7, 8, 9]) & ~df.calma_10m]
    print(f"Conferência com Liberato (2019; seca = jul–set): ERA5 10 m de SE, S ou SW em "
          f"{100 * jas.setor_10m.isin(['SE', 'S', 'SW']).mean():.1f}% das horas sem calmaria; "
          f"calmaria (<0,5 m/s) em {100 * df[df.mes.isin([7, 8, 9])].calma_10m.mean():.1f}% das horas")

    ok = df[~df.calma_10m]
    t3 = pd.DataFrame({"geral": faixas(ok.dif_dir), "dia": faixas(ok[ok.periodo == "dia"].dif_dir),
                       "noite": faixas(ok[ok.periodo == "noite"].dif_dir)}).T
    d = df[["u10", "v10", "u", "v"]].resample("D").mean()
    t3.loc["diario"] = faixas(difang(direcao(d.u, d.v), direcao(d.u10, d.v10)))
    t3.round(1).to_csv(SAIDA / f"T3_diferenca_direcao_{nome}.csv")
    print("\nT3 diferença absoluta de direção 850 hPa × 10 m (graus; horas com calmaria a 10 m excluídas)\n",
          t3.round(1).to_string())
    th = ok.groupby("hora_local").dif_dir.apply(faixas).unstack().round(1)
    th.to_csv(SAIDA / f"T3_por_hora_{nome}.csv")
    print("\nT3 por hora local\n", th[["n", "mediana_graus", "pct_ate_45", "pct_acima_90"]].to_string())
    tm = ok.groupby("mes").dif_dir.apply(faixas).unstack().round(1)
    tm.to_csv(SAIDA / f"T3_por_mes_{nome}.csv")
    print("\nT3 por mês\n", tm.to_string())

    linhas = []
    for nome_s, c in zip(SETORES, CENTROS):
        w = {}
        for n in ("10m", "850"):
            w[n] = np.maximum(0, df[f"vel_{n}"] * np.cos(np.radians(df[f"dir_{n}"] - c)))
        linhas.append({"setor": nome_s,
                       "peso_medio_10m": w["10m"].mean(), "peso_medio_850": w["850"].mean(),
                       "pct_horas_positivo_10m": 100 * (w["10m"] > 0).mean(),
                       "pct_horas_positivo_850": 100 * (w["850"] > 0).mean(),
                       "spearman_10m_850": pd.Series(w["10m"]).corr(pd.Series(w["850"]), method="spearman")})
    t4 = pd.DataFrame(linhas).set_index("setor")
    for n in ("10m", "850"):
        t4[f"parcela_pct_{n}"] = 100 * t4[f"peso_medio_{n}"] / t4[f"peso_medio_{n}"].sum()
    t4 = t4.round(2)
    t4.to_csv(SAIDA / f"T4_peso_por_setor_{nome}.csv")
    print("\nT4 peso no FRP transportado: max(0, vel·cos(dir − setor)); setor = de onde o fogo está\n",
          t4.to_string())


def main():
    SAIDA.mkdir(parents=True, exist_ok=True)
    ds = carrega()
    print(f"Período carregado: {str(ds.valid_time.values.min())[:19]} a {str(ds.valid_time.values.max())[:19]} UTC; "
          f"{ds.sizes['valid_time']} horas; meses {sorted(set(ds.valid_time.dt.month.values.tolist()))}; "
          f"anos {sorted(set(ds.valid_time.dt.year.values.tolist()))}")
    falt = int(sum(int(ds[v].isnull().sum()) for v in ("u10", "v10", "u", "v")))
    print(f"Valores ausentes em u10, v10, u, v: {falt}")
    esperado = sum(pd.Timestamp(a, 11, 1).dayofyear - pd.Timestamp(a, 7, 1).dayofyear for a in ANOS) * 24
    print(f"Horas esperadas: {esperado}")
    for nome in ("ponto", "regiao"):
        analisa(tabela_ponto(ds, nome), nome)


if __name__ == "__main__":
    main()
