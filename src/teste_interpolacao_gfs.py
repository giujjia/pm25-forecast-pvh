"""Test of the GFS 3 h -> 1 h interpolation (gap 11), without PM2.5.

Two questions, both at the ERA5/GFS cell of sensor 39077 (lat -8.75, lon -64.0):

A. How much does linear interpolation lose by itself? The hourly ERA5 (2020 to Oct 2026)
   is sampled every 3 h (00, 03, ..., 21 UTC, the valid times of the 18 UTC GFS run), the
   state variables are interpolated linearly back to 1 h, and the rain of each 3 h window
   is split equally over its 3 hours. The error against the true hourly ERA5 is the cost
   of the interpolation alone.

B. How large is that cost next to the error of the GFS forecast itself? For each 18 UTC
   run of 2022 to Oct 2026 (complete on disk), the steps that cover the decoder horizon
   (issuance at 04 UTC of the next day, +1 to +48 h, i.e. run + 11 to run + 58 h) are
   compared with the hourly ERA5:
   - GFS at its native 3 h steps (forecast error only);
   - GFS interpolated to every hour (forecast error + interpolation).

Metrics: MAE and RMSE per variable, in the whole period and in the dry season (Jul-Oct).
Rain is also compared as the daily maximum hourly rate. ERA5 rain is converted from m to mm.

Usage: python src/teste_interpolacao_gfs.py
"""
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

RAIZ = Path(__file__).resolve().parents[1]
LAT, LON = -8.75, -64.0
ESTADO = ["blh", "u10", "v10", "ws10", "t2m", "d2m", "u850", "v850", "vent"]


def era5_horario():
    """Hourly ERA5 series at the cell, 2020 to Oct 2026."""
    partes = []
    for tipo in ("sl", "pl"):
        series = []
        for f in sorted(glob.glob(str(RAIZ / f"data/raw/era5/era5_{tipo}_20*.nc"))):
            ds = xr.open_dataset(f).sel(latitude=LAT, longitude=LON)
            if "pressure_level" in ds.dims:
                ds = ds.isel(pressure_level=0).rename({"u": "u850", "v": "v850"})
            keep = [v for v in ("tp", "blh", "u10", "v10", "t2m", "d2m", "u850", "v850") if v in ds]
            series.append(ds[keep].to_dataframe()[keep])
        df = pd.concat(series)
        partes.append(df[~df.index.duplicated()].sort_index())
    df = partes[0].join(partes[1], how="inner")
    df["tp"] = df["tp"] * 1000.0  # m -> mm
    df["ws10"] = np.hypot(df["u10"], df["v10"])
    df["vent"] = df["blh"] * df["ws10"]
    esperado = pd.date_range(df.index[0], df.index[-1], freq="h")
    assert len(df) == len(esperado), f"ERA5 com buracos: {len(df)} de {len(esperado)} horas"
    return df


def interpolar(df3, horas):
    """Linear interpolation of state variables to the hours; rain split equally."""
    out = pd.DataFrame(index=horas)
    for v in ("blh", "u10", "v10", "t2m", "d2m", "u850", "v850"):
        out[v] = np.interp(horas.asi8, df3.index.asi8, df3[v].to_numpy())
    out["ws10"] = np.hypot(out["u10"], out["v10"])
    out["vent"] = out["blh"] * out["ws10"]
    # rain: df3["tp3"] is the 3 h total ending at each 3 h valid time
    tp = pd.Series(np.nan, index=horas)
    for t, total in df3["tp3"].items():
        for k in range(3):
            h = t - pd.Timedelta(hours=k)
            if h in tp.index:
                tp[h] = total / 3.0
    out["tp"] = tp
    return out


def metricas(prev, verdade, rotulo):
    linhas = []
    for nome, mask in (("ano todo", slice(None)), ("seca jul-out", verdade.index.month.isin([7, 8, 9, 10]))):
        p, o = prev.loc[mask], verdade.loc[mask]
        for v in ESTADO + ["tp"]:
            e = (p[v] - o[v]).dropna()
            linhas.append({"teste": rotulo, "periodo": nome, "variavel": v, "n": len(e),
                           "MAE": e.abs().mean(), "RMSE": np.sqrt((e ** 2).mean()),
                           "dp_verdade": o[v].std()})
        # daily maximum hourly rain
        pmax = p["tp"].resample("D").max()
        omax = o["tp"].resample("D").max()
        e = (pmax - omax).dropna()
        linhas.append({"teste": rotulo, "periodo": nome, "variavel": "tp_max_diario", "n": len(e),
                       "MAE": e.abs().mean(), "RMSE": np.sqrt((e ** 2).mean()), "dp_verdade": omax.std()})
    return linhas


def teste_a(era):
    df3 = era[era.index.hour % 3 == 0].copy()
    df3["tp3"] = era["tp"].rolling(3).sum().reindex(df3.index)
    df3 = df3.iloc[1:]  # the first window is incomplete
    horas = era.index[(era.index >= df3.index[0] - pd.Timedelta(hours=2)) & (era.index <= df3.index[-1])]
    interp = interpolar(df3, horas)
    return metricas(interp, era.loc[horas], "A: interpolação do ERA5")


def teste_b(era):
    nativo, interp_all = [], []
    arquivos = sorted(f for a in range(2022, 2027) for f in glob.glob(str(RAIZ / f"data/raw/gfs/{a}/gfs_*18.nc")))
    for f in arquivos:
        g = xr.open_dataset(f).sel(lat=LAT, lon=LON)
        rodada = pd.Timestamp(g.attrs["rodada"].replace("Z", "")).tz_localize(None)
        tempos = rodada + pd.to_timedelta(g["passo"].values.astype(int), unit="h")
        d = pd.DataFrame({"blh": g["hpbl"].values, "u10": g["u10"].values, "v10": g["v10"].values,
                          "t2m": g["t2m"].values, "d2m": g["d2m"].values, "u850": g["u850"].values,
                          "v850": g["v850"].values, "tp0": g["tp_0a"].values}, index=tempos)
        d["tp3"] = d["tp0"].diff()  # 3 h total ending at each step
        d["ws10"] = np.hypot(d["u10"], d["v10"])
        d["vent"] = d["blh"] * d["ws10"]
        ini, fim = rodada + pd.Timedelta(hours=11), rodada + pd.Timedelta(hours=58)
        janela = d.loc[(d.index >= ini) & (d.index <= fim + pd.Timedelta(hours=2))]
        horas = pd.date_range(ini, fim, freq="h")
        if not horas.isin(era.index).all() or d.loc[d.index >= ini - pd.Timedelta(hours=3), "tp3"].isna().any():
            continue
        nat = d.loc[(d.index >= ini) & (d.index <= fim)].copy()
        nat["tp"] = nat["tp3"] / 3.0  # comparable with hourly ERA5 rain as a mean rate
        nativo.append(nat)
        base = d.loc[(d.index >= ini - pd.Timedelta(hours=3)) & (d.index <= fim + pd.Timedelta(hours=3))]
        interp_all.append(interpolar(base, horas))
    nat = pd.concat(nativo)
    itp = pd.concat(interp_all)
    # native: compare state at the step hour; rain as 3 h mean rate against ERA5 3 h mean rate
    era_nat = era.loc[nat.index].copy()
    era_nat["tp"] = era["tp"].rolling(3).mean().reindex(nat.index)
    linhas = metricas(nat, era_nat, "B1: GFS nos passos de 3 h")
    linhas += metricas(itp, era.loc[itp.index], "B2: GFS interpolado para 1 h")
    return linhas, len(nativo)


def main():
    era = era5_horario()
    print(f"ERA5: {len(era)} horas, de {era.index[0]} a {era.index[-1]}; célula {LAT}, {LON}")
    linhas = teste_a(era)
    lb, n = teste_b(era)
    print(f"GFS: {n} rodadas das 18 UTC (2022 a out/2026) com a janela +11 a +58 h completa")
    res = pd.DataFrame(linhas + lb)
    res["MAE/dp"] = res["MAE"] / res["dp_verdade"]
    pd.set_option("display.width", 200, "display.max_rows", 200)
    for periodo in ("ano todo", "seca jul-out"):
        t = res[res["periodo"] == periodo].pivot_table(index="variavel", columns="teste", values="MAE")
        print(f"\nMAE por variável ({periodo}); unidades: m, m/s, K, mm/h, m²/s")
        print(t.round(3).to_string())
    saida = RAIZ / "data/testes/teste_interpolacao_gfs.csv"
    saida.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(saida, index=False)
    print(f"\nTabela completa: {saida.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
