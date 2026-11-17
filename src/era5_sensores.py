"""ERA5 variables per sensor and hour, with the derived ones of METHODS.md §4.

For each PurpleAir sensor, builds an hourly series with:
- the downloaded variables, at the 0.25° cell nearest the sensor: PBLH (blh, m),
  wind at 10 m and 850 hPa (m/s), 2 m temperature and dew point (°C) and
  hourly precipitation (mm);
- the same variables averaged over the downloaded region (lat −10.5 to −8.0; lon −65.0 to −62.5),
  with suffix _reg; the regional wind is the vector mean (mean of u and of v);
- the derived ones, computed at the sensor cell:
  * relative humidity, from T2m and D2m (saturation pressure over water from IFS/ECMWF);
  * accumulated rainfall over 24 h, 72 h and 7 days;
  * hours since the last hour with at least 1 mm of rain;
  * ventilation index: PBLH × 10 m wind speed.

Time convention: tp with valid_time = t is the rain of (t − 1 h, t];
the other variables are instantaneous at t. Every window ends at t and uses only hours
≤ t, so no value in the table depends on the future (tested in tests/).

Does not read PurpleAir or PM2.5. Usage (from the root): .venv/bin/python src/era5_sensores.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

RAIZ = Path(__file__).resolve().parents[1]
RAW = RAIZ / "data" / "raw" / "era5"
OUT = RAIZ / "data" / "processed" / "era5_sensores"
INVENTARIO = RAIZ / "data" / "raw" / "purpleair" / "inventario.csv"

LIMIAR_CHUVA_MM = 1.0  # METHODS.md §4: "at least 1 mm"
JANELAS_CHUVA_H = {"chuva_24h": 24, "chuva_72h": 72, "chuva_7d": 168}


def pressao_saturacao_agua(t_k):
    """Saturation vapor pressure over water (Pa), in the Tetens form used by the IFS.

    e_s = 611.21 · exp(17.502 · (T − 273.16) / (T − 32.19)), with T in K.
    """
    t_k = np.asarray(t_k, dtype=float)
    return 611.21 * np.exp(17.502 * (t_k - 273.16) / (t_k - 32.19))


def umidade_relativa(t2m_k, d2m_k):
    """Relative humidity (%) from T2m and D2m, in K, clipped to [0, 100]."""
    ur = 100.0 * pressao_saturacao_agua(d2m_k) / pressao_saturacao_agua(t2m_k)
    return np.clip(ur, 0.0, 100.0)


def chuva_acumulada(tp_mm, horas):
    """Sum of the rain over the horas ending at t (causal window; NaN until the window is full)."""
    return pd.Series(tp_mm).rolling(window=horas, min_periods=horas).sum().to_numpy()


def horas_sem_chuva(tp_mm, limiar=LIMIAR_CHUVA_MM):
    """Hours since the last hour with rain ≥ limiar (0 in that hour; NaN before the first)."""
    tp = np.asarray(tp_mm, dtype=float)
    idx = np.arange(len(tp), dtype=float)
    ultima = np.where(tp >= limiar, idx, np.nan)
    ultima = pd.Series(ultima).ffill().to_numpy()
    return idx - ultima


def _abrir(prefixo):
    """Opens and joins the era5_<prefixo>_*.nc files (no dask), in time order and without repeated hours."""
    partes = []
    for arq in sorted(RAW.glob(f"era5_{prefixo}_*.nc")):
        ds = xr.open_dataset(arq).load()
        ds = ds.drop_vars([c for c in ("number", "expver") if c in ds.coords or c in ds.data_vars])
        if "pressure_level" in ds.dims:
            ds = ds.squeeze("pressure_level", drop=True)
        partes.append(ds)
    ds = xr.concat(partes, dim="valid_time").sortby("valid_time")
    _, unicos = np.unique(ds.valid_time.values, return_index=True)
    return ds.isel(valid_time=np.sort(unicos))


def tabela_sensor(sl, pl, lat, lon):
    """Hourly series of one sensor at (lat, lon), from the surface (sl) and 850 hPa (pl) datasets."""
    sl, pl = xr.align(sl, pl, join="inner")
    c_sl = sl.sel(latitude=lat, longitude=lon, method="nearest")
    c_pl = pl.sel(latitude=lat, longitude=lon, method="nearest")
    r_sl = sl.mean(["latitude", "longitude"])
    r_pl = pl.mean(["latitude", "longitude"])
    ur_campo = xr.apply_ufunc(umidade_relativa, sl.t2m, sl.d2m)

    df = pd.DataFrame({"time_stamp": pd.to_datetime(sl.valid_time.values).tz_localize("UTC")})
    df["lat_celula"] = float(c_sl.latitude)
    df["lon_celula"] = float(c_sl.longitude)
    tp_mm = np.clip(c_sl.tp.values * 1000.0, 0.0, None)  # m → mm; negative packing residuals become 0
    df["blh"] = c_sl.blh.values
    df["u10"], df["v10"] = c_sl.u10.values, c_sl.v10.values
    df["vel10"] = np.hypot(df.u10, df.v10)
    df["t2m_c"] = c_sl.t2m.values - 273.15
    df["d2m_c"] = c_sl.d2m.values - 273.15
    df["ur"] = umidade_relativa(c_sl.t2m.values, c_sl.d2m.values)
    df["tp_mm"] = tp_mm
    df["u850"], df["v850"] = c_pl.u.values, c_pl.v.values
    df["vel850"] = np.hypot(df.u850, df.v850)
    for nome, h in JANELAS_CHUVA_H.items():
        df[nome] = chuva_acumulada(tp_mm, h)
    df["horas_sem_chuva"] = horas_sem_chuva(tp_mm)
    df["ventilacao"] = df.blh * df.vel10
    # regional mean (wind: vector mean)
    df["blh_reg"] = r_sl.blh.values
    df["u10_reg"], df["v10_reg"] = r_sl.u10.values, r_sl.v10.values
    df["vel10_reg"] = np.hypot(df.u10_reg, df.v10_reg)
    df["t2m_c_reg"] = r_sl.t2m.values - 273.15
    df["ur_reg"] = ur_campo.mean(["latitude", "longitude"]).values
    df["tp_mm_reg"] = np.clip(r_sl.tp.values * 1000.0, 0.0, None)
    df["u850_reg"], df["v850_reg"] = r_pl.u.values, r_pl.v.values
    df["vel850_reg"] = np.hypot(df.u850_reg, df.v850_reg)
    return df


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sl, pl = _abrir("sl"), _abrir("pl")
    print(f"superfície: {sl.sizes['valid_time']} h, {str(sl.valid_time.values[0])[:13]} → {str(sl.valid_time.values[-1])[:13]}")
    print(f"850 hPa:    {pl.sizes['valid_time']} h, {str(pl.valid_time.values[0])[:13]} → {str(pl.valid_time.values[-1])[:13]}")
    neg = int((sl.tp.values < 0).sum())
    print(f"valores de tp < 0 (zerados): {neg}")
    sensores = pd.read_csv(INVENTARIO)[["sensor_index", "latitude", "longitude"]]
    tabelas = []
    for s in sensores.itertuples(index=False):
        df = tabela_sensor(sl, pl, s.latitude, s.longitude)
        df.insert(0, "sensor", int(s.sensor_index))
        tabelas.append(df)
        print(f"{s.sensor_index}: célula lat {df.lat_celula.iat[0]:.2f}, lon {df.lon_celula.iat[0]:.2f}; {len(df)} h")
    tab = pd.concat(tabelas, ignore_index=True)
    arq = OUT / "era5_sensores_2020_2024.parquet"
    tab.to_parquet(arq, index=False)
    nulos = tab.isna().sum()
    print("valores ausentes por coluna (as janelas de chuva começam vazias):")
    print(nulos[nulos > 0].to_string())
    print(f"{len(tab)} linhas, {tab.shape[1]} colunas → {arq}")


if __name__ == "__main__":
    main()
