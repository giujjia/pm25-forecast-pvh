"""A/B quality control and PM2.5 corrections for PurpleAir sensors.

Three published corrections, for sensitivity analysis (none was validated
in the Amazon against a reference monitor):

- barkjohn2021: Barkjohn, Gantt & Clements (2021), AMT 14:4617. Input cf_1.
- barkjohn2022: Barkjohn et al. (2022), Sensors 22:9669, section 3.1.3 (Eqs. 1–3).
  Input cf_1. Fitted on hourly means.
- airnow_fsm: 5-piece correction used in the AirNow Fire and Smoke Map (EPA).
  Input cf_atm (Jaffe et al., 2023, AMT 16:1311). Coefficients to be checked against the
  EPA technical report before final use.

RH is the humidity measured by the sensor itself, as in the sources above, with one exception
decided on 16/11/2026 (METHODS.md §3, item 4): 56831 uses the humidity of 39077 from 01/10/2022
(REGRA_UR). Negatives (item 6): the corrected value keeps its sign; below
LIMIAR_INVALIDO the hour is invalidated; the floor at zero only enters in transformar_log, in the network.
"""
import numpy as np
import pandas as pd


# AirNow convention for negatives (Jiang et al., 2023, p. 6).
LIMIAR_INVALIDO = -4.99
# sensor -> (humidity donor sensor, from when, UTC). METHODS.md §3.3, item 4.
REGRA_UR = {"56831": ("39077", "2022-10-01")}


def qc_ab(a, b, abs_max=5.0, rel_max=0.7):
    """True where channels A and B agree (discards if they differ by > 5 µg/m³ AND > 70%)."""
    diff = (a - b).abs()
    media = (a + b) / 2
    rel = diff.where(media > 0, 0.0) / media.where(media > 0, 1.0)
    return ~((diff > abs_max) & (rel > rel_max))


def qc_umidade(rh, minimo=0.0, maximo=100.0):
    """True where relative humidity is finite and within [0, 100]%."""
    arr = pd.to_numeric(rh, errors="coerce")
    return arr.notna() & np.isfinite(arr) & (arr >= minimo) & (arr <= maximo)


def aplicar_piso(valores, piso=0.0):
    """Applies a lower floor (e.g. 0.0 before log(1 + y)), preserving NaNs."""
    arr = np.asarray(valores, dtype=float)
    return np.where(np.isnan(arr), np.nan, np.maximum(arr, piso))


def resumo_truncamento(valores, piso=0.0):
    """Returns (n_truncated, n_valid, truncated_fraction) below the floor."""
    arr = np.asarray(valores, dtype=float)
    validos = arr[~np.isnan(arr)]
    if validos.size == 0:
        return 0, 0, 0.0
    n_trunc = int(np.sum(validos < piso))
    return n_trunc, int(validos.size), float(n_trunc / validos.size)


def substituir_ur(df, df_doador, desde):
    """Replaces the humidity of df with that of df_doador (same time_stamp) from desde (UTC).

    Hours in which the donor has no humidity are left without humidity and drop out in QC. Before desde, nothing changes.
    """
    out = df.copy()
    ts = pd.to_datetime(out["time_stamp"], utc=True)
    m = (ts >= pd.Timestamp(desde, tz="UTC")).to_numpy()
    doador = df_doador.set_index(pd.to_datetime(df_doador["time_stamp"], utc=True))["humidity"]
    doador = doador[~doador.index.duplicated()]
    out["humidity"] = out["humidity"].astype(float)
    out.loc[m, "humidity"] = doador.reindex(ts[m]).to_numpy()
    return out


def transformar_log(valores):
    """log(1 + max(y, 0)): the floor at zero exists only inside the network transformation (METHODS.md §3, item 6)."""
    return np.log1p(aplicar_piso(valores, piso=0.0))


def barkjohn2021(cf1, rh):
    return 0.524 * cf1 - 0.0862 * rh + 5.75


def barkjohn2022(cf1, rh):
    x, rh = np.asarray(cf1, float), np.asarray(rh, float)
    eq1 = 0.524 * x - 0.0862 * rh + 5.75
    eq3 = 4.21e-4 * x**2 + 0.392 * x + 3.44
    w = np.clip(0.0244 * x - 13.9, 0, 1)  # transition 570 ≤ cf_1 < 611
    return np.where(x < 570, eq1, np.where(x >= 611, eq3, w * eq3 + (1 - w) * eq1))


def airnow_fsm(atm, rh):
    x, rh = np.asarray(atm, float), np.asarray(rh, float)
    y = np.empty_like(x)
    a = x < 30
    y[a] = 0.524 * x[a] - 0.0862 * rh[a] + 5.75
    b = (x >= 30) & (x < 50)
    w = x[b] / 20 - 1.5
    y[b] = (0.786 * w + 0.524 * (1 - w)) * x[b] - 0.0862 * rh[b] + 5.75
    c = (x >= 50) & (x < 210)
    y[c] = 0.786 * x[c] - 0.0862 * rh[c] + 5.75
    d = (x >= 210) & (x < 260)
    w = x[d] / 50 - 4.2
    y[d] = ((0.69 * w + 0.786 * (1 - w)) * x[d] - 0.0862 * rh[d] * (1 - w)
            + 2.966 * w + 5.75 * (1 - w) + 8.84e-4 * x[d] ** 2 * w)
    e = x >= 260
    y[e] = 2.966 + 0.69 * x[e] + 8.84e-4 * x[e] ** 2
    return y


def corrigir(df, piso=None, invalidar_abaixo=LIMIAR_INVALIDO):
    """Applies A/B QC, the 0–100% RH filter and the three corrections.

    Decided on 16/11/2026 (METHODS.md §3, item 6): by default it returns the signed correction
    and invalidates (NaN) values below invalidar_abaixo (−4.99, AirNow convention).
    The floor at zero does not enter the data: the network uses transformar_log. piso remains
    available only for sensitivity analyses. For the 56831 humidity rule, apply substituir_ur
    before this function.
    """
    ok = qc_ab(df["pm2.5_cf_1_a"], df["pm2.5_cf_1_b"]) & qc_umidade(df["humidity"])
    df = df[ok].copy()
    cf1 = (df["pm2.5_cf_1_a"] + df["pm2.5_cf_1_b"]) / 2
    atm = (df["pm2.5_atm_a"] + df["pm2.5_atm_b"]) / 2
    df["cf1"] = cf1
    b21 = barkjohn2021(cf1, df["humidity"])
    b22 = barkjohn2022(cf1, df["humidity"])
    fsm = airnow_fsm(atm, df["humidity"])
    if invalidar_abaixo is not None:
        b21 = np.where(np.asarray(b21, float) < invalidar_abaixo, np.nan, b21)
        b22 = np.where(np.asarray(b22, float) < invalidar_abaixo, np.nan, b22)
        fsm = np.where(np.asarray(fsm, float) < invalidar_abaixo, np.nan, fsm)
    if piso is not None:
        b21 = aplicar_piso(b21, piso=piso)
        b22 = aplicar_piso(b22, piso=piso)
        fsm = aplicar_piso(fsm, piso=piso)
    df["barkjohn2021"] = b21
    df["barkjohn2022"] = b22
    df["airnow_fsm"] = fsm
    return df
