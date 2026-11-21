"""Tabular inputs for XGBoost and Ridge (doc 04 §3.5): the same information as the network, as lags
and moving means, plus the future meteorology of the target hour.

IMPLEMENTAÇÃO (confirmed by the author on 18/11/2026):
- PM2.5 (log(1 + max(y, 0)), gaps <= 3 h interpolated): values at t, t−1, t−2, t−3, t−6, t−12, t−24,
  t−48 and t−71; means of the last 6, 24 and 72 h; maximum of the last 24 h;
- past meteorology and fire: the version's columns at t (the derived ones already accumulate the past);
  FRP as log(1 + FRP); outage stays NaN for XGBoost (indicator columns included);
- calendar of t and of the target hour t + h; horizon h (XGBoost only);
- future meteorology (variant "era5"): the fixed ERA5 set at t + h.
"""
import numpy as np
import pandas as pd

import conjunto

DEFASAGENS = (0, 1, 2, 3, 6, 12, 24, 48, 71)
MEDIAS = (6, 24, 72)


def _pm_log(serie):
    return np.log1p(np.maximum(serie.pm_in, 0.0))


def base(serie, idx, conj):
    """[N, F] features of issuance t = tempo[i] that do not depend on h."""
    pm = _pm_log(serie)
    cols, nomes = [], []
    for k in DEFASAGENS:
        cols.append(pm[idx - k]); nomes.append(f"pm_lag{k}")
    cs = np.concatenate([[0.0], np.cumsum(np.nan_to_num(pm))])
    for w in MEDIAS:
        cols.append((cs[idx + 1] - cs[idx + 1 - w]) / w); nomes.append(f"pm_media{w}")
    cols.append(np.array([pm[i - 23:i + 1].max() for i in idx])); nomes.append("pm_max24")
    raw = serie.enc_raw
    for c in raw.columns:
        v = raw[c].to_numpy(float)[idx]
        if c.startswith("frp"):
            v = np.log1p(v)
        cols.append(v); nomes.append(c)
    cal = conjunto.calendario(serie.tempo[idx])
    for j, c in enumerate(conjunto.CALENDARIO):
        cols.append(cal[:, j]); nomes.append("t_" + c)
    return np.column_stack(cols).astype(np.float32), nomes


def futuro(serie, idx, h, conj):
    """[N, F] features of the target hour t + h (calendar and, in variant era5, ERA5)."""
    j = idx + h
    cal = conjunto.calendario(serie.tempo[j])
    cols, nomes = [cal[:, k] for k in range(cal.shape[1])], ["alvo_" + c for c in conjunto.CALENDARIO]
    fut = [n for n in conj.nomes_dec if n.startswith("fut_")]
    if fut:
        # unstandardise the decoder ERA5 back to physical units (trees and the Ridge scaler do their own scaling)
        k0 = [conj.nomes_dec.index(n) for n in fut]
        v = serie.dec[j][:, k0] * np.array([conj.desvio[n] for n in fut]) + np.array([conj.media[n] for n in fut])
        for k, n in enumerate(fut):
            cols.append(v[:, k]); nomes.append(n)
    return np.column_stack(cols).astype(np.float32), nomes


def matriz_horizontes(conj, idx_por_sensor, horizontes=range(1, conjunto.HORIZONTE_H + 1)):
    """Stacked rows (sensor, i, h) with features, h and the signed target. Returns X, y, meta, names."""
    Xs, ys, metas, nomes = [], [], [], None
    for s, idx in idx_por_sensor.items():
        if len(idx) == 0:
            continue
        serie = conj.series[s]
        b, nb = base(serie, idx, conj)
        for h in horizontes:
            f, nf = futuro(serie, idx, h, conj)
            Xs.append(np.column_stack([b, f, np.full(len(idx), h, np.float32)]))
            ys.append(serie.y[idx + h])
            metas.append(pd.DataFrame({"sensor": s, "i": idx, "t_emissao": serie.tempo[idx], "h": h}))
            nomes = nb + nf + ["h"]
    return np.concatenate(Xs), np.concatenate(ys), pd.concat(metas, ignore_index=True), nomes
