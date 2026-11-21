"""Common evaluation (T21, doc 04 §3.6–3.7): main metric, curves, uncertainty, tail and Diebold-Mariano.

Predictions are long tables with columns: modelo, versao, variante, semente, dobra, sensor, t_emissao,
h, q10, q50, q90 (q10/q90 NaN for median-only adversaries) and y (signed observed target, NaN if missing).

- Main metric: MAE of the median, mean over the available hours +1 … +48, for issuances at 0h local
  (04 UTC) in the dry season (Jul–Oct); aggregated over all issuance–sensor pairs (lacuna 18).
- Diebold-Mariano with the Harvey, Leybourne & Newbold (1997) correction and HAC variance, on the series
  of the mean (over sensors) loss differential per issuance (lacuna 6); 95% CI of the MAE difference from
  the test itself (lacuna 7); Holm over adversaries.
- The 5 (first delivery: 3) seeds are averaged quantile by quantile before evaluation (lacuna 9).
"""
import numpy as np
import pandas as pd
from scipy import stats

HORA_EMISSAO_UTC = 4
MESES_SECA = (7, 8, 9, 10)
LIMIAR_CAUDA = 50.0


def media_sementes(pred):
    chaves = [c for c in ["modelo", "versao", "variante", "dobra", "sensor", "t_emissao", "h"] if c in pred.columns]
    agg = {c: "mean" for c in ["q10", "q50", "q90"] if c in pred.columns}
    agg["y"] = "first"
    return pred.groupby(chaves, as_index=False, dropna=False).agg(agg)


def filtro_principal(pred):
    t = pd.to_datetime(pred["t_emissao"], utc=True)
    return pred[(t.dt.hour == HORA_EMISSAO_UTC) & t.dt.month.isin(MESES_SECA)]


def erro_por_par(pred, coluna="q50", perda="mae"):
    """Per (sensor, t_emissao): mean absolute error over the available hours."""
    d = pred[pred["y"].notna()].copy()
    if perda == "mae":
        d["e"] = (d[coluna] - d["y"]).abs()
    elif perda == "mae50":
        d["e"] = (np.maximum(d[coluna], LIMIAR_CAUDA) - np.maximum(d["y"], LIMIAR_CAUDA)).abs()
    return d.groupby(["sensor", "t_emissao"])["e"].mean()


def metrica_principal(pred, perda="mae"):
    """(value, number of pairs). pred: one model (seeds already averaged), any number of folds."""
    e = erro_por_par(filtro_principal(pred), perda=perda)
    return float(e.mean()), int(len(e))


def curva_horizonte(pred):
    d = pred[pred["y"].notna()]
    g = d.groupby("h")
    def r2(x):
        ss = ((x["y"] - x["y"].mean()) ** 2).sum()
        return 1 - ((x["y"] - x["q50"]) ** 2).sum() / ss if ss > 0 else np.nan
    return pd.DataFrame({"mae": g.apply(lambda x: (x["q50"] - x["y"]).abs().mean()),
                         "rmse": g.apply(lambda x: np.sqrt(((x["q50"] - x["y"]) ** 2).mean())),
                         "r2": g.apply(r2), "n": g.size()})


def incerteza(pred, blocos=((1, 12), (13, 24), (25, 36), (37, 48))):
    """Pinball (µg/m³), Q10–Q90 coverage and mean width, total and per horizon block."""
    d = pred[pred["y"].notna() & pred["q10"].notna()]
    def resumo(x):
        pb = 0.0
        for q, c in ((0.1, "q10"), (0.5, "q50"), (0.9, "q90")):
            e = x["y"] - x[c]
            pb += np.maximum(q * e, (q - 1) * e).mean() / 3
        return pd.Series({"pinball": pb, "cobertura": ((x["y"] >= x["q10"]) & (x["y"] <= x["q90"])).mean(),
                          "largura": (x["q90"] - x["q10"]).mean(), "n": len(x)})
    linhas = {"total": resumo(d)}
    for a, b in blocos:
        linhas[f"+{a}–{b}"] = resumo(d[(d["h"] >= a) & (d["h"] <= b)])
    return pd.DataFrame(linhas).T


def serie_diferencial(pred_a, pred_b, perda="mae"):
    """Per issuance: mean over sensors of (loss A − loss B), on the issuance–sensor pairs both have."""
    ea = erro_por_par(filtro_principal(pred_a), perda=perda)
    eb = erro_por_par(filtro_principal(pred_b), perda=perda)
    comum = ea.index.intersection(eb.index)
    d = (ea[comum] - eb[comum]).groupby(level="t_emissao").mean()
    return d.sort_index()


def diebold_mariano(d, h=2):
    """HLN-corrected DM on the loss differential series d (one value per daily issuance).

    h = 2 days: a 48 h forecast issued daily overlaps the next one, so the HAC variance uses h − 1 = 1
    autocovariance (rectangular kernel, as in Diebold & Mariano, 1995). Returns the statistic, two-sided
    p-value (Student t with n − 1 d.f.), mean difference and its 95% CI.
    """
    d = np.asarray(d, float)
    d = d[~np.isnan(d)]
    n = len(d)
    m = d.mean()
    dc = d - m
    gam = [np.sum(dc[k:] * dc[:n - k]) / n for k in range(h)]
    var = (gam[0] + 2 * sum(gam[1:])) / n
    if var <= 0:
        var = gam[0] / n
    dm = m / np.sqrt(var)
    fator = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    est = dm * fator
    p = 2 * stats.t.sf(abs(est), df=n - 1)
    tcrit = stats.t.ppf(0.975, df=n - 1)
    ep = np.sqrt(var) / fator
    return {"n": n, "dif_media": m, "estatistica": est, "p": p, "ic95": (m - tcrit * ep, m + tcrit * ep)}


def holm(pvalores):
    """Holm-adjusted p-values (same order as the input)."""
    p = np.asarray(pvalores, float)
    ordem = np.argsort(p)
    aj = np.empty_like(p)
    acum = 0.0
    for k, i in enumerate(ordem):
        acum = max(acum, min(1.0, (len(p) - k) * p[i]))
        aj[i] = acum
    return aj
