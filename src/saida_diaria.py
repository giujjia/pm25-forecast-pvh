"""Daily output by heteroscedastic extended logistic regression (doc 04 §3.5, lacuna 25).

For the issuance at 0h local (04 UTC of day D): day D = hours +1 … +24 and day D+1 = +25 … +48.
P(daily mean <= q) = Λ((θ · log(1 + q) − x'β) / exp(z'γ)) (Wilks, 2009; Messner et al., 2014), with
- position x = [1, log(1 + mean of the hourly predicted medians of the day), log(1 + max(mean PM2.5 of
  the 24 h before, 0))];
- scale z = [1, log(1 + mean width Q90 − Q10 of the day)];
- the threshold as a predictor, in log(1 + q). Fitted separately for D and D+1.
The observed daily mean only counts on a valid day (>= 18 of 24 hours).

IMPLEMENTAÇÃO (confirmed by the author on 18/11/2026): fitting thresholds = deciles 10–90% of the observed daily means of the fitting
data plus 50, 60 and 125 µg/m³; maximum likelihood (BFGS). For the selection Brier in the folds, the
regression of each fold is fitted on the out-of-fold predictions of the OTHER two folds (cross-fitting),
so no fold scores probabilities fitted on itself; for 2024 it is fitted on the three folds.
"""
import numpy as np
import pandas as pd
from scipy import optimize, special

LIMIARES_FIXOS = (50.0, 60.0, 125.0)
MIN_HORAS_DIA = 18


def tabela_diaria(pred, hist24):
    """pred: long table of one model (04 UTC issuances); hist24: Series indexed by (sensor, t_emissao).

    Returns one row per (sensor, t_emissao, dia ∈ {0, 1}) with m, largura, hist, ybar (NaN if invalid day).
    """
    d = pred.copy()
    d["dia"] = (d["h"] > 24).astype(int)
    d["larg"] = d["q90"] - d["q10"]
    g = d.groupby(["sensor", "t_emissao", "dia"])
    out = pd.DataFrame({"m": g["q50"].mean(), "largura": g["larg"].mean(),
                        "n_obs": g["y"].count(), "ybar": g["y"].mean()}).reset_index()
    out.loc[out["n_obs"] < MIN_HORAS_DIA, "ybar"] = np.nan
    out["hist"] = hist24.reindex(pd.MultiIndex.from_frame(out[["sensor", "t_emissao"]])).to_numpy()
    return out


def _mats(t):
    x = np.column_stack([np.ones(len(t)), np.log1p(np.maximum(t["m"], 0)), np.log1p(np.maximum(t["hist"], 0))])
    z = np.column_stack([np.ones(len(t)), np.log1p(np.maximum(t["largura"], 0))])
    return x, z


class HXLR:
    def ajustar(self, t):
        t = t.dropna(subset=["ybar", "m", "largura", "hist"])
        self.limiares = np.unique(np.concatenate([np.percentile(t["ybar"], np.arange(10, 100, 10)), LIMIARES_FIXOS]))
        x, z = _mats(t)
        y = t["ybar"].to_numpy()
        g = np.log1p(self.limiares)
        ind = (y[:, None] <= self.limiares[None, :]).astype(float)

        def nll(p):
            th, b, c = p[0], p[1:1 + x.shape[1]], p[1 + x.shape[1]:]
            eta = (th * g[None, :] - (x @ b)[:, None]) / np.exp(z @ c)[:, None]
            return -np.sum(ind * special.log_expit(eta) + (1 - ind) * special.log_expit(-eta))

        p0 = np.concatenate([[1.0], [0.0, 1.0, 0.0], [0.0, 0.0]])
        r = optimize.minimize(nll, p0, method="BFGS")
        self.p, self.ok = r.x, r.success
        return self

    def prob_acima(self, t, q):
        x, z = _mats(t)
        th, b, c = self.p[0], self.p[1:4], self.p[4:]
        eta = (th * np.log1p(q) - x @ b) / np.exp(z @ c)
        return 1 - special.expit(eta)

    def quantil(self, t, tau):
        x, z = _mats(t)
        th, b, c = self.p[0], self.p[1:4], self.p[4:]
        return np.expm1((x @ b + np.exp(z @ c) * special.logit(tau)) / th)


def brier(prob, ybar, q):
    ok = ~np.isnan(ybar) & ~np.isnan(prob)
    return float(np.mean((prob[ok] - (ybar[ok] > q)) ** 2)), int(ok.sum())
