"""Ridge per horizon (doc 04 §3.5, lacuna 21): L2 linear regression on log(1 + max(y, 0)), one per
horizon, same inputs as XGBoost (without h). Median only, main metric only.

IMPLEMENTAÇÃO (confirmed by the author on 18/11/2026): alpha = 1.0 on standardised inputs (fixed, like the other hyperparameters of
the first delivery); missing values (FIRMS outage) are replaced by the training mean, and the outage
indicators are among the inputs.
"""
import numpy as np
from sklearn.linear_model import Ridge

ALPHA = 1.0


class RidgePorHorizonte:
    def ajustar(self, Xs, ys):
        """Xs, ys: dicts h -> training matrix/target."""
        self.m, self.mu, self.sd = {}, {}, {}
        for h, X in Xs.items():
            y = ys[h]
            ok = ~np.isnan(y)
            mu = np.nanmean(X[ok], axis=0)
            sd = np.nanstd(X[ok], axis=0)
            sd[sd == 0] = 1.0
            Z = np.nan_to_num((X[ok] - mu) / sd)
            self.m[h] = Ridge(alpha=ALPHA).fit(Z, np.log1p(np.maximum(y[ok], 0.0)))
            self.mu[h], self.sd[h] = mu, sd
        return self

    def prever(self, Xs):
        return {h: np.expm1(self.m[h].predict(np.nan_to_num((X - self.mu[h]) / self.sd[h]))) for h, X in Xs.items()}
