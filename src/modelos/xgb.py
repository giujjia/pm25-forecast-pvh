"""XGBoost with the horizon as input and 3 quantiles (T23, doc 04 §3.5–3.6).

One multi-quantile model (reg:quantileerror, alpha = 0.1, 0.5, 0.9) on log(1 + max(y, 0)), the same
pinball loss as the network. Early stopping on the fold's evaluated year, like the network's epochs
(doc 04 §3.6: the folds choose the number of epochs/trees).

IMPLEMENTAÇÃO (confirmed by the author on 18/11/2026; doc 04 §3.6 gives only the search space, and the first delivery has fixed
hyperparameters): learning_rate 0.05, max_depth 6, min_child_weight 5, subsample 0.8,
colsample_bytree 0.8, reg_lambda 1, up to 2000 trees, early stopping after 50 rounds.
Training rows: issuances every 3 h (hours ≡ 1 mod 3 UTC, which include 04 UTC) × 48 horizons, to keep
memory and time manageable on the CPU; consecutive issuances share almost all of their information.
Quantiles are sorted afterwards to guarantee Q10 <= Q50 <= Q90.
"""
import numpy as np
import xgboost as xgb

QUANTIS = (0.1, 0.5, 0.9)
PARAMS = dict(objective="reg:quantileerror", quantile_alpha=np.array(QUANTIS), tree_method="hist",
              learning_rate=0.05, max_depth=6, min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
              reg_lambda=1.0, nthread=8)
N_ARVORES = 2000
PACIENCIA = 50
PASSO_EMISSAO_TREINO_H = 3


def alvo_log(y):
    return np.log1p(np.maximum(y, 0.0))


def treinar(Xtr, ytr, Xav=None, yav=None, semente=0, n_arvores=None):
    mtr = ~np.isnan(ytr)
    dtr = xgb.QuantileDMatrix(Xtr[mtr], alvo_log(ytr[mtr]))
    p = dict(PARAMS, seed=int(semente))
    if Xav is not None:
        mav = ~np.isnan(yav)
        dav = xgb.QuantileDMatrix(Xav[mav], alvo_log(yav[mav]), ref=dtr)
        b = xgb.train(p, dtr, num_boost_round=N_ARVORES, evals=[(dav, "aval")],
                      early_stopping_rounds=PACIENCIA, verbose_eval=False)
    else:
        b = xgb.train(p, dtr, num_boost_round=int(n_arvores))
    return b


def prever(b, X):
    it = (0, b.best_iteration + 1) if getattr(b, "best_iteration", None) is not None else (0, 0)
    q = b.predict(xgb.DMatrix(X), iteration_range=it)
    q = np.sort(q, axis=1)
    return np.expm1(q)  # back to µg/m³ (quantiles are preserved by the monotone transform)
