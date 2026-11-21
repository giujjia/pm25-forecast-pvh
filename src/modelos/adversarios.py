"""Simple adversaries (T22, doc 04 §3.5): persistence, 24 h persistence and climatology. Median only.

All use the input PM2.5 series (signed, gaps <= 3 h interpolated), only up to t.
"""
import numpy as np
import pandas as pd

import conjunto


def persistencia(serie, idx):
    """Repeats the last observed value (the value labelled t) for the 48 h."""
    return np.repeat(serie.pm_in[idx][:, None], conjunto.HORIZONTE_H, axis=1)


def persistencia_24h(serie, idx):
    """Repeats the curve of the last 24 h: hour t + h gets the value of t + h − 24 (h <= 24) or t + h − 48."""
    h = np.arange(1, conjunto.HORIZONTE_H + 1)
    desloc = np.where(h <= 24, h - 24, h - 48)
    return np.stack([serie.pm_in[i + desloc] for i in idx])


class Climatologia:
    """Mean per local hour of day and month, computed on the training issuances' hours only."""

    def ajustar(self, conj, idx_treino):
        partes = []
        for s, idx in idx_treino.items():
            serie = conj.series[s]
            horas = np.unique((idx[:, None] + np.arange(1, conjunto.HORIZONTE_H + 1)[None, :]).ravel())
            loc = serie.tempo[horas] - pd.Timedelta(hours=1) + pd.Timedelta(hours=conjunto.FUSO_H)
            partes.append(pd.DataFrame({"mes": loc.month, "hora": loc.hour, "y": serie.y[horas]}))
        d = pd.concat(partes).dropna()
        self.tab = d.groupby(["mes", "hora"])["y"].mean()
        self.geral = d["y"].mean()
        return self

    def prever(self, serie, idx):
        j = idx[:, None] + np.arange(1, conjunto.HORIZONTE_H + 1)[None, :]
        loc = serie.tempo[j.ravel()] - pd.Timedelta(hours=1) + pd.Timedelta(hours=conjunto.FUSO_H)
        chave = pd.MultiIndex.from_arrays([loc.month, loc.hour])
        v = self.tab.reindex(chave).fillna(self.geral).to_numpy()
        return v.reshape(j.shape)
