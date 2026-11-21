"""Runs one model in one cross-validation fold and writes its predictions (T21–T25).

Folds (doc 04 §3.6): train 2020 / 2020–2021 / 2020–2022 (39077 and 56831, 48 h embargo), evaluate
2021 / 2022 / 2023. Inputs chosen per fold by T20b (data/processed/selecao/escolha_<dobra>.json).

Usage (from the root):
    .venv/bin/python src/validacao_cruzada.py adversarios --dobra dobra1
    .venv/bin/python src/validacao_cruzada.py ridge  --dobra dobra1 --versao M3 --variante sem
    .venv/bin/python src/validacao_cruzada.py xgb    --dobra dobra1 --versao M3 --variante sem --semente 0
    .venv/bin/python src/validacao_cruzada.py bilstm --dobra dobra1 --versao M3 --variante sem --semente 0
Output: data/processed/vc/pred_<modelo>_<versao>_<variante>_<dobra>_s<semente>.parquet
(+ .json with the best epoch/number of trees and the training history).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "src" / "modelos"))
import conjunto  # noqa: E402

OUT = RAIZ / "data" / "processed" / "vc"


def longa(conj, idx_por_sensor, q_por_sensor, **meta):
    """dict sensor -> [N, 48] (median) or [N, 48, 3] -> long table with the observed target."""
    partes = []
    for s, idx in idx_por_sensor.items():
        if not len(idx):
            continue
        serie = conj.series[s]
        q = np.asarray(q_por_sensor[s])
        y = conjunto.alvo_janela(serie, idx)
        n = len(idx)
        d = pd.DataFrame({"sensor": s, "t_emissao": np.repeat(serie.tempo[idx], conjunto.HORIZONTE_H),
                          "h": np.tile(np.arange(1, conjunto.HORIZONTE_H + 1), n), "y": y.ravel()})
        if q.ndim == 3:
            d["q10"], d["q50"], d["q90"] = q[..., 0].ravel(), q[..., 1].ravel(), q[..., 2].ravel()
        else:
            d["q10"], d["q50"], d["q90"] = np.nan, q.ravel(), np.nan
        partes.append(d)
    out = pd.concat(partes, ignore_index=True)
    for k, v in meta.items():
        out.insert(0, k, v)
    return out


def rodar_adversarios(tab, dobra):
    import adversarios
    conj = conjunto.montar(tab, dobra, "M1", "sem")
    tr, av = conjunto.particao(conj)
    clim = adversarios.Climatologia().ajustar(conj, tr)
    saidas = []
    for nome, f in (("persistencia", adversarios.persistencia), ("persistencia_24h", adversarios.persistencia_24h),
                    ("climatologia", clim.prever)):
        q = {s: f(conj.series[s], idx) for s, idx in av.items() if len(idx)}
        saidas.append(longa(conj, av, q, modelo=nome, versao="-", variante="-", dobra=dobra, semente=0))
    return pd.concat(saidas, ignore_index=True), {}


def _a_cada(idx_por_sensor, conj, passo):
    return {s: idx[(conj.series[s].tempo[idx].hour % passo) == (4 % passo)] for s, idx in idx_por_sensor.items()}


def rodar_xgb(tab, dobra, versao, variante, semente):
    import tabular
    import xgb
    conj = conjunto.montar(tab, dobra, versao, variante)
    tr, av = conjunto.particao(conj)
    Xtr, ytr, _, nomes = tabular.matriz_horizontes(conj, _a_cada(tr, conj, xgb.PASSO_EMISSAO_TREINO_H))
    Xes, yes, _, _ = tabular.matriz_horizontes(conj, _a_cada(av, conj, xgb.PASSO_EMISSAO_TREINO_H))
    b = xgb.treinar(Xtr, ytr, Xes, yes, semente=semente)
    del Xtr, Xes
    q = {}
    for s, idx in av.items():
        if not len(idx):
            continue
        X, _, meta, _ = tabular.matriz_horizontes(conj, {s: idx})
        p = xgb.prever(b, X)  # rows ordered h-major
        n = len(idx)
        q[s] = p.reshape(conjunto.HORIZONTE_H, n, 3).transpose(1, 0, 2)
    pred = longa(conj, av, q, modelo="xgb", versao=versao, variante=variante, dobra=dobra, semente=semente)
    return pred, {"melhor_n_arvores": int(b.best_iteration + 1), "n_linhas_treino": int(len(ytr)), "n_entradas": len(nomes)}


def rodar_ridge(tab, dobra, versao, variante):
    import ridge
    import tabular
    conj = conjunto.montar(tab, dobra, versao, variante)
    tr, av = conjunto.particao(conj)
    Xs, ys = {}, {}
    for h in range(1, conjunto.HORIZONTE_H + 1):
        X, y, _, _ = tabular.matriz_horizontes(conj, tr, horizontes=[h])
        Xs[h] = X[:, :-1]
        ys[h] = y
    m = ridge.RidgePorHorizonte().ajustar(Xs, ys)
    q = {}
    for s, idx in av.items():
        if not len(idx):
            continue
        Xa = {h: tabular.matriz_horizontes(conj, {s: idx}, horizontes=[h])[0][:, :-1] for h in range(1, conjunto.HORIZONTE_H + 1)}
        p = m.prever(Xa)
        q[s] = np.column_stack([p[h] for h in range(1, conjunto.HORIZONTE_H + 1)])
    return longa(conj, av, q, modelo="ridge", versao=versao, variante=variante, dobra=dobra, semente=0), {}


def rodar_bilstm(tab, dobra, versao, variante, semente):
    import torch
    import bilstm
    torch.set_num_threads(8)
    print(f"dispositivo: {bilstm.DISPOSITIVO}", flush=True)
    conj = conjunto.montar(tab, dobra, versao, variante)
    tr, av = conjunto.particao(conj)
    modelo, ep, hist = bilstm.treinar(conj, tr, av, semente=semente, log=lambda m: print(m, flush=True))
    q = bilstm.prever(modelo, conj, av)
    pred = longa(conj, av, q, modelo="bilstm", versao=versao, variante=variante, dobra=dobra, semente=semente)
    return pred, {"melhor_epoca": int(ep), "historico": hist, "n_amostras_treino": int(sum(len(v) for v in tr.values()))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("modelo", choices=["adversarios", "ridge", "xgb", "bilstm"])
    ap.add_argument("--dobra", required=True, choices=["dobra1", "dobra2", "dobra3"])
    ap.add_argument("--versao", default="M3", choices=["M1", "M2", "M3"])
    ap.add_argument("--variante", default="sem", choices=["sem", "era5"])
    ap.add_argument("--semente", type=int, default=0)
    a = ap.parse_args()
    tab = pd.read_parquet(conjunto.ENTRADAS)
    t0 = time.time()
    if a.modelo == "adversarios":
        pred, info = rodar_adversarios(tab, a.dobra)
        nome = f"adversarios_-_-_{a.dobra}_s0"
    elif a.modelo == "ridge":
        pred, info = rodar_ridge(tab, a.dobra, a.versao, a.variante)
        nome = f"ridge_{a.versao}_{a.variante}_{a.dobra}_s0"
    elif a.modelo == "xgb":
        pred, info = rodar_xgb(tab, a.dobra, a.versao, a.variante, a.semente)
        nome = f"xgb_{a.versao}_{a.variante}_{a.dobra}_s{a.semente}"
    else:
        pred, info = rodar_bilstm(tab, a.dobra, a.versao, a.variante, a.semente)
        nome = f"bilstm_{a.versao}_{a.variante}_{a.dobra}_s{a.semente}"
    OUT.mkdir(parents=True, exist_ok=True)
    pred.to_parquet(OUT / f"pred_{nome}.parquet", index=False)
    info["duracao_s"] = round(time.time() - t0)
    (OUT / f"pred_{nome}.json").write_text(json.dumps(info, indent=1, default=float))
    print(f"{nome}: {len(pred)} linhas, {info['duracao_s']} s; {({k: v for k, v in info.items() if k != 'historico'})}")


if __name__ == "__main__":
    main()
