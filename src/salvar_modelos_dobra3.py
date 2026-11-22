"""Reruns fold 3 of the frozen 2024 configurations, saves model weights and checks reproducibility (T28).

Per METHODS.md §6 (decided 18/11/2026) and doc 05 §2.1 (frozen 20/11/2026):
- Models evaluated in 2024 are the fold-3 models (trained on 2020–2022, early stopping on 2023),
  with the same 3 seeds and on the same device as CV:
  * --lote local : Ridge (M2 era5, M1 sem) and XGBoost (M2 era5, M2 sem × seeds 0–2) on local CPU;
  * --lote colab : Bi-LSTM (M2 era5, M1 sem, M2 sem × seeds 0–2) on Colab Tesla T4 GPU.
- Before saving as valid, each model's fold-3 predictions are compared against data/processed/vc/
  to verify exact reproduction (max |diff| <= 1e-4 µg/m³).

Usage (from the root):
    .venv/bin/python -u src/salvar_modelos_dobra3.py --lote local
    .venv/bin/python -u src/salvar_modelos_dobra3.py --lote colab
Outputs: resultados/pesos/{<modelo>_<versao>_<variante>_dobra3_s<sem>.{pkl|ubj|pt}, *.meta.json,
         conferencia_dobra3_<lote>.csv}
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "src" / "modelos"))
import conjunto  # noqa: E402
import validacao_cruzada as vc  # noqa: E402

VC_DIR = RAIZ / "data" / "processed" / "vc"
OUT = RAIZ / "resultados" / "pesos"
DOBRA = "dobra3"
TOL = 1e-4

LOTES = {
    "local": [
        ("ridge", "M2", "era5", 0),
        ("ridge", "M1", "sem", 0),
        ("xgb", "M2", "era5", 0),
        ("xgb", "M2", "era5", 1),
        ("xgb", "M2", "era5", 2),
        ("xgb", "M2", "sem", 0),
        ("xgb", "M2", "sem", 1),
        ("xgb", "M2", "sem", 2),
    ],
    "colab": [
        ("bilstm", "M2", "era5", 0),
        ("bilstm", "M2", "era5", 1),
        ("bilstm", "M2", "era5", 2),
        ("bilstm", "M1", "sem", 0),
        ("bilstm", "M1", "sem", 1),
        ("bilstm", "M1", "sem", 2),
        ("bilstm", "M2", "sem", 0),
        ("bilstm", "M2", "sem", 1),
        ("bilstm", "M2", "sem", 2),
    ],
}


def conferir_contra_vc(pred_novo, modelo, versao, variante, semente):
    """Compares fold-3 predictions against the saved VC parquet. Returns max abs diff across quantiles."""
    arq_vc = VC_DIR / f"pred_{modelo}_{versao}_{variante}_{DOBRA}_s{semente}.parquet"
    ref = pd.read_parquet(arq_vc)
    chaves = ["sensor", "t_emissao", "h"]
    a = pred_novo.sort_values(chaves).reset_index(drop=True)
    b = ref.sort_values(chaves).reset_index(drop=True)
    assert len(a) == len(b), f"Linhas divergentes: {len(a)} vs {len(b)}"
    assert (a["sensor"].to_numpy() == b["sensor"].to_numpy()).all()
    assert (a["h"].to_numpy() == b["h"].to_numpy()).all()
    assert (pd.to_datetime(a["t_emissao"], utc=True) == pd.to_datetime(b["t_emissao"], utc=True)).all()
    diffs = {}
    for col in ("q10", "q50", "q90"):
        va, vb = a[col].to_numpy(dtype=float), b[col].to_numpy(dtype=float)
        if np.isnan(va).all() and np.isnan(vb).all():
            diffs[col] = 0.0
        else:
            assert np.array_equal(np.isnan(va), np.isnan(vb)), f"NaN divergente em {col}"
            ok = ~np.isnan(va)
            diffs[col] = float(np.max(np.abs(va[ok] - vb[ok])))
    return max(diffs.values()), diffs


def treinar_e_salvar(tab, modelo, versao, variante, semente):
    t0 = time.perf_counter()
    tag = f"{modelo}_{versao}_{variante}_{DOBRA}_s{semente}"
    conj = conjunto.montar(tab, DOBRA, versao, variante)
    tr, av = conjunto.particao(conj)
    meta = {
        "modelo": modelo,
        "versao": versao,
        "variante": variante,
        "dobra": DOBRA,
        "semente": semente,
        "nomes_enc": conj.nomes_enc,
        "nomes_dec": conj.nomes_dec,
        "media": {k: float(v) for k, v in conj.media.items()},
        "desvio": {k: float(v) for k, v in conj.desvio.items()},
    }

    if modelo == "ridge":
        import ridge
        import tabular
        Xs, ys = {}, {}
        for h in range(1, conjunto.HORIZONTE_H + 1):
            X, y, _, nomes = tabular.matriz_horizontes(conj, tr, horizontes=[h])
            Xs[h] = X[:, :-1]
            ys[h] = y
        m = ridge.RidgePorHorizonte().ajustar(Xs, ys)
        q = {}
        for s, idx in av.items():
            if not len(idx):
                continue
            Xa = {h: tabular.matriz_horizontes(conj, {s: idx}, horizontes=[h])[0][:, :-1]
                  for h in range(1, conjunto.HORIZONTE_H + 1)}
            p = m.prever(Xa)
            q[s] = np.column_stack([p[h] for h in range(1, conjunto.HORIZONTE_H + 1)])
        pred = vc.longa(conj, av, q, modelo="ridge", versao=versao, variante=variante, dobra=DOBRA, semente=0)
        max_diff, diffs = conferir_contra_vc(pred, modelo, versao, variante, semente)
        assert max_diff <= TOL, f"{tag}: reprodutibilidade falhou (max_diff={max_diff:.6g})"
        arq_peso = OUT / f"{tag}.pkl"
        with open(arq_peso, "wb") as f:
            pickle.dump(m, f)
        meta["nomes_tabular"] = nomes[:-1]

    elif modelo == "xgb":
        import tabular
        import xgb
        Xtr, ytr, _, nomes = tabular.matriz_horizontes(conj, vc._a_cada(tr, conj, xgb.PASSO_EMISSAO_TREINO_H))
        Xes, yes, _, _ = tabular.matriz_horizontes(conj, vc._a_cada(av, conj, xgb.PASSO_EMISSAO_TREINO_H))
        b = xgb.treinar(Xtr, ytr, Xes, yes, semente=semente)
        del Xtr, Xes
        q = {}
        for s, idx in av.items():
            if not len(idx):
                continue
            X, _, _, _ = tabular.matriz_horizontes(conj, {s: idx})
            p = xgb.prever(b, X)
            n = len(idx)
            q[s] = p.reshape(conjunto.HORIZONTE_H, n, 3).transpose(1, 0, 2)
        pred = vc.longa(conj, av, q, modelo="xgb", versao=versao, variante=variante, dobra=DOBRA, semente=semente)
        max_diff, diffs = conferir_contra_vc(pred, modelo, versao, variante, semente)
        assert max_diff <= TOL, f"{tag}: reprodutibilidade falhou (max_diff={max_diff:.6g})"
        arq_peso = OUT / f"{tag}.ubj"
        b.save_model(str(arq_peso))
        meta["melhor_n_arvores"] = int(b.best_iteration + 1)
        meta["nomes_tabular"] = nomes

    elif modelo == "bilstm":
        import torch
        import bilstm
        torch.set_num_threads(8)
        m, ep, hist = bilstm.treinar(conj, tr, av, semente=semente, log=lambda msg: print(msg, flush=True))
        q = bilstm.prever(m, conj, av)
        pred = vc.longa(conj, av, q, modelo="bilstm", versao=versao, variante=variante, dobra=DOBRA, semente=semente)
        max_diff, diffs = conferir_contra_vc(pred, modelo, versao, variante, semente)
        assert max_diff <= TOL, f"{tag}: reprodutibilidade falhou (max_diff={max_diff:.6g})"
        arq_peso = OUT / f"{tag}.pt"
        torch.save({
            "state_dict": {k: v.cpu() for k, v in m.state_dict().items()},
            "hp": bilstm.HP,
            "n_enc": len(conj.nomes_enc),
            "n_dec": len(conj.nomes_dec),
            "melhor_epoca": int(ep),
            "dispositivo": str(bilstm.DISPOSITIVO),
        }, arq_peso)
        meta["melhor_epoca"] = int(ep)
        meta["dispositivo"] = str(bilstm.DISPOSITIVO)

    else:
        raise ValueError(f"Modelo desconhecido: {modelo}")

    dt = time.perf_counter() - t0
    meta["arquivo_peso"] = arq_peso.name
    meta["max_abs_diff_vc"] = max_diff
    meta["diffs_vc"] = diffs
    meta["duracao_s"] = round(dt, 2)
    (OUT / f"{tag}.meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"[OK] {tag}: max_diff_vc={max_diff:.2e} ({dt:.1f} s) -> {arq_peso.name}", flush=True)
    return {"tag": tag, "modelo": modelo, "versao": versao, "variante": variante, "semente": semente,
            "max_abs_diff_vc": max_diff, "duracao_s": round(dt, 2), "arquivo": arq_peso.name}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lote", choices=tuple(LOTES), required=True)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    tab = pd.read_parquet(conjunto.ENTRADAS)
    linhas = []
    for modelo, versao, variante, semente in LOTES[args.lote]:
        linhas.append(treinar_e_salvar(tab, modelo, versao, variante, semente))
    df = pd.DataFrame(linhas)
    df.to_csv(OUT / f"conferencia_dobra3_{args.lote}.csv", index=False)
    print("\n=== CONFERENCIA DOBRA 3 ===", flush=True)
    print(df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
