"""Checks 1 and 3 (18/11/2026) on the cross-validation predictions, WITHOUT computing any error.

Memory: one file at a time, only the needed columns, keys as int64 (the first version, run E155,
kept every key of every file as text tuples and ran out of memory).

Check 1, integrity (data/processed/vc/pred_*.parquet):
  - quantiles finite and ordered (q10 <= q50 <= q90); horizons 1..48;
  - issuances only inside the evaluated year of the fold; nothing from 2024;
  - the target y equals that of the adversaries at the same (sensor, issuance, horizon);
  - the set of (sensor, issuance, horizon) of each file against the adversaries of the same fold;
  - seeds per file; issuances at 04 UTC present.
Check 3, stability of the 2024 rule (data/processed/vc/pred_*.json): per configuration, the 9 values
  (3 folds × 3 seeds) of the best epoch (network) and best number of trees (XGBoost).
Outputs: data/processed/conferencias/integridade_vc.csv and epocas_arvores_vc.csv
Usage: .venv/bin/python src/conferir_previsoes_vc.py
"""
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import selecao_entradas as sel

RAIZ = Path(__file__).resolve().parents[1]
VC = RAIZ / "data" / "processed" / "vc"
OUT = RAIZ / "data" / "processed" / "conferencias"
COLS = ["modelo", "semente", "sensor", "t_emissao", "h", "y", "q10", "q50", "q90"]


def chave(df):
    """int64 key of (sensor, issuance hour, horizon)."""
    horas = df["t_emissao"].astype("int64").to_numpy() // 3_600_000_000  # µs -> h
    sens = df["sensor"].astype("int64").to_numpy()  # sensor ids < 1 000 000
    return (horas * 1_000_000 + sens) * 100 + df["h"].to_numpy()


def referencia(dobra):
    t = pq.read_table(VC / f"pred_adversarios_-_-_{dobra}_s0.parquet",
                      columns=["modelo", "sensor", "t_emissao", "h", "y"]).to_pandas()
    t = t[t["modelo"] == "persistencia"]
    k = chave(t)
    ordem = np.argsort(k)
    return k[ordem], t["y"].to_numpy(float)[ordem]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    linhas = []
    for dobra in ("dobra1", "dobra2", "dobra3"):
        ref_k, ref_y = referencia(dobra)
        anos_av = set(sel.DOBRAS[dobra]["avaliacao"])
        for f in sorted(VC.glob(f"pred_*_{dobra}_s*.parquet")):
            p = pq.read_table(f, columns=COLS).to_pandas()
            for mod, g in p.groupby("modelo"):
                q = g[["q10", "q50", "q90"]].to_numpy(float)
                anos = set(g["t_emissao"].dt.year.unique())
                k = chave(g)
                uk = np.unique(k)
                pos = np.searchsorted(ref_k, k)
                pos = np.clip(pos, 0, len(ref_k) - 1)
                achou = ref_k[pos] == k
                ydif = np.abs(g["y"].to_numpy(float)[achou] - ref_y[pos[achou]])
                ydif = ydif[np.isfinite(ydif)]
                emis = g["t_emissao"].drop_duplicates()
                linhas.append(dict(
                    arquivo=f.name, modelo=mod, dobra=dobra, linhas=len(g),
                    sementes=g["semente"].nunique(),
                    quantis_nao_finitos=int((~np.isfinite(q)).sum()),
                    quantis_fora_de_ordem=int(((q[:, 0] > q[:, 1] + 1e-6) | (q[:, 1] > q[:, 2] + 1e-6)).sum()),
                    y_ausente=int(g["y"].isna().sum()),
                    h_fora_1_48=int((~g["h"].between(1, 48)).sum()),
                    anos_fora_da_avaliacao=",".join(str(a) for a in sorted(anos - anos_av)),
                    tem_2024=int(2024 in anos),
                    emissoes=len(emis), emissoes_04utc=int((emis.dt.hour == 4).sum()),
                    pares=len(uk), pares_adversarios=len(ref_k),
                    so_no_modelo=int((~np.isin(uk, ref_k, assume_unique=True)).sum()),
                    so_nos_adversarios=int((~np.isin(ref_k, uk, assume_unique=True)).sum()),
                    y_diferente_dos_adversarios=int((ydif > 1e-4).sum())))
            del p
            gc.collect()
    integ = pd.DataFrame(linhas)
    integ.to_csv(OUT / "integridade_vc.csv", index=False)

    ep = []
    for f in sorted(VC.glob("pred_*.json")):
        partes = f.stem.split("_")  # pred_<modelo>_<versao>_<variante>_<dobra>_s<semente>
        if partes[1] not in ("bilstm", "xgb"):
            continue
        j = json.loads(f.read_text())
        ep.append(dict(modelo=partes[1], versao=partes[2], variante=partes[3], dobra=partes[4],
                       semente=int(partes[5][1:]), valor=j.get("melhor_epoca", j.get("melhor_n_arvores"))))
    ep = pd.DataFrame(ep)
    res = ep.groupby(["modelo", "versao", "variante"])["valor"].agg(
        n="size", minimo="min", mediana="median", maximo="max").reset_index()
    pdob = ep.groupby(["modelo", "versao", "variante", "dobra"])["valor"].median().unstack("dobra")
    res = res.merge(pdob.add_prefix("mediana_").reset_index(), on=["modelo", "versao", "variante"])
    res.to_csv(OUT / "epocas_arvores_vc.csv", index=False)

    pd.set_option("display.width", 250)
    print("== Checagem 1: integridade (somas sobre", len(integ), "arquivo×modelo)")
    print(integ[["quantis_nao_finitos", "quantis_fora_de_ordem", "h_fora_1_48", "tem_2024",
                 "y_diferente_dos_adversarios"]].sum().to_string())
    print("anos fora da avaliação:", sorted(set(integ["anos_fora_da_avaliacao"]) - {""}))
    print("sementes por modelo:", integ.groupby("modelo")["sementes"].unique().to_dict())
    print(integ.groupby("dobra")[["emissoes", "emissoes_04utc", "y_ausente", "pares", "pares_adversarios"]]
          .agg(["min", "max"]).to_string())
    print("\npares diferentes dos adversários (por modelo, versão e dobra; máximo entre arquivos):")
    integ["config"] = integ["arquivo"].str.replace(r"_s\d+\.parquet$", "", regex=True)
    print(integ.groupby(["dobra", "config", "modelo"])[["pares", "so_no_modelo", "so_nos_adversarios"]]
          .max().reset_index().to_string(index=False))
    print("\n== Checagem 3: melhor época (rede) e melhor número de árvores (XGBoost), 3 dobras × 3 sementes")
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
