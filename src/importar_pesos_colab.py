"""Imports the Bi-LSTM fold-3 weights produced on Colab (T28) into resultados/pesos/.

Checks, before copying anything:
- conferencia_dobra3_colab.csv lists the 9 frozen networks, all with max_abs_diff_vc <= 1e-4;
- every .pt loads on CPU, matches its .meta.json (file name, best epoch, device) and has the
  expected input sizes (n_enc = len(nomes_enc), n_dec = len(nomes_dec));
- no existing file in resultados/pesos/ is overwritten.
Prints the SHA-256 of every imported file.

Usage: .venv/bin/python src/importar_pesos_colab.py <extracted t28_pesos folder>
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import torch

RAIZ = Path(__file__).resolve().parents[1]
DESTINO = RAIZ / "resultados" / "pesos"
ESPERADAS = {f"bilstm_{v}_{var}_dobra3_s{s}" for v, var in (("M2", "era5"), ("M1", "sem"), ("M2", "sem")) for s in (0, 1, 2)}


def sha256(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    origem = Path(sys.argv[1])
    conf = pd.read_csv(origem / "conferencia_dobra3_colab.csv")
    assert set(conf["tag"]) == ESPERADAS and len(conf) == 9, "Conferência não lista as 9 redes congeladas"
    assert (conf["max_abs_diff_vc"] <= 1e-4).all(), "Alguma rede não reproduziu a VC"
    arquivos = [origem / "conferencia_dobra3_colab.csv"]
    for tag in sorted(ESPERADAS):
        meta = json.loads((origem / f"{tag}.meta.json").read_text())
        ck = torch.load(origem / f"{tag}.pt", map_location="cpu")
        assert meta["arquivo_peso"] == f"{tag}.pt"
        assert ck["melhor_epoca"] == meta["melhor_epoca"] and ck["dispositivo"] == meta["dispositivo"]
        assert ck["n_enc"] == len(meta["nomes_enc"]) and ck["n_dec"] == len(meta["nomes_dec"])
        assert meta["max_abs_diff_vc"] <= 1e-4
        print(f"{tag}: época {meta['melhor_epoca']}, {meta['dispositivo']}, n_enc={ck['n_enc']}, "
              f"n_dec={ck['n_dec']}, max_dif_vc={meta['max_abs_diff_vc']:.1e}")
        arquivos += [origem / f"{tag}.pt", origem / f"{tag}.meta.json"]
    for f in arquivos:
        assert not (DESTINO / f.name).exists(), f"{f.name} já existe em resultados/pesos/"
    for f in arquivos:
        shutil.copy2(f, DESTINO / f.name)
        assert sha256(DESTINO / f.name) == sha256(f)
        print(f"copiado {f.name} sha256={sha256(f)}")
    print(f"\n{len(arquivos)} arquivos importados para {DESTINO}")


if __name__ == "__main__":
    main()
