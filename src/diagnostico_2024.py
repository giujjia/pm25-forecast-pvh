"""Diagnostics of the 2024 test (descriptive, run after the test): how far 2024 was from the training range.

Reads only saved outputs (data/processed/teste_2024/previsoes.parquet and the 2020–2023 input table);
tests nothing and changes no result. For the dry season (Jul–Oct), 04 UTC issuances, training sensors,
over all 48 horizons:
- mean and maximum of the PM2.5 measured by the corrected PurpleAir sensors in 2024;
- 99th percentile and maximum of the training target (2020–2022, training sensors, all hours);
- per model: maximum of the median forecast and mean bias (median − measured), seeds averaged.

Usage (from the root): .venv/bin/python src/diagnostico_2024.py
Output: data/processed/teste_2024/diagnostico_faixa.csv
"""
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
import avaliacao as av  # noqa: E402
import selecao_entradas as sel  # noqa: E402

TESTE = RAIZ / "data" / "processed" / "teste_2024"
MODELOS = [("bilstm", "M1", "sem"), ("bilstm", "M2", "era5"), ("bilstm", "M2", "sem"), ("xgb", "M2", "era5"),
           ("xgb", "M2", "sem"), ("ridge", "M2", "era5"), ("ridge", "M1", "sem"), ("persistencia", "-", "-"),
           ("cams_vies", "N7", "-"), ("cams_bruto", "-", "-")]


def main():
    tab = pd.read_parquet(RAIZ / "data" / "processed" / "entradas" / "entradas_horario_2020_2023.parquet",
                          columns=["sensor", "time_stamp", "barkjohn2022"])
    ts = pd.to_datetime(tab["time_stamp"], utc=True)
    treino = tab.loc[tab["sensor"].isin(sel.SENSORES_TREINO) & ts.dt.year.isin(sel.DOBRAS["dobra3"]["treino"]),
                     "barkjohn2022"].dropna()
    p = pd.read_parquet(TESTE / "previsoes.parquet",
                        columns=["modelo", "versao", "variante", "semente", "dobra", "sensor", "t_emissao", "h", "y", "q50"])
    p = p[p["dobra"] == "teste"]
    linhas = []
    obs = None
    for ch in MODELOS:
        g = p[(p["modelo"] == ch[0]) & (p["versao"] == ch[1]) & (p["variante"] == ch[2])]
        if g.empty:
            continue
        g = av.media_sementes(g) if g["semente"].nunique() > 1 else g
        s = av.filtro_principal(g).dropna(subset=["y"])
        if obs is None:
            obs = s
        linhas.append({"modelo": "_".join(ch), "n_horas": len(s), "max_mediana": s["q50"].max(),
                       "vies_medio": (s["q50"] - s["y"]).mean(), "max_medido": s["y"].max(), "media_medida": s["y"].mean()})
    d = pd.DataFrame(linhas)
    d["p99_treino_2020_2022"] = treino.quantile(0.99)
    d["max_treino_2020_2022"] = treino.max()
    d.to_csv(TESTE / "diagnostico_faixa.csv", index=False)
    print(f"treino 2020–2022 (sensores do treino, todas as horas): p99 = {treino.quantile(0.99):.1f}, máx = {treino.max():.1f}")
    print(f"2024, seca, emissões às 04 UTC, horas +1 a +48: média medida = {obs['y'].mean():.1f}, máx = {obs['y'].max():.1f}")
    print(d.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
