"""Test of 2024 (T29–T30), run once, with the configuration frozen on 20/11/2026 (doc 05 §2.1).

Applies the fold-3 models saved by src/salvar_modelos_dobra3.py (T28) to 2024 and evaluates only the
six frozen comparisons of the first delivery (doc 05 §2.1, "Primeira entrega" table):
  1. bilstm M2 era5 (upper bound, perfect weather) × persistence, 24 h persistence, climatology,
     ridge M2 era5, xgb M2 era5, raw CAMS (12 UTC D−1) and CAMS with moving bias (N = 7);
     N ∈ {1, 3, 14} as sensitivity (reported, outside the Holm family);
  2. bilstm M1 sem (no future weather) × the same adversaries, with ridge M1 sem and xgb M2 sem;
  3. bilstm M2 era5 × bilstm M2 sem (value of future weather, OE6);
  4. curve by horizon (MAE, RMSE, R²), MAE > 50, skill, daily alerts at 50, 60 and 125 µg/m³
     (precision, recall, F1 with block bootstrap, Brier and its skill, PR curve, daily × hourly
     coherence) and the descriptive rain cut (>= 1 mm in the 24 h before t, ERA5 at the sensor cell);
  5. uncertainty (OE2) of bilstm M2 era5, bilstm M1 sem and xgb M2 era5: pinball, Q10–Q90 coverage
     and width, total, per 12 h block and per regime known at t;
  6. sensors outside training (OE5): bilstm M2 era5 and bilstm M1 sem on 219595, 219981, 225327
     (mean of the three per issuance) and 128603 (Cuniã, apart), against the local persistences and
     the noise of the co-located pair (219981 × 225327).

Guards:
- 2024 requires --abrir and the 9 Bi-LSTM weights checked in T28 (Colab, T4 GPU).
- Load check: every saved model is reloaded and its fold-3 (2023) prediction is compared with the CV
  parquet in data/processed/vc/ before anything is predicted for the test year. Ridge, XGBoost and the
  simple adversaries must match exactly (<= 1e-4 µg/m³); the Bi-LSTM was trained on GPU and is run here
  on CPU, so it must match within TOL_BILSTM (floating point is not bit-identical across devices,
  doc 04 §7); the observed difference is saved.
- Input check (2024): the 2020–2023 rows of the 2020–2024 input table must equal the frozen
  2020–2023 table, and each model's standardisation must equal the one saved with its weights.
- --ensaio runs the whole pipeline on 2023 (the fold-3 year, already seen in CV) to test the code
  without opening 2024. Its numbers are not results: the daily output (HXLR) is fitted on folds that
  include 2023, and the OE5 block uses the training sensors as stand-ins. Output goes to a separate folder.

IMPLEMENTAÇÃO (choices not fixed by the frozen table; confirmed by the author on 22/11/2026, before opening
2024, and recorded in doc 05 §2.1):
- every model applied in 2024 is the fold-3 one, including the climatology (fitted on the fold-3
  training issuances, 2020–2022), as in CV;
- the main metric is reported on each model's own pairs and on the pairs common to every model of the
  comparison (headline: common pairs, so all models are scored on the same issuances);
- alerts, Brier and coherence on the 04 UTC issuances of Jul–Oct, for D and D+1 separately;
  "above 50/60" is ybar > q and "Atenção" is ybar >= 125; the Brier climatology is the monthly
  frequency of the event over valid days of the training sensors in 2020–2022;
- bootstrap: 1000 resamples of blocks of 7 consecutive issuance days (both sensors and D/D+1 of a day
  stay together), seed SEMENTE_BOOTSTRAP, 95% percentile interval;
- Holm per comparison family (row 1, row 2; row 6 per model and group); row 3 is a single test.

Usage (from the root, through executar.py):
    .venv/bin/python -u src/teste_2024.py --ensaio                       # 2023, code test
    .venv/bin/python -u src/teste_2024.py --ano 2024 --abrir
Outputs: data/processed/teste_2024/ (or data/processed/ensaio_teste_2023/).
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "src" / "modelos"))
import avaliacao as av  # noqa: E402
import cams_vies_movel as cvm  # noqa: E402
import conjunto  # noqa: E402
import saida_diaria as sd  # noqa: E402
import selecao_entradas as sel  # noqa: E402
import validacao_cruzada as vc  # noqa: E402

PESOS = RAIZ / "resultados" / "pesos"
VC_DIR = RAIZ / "data" / "processed" / "vc"
ENTRADAS_DIR = RAIZ / "data" / "processed" / "entradas"
INVENTARIO = RAIZ / "data" / "raw" / "purpleair" / "inventario.csv"

DOBRA = "dobra3"
SEMENTES = (0, 1, 2)
TOL_EXATA = 1e-4
TOL_BILSTM = 0.05  # µg/m³, GPU (training) × CPU (here)
SEMENTE_BOOTSTRAP = 20261122
N_BOOTSTRAP = 1000
BLOCO_DIAS = 7
LIMIARES = {50.0: 0.50, 60.0: 0.50, 125.0: 0.33}  # threshold -> p*
LIMIAR_INCLUSIVO = {125.0}  # "dia >= 125" (Atenção); the others are "acima de"
URBANOS_OE5 = (219595, 219981, 225327)
CUNIA = 128603
PAR_COLOCALIZADO = (219981, 225327)
N_CAMS = (7, 1, 3, 14)

# frozen configurations (doc 05 §2.1): (modelo, versao, variante) -> seeds
CONFIGS = {
    ("bilstm", "M2", "era5"): SEMENTES,
    ("bilstm", "M1", "sem"): SEMENTES,
    ("bilstm", "M2", "sem"): SEMENTES,
    ("ridge", "M2", "era5"): (0,),
    ("ridge", "M1", "sem"): (0,),
    ("xgb", "M2", "era5"): SEMENTES,
    ("xgb", "M2", "sem"): SEMENTES,
}
ADV_SIMPLES = ("persistencia", "persistencia_24h", "climatologia")
COMPARACOES = {
    "1_principal_era5": (("bilstm", "M2", "era5"),
                         [("persistencia", "-", "-"), ("persistencia_24h", "-", "-"), ("climatologia", "-", "-"),
                          ("ridge", "M2", "era5"), ("xgb", "M2", "era5"), ("cams_bruto", "-", "-"),
                          ("cams_vies", "N7", "-")]),
    "2_piso_sem": (("bilstm", "M1", "sem"),
                   [("persistencia", "-", "-"), ("persistencia_24h", "-", "-"), ("climatologia", "-", "-"),
                    ("ridge", "M1", "sem"), ("xgb", "M2", "sem"), ("cams_bruto", "-", "-"),
                    ("cams_vies", "N7", "-")]),
}
PROBABILISTICOS = [("bilstm", "M2", "era5"), ("bilstm", "M1", "sem"), ("xgb", "M2", "era5"), ("xgb", "M2", "sem")]
INCERTEZA_OE2 = [("bilstm", "M2", "era5"), ("bilstm", "M1", "sem"), ("xgb", "M2", "era5")]


def rotulo(ch):
    return "_".join(ch)


# ---------------------------------------------------------------- inputs

def carregar_tabela(ano):
    if ano == 2023:
        tab = pd.read_parquet(ENTRADAS_DIR / "entradas_horario_2020_2023.parquet")
        tab["time_stamp"] = pd.to_datetime(tab["time_stamp"], utc=True)
        return tab
    nova = pd.read_parquet(ENTRADAS_DIR / "entradas_horario_2020_2024.parquet")
    nova["time_stamp"] = pd.to_datetime(nova["time_stamp"], utc=True)
    velha = pd.read_parquet(ENTRADAS_DIR / "entradas_horario_2020_2023.parquet")
    velha["time_stamp"] = pd.to_datetime(velha["time_stamp"], utc=True)
    fim = velha["time_stamp"].max()
    a = velha.sort_values(["sensor", "time_stamp"]).reset_index(drop=True)
    b = nova[nova["time_stamp"] <= fim].sort_values(["sensor", "time_stamp"]).reset_index(drop=True)
    assert len(a) == len(b), f"2020–2023 mudou de tamanho: {len(a)} × {len(b)}"
    assert list(a.columns) == list(b.columns), "Colunas da tabela de entradas mudaram"
    for c in a.columns:
        if c == "dia_local":
            continue  # the last local day of 2023 gets more hours once 2024 exists
        x, y = a[c], b[c]
        if pd.api.types.is_numeric_dtype(x) and not pd.api.types.is_bool_dtype(x):
            ok = np.allclose(x.to_numpy(float), y.to_numpy(float), rtol=0, atol=1e-9, equal_nan=True)
        else:
            ok = bool((x.astype(str).to_numpy() == y.astype(str).to_numpy()).all())
        if not ok and c in ("dia_valido", "horas_validas_dia"):
            # only the local day that straddles 31/12/2023 → 01/01/2024 may change
            dif = a.loc[x.astype(str).to_numpy() != y.astype(str).to_numpy(), "time_stamp"]
            ok = bool((dif >= pd.Timestamp("2023-12-31 04:00", tz="UTC")).all())
        assert ok, f"Coluna {c} de 2020–2023 difere da tabela congelada"
    print(f"conferência de entradas: 2020–2023 idêntico à tabela congelada ({len(a)} linhas)", flush=True)
    return nova


def hist24_de(tab):
    """Mean observed PM2.5 over (t − 24 h, t] with >= 18 valid hours, per sensor (as src/avaliar_vc.py)."""
    partes = []
    for s, g in tab[["sensor", "time_stamp", "barkjohn2022"]].groupby("sensor"):
        g = g.set_index("time_stamp").sort_index()
        g = g.reindex(pd.date_range(g.index.min(), g.index.max(), freq="h"))
        h24 = g["barkjohn2022"].rolling(24, min_periods=18).mean()
        partes.append(pd.DataFrame({"sensor": s, "t_emissao": h24.index, "hist24": h24.to_numpy()}))
    return pd.concat(partes, ignore_index=True).set_index(["sensor", "t_emissao"])["hist24"]


# ---------------------------------------------------------------- models

def meta_de(modelo, versao, variante, semente):
    return json.loads((PESOS / f"{modelo}_{versao}_{variante}_{DOBRA}_s{semente}.meta.json").read_text())


def conferir_padronizacao(conj, meta):
    assert conj.nomes_enc == meta["nomes_enc"] and conj.nomes_dec == meta["nomes_dec"], "Entradas mudaram"
    for k, v in meta["media"].items():
        assert abs(conj.media[k] - v) <= 1e-9 * max(1.0, abs(v)), f"Média de {k} mudou"
        assert abs(conj.desvio[k] - meta["desvio"][k]) <= 1e-9 * max(1.0, abs(meta["desvio"][k])), f"Desvio de {k} mudou"


def prever_ridge(conj, idx, m):
    q = {}
    for s, ii in idx.items():
        if not len(ii):
            continue
        import tabular
        Xa = {h: tabular.matriz_horizontes(conj, {s: ii}, horizontes=[h])[0][:, :-1]
              for h in range(1, conjunto.HORIZONTE_H + 1)}
        p = m.prever(Xa)
        q[s] = np.column_stack([p[h] for h in range(1, conjunto.HORIZONTE_H + 1)])
    return q


def prever_xgb(conj, idx, b, n_arvores):
    import tabular
    import xgboost
    q = {}
    for s, ii in idx.items():
        if not len(ii):
            continue
        X, _, _, _ = tabular.matriz_horizontes(conj, {s: ii})
        p = np.expm1(np.sort(b.predict(xgboost.DMatrix(X), iteration_range=(0, n_arvores)), axis=1))
        q[s] = p.reshape(conjunto.HORIZONTE_H, len(ii), 3).transpose(1, 0, 2)
    return q


def carregar_bilstm(arq):
    import torch
    import bilstm
    ck = torch.load(arq, map_location="cpu")
    m = bilstm.BiLSTMQuantis(ck["n_enc"], ck["n_dec"], ck["hp"])
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m


def so_validos(idx):
    return {s: ii for s, ii in idx.items() if len(ii)}


def comparar_com_vc(pred, arq_vc, tol):
    """Max |diff| of the quantiles on the CV keys; every CV key must be present in pred."""
    ref = pd.read_parquet(arq_vc)
    chaves = ["modelo", "sensor", "t_emissao", "h"]
    for d in (ref, pred):
        d["t_emissao"] = pd.to_datetime(d["t_emissao"], utc=True)
    m = ref[chaves + ["q10", "q50", "q90"]].merge(pred[chaves + ["q10", "q50", "q90"]], on=chaves,
                                                    how="left", suffixes=("_vc", "_novo"), indicator=True)
    assert (m["_merge"] == "both").all(), f"{arq_vc.name}: {int((m['_merge'] != 'both').sum())} chaves da VC ausentes"
    dif = 0.0
    for c in ("q10", "q50", "q90"):
        a, b = m[f"{c}_vc"].to_numpy(float), m[f"{c}_novo"].to_numpy(float)
        assert np.array_equal(np.isnan(a), np.isnan(b)), f"{arq_vc.name}: NaN divergente em {c}"
        ok = ~np.isnan(a)
        if ok.any():
            dif = max(dif, float(np.max(np.abs(a[ok] - b[ok]))))
    assert dif <= tol, f"{arq_vc.name}: carga não reproduz a VC (max |dif| = {dif:.3g} > {tol})"
    return dif, len(ref)


def prever_todos(tab, ano, ensaio):
    """Predictions of every frozen model and simple adversary for the test year (training sensors and
    OE5 sensors), after the load check on fold 3. Returns (long table, load-check table)."""
    sens_oe5 = sel.SENSORES_TREINO if ensaio else sel.SENSORES_AVALIACAO_OE5
    previsoes, carga = [], []
    por_conj = {}
    for (modelo, versao, variante), sementes in CONFIGS.items():
        por_conj.setdefault((versao, variante), []).append((modelo, sementes))
    for (versao, variante), modelos in por_conj.items():
        conj = conjunto.montar(tab, DOBRA, versao, variante)
        _, av23 = conjunto.particao(conj)
        idx_teste = {s: conjunto.emissoes_validas(conj.series[s], anos=[ano]) for s in sel.SENSORES_TREINO}
        idx_oe5 = {s: conjunto.emissoes_validas(conj.series[s], anos=[ano]) for s in sens_oe5 if s in conj.series}
        conjuntos_idx = {"teste": so_validos(idx_teste), "oe5": so_validos(idx_oe5)}
        if not ensaio:
            assert all(len(v) for v in conjuntos_idx["oe5"].values()) and len(conjuntos_idx["oe5"]) == 4, \
                "Sensores da OE5 sem emissões em 2024"
        for modelo, sementes in modelos:
            for s_ in sementes:
                arq_meta = PESOS / f"{modelo}_{versao}_{variante}_{DOBRA}_s{s_}.meta.json"
                if not arq_meta.exists():
                    if ensaio and modelo == "bilstm":
                        # stand-in: in 2023 the CV parquet is exactly what the saved weights would give
                        print(f"[ensaio] sem pesos de {arq_meta.stem}; usa a previsão da VC como substituta", flush=True)
                        p = pd.read_parquet(VC_DIR / f"pred_{modelo}_{versao}_{variante}_{DOBRA}_s{s_}.parquet")
                        previsoes += [p.assign(dobra="teste"), p.assign(dobra="oe5")]
                        continue
                    raise SystemExit(f"Faltam os pesos de {arq_meta.stem} (T28)")
                meta = meta_de(modelo, versao, variante, s_)
                conferir_padronizacao(conj, meta)
                arq = PESOS / meta["arquivo_peso"]
                if modelo == "ridge":
                    with open(arq, "rb") as f:
                        m = pickle.load(f)
                    fazer = lambda idx, m=m: prever_ridge(conj, idx, m)  # noqa: E731
                    tol = TOL_EXATA
                elif modelo == "xgb":
                    import xgboost
                    b = xgboost.Booster()
                    b.load_model(str(arq))
                    n = int(meta["melhor_n_arvores"])
                    fazer = lambda idx, b=b, n=n: prever_xgb(conj, idx, b, n)  # noqa: E731
                    tol = TOL_EXATA
                else:
                    import bilstm
                    m = carregar_bilstm(arq)
                    fazer = lambda idx, m=m: bilstm.prever(m, conj, idx)  # noqa: E731
                    tol = TOL_BILSTM
                kw = dict(modelo=modelo, versao=versao, variante=variante, semente=s_)
                p23 = vc.longa(conj, so_validos(av23), fazer(so_validos(av23)), dobra=DOBRA, **kw)
                dif, n = comparar_com_vc(p23, VC_DIR / f"pred_{modelo}_{versao}_{variante}_{DOBRA}_s{s_}.parquet", tol)
                carga.append({**kw, "max_abs_dif_vc": dif, "linhas_vc": n, "tolerancia": tol})
                print(f"carga {modelo} {versao} {variante} s{s_}: max |dif| VC = {dif:.2e}", flush=True)
                for nome, idx in conjuntos_idx.items():
                    if idx and (nome == "teste" or modelo == "bilstm"):
                        previsoes.append(vc.longa(conj, idx, fazer(idx), dobra=nome, **kw))
        if (versao, variante) == ("M1", "sem"):
            import adversarios
            tr, _ = conjunto.particao(conj)
            clim = adversarios.Climatologia().ajustar(conj, tr)
            fs = {"persistencia": adversarios.persistencia, "persistencia_24h": adversarios.persistencia_24h,
                  "climatologia": clim.prever}
            p23 = []
            for nome, f in fs.items():
                q = {s: f(conj.series[s], ii) for s, ii in so_validos(av23).items()}
                p23.append(vc.longa(conj, so_validos(av23), q, modelo=nome, versao="-", variante="-", dobra=DOBRA, semente=0))
                for grupo, idx in conjuntos_idx.items():
                    if grupo == "oe5" and nome == "climatologia":
                        continue  # doc 04 §3.8: no climatology on the new sensors
                    q = {s: f(conj.series[s], ii) for s, ii in idx.items()}
                    previsoes.append(vc.longa(conj, idx, q, modelo=nome, versao="-", variante="-", dobra=grupo, semente=0))
            dif, n = comparar_com_vc(pd.concat(p23, ignore_index=True), VC_DIR / f"pred_adversarios_-_-_{DOBRA}_s0.parquet", TOL_EXATA)
            carga.append({"modelo": "adversarios", "versao": "-", "variante": "-", "semente": 0,
                          "max_abs_dif_vc": dif, "linhas_vc": n, "tolerancia": TOL_EXATA})
            print(f"carga adversarios simples: max |dif| VC = {dif:.2e}", flush=True)
            series_m1 = conj.series
    pred = pd.concat(previsoes, ignore_index=True)
    pred["t_emissao"] = pd.to_datetime(pred["t_emissao"], utc=True)
    return pred, pd.DataFrame(carga), series_m1


def prever_cams(series, ano):
    """Raw CAMS and CAMS with moving bias (N = 7, 1, 3, 14) for the 04 UTC issuances of Jul–Oct."""
    inv = pd.read_csv(INVENTARIO).set_index("sensor_index")
    arquivos = sorted((RAIZ / "data" / "raw" / "cams").glob(f"cams_pm25_{ano}*.nc"))
    assert len(arquivos) == 6, f"CAMS de {ano}: {len(arquivos)} arquivos"
    emissoes = pd.date_range(f"{ano}-07-01 04:00", f"{ano}-10-31 04:00", freq="D", tz="UTC")
    partes = []
    for s in sel.SENSORES_TREINO:
        serie = series[s]
        medido = pd.Series(serie.y, index=serie.tempo)
        cams = cvm.cams_na_celula(arquivos, inv.loc[s, "latitude"], inv.loc[s, "longitude"])
        for n in N_CAMS:
            d = cvm.prever(cams, medido, emissoes, n_dias=n)
            d["y"] = medido.reindex(d["t_emissao"] + pd.to_timedelta(d["h"], unit="h")).to_numpy()
            if n == N_CAMS[0]:
                partes.append(pd.DataFrame({"modelo": "cams_bruto", "versao": "-", "variante": "-", "semente": 0,
                                            "dobra": "teste", "sensor": s, "t_emissao": d["t_emissao"], "h": d["h"],
                                            "y": d["y"], "q10": np.nan, "q50": d["cams_bruto"], "q90": np.nan}))
            partes.append(pd.DataFrame({"modelo": "cams_vies", "versao": f"N{n}", "variante": "-", "semente": 0,
                                        "dobra": "teste", "sensor": s, "t_emissao": d["t_emissao"], "h": d["h"],
                                        "y": d["y"], "q10": np.nan, "q50": d["cams_vies"], "q90": np.nan,
                                        "fator": d["fator"]}))
    out = pd.concat(partes, ignore_index=True)
    out["t_emissao"] = pd.to_datetime(out["t_emissao"], utc=True)
    return out


# ---------------------------------------------------------------- evaluation helpers

def por_chave(pred):
    """dict (modelo, versao, variante) -> seed-averaged long table."""
    out = {}
    for ch, g in pred.groupby(["modelo", "versao", "variante"], sort=False):
        g = g.drop(columns=[c for c in ("fator",) if c in g.columns])
        out[ch] = av.media_sementes(g) if g["semente"].nunique() > 1 else g
    return out


def erro_pares(p, perda="mae"):
    return av.erro_por_par(av.filtro_principal(p), perda=perda)


def tabela_principal(modelos, chaves, nome):
    erros = {ch: erro_pares(modelos[ch]) for ch in chaves}
    comum = None
    for e in erros.values():
        comum = e.index if comum is None else comum.intersection(e.index)
    linhas = []
    for ch in chaves:
        e50 = erro_pares(modelos[ch], perda="mae50")
        linhas.append({"comparacao": nome, "modelo": rotulo(ch), "mae_comum": float(erros[ch][comum].mean()),
                       "n_pares_comum": len(comum), "mae50_comum": float(e50.reindex(comum).mean()),
                       "mae_proprio": float(erros[ch].mean()), "n_pares_proprio": len(erros[ch])})
    return pd.DataFrame(linhas)


def linha_dm(grupo, a, b, pa, pb):
    r = av.diebold_mariano(av.serie_diferencial(pa, pb))
    return {"grupo": grupo, "a": a, "b": b, "n_emissoes": r["n"], "dif_mae_a_menos_b": r["dif_media"],
            "ic95_inf": r["ic95"][0], "ic95_sup": r["ic95"][1], "estatistica_dm": r["estatistica"], "p_valor": r["p"]}


def com_holm(df, colunas_familia=("grupo",)):
    df = df.copy()
    df["p_holm"] = np.nan
    fam = df[df["familia_holm"]]
    chave = colunas_familia[0] if len(colunas_familia) == 1 else list(colunas_familia)
    for _, idx in fam.groupby(chave).groups.items():
        df.loc[idx, "p_holm"] = av.holm(df.loc[idx, "p_valor"].to_numpy())
    return df


# ---- daily output and alerts

def evento(ybar, q):
    return (ybar >= q) if q in LIMIAR_INCLUSIVO else (ybar > q)


def dias_seca04(pred, hist24):
    t = pd.to_datetime(pred["t_emissao"], utc=True)
    p = pred[(t.dt.hour == av.HORA_EMISSAO_UTC) & t.dt.month.isin(av.MESES_SECA)]
    tb = sd.tabela_diaria(p, hist24)
    tb["t_emissao"] = pd.to_datetime(tb["t_emissao"], utc=True)
    return tb


def ajustar_hxlr(ch, hist24_vc):
    """HXLR per day (D, D+1) fitted on the out-of-fold CV predictions of the three folds (doc 05 §2.1)."""
    modelo, versao, variante = ch
    ps = [VC_DIR / f"pred_{modelo}_{versao}_{variante}_dobra{d}_s{s}.parquet" for d in (1, 2, 3) for s in SEMENTES]
    p = av.media_sementes(pd.concat([pd.read_parquet(f) for f in ps], ignore_index=True))
    p["t_emissao"] = pd.to_datetime(p["t_emissao"], utc=True)
    assert p["t_emissao"].dt.year.max() <= 2023
    p = p[p["t_emissao"].dt.hour == av.HORA_EMISSAO_UTC]
    tb = sd.tabela_diaria(p, hist24_vc)
    return {dia: sd.HXLR().ajustar(tb[tb["dia"] == dia]) for dia in (0, 1)}


def freq_climatologica(tab):
    """Monthly frequency of each daily event over valid days of the training sensors, 2020–2022."""
    t = tab[tab["sensor"].isin(sel.SENSORES_TREINO) & tab["dia_valido"].astype(bool)]
    t = t[pd.to_datetime(t["dia_local"]).dt.year.isin(sel.DOBRAS[DOBRA]["treino"])]
    dia = t.groupby(["sensor", "dia_local"])["barkjohn2022"].mean().reset_index()
    dia["mes"] = pd.to_datetime(dia["dia_local"]).dt.month
    return {q: dia.assign(e=evento(dia["barkjohn2022"], q)).groupby("mes")["e"].mean() for q in LIMIARES}


def prf(alerta, real):
    tp = int((alerta & real).sum()); fp = int((alerta & ~real).sum()); fn = int((~alerta & real).sum())
    tn = int((~alerta & ~real).sum())
    pr = tp / (tp + fp) if tp + fp else np.nan
    rc = tp / (tp + fn) if tp + fn else np.nan
    f1 = 2 * pr * rc / (pr + rc) if (pr == pr and rc == rc and pr + rc > 0) else np.nan
    return tp, fp, fn, tn, pr, rc, f1


def blocos_bootstrap(dias_unicos, rng):
    n = len(dias_unicos)
    if n <= BLOCO_DIAS:
        return dias_unicos[rng.integers(0, n, n)]
    inicios = rng.integers(0, n - BLOCO_DIAS + 1, int(np.ceil(n / BLOCO_DIAS)))
    pos = (inicios[:, None] + np.arange(BLOCO_DIAS)[None, :]).ravel()[:n]
    return dias_unicos[pos]


def ic_bootstrap(df_alerta, rng):
    """95% CI of precision, recall and F1 by block bootstrap of issuance days. df: t_emissao, alerta, real."""
    dias = np.array(sorted(df_alerta["t_emissao"].unique()))
    grupos = {d: g for d, g in df_alerta.groupby("t_emissao")}
    reps = []
    for _ in range(N_BOOTSTRAP):
        amostra = pd.concat([grupos[d] for d in blocos_bootstrap(dias, rng)], ignore_index=True)
        reps.append(prf(amostra["alerta"].to_numpy(bool), amostra["real"].to_numpy(bool))[4:])
    reps = np.array(reps, float)
    lo, hi = np.nanpercentile(reps, 2.5, axis=0), np.nanpercentile(reps, 97.5, axis=0)
    return dict(precisao_ic95=(lo[0], hi[0]), recall_ic95=(lo[1], hi[1]), f1_ic95=(lo[2], hi[2]))


def avaliar_alertas(modelos, hist24_teste, hist24_vc, freq_clim, ensaio):
    rng = np.random.default_rng(SEMENTE_BOOTSTRAP)
    linhas, brier_l, pr_l, coer_l = [], [], [], []
    chaves = [ch for ch in modelos if modelos[ch]["sensor"].isin(sel.SENSORES_TREINO).any()]
    for ch in chaves:
        p = modelos[ch][modelos[ch]["sensor"].isin(sel.SENSORES_TREINO)]
        tb = dias_seca04(p, hist24_teste)
        tb = tb.dropna(subset=["ybar"]).copy()
        prob = ch in PROBABILISTICOS
        if prob:
            hx = ajustar_hxlr(ch, hist24_vc)
            tb = tb.dropna(subset=["m", "largura", "hist"])
        for dia in (0, 1):
            t = tb[tb["dia"] == dia]
            mes = (t["t_emissao"] + pd.to_timedelta(dia, unit="D")).dt.month.to_numpy()
            for q, pstar in LIMIARES.items():
                real = evento(t["ybar"], q).to_numpy(bool)
                det = evento(t["m"], q).to_numpy(bool)
                if prob:
                    pr = hx[dia].prob_acima(t, q)
                    alerta = pr > pstar
                    bs = float(np.mean((pr - real) ** 2))
                    pc = freq_clim[q].reindex(mes).to_numpy(float)
                    bs_clim = float(np.mean((pc - real) ** 2))
                    brier_l.append({"modelo": rotulo(ch), "dia": "D" if dia == 0 else "D+1", "limiar": q,
                                    "brier": bs, "brier_clim_mensal": bs_clim,
                                    "skill_brier": 1 - bs / bs_clim if bs_clim > 0 else np.nan,
                                    "n_dias": len(t), "n_eventos": int(real.sum()), "hxlr_ok": bool(hx[dia].ok)})
                    for ps in np.round(np.arange(0.01, 1.0, 0.01), 2):
                        _, _, _, _, a, b, f = prf(pr > ps, real)
                        pr_l.append({"modelo": rotulo(ch), "dia": "D" if dia == 0 else "D+1", "limiar": q,
                                     "p_estrela": ps, "precisao": a, "recall": b, "f1": f})
                    coer_l.append({"modelo": rotulo(ch), "dia": "D" if dia == 0 else "D+1", "limiar": q,
                                   "p_estrela": pstar, "fracao_discordancia": float(np.mean(alerta != det)),
                                   "n_dias": len(t)})
                else:
                    alerta = det
                tp, fp, fn, tn, a, b, f = prf(alerta, real)
                ic = ic_bootstrap(pd.DataFrame({"t_emissao": t["t_emissao"].to_numpy(), "alerta": alerta,
                                                "real": real}), rng)
                linhas.append({"modelo": rotulo(ch), "tipo_alerta": "probabilidade" if prob else "media_diaria",
                               "dia": "D" if dia == 0 else "D+1", "limiar": q, "p_estrela": pstar if prob else np.nan,
                               "tp": tp, "fp": fp, "fn": fn, "tn": tn, "precisao": a, "recall": b, "f1": f,
                               "precisao_ic95_inf": ic["precisao_ic95"][0], "precisao_ic95_sup": ic["precisao_ic95"][1],
                               "recall_ic95_inf": ic["recall_ic95"][0], "recall_ic95_sup": ic["recall_ic95"][1],
                               "f1_ic95_inf": ic["f1_ic95"][0], "f1_ic95_sup": ic["f1_ic95"][1]})
    return pd.DataFrame(linhas), pd.DataFrame(brier_l), pd.DataFrame(pr_l), pd.DataFrame(coer_l)


# ---- uncertainty by regime

def mediana_frp300(tab):
    cols = list(sel.FRP_FIXAS)
    t = tab[tab["sensor"].isin(sel.SENSORES_TREINO)]
    ts = pd.to_datetime(t["time_stamp"], utc=True)
    m = ts.dt.year.isin(sel.DOBRAS[DOBRA]["treino"]) & ts.dt.month.isin(av.MESES_SECA)
    return float(t.loc[m, cols].sum(axis=1, min_count=len(cols)).median())


def regimes_em_t(tab, hist24, med_frp):
    cols = list(sel.FRP_FIXAS)
    r = tab[["sensor", "time_stamp", "chuva_24h"]].copy()
    r["frp300"] = tab[cols].sum(axis=1, min_count=len(cols))
    r = r.rename(columns={"time_stamp": "t_emissao"})
    r["t_emissao"] = pd.to_datetime(r["t_emissao"], utc=True)
    r = r.set_index(["sensor", "t_emissao"])
    r["hist24"] = hist24.reindex(r.index)
    r["regime_pm24"] = np.where(r["hist24"].isna(), "sem_dado", np.where(r["hist24"] > 50, "pm24>50", "pm24<=50"))
    r["regime_frp300"] = np.where(r["frp300"].isna(), "falha_firms",
                                  np.where(r["frp300"] > med_frp, "frp300>mediana", "frp300<=mediana"))
    r["chuva_24h_1mm"] = np.where(r["chuva_24h"].isna(), "sem_dado", np.where(r["chuva_24h"] >= 1.0, "com_chuva", "sem_chuva"))
    return r


def incerteza_oe2(modelos, reg):
    linhas = []
    for ch in INCERTEZA_OE2:
        if ch not in modelos:
            continue
        p = modelos[ch][modelos[ch]["sensor"].isin(sel.SENSORES_TREINO)]
        for escopo, sub in (("ano_todo", p), ("seca_04utc", av.filtro_principal(p))):
            t = av.incerteza(sub).reset_index().rename(columns={"index": "bloco"})
            t.insert(0, "regime", "todos"); t.insert(0, "escopo", escopo); t.insert(0, "modelo", rotulo(ch))
            linhas.append(t)
        sub = av.filtro_principal(p).join(reg[["regime_pm24", "regime_frp300"]], on=["sensor", "t_emissao"])
        for col in ("regime_pm24", "regime_frp300"):
            for nome, g in sub.groupby(col):
                t = av.incerteza(g).loc[["total"]].reset_index().rename(columns={"index": "bloco"})
                t.insert(0, "regime", nome); t.insert(0, "escopo", "seca_04utc"); t.insert(0, "modelo", rotulo(ch))
                linhas.append(t)
    return pd.concat(linhas, ignore_index=True)


# ---- OE5

def ruido_par(series, ano):
    a, b = (series[s] for s in PAR_COLOCALIZADO)
    ya = pd.Series(a.y, index=a.tempo)
    yb = pd.Series(b.y, index=b.tempo)
    emissoes = pd.date_range(f"{ano}-07-01 04:00", f"{ano}-10-31 04:00", freq="D", tz="UTC")
    erros = []
    for t in emissoes:
        h = pd.date_range(t + pd.Timedelta(hours=1), periods=conjunto.HORIZONTE_H, freq="h")
        d = (ya.reindex(h) - yb.reindex(h)).abs().dropna()
        if len(d):
            erros.append(d.mean())
    return float(np.mean(erros)) if erros else np.nan, len(erros)


def avaliar_oe5(pred_oe5, series, ano, ensaio, mae_treino):
    urbanos = list(sel.SENSORES_TREINO) if ensaio else list(URBANOS_OE5)
    cunia = None if ensaio else CUNIA
    modelos = por_chave(pred_oe5)
    redes = [("bilstm", "M2", "era5"), ("bilstm", "M1", "sem")]
    refs = [("persistencia", "-", "-"), ("persistencia_24h", "-", "-")]
    linhas, dm = [], []
    grupos = {"urbanos": urbanos}
    if cunia is not None:
        grupos["cunia"] = [cunia]
    for gnome, sens in grupos.items():
        for ch in redes + refs:
            if ch not in modelos:
                continue
            p = modelos[ch][modelos[ch]["sensor"].isin(sens)]
            e = erro_pares(p)
            linhas.append({"grupo": gnome, "sensor": "todos", "modelo": rotulo(ch), "mae_principal": float(e.mean()),
                           "n_pares": len(e), "mae_mesmo_modelo_sensores_treino": mae_treino.get(rotulo(ch), np.nan)})
            for s, es in e.groupby(level="sensor"):
                linhas.append({"grupo": gnome, "sensor": s, "modelo": rotulo(ch), "mae_principal": float(es.mean()),
                               "n_pares": len(es), "mae_mesmo_modelo_sensores_treino": np.nan})
        for ch in redes:
            if ch not in modelos:
                continue
            for ref in refs:
                pa = modelos[ch][modelos[ch]["sensor"].isin(sens)]
                pb = modelos[ref][modelos[ref]["sensor"].isin(sens)]
                r = linha_dm(f"6_oe5_{gnome}", rotulo(ch), rotulo(ref), pa, pb)
                r["familia_holm"] = True
                dm.append(r)
    ruido, n = (np.nan, 0) if ensaio else ruido_par(series, ano)
    linhas.append({"grupo": "ruido_par_colocalizado", "sensor": f"{PAR_COLOCALIZADO[0]}x{PAR_COLOCALIZADO[1]}",
                   "modelo": "diferenca_entre_sensores", "mae_principal": ruido, "n_pares": n,
                   "mae_mesmo_modelo_sensores_treino": np.nan})
    dmdf = pd.DataFrame(dm)
    if len(dmdf):
        dmdf = com_holm(dmdf, ("grupo", "a"))
    return pd.DataFrame(linhas), dmdf


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ano", type=int, choices=(2023, 2024), default=None)
    ap.add_argument("--ensaio", action="store_true", help="roda em 2023 (ano da dobra 3) só para testar o código")
    ap.add_argument("--abrir", action="store_true")
    a = ap.parse_args()
    ano = 2023 if a.ensaio else a.ano
    if ano is None or (ano == 2023 and not a.ensaio):
        raise SystemExit("Use --ensaio (2023) ou --ano 2024 --abrir.")
    if ano == 2024:
        if not a.abrir:
            raise SystemExit("2024 só é aberto depois do pré-registro (T27). Use --abrir.")
        conf = PESOS / "conferencia_dobra3_colab.csv"
        if not conf.exists():
            raise SystemExit("Faltam os pesos da Bi-LSTM conferidos na T28 (resultados/pesos/conferencia_dobra3_colab.csv).")
        c = pd.read_csv(conf)
        assert len(c) == 9 and (c["max_abs_diff_vc"] <= 1e-4).all(), "T28 (Colab) não reproduziu a VC"
    ensaio = a.ensaio
    out = RAIZ / "data" / "processed" / ("ensaio_teste_2023" if ensaio else "teste_2024")
    out.mkdir(parents=True, exist_ok=True)
    if ensaio:
        print("ENSAIO em 2023: testa o código; os números NÃO são resultado (2023 foi visto na VC).", flush=True)

    tab = carregar_tabela(ano)
    pred, carga, series = prever_todos(tab, ano, ensaio)
    carga.to_csv(out / "conferencia_carga.csv", index=False)
    cams = prever_cams(series, ano)
    todas = pd.concat([pred, cams], ignore_index=True)
    todas.to_parquet(out / "previsoes.parquet", index=False)

    teste = todas[todas["dobra"] == "teste"]
    modelos = por_chave(teste)
    hist24_t = hist24_de(tab)
    hist24_vc = hist24_de(pd.read_parquet(ENTRADAS_DIR / "entradas_horario_2020_2023.parquet").assign(
        time_stamp=lambda d: pd.to_datetime(d["time_stamp"], utc=True)))

    # rows 1 and 2: main metric and DM + Holm; CAMS N in {1, 3, 14} as sensitivity
    tabs, dm = [], []
    for nome, (rede, advs) in COMPARACOES.items():
        presentes = [ch for ch in [rede] + advs if ch in modelos]
        tabs.append(tabela_principal(modelos, presentes, nome))
        if rede not in modelos:
            continue
        for adv in advs:
            r = linha_dm(nome, rotulo(rede), rotulo(adv), modelos[rede], modelos[adv]); r["familia_holm"] = True
            dm.append(r)
        for n in N_CAMS[1:]:
            adv = ("cams_vies", f"N{n}", "-")
            r = linha_dm(f"{nome}_sensibilidade_cams", rotulo(rede), rotulo(adv), modelos[rede], modelos[adv])
            r["familia_holm"] = False
            dm.append(r)
    # row 3: value of future weather
    a3, b3 = ("bilstm", "M2", "era5"), ("bilstm", "M2", "sem")
    if a3 in modelos and b3 in modelos:
        tabs.append(tabela_principal(modelos, [a3, b3], "3_oe6_era5_vs_sem"))
        r = linha_dm("3_oe6_era5_vs_sem", rotulo(a3), rotulo(b3), modelos[a3], modelos[b3]); r["familia_holm"] = False
        dm.append(r)
    principal = pd.concat(tabs, ignore_index=True)
    principal.to_csv(out / "metrica_principal.csv", index=False)
    dmdf = com_holm(pd.DataFrame(dm)) if dm else pd.DataFrame()

    # row 4: skill, curves, rain cut, alerts
    skill = []
    for nome, (rede, advs) in COMPARACOES.items():
        sub = principal[principal["comparacao"] == nome].set_index("modelo")
        if rotulo(rede) not in sub.index:
            continue
        for adv in advs:
            if rotulo(adv) in sub.index:
                skill.append({"comparacao": nome, "modelo": rotulo(rede), "referencia": rotulo(adv),
                              "skill_mae": 1 - sub.loc[rotulo(rede), "mae_comum"] / sub.loc[rotulo(adv), "mae_comum"]})
    pd.DataFrame(skill).to_csv(out / "skill.csv", index=False)

    curvas = []
    for ch, p in modelos.items():
        t = pd.to_datetime(p["t_emissao"], utc=True)
        escopos = [("seca_04utc", av.filtro_principal(p))]
        if not ch[0].startswith("cams"):
            escopos += [("ano_todo", p), ("seca_todas_emissoes", p[t.dt.month.isin(av.MESES_SECA)])]
        for escopo, sub in escopos:
            c = av.curva_horizonte(sub).reset_index()
            c.insert(0, "escopo", escopo); c.insert(0, "modelo", rotulo(ch))
            curvas.append(c)
    pd.concat(curvas, ignore_index=True).to_csv(out / "curvas_horizonte.csv", index=False)

    reg = regimes_em_t(tab, hist24_t, mediana_frp300(tab))
    chuva = []
    for ch, p in modelos.items():
        sub = av.filtro_principal(p).join(reg[["chuva_24h_1mm"]], on=["sensor", "t_emissao"])
        for nome, g in sub.groupby("chuva_24h_1mm"):
            e = av.erro_por_par(g)
            chuva.append({"modelo": rotulo(ch), "chuva_24h": nome, "mae_principal": float(e.mean()), "n_pares": len(e)})
    pd.DataFrame(chuva).to_csv(out / "recorte_chuva.csv", index=False)

    alertas, brier, curva_pr, coer = avaliar_alertas(modelos, hist24_t, hist24_vc, freq_climatologica(tab), ensaio)
    alertas.to_csv(out / "alertas.csv", index=False)
    brier.to_csv(out / "brier.csv", index=False)
    curva_pr.to_csv(out / "curva_precisao_recall.csv", index=False)
    coer.to_csv(out / "coerencia_diaria_horaria.csv", index=False)

    # row 5: uncertainty
    inc = incerteza_oe2(modelos, reg)
    inc.to_csv(out / "incerteza.csv", index=False)

    # row 6: OE5
    mae_treino = principal.drop_duplicates("modelo").set_index("modelo")["mae_comum"].to_dict()
    oe5, dm6 = avaliar_oe5(todas[todas["dobra"] == "oe5"], series, ano, ensaio, mae_treino)
    oe5.to_csv(out / "oe5_sensores_novos.csv", index=False)
    if len(dm6):
        dmdf = pd.concat([dmdf, dm6], ignore_index=True)
    dmdf.to_csv(out / "diebold_mariano.csv", index=False)

    resumo = {"ano": ano, "ensaio": ensaio, "configuracao_congelada": "doc 05 §2.1 (20/11/2026, commit c26125a)",
              "carga_max_abs_dif": carga.groupby("modelo")["max_abs_dif_vc"].max().to_dict(),
              "n_linhas_previsoes": int(len(todas))}
    (out / "resumo.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False, default=float))

    f = lambda x: f"{x:.4f}"  # noqa: E731
    print("\n=== CONFERÊNCIA DE CARGA (dobra 3, 2023) ===")
    print(carga.to_string(index=False, float_format=lambda x: f"{x:.2e}"))
    print(f"\n=== MÉTRICA PRINCIPAL ({'ENSAIO 2023' if ensaio else '2024'}) ===")
    print(principal.to_string(index=False, float_format=f))
    print("\n=== DIEBOLD-MARIANO (HLN, HAC) + HOLM ===")
    print(dmdf.to_string(index=False, float_format=f))
    print("\n=== ALERTAS (D e D+1, seca, 04 UTC) ===")
    print(alertas[["modelo", "dia", "limiar", "tp", "fp", "fn", "precisao", "recall", "f1"]].to_string(index=False, float_format=f))
    print("\n=== INCERTEZA (total) ===")
    print(inc[(inc["bloco"] == "total") & (inc["regime"] == "todos")].to_string(index=False, float_format=f))
    print("\n=== OE5 ===")
    print(oe5[oe5["sensor"].astype(str) == "todos"].to_string(index=False, float_format=f))
    print(oe5[oe5["grupo"] == "ruido_par_colocalizado"].to_string(index=False, float_format=f))


if __name__ == "__main__":
    main()
