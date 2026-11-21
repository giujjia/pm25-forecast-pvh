"""Common dataset for every model (T21): issuances, windows, versions M1–M3 and decoder variants.

doc 04 §3.4–3.6. One sample = (sensor, issuance hour t). The encoder reads the 72 h up to t
(labels t − 71 … t, each label covering the hour that ends there); the decoder receives, for each
hour t + 1 … t + 48, the calendar and, in the "era5" variant, the ERA5 of that hour ("perfect
weather"). The target is the corrected PM2.5 (Barkjohn 2022) of t + 1 … t + 48, never interpolated.

Versions (doc 04 §3.8): M1 = PM2.5 history + calendar; M2 = M1 + meteorology (past and future);
M3 = M2 + FIRMS fire. Fixed inputs always enter; candidate blocks enter if chosen in that fold (T20b).

IMPLEMENTAÇÃO (confirmed by the author on 18/11/2026):
- PM2.5 input: gaps of up to 3 h are linearly interpolated (input only); an issuance whose 72 h window
  still has a gap is dropped. The input is log(1 + max(y, 0)), as the target in the loss.
- FRP columns enter as log(1 + FRP). An outage (NaN fire, with indicator) becomes 0 after
  standardisation (the training mean) and the indicator tells the network; XGBoost receives NaN.
- The future meteorology of the "era5" variant is the fixed ERA5 set (cell and region); the derived
  ones (RH, accumulated rain, ventilation) enter only in the past.
- Standardisation: mean and standard deviation of the training rows of the fold (training sensors and
  years, before the embargo), never of the evaluated year.
"""
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import correcao  # noqa: E402
import selecao_entradas as sel  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
ENTRADAS = RAIZ / "data" / "processed" / "entradas" / "entradas_horario_2020_2023.parquet"
SELECAO = RAIZ / "data" / "processed" / "selecao"

JANELA_H = 72
HORIZONTE_H = 48
LACUNA_MAX_H = 3
ALVO = "barkjohn2022"
INDICADORES_FALHA = ["falha_firms_24h", "falha_firms_d24_48", "falha_firms_d48_72"]
CALENDARIO = ["hora_sin", "hora_cos", "doy_sin", "doy_cos"]
FUSO_H = -4


def calendario(ts):
    loc = pd.DatetimeIndex(ts) + pd.Timedelta(hours=FUSO_H)
    h = loc.hour.to_numpy() + loc.minute.to_numpy() / 60
    d = loc.dayofyear.to_numpy()
    return np.column_stack([np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24),
                            np.sin(2 * np.pi * d / 365.25), np.cos(2 * np.pi * d / 365.25)]).astype(np.float32)


def colunas_versao(versao, escolhidos):
    """Past (encoder) columns of the table for the version, besides PM2.5 and calendar."""
    meteo = list(sel.ERA5_FIXAS) + [c for b in escolhidos if not b.startswith("frp") for c in sel.BLOCOS_CANDIDATOS[b]]
    fogo = list(sel.FRP_FIXAS) + [c for b in escolhidos if b.startswith("frp") for c in sel.BLOCOS_CANDIDATOS[b]]
    if versao == "M1":
        return [], []
    if versao == "M2":
        return meteo, []
    if versao == "M3":
        return meteo, fogo
    raise ValueError(f"Versão desconhecida: {versao}")


def escolhidos_dobra(dobra):
    return json.loads((SELECAO / f"escolha_{dobra}.json").read_text())["escolhidos"]


@dataclass
class Serie:
    """Hourly arrays of one sensor (continuous grid)."""
    sensor: int
    tempo: pd.DatetimeIndex
    y: np.ndarray            # signed target (µg/m³), NaN where missing
    pm_in: np.ndarray        # input PM2.5 (signed, gaps <= 3 h interpolated)
    enc: np.ndarray          # [T, F_enc] standardised past inputs (without NaN)
    dec: np.ndarray          # [T, F_dec] standardised future inputs
    enc_raw: pd.DataFrame = field(repr=False, default=None)  # untransformed, for XGBoost/Ridge


@dataclass
class Conjunto:
    versao: str
    variante: str
    dobra: str
    nomes_enc: list
    nomes_dec: list
    series: dict
    media: dict
    desvio: dict


def _interpolar_curto(y):
    s = pd.Series(y)
    return s.interpolate(limit=LACUNA_MAX_H, limit_area="inside").to_numpy()


def montar(tab, dobra, versao, variante, escolhidos=None, sensores=None):
    """Builds the standardised hourly arrays of each sensor for one fold, version and variant."""
    if variante not in ("sem", "era5"):
        raise ValueError("variante deve ser 'sem' (sem meteorologia futura) ou 'era5'")
    escolhidos = escolhidos_dobra(dobra) if escolhidos is None else escolhidos
    meteo, fogo = colunas_versao(versao, escolhidos)
    futuro = list(sel.ERA5_FIXAS) if (variante == "era5" and versao != "M1") else []
    treino_anos = sel.DOBRAS[dobra]["treino"]
    limite = pd.Timestamp(min(sel.DOBRAS[dobra]["avaliacao"]), 1, 1, tz="UTC") - pd.Timedelta(hours=sel.EMBARGO_HORAS)
    sensores = sensores or sorted(tab["sensor"].unique())

    brutos = {}
    for s in sensores:
        g = tab[tab["sensor"] == s].set_index("time_stamp").sort_index()
        g = g.reindex(pd.date_range(g.index.min(), g.index.max(), freq="h"))
        brutos[s] = g

    # transformations
    def transformar(g):
        d = pd.DataFrame(index=g.index)
        d["pm_log"] = correcao.transformar_log(_interpolar_curto(g[ALVO].to_numpy()))
        for c in meteo:
            d[c] = g[c].astype(float)
        for c in fogo:
            d[c] = np.log1p(g[c].astype(float))
        if fogo:
            for c in INDICADORES_FALHA:
                d[c] = g[c].fillna(1).astype(float)
        for c in futuro:
            d["fut_" + c] = g[c].astype(float)
        return d

    trans = {s: transformar(g) for s, g in brutos.items()}
    # standardisation statistics: training rows only
    linhas_treino = []
    for s in sensores:
        if s not in sel.SENSORES_TREINO:
            continue
        d = trans[s]
        m = d.index.year.isin(treino_anos) & (d.index <= limite)
        linhas_treino.append(d[m])
    ref = pd.concat(linhas_treino)
    cols_padronizar = [c for c in ref.columns if c not in INDICADORES_FALHA]
    media = ref[cols_padronizar].mean().to_dict()
    desvio = ref[cols_padronizar].std().replace(0, 1).to_dict()

    nomes_enc = ["pm_log"] + meteo + fogo + (INDICADORES_FALHA if fogo else []) + CALENDARIO
    nomes_dec = ["fut_" + c for c in futuro] + CALENDARIO
    series = {}
    for s in sensores:
        d = trans[s].copy()
        for c in cols_padronizar:
            d[c] = (d[c] - media[c]) / desvio[c]
        cal = calendario(d.index)
        enc_cols = ["pm_log"] + meteo + fogo + (INDICADORES_FALHA if fogo else [])
        enc = np.column_stack([d[enc_cols].to_numpy(np.float32), cal])
        # outage: fire NaN -> 0 (training mean) after standardisation; the indicator carries the information
        if fogo:
            idx = [enc_cols.index(c) for c in fogo]
            enc[:, idx] = np.nan_to_num(enc[:, idx], nan=0.0)
        dec = np.column_stack([d[["fut_" + c for c in futuro]].to_numpy(np.float32), cal]) if futuro else cal
        g = brutos[s]
        series[s] = Serie(sensor=int(s), tempo=d.index, y=g[ALVO].to_numpy(float),
                          pm_in=_interpolar_curto(g[ALVO].to_numpy(float)), enc=enc, dec=dec,
                          enc_raw=g[meteo + fogo + ([c for c in INDICADORES_FALHA] if fogo else [])].copy())
    return Conjunto(versao, variante, dobra, nomes_enc, nomes_dec, series, media, desvio)


def emissoes_validas(serie, anos=None, ate=None, desde=None, so_hora=None):
    """Row indices i (issuance t = tempo[i]) with complete 72 h window and complete decoder inputs.

    The encoder window is i − 71 … i; the decoder i + 1 … i + 48 (must exist in the grid).
    """
    T = len(serie.tempo)
    ok_enc = ~np.isnan(serie.enc).any(axis=1)
    ok_dec = ~np.isnan(serie.dec).any(axis=1)
    # rolling "all ok" over the window
    def todos(ok, n):
        c = np.concatenate([[0], np.cumsum(~ok)])
        return c
    ce = todos(ok_enc, JANELA_H)
    cd = todos(ok_dec, HORIZONTE_H)
    idx = np.arange(JANELA_H - 1, T - HORIZONTE_H)
    enc_ok = (ce[idx + 1] - ce[idx + 1 - JANELA_H]) == 0
    dec_ok = (cd[idx + 1 + HORIZONTE_H] - cd[idx + 1]) == 0
    idx = idx[enc_ok & dec_ok]
    t = serie.tempo[idx]
    m = np.ones(len(idx), bool)
    if anos is not None:
        m &= np.isin(t.year, list(anos))
    if ate is not None:
        m &= t <= ate
    if desde is not None:
        m &= t >= desde
    if so_hora is not None:
        m &= t.hour == so_hora
    return idx[m]


def alvo_janela(serie, idx):
    """[N, 48] signed target of t + 1 … t + 48 (NaN where missing)."""
    return np.stack([serie.y[i + 1:i + 1 + HORIZONTE_H] for i in idx]) if len(idx) else np.zeros((0, HORIZONTE_H))


def particao(conj, dobra=None):
    """Training issuances (training sensors, training years, t <= eval start − 48 h) and evaluation
    issuances (training sensors, evaluated year)."""
    dobra = dobra or conj.dobra
    treino_anos = sel.DOBRAS[dobra]["treino"]
    aval_anos = sel.DOBRAS[dobra]["avaliacao"]
    limite = pd.Timestamp(min(aval_anos), 1, 1, tz="UTC") - pd.Timedelta(hours=sel.EMBARGO_HORAS)
    tr, av = {}, {}
    for s, serie in conj.series.items():
        if s not in sel.SENSORES_TREINO:
            continue
        i_tr = emissoes_validas(serie, anos=treino_anos, ate=limite)
        y = alvo_janela(serie, i_tr)
        tr[s] = i_tr[~np.isnan(y).all(axis=1)] if len(i_tr) else i_tr
        av[s] = emissoes_validas(serie, anos=aval_anos)
    return tr, av
