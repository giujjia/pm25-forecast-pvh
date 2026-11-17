"""Temporal availability rules at t and FIRMS filters for assembling inputs.

Audited project conventions:
- PurpleAir (average=60): time_stamp = T marks the START of the hour [T, T + 1h).
  At issuance time t, the latest hourly mean available without future leakage is T <= t - 1h.
- ERA5 total_precipitation (tp): valid_time = T accumulates (T - 1h, T]; the other variables
  are instantaneous at T. In real operation ERA5 has a ~5 day delay.
- FIRMS NRT: published up to ~3 h after the overpass; the SP history includes a
  type column (0 = presumed vegetation), absent from the operational NRT feed.
  In our parquets (version 2) type has a known defect: some static sources came as 0
.

The FIRMS filters (type, satellite, confidence) have no default value and must be passed
explicitly. The author's decisions of 16/11/2026 (METHODS.md §4) are in the constants
SATELITES_FIRMS_DECIDIDOS (NOAA-20 only) and CONFIANCAS_FIRMS_DECIDIDAS (None: 'l' kept), with
type not filtered (NRT lacks it) and replaced by the fixed spatial mask of static sources
(aplicar_mascara_estatica, ~400 m grid built from 2020–2023 by src/analise_firms_n20_mascara.py).
A day with no NOAA-20 detection in the 1000 km box is an outage (dias_falha_firms):
the fire inputs of the windows that touch it are missing, never zero.
- PurpleAir, label (decided on 16/11/2026; METHODS.md §3, item 7): the record is now labelled
  by the end of the hour (rotular_purpleair_pelo_fim); after that, the availability delay is zero.
- CAMS global forecast: the 00 UTC run guaranteed at 10 UTC; the 12 UTC run guaranteed
  at 22 UTC.
"""
from pathlib import Path

import numpy as np
import pandas as pd

CONVENCAO_PURPLEAIR = "inicio"
ATRASO_PADRAO_HORAS = {
    "purpleair": 1.0,  # stamp T closes at T + 1h
    "era5_janela": 0.0,  # valid_time T represents instant T or (T-1h, T]
    "era5_operacao": 120.0,  # ~5 days in ERA5T
    "firms_nrt": 3.0,  # best-effort <= 3 h
    "cams_previsao": 10.0,  # 00 UTC -> 10 UTC; 12 UTC -> 22 UTC
}

SATELITES_FIRMS_VALIDOS = ("N", "N20")
SATELITES_FIRMS_DECIDIDOS = ("N20",)  # 16/11/2026: NOAA-20 only (METHODS.md §4)
CONFIANCAS_FIRMS_DECIDIDAS = None  # 16/11/2026: 'l' kept (Schroeder et al., 2014)
CELULA_MASCARA_GRAUS = 0.0036  # ~400 m, like the FIRMS static anomalies layer
MASCARA_ESTATICA_CSV = Path(__file__).resolve().parents[1] / "data" / "processed" / "firms_mascara" / "mascara_uniao_celulas.csv"
TIPOS_FIRMS_VEGETACAO = (0,)
CONFIANCAS_FIRMS_VALIDAS = ("l", "n", "h")


def _as_utc_series(valores):
    ts = pd.to_datetime(valores, utc=True)
    if isinstance(ts, pd.DatetimeIndex):
        return pd.Series(ts)
    return ts


def calcular_disponivel_em(timestamps, fonte, atraso_horas=None):
    """Returns the UTC instant at which each record of fonte becomes available."""
    if fonte not in ATRASO_PADRAO_HORAS:
        raise ValueError(f"Fonte desconhecida '{fonte}'. Válidas: {sorted(ATRASO_PADRAO_HORAS)}")
    atraso = ATRASO_PADRAO_HORAS[fonte] if atraso_horas is None else float(atraso_horas)
    if fonte == "purpleair" and atraso < 1.0:
        raise ValueError("PurpleAir (average=60) marca início da hora [T, T+1h); atraso mínimo é 1 h.")
    ts = _as_utc_series(timestamps)
    return ts + pd.to_timedelta(atraso, unit="h")


def validar_disponibilidade_janela(timestamps, t_emissao, fonte, atraso_horas=None):
    """Raises ValueError if any record is available only after t_emissao."""
    t_ref = pd.Timestamp(t_emissao)
    if t_ref.tzinfo is None:
        t_ref = t_ref.tz_localize("UTC")
    else:
        t_ref = t_ref.tz_convert("UTC")

    ts = _as_utc_series(timestamps)
    disp = calcular_disponivel_em(ts, fonte=fonte, atraso_horas=atraso_horas)
    vazados = (disp > t_ref) | (ts > t_ref)
    if bool(vazados.any()):
        pior_ts = ts[vazados].max()
        pior_disp = disp[vazados].max()
        raise ValueError(
            f"Vazamento temporal em '{fonte}' para emissão {t_ref.isoformat()}: "
            f"{int(vazados.sum())} registro(s) com timestamp até {pior_ts.isoformat()} "
            f"(disponível apenas em {pior_disp.isoformat()})."
        )


def selecionar_janela_disponivel(df, t_emissao, fonte, janela_horas=72, coluna_tempo="time_stamp", atraso_horas=None):
    """Slices [t_emissao - janela_horas, t_emissao] ensuring availability at t_emissao."""
    t_ref = pd.Timestamp(t_emissao)
    if t_ref.tzinfo is None:
        t_ref = t_ref.tz_localize("UTC")
    else:
        t_ref = t_ref.tz_convert("UTC")

    ts = _as_utc_series(df[coluna_tempo])
    disp = calcular_disponivel_em(ts, fonte=fonte, atraso_horas=atraso_horas)
    ini = t_ref - pd.Timedelta(hours=float(janela_horas))
    # For intervals starting at T (PurpleAir), T = t_ref - janela_horas covers [ini, ini+1h) and avail <= t_ref requires T <= t_ref - 1h.
    mask = ((ts >= ini) & (disp <= t_ref) & (ts <= t_ref)).to_numpy()
    recorte = df.loc[mask].copy()
    validar_disponibilidade_janela(recorte[coluna_tempo], t_emissao=t_ref, fonte=fonte, atraso_horas=atraso_horas)
    return recorte


def validar_rodada_cams(rodada_utc, t_emissao, garantia_horas=10.0):
    """Ensures the CAMS run (00 or 12 UTC) was already guaranteed published at t_emissao."""
    validar_disponibilidade_janela([rodada_utc], t_emissao=t_emissao, fonte="cams_previsao", atraso_horas=garantia_horas)


def construir_timestamp_firms(df):
    """Builds a UTC timestamp from acq_date (YYYY-MM-DD) and acq_time (HMM or HHMM)."""
    horas_str = df["acq_time"].astype(int).astype(str).str.zfill(4)
    texto = df["acq_date"].astype(str).str.strip() + " " + horas_str
    return pd.to_datetime(texto, format="%Y-%m-%d %H%M", utc=True)


def filtrar_firms(df, *, tipos_permitidos, satelites_permitidos, confiancas_permitidas):
    """Filters type, satellite, confidence and frp >= 0.

    The three filters are mandatory and have no default (pending decision). None in
    tipos_permitidos or confiancas_permitidas means "do not filter" and must be
    written explicitly by the caller. Asking for a type filter on a DataFrame without the column
    (the NRT case) is an error.
    """
    if not satelites_permitidos:
        raise ValueError("`satelites_permitidos` não pode ser vazio (evitar dupla contagem ou omissão implícita).")

    invalid_sat = set(satelites_permitidos) - set(SATELITES_FIRMS_VALIDOS)
    if invalid_sat:
        raise ValueError(f"Satélite(s) inválido(s) no filtro FIRMS: {sorted(invalid_sat)}")

    mask = pd.Series(True, index=df.index)
    if tipos_permitidos is not None:
        if "type" not in df.columns:
            raise ValueError("Column `type` missing from the FIRMS DataFrame: NRT does not have this field.")
        mask &= df["type"].isin(list(tipos_permitidos))

    mask &= df["satellite"].isin(list(satelites_permitidos))

    if confiancas_permitidas is not None:
        invalid_conf = set(confiancas_permitidas) - set(CONFIANCAS_FIRMS_VALIDAS)
        if invalid_conf:
            raise ValueError(f"Confiança(s) inválida(s): {sorted(invalid_conf)}")
        mask &= df["confidence"].isin(list(confiancas_permitidas))

    mask &= df["frp"].notna() & (df["frp"] >= 0)
    return df.loc[mask].copy()


def validar_firms_agregado(df, *, tipos_permitidos, satelites_permitidos, t_emissao=None, atraso_horas=3.0):
    """Checks that a FIRMS subset ready for aggregation violates neither filters nor availability."""
    if tipos_permitidos is not None and "type" in df.columns:
        fora_tipo = ~df["type"].isin(list(tipos_permitidos))
        if bool(fora_tipo.any()):
            raise ValueError(f"FIRMS agregado contém `type` proibido(s): {sorted(df.loc[fora_tipo, 'type'].unique().tolist())}")

    fora_sat = ~df["satellite"].isin(list(satelites_permitidos))
    if bool(fora_sat.any()):
        raise ValueError(f"FIRMS agregado contém satélite fora da lista: {sorted(df.loc[fora_sat, 'satellite'].unique().tolist())}")

    if t_emissao is not None:
        ts = df["timestamp_utc"] if "timestamp_utc" in df.columns else construir_timestamp_firms(df)
        validar_disponibilidade_janela(ts, t_emissao=t_emissao, fonte="firms_nrt", atraso_horas=atraso_horas)


def agregar_firms_janela(
    df,
    t_emissao,
    janela_horas=24,
    atraso_horas=3.0,
    *,
    tipos_permitidos,
    satelites_permitidos,
    confiancas_permitidas,
    grupo_cols=None,
):
    """Filters FIRMS detections by type/satellite/availability and aggregates FRP over the causal window up to t_emissao."""
    filtrado = filtrar_firms(
        df,
        tipos_permitidos=tipos_permitidos,
        satelites_permitidos=satelites_permitidos,
        confiancas_permitidas=confiancas_permitidas,
    )
    filtrado["timestamp_utc"] = construir_timestamp_firms(filtrado)

    recorte = selecionar_janela_disponivel(
        filtrado,
        t_emissao=t_emissao,
        fonte="firms_nrt",
        janela_horas=janela_horas,
        coluna_tempo="timestamp_utc",
        atraso_horas=atraso_horas,
    )
    validar_firms_agregado(
        recorte,
        tipos_permitidos=tipos_permitidos,
        satelites_permitidos=satelites_permitidos,
        t_emissao=t_emissao,
        atraso_horas=atraso_horas,
    )

    if grupo_cols:
        if len(recorte) == 0:
            cols = list(grupo_cols) + ["frp_total", "n_focos"]
            return pd.DataFrame(columns=cols)
        return (
            recorte.groupby(list(grupo_cols), as_index=False)
            .agg(frp_total=("frp", "sum"), n_focos=("frp", "size"))
        )

    return {"frp_total": float(recorte["frp"].sum()), "n_focos": int(len(recorte)), "focos": recorte}


def rotular_purpleair_pelo_fim(df, coluna="time_stamp"):
    """Labels the PurpleAir hourly mean by the end of the hour: T (start) -> T + 1 h.

    After that, the labelled value t covers (t − 1 h, t], like ERA5 rain, and is
    available at t with no additional delay.
    """
    out = df.copy()
    out[coluna] = pd.to_datetime(out[coluna], utc=True) + pd.Timedelta(hours=1)
    return out


def celula_mascara(lat, lon, passo=CELULA_MASCARA_GRAUS):
    """Indices (i_lat, i_lon) of the ~400 m cell, in the same convention as src/analise_firms_n20_mascara.py."""
    return (np.floor(np.asarray(lat, float) / passo).astype(int),
            np.floor(np.asarray(lon, float) / passo).astype(int))


def carregar_mascara_estatica(caminho=MASCARA_ESTATICA_CSV):
    """Set of cells (i_lat, i_lon) of the fixed mask of static sources."""
    m = pd.read_csv(caminho)
    return set(zip(m["i_lat"].astype(int), m["i_lon"].astype(int)))


def aplicar_mascara_estatica(df, celulas):
    """Removes the detections falling in a mask cell. Applies equally to SP and NRT (does not use type)."""
    i_lat, i_lon = celula_mascara(df["latitude"], df["longitude"])
    dentro = np.fromiter(((a, b) in celulas for a, b in zip(i_lat, i_lon)), bool, count=len(df))
    return df.loc[~dentro].copy()


def dias_falha_firms(df, dias, satelite="N20"):
    """Days (UTC dates) of dias with no detection from satelite in df (the whole 1000 km box).

    Treated as a satellite outage: the fire inputs of the windows touching those days are
    missing, with an indicator, and do not become zero (METHODS.md §4).
    """
    tem = set(pd.to_datetime(df.loc[df["satellite"] == satelite, "acq_date"]).dt.date)
    return sorted(d for d in pd.to_datetime(pd.Index(dias)).date if d not in tem)

