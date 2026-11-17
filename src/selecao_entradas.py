"""Per-fold isolation contracts and fixed parameters for input selection.

Fixed rule decided on 13/11/2026 (METHODS.md §4 and the pre-registration):
- Selection data: only training sensors (39077 and 56831), in the dry months (Jul–Oct)
  and only in the training years of the corresponding fold, respecting a 48 h embargo.
- Sensors outside training (219595, 219981, 225327 and Cuniã 128603) and the year evaluated in the fold
  (or >= 2024) never take part in input selection.
"""
import pandas as pd

SENSORES_TREINO = (39077, 56831)
SENSORES_AVALIACAO_OE5 = (219595, 219981, 225327, 128603)

DOBRAS = {
    "dobra1": {"treino": (2020,), "avaliacao": (2021,)},
    "dobra2": {"treino": (2020, 2021), "avaliacao": (2022,)},
    "dobra3": {"treino": (2020, 2021, 2022), "avaliacao": (2023,)},
    "final": {"treino": (2020, 2021, 2022, 2023), "avaliacao": (2024,)},
}

MESES_SECA = (7, 8, 9, 10)
EMBARGO_HORAS = 48

# Parameters pre-registered on 13/11/2026 (the pre-registration)
N_BLOCOS_CANDIDATOS = 19
LIMIAR_RHO_PARCIAL = 0.15
IC_BOOTSTRAP = 0.95
BLOCO_BOOTSTRAP_DIAS = 7
N_BOOTSTRAP = 1000
LIMIAR_REDUNDANCIA_RHO = 0.90
K_MAX_BLOCOS = 8


def validar_isolamento_dobra(
    df,
    dobra,
    embargo_horas=EMBARGO_HORAS,
    coluna_tempo="time_stamp",
    coluna_sensor="sensor_index",
    sensores_permitidos=SENSORES_TREINO,
):
    """Raises ValueError if df contains a sensor, year or timestamp outside the training fold."""
    if dobra not in DOBRAS:
        raise ValueError(f"Dobra desconhecida '{dobra}'. Válidas: {sorted(DOBRAS)}")

    anos_treino = set(DOBRAS[dobra]["treino"])
    primeiro_ano_aval = min(DOBRAS[dobra]["avaliacao"])
    limite_embargo = pd.Timestamp(primeiro_ano_aval, 1, 1, tz="UTC") - pd.Timedelta(hours=int(embargo_horas))

    if coluna_sensor in df.columns:
        sensores_presentes = set(pd.Series(df[coluna_sensor]).dropna().astype(int).unique().tolist())
        proibidos = sensores_presentes - set(sensores_permitidos)
        if proibidos:
            raise ValueError(
                f"Vazamento espacial na {dobra}: sensores fora do treino presentes: {sorted(proibidos)}"
            )

    ts = pd.to_datetime(df[coluna_tempo], utc=True)
    anos_presentes = set(ts.dt.year.dropna().astype(int).unique().tolist())
    anos_fora = anos_presentes - anos_treino
    if anos_fora:
        raise ValueError(
            f"Vazamento temporal na {dobra}: anos fora do treino {sorted(anos_treino)}: {sorted(anos_fora)}"
        )

    viola_embargo = ts > limite_embargo
    if bool(viola_embargo.any()):
        raise ValueError(
            f"Violação do embargo de {embargo_horas} h na {dobra}: "
            f"{int(viola_embargo.sum())} registro(s) após {limite_embargo.isoformat()}."
        )


def selecionar_amostra_dobra(
    df,
    dobra,
    meses_seca=MESES_SECA,
    embargo_horas=EMBARGO_HORAS,
    coluna_tempo="time_stamp",
    coluna_sensor="sensor_index",
    sensores_permitidos=SENSORES_TREINO,
):
    """Extracts exclusively the training data allowed for dobra and validates the isolation."""
    if dobra not in DOBRAS:
        raise ValueError(f"Dobra desconhecida '{dobra}'. Válidas: {sorted(DOBRAS)}")

    anos_treino = set(DOBRAS[dobra]["treino"])
    primeiro_ano_aval = min(DOBRAS[dobra]["avaliacao"])
    limite_embargo = pd.Timestamp(primeiro_ano_aval, 1, 1, tz="UTC") - pd.Timedelta(hours=int(embargo_horas))

    ts = pd.to_datetime(df[coluna_tempo], utc=True)
    mask = ts.dt.year.isin(anos_treino) & (ts <= limite_embargo)
    if meses_seca is not None:
        mask &= ts.dt.month.isin(list(meses_seca))
    if coluna_sensor in df.columns:
        mask &= df[coluna_sensor].astype(int).isin(list(sensores_permitidos))

    amostra = df.loc[mask].copy()
    validar_isolamento_dobra(
        amostra,
        dobra=dobra,
        embargo_horas=embargo_horas,
        coluna_tempo=coluna_tempo,
        coluna_sensor=coluna_sensor,
        sensores_permitidos=sensores_permitidos,
    )
    return amostra
