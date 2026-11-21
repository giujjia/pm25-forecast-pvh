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


# ---------------------------------------------------------------------------------------------
# Selection rule (T20b), as pre-registered on 13/11/2026 (doc 04 §3.4, "Escolha das entradas candidatas";
# doc 05 §2.1). Implementation details not fixed by the text are marked "IMPLEMENTAÇÃO".
# ---------------------------------------------------------------------------------------------
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
ENTRADAS = RAIZ / "data" / "processed" / "entradas"
OUT = RAIZ / "data" / "processed" / "selecao"

HORA_EMISSAO_UTC = 4  # 0h local (UTC−4)
HORIZONTE_H = 48
ALVO = "barkjohn2022"
SEMENTE_BOOTSTRAP = 20261113
# IMPLEMENTAÇÃO (confirmed by the author on 18/11/2026): like the valid day (>= 18 of 24 h, 75%), a 48 h mean needs >= 36 hours and
# the 24 h history mean needs >= 18 hours.
MIN_HORAS_ALVO = 36
MIN_HORAS_HIST = 18

ANEIS = ("a0_50", "a50_150", "a150_300", "a300_600", "a600_1000")
SETORES = range(8)


def _cols(prefixo, anel):
    return [f"{prefixo}_{anel}_s{s}" for s in SETORES]


# The 19 candidate blocks: name -> columns (the block is tested by the sum of its columns).
BLOCOS_CANDIDATOS = {
    "ur": ["ur"],
    "chuva_24h": ["chuva_24h"],
    "chuva_72h": ["chuva_72h"],
    "chuva_7d": ["chuva_7d"],
    "horas_sem_chuva": ["horas_sem_chuva"],
    "ventilacao": ["ventilacao"],
    "frp24_a300_600": _cols("frp24", "a300_600"),
    "frp24_a600_1000": _cols("frp24", "a600_1000"),
    **{f"frp24_d24_48_{a}": _cols("frp24_d24_48", a) for a in ANEIS},
    **{f"frp24_d48_72_{a}": _cols("frp24_d48_72", a) for a in ANEIS},
    "frp_transportado": ["frp_transportado"],
}
assert len(BLOCOS_CANDIDATOS) == N_BLOCOS_CANDIDATOS

# Fixed inputs (always enter). ERA5 at the cell and the regional mean; the regional D2m was not
# extracted in E057 (only the regional RH), so the regional block has no D2m (declared).
ERA5_FIXAS = ["blh", "u10", "v10", "u850", "v850", "t2m_c", "d2m_c", "tp_mm",
              "blh_reg", "u10_reg", "v10_reg", "u850_reg", "v850_reg", "t2m_c_reg", "tp_mm_reg"]
FRP_FIXAS = [c for a in ("a0_50", "a50_150", "a150_300") for c in _cols("frp24", a)]


def pares_emissao(tab, alvo=ALVO):
    """One row per (sensor, issuance at 04 UTC): candidates at t, mean target over t+1..t+48, mean of the 24 h before."""
    linhas = []
    for s, g in tab.groupby("sensor"):
        g = g.set_index("time_stamp").sort_index()
        # continuous hourly grid: the sample has only Jul–Oct, and a row-based window would join years
        g = g.reindex(pd.date_range(g.index.min(), g.index.max(), freq="h"))
        y = g[alvo]
        fut = y[::-1].rolling(HORIZONTE_H, min_periods=MIN_HORAS_ALVO).mean()[::-1].shift(-1)  # mean of t+1..t+48
        hist = y.rolling(24, min_periods=MIN_HORAS_HIST).mean()  # mean of t−23..t (labels by the end: known at t)
        emis = g.index[g.index.hour == HORA_EMISSAO_UTC]
        d = g.loc[emis].copy()
        d.index.name = "time_stamp"
        d["y48"] = fut.loc[emis]
        d["hist24"] = hist.loc[emis]
        d["sensor"] = s
        linhas.append(d.reset_index())
    return pd.concat(linhas, ignore_index=True)


def valor_bloco(df, bloco):
    cols = BLOCOS_CANDIDATOS[bloco]
    return df[cols].sum(axis=1, min_count=len(cols))


def rho_parcial_spearman(x, y, z):
    """Partial Spearman: Pearson of the residuals of rank(x) and rank(y) regressed on rank(z)."""
    rx, ry, rz = (stats.rankdata(v) for v in (x, y, z))
    A = np.column_stack([np.ones_like(rz), rz])
    ex = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]
    ey = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
    den = np.sqrt((ex ** 2).sum() * (ey ** 2).sum())
    return float((ex * ey).sum() / den) if den > 0 else np.nan


def indices_bootstrap_blocos(dias, n_reamostras=N_BOOTSTRAP, bloco=BLOCO_BOOTSTRAP_DIAS, semente=SEMENTE_BOOTSTRAP):
    """Moving-block bootstrap over the ordered list of days; all sensors of a drawn day go together.

    IMPLEMENTAÇÃO (confirmed by the author on 18/11/2026): blocks of 7 consecutive days within the ordered list of dry-season days of the sample
    (a block may join the end of one dry season with the start of the next, which is rare).
    """
    rng = np.random.default_rng(semente)
    n = len(dias)
    n_blocos = int(np.ceil(n / bloco))
    for _ in range(n_reamostras):
        ini = rng.integers(0, n - bloco + 1, size=n_blocos)
        idx = (ini[:, None] + np.arange(bloco)[None, :]).ravel()[:n]
        yield dias[idx]


def avaliar_candidatos(pares):
    dias = np.array(sorted(pares["dia"].unique()))
    por_dia = {d: i for d, i in pares.groupby("dia").groups.items()}
    linhas = []
    for b in BLOCOS_CANDIDATOS:
        v = valor_bloco(pares, b)
        ok = v.notna() & pares["y48"].notna() & pares["hist24"].notna()
        x, y, z = v[ok].to_numpy(), pares.loc[ok, "y48"].to_numpy(), pares.loc[ok, "hist24"].to_numpy()
        rho = rho_parcial_spearman(x, y, z)
        okidx = set(np.flatnonzero(ok.to_numpy()))
        boot = []
        for amostra in indices_bootstrap_blocos(dias):
            idx = [i for d in amostra for i in por_dia[d] if i in okidx]
            boot.append(rho_parcial_spearman(v.iloc[idx].to_numpy(), pares["y48"].iloc[idx].to_numpy(),
                                             pares["hist24"].iloc[idx].to_numpy()))
        lo, hi = np.nanpercentile(boot, [2.5, 97.5])
        linhas.append({"bloco": b, "n": int(ok.sum()), "rho_parcial": rho, "ic95_inf": lo, "ic95_sup": hi,
                       "passa_a": abs(rho) >= LIMIAR_RHO_PARCIAL, "passa_b": not (lo <= 0 <= hi)})
    return pd.DataFrame(linhas)


def escolher(pares, aval):
    """Cut (a and b), order by |rho|, skip if |Spearman| >= 0.9 with a chosen one, at most K = 8."""
    cand = aval[aval["passa_a"] & aval["passa_b"]].copy()
    cand = cand.reindex(cand["rho_parcial"].abs().sort_values(ascending=False).index)
    escolhidos, pulados = [], []
    for b in cand["bloco"]:
        vb = valor_bloco(pares, b)
        red = None
        for e in escolhidos:
            ve = valor_bloco(pares, e)
            ok = vb.notna() & ve.notna()
            r = stats.spearmanr(vb[ok], ve[ok]).statistic
            if abs(r) >= LIMIAR_REDUNDANCIA_RHO:
                red = (e, float(r))
                break
        if red:
            pulados.append({"bloco": b, "redundante_com": red[0], "rho": red[1]})
            continue
        escolhidos.append(b)
        if len(escolhidos) == K_MAX_BLOCOS:
            break
    return escolhidos, pulados


def rodar(dobra, tab):
    amostra = selecionar_amostra_dobra(tab, dobra, coluna_sensor="sensor")
    # The sample stops at the embargo limit and at the end of October, so no 48 h target window
    # reaches the evaluated year: the hours after the sample simply do not exist here.
    pares = pares_emissao(amostra)
    pares = pares[pares["time_stamp"].dt.month.isin(MESES_SECA)].reset_index(drop=True)
    pares["dia"] = pares["time_stamp"].dt.normalize()
    aval = avaliar_candidatos(pares)
    escolhidos, pulados = escolher(pares, aval)
    return pares, aval, escolhidos, pulados


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in DOBRAS:
        raise SystemExit(f"Uso: selecao_entradas.py <{'|'.join(DOBRAS)}>")
    dobra = sys.argv[1]
    tab = pd.read_parquet(ENTRADAS / "entradas_horario_2020_2023.parquet")
    pares, aval, escolhidos, pulados = rodar(dobra, tab)
    OUT.mkdir(parents=True, exist_ok=True)
    aval.to_csv(OUT / f"candidatos_{dobra}.csv", index=False)
    res = {"dobra": dobra, "anos": list(DOBRAS[dobra]["treino"]), "sensores": sorted(pares["sensor"].unique().tolist()),
           "n_pares": int(len(pares)), "n_pares_com_alvo": int(pares["y48"].notna().sum()),
           "escolhidos": escolhidos, "pulados_por_redundancia": pulados}
    (OUT / f"escolha_{dobra}.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    pd.set_option("display.width", 200)
    print(aval.round(3).to_string(index=False))
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
