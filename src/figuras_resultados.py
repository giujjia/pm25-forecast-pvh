"""Figures of the 2024 test for the report and the page (T35–T36).

Reads only the saved forecasts of E180 (data/processed/teste_2024/previsoes.parquet).
- fig_mae_horizonte_2024.png: MAE of the median by horizon (+1 … +48 h), 04 UTC issuances, Jul–Oct 2024,
  training sensors, for bilstm M1 sem, bilstm M2 era5, persistence, CAMS with moving bias (N = 7) and
  raw CAMS, on the issuance–sensor pairs common to the five models (the same 197 pairs as the main
  table). curvas_horizonte.csv is NOT used: it scores each model on its own pairs (CAMS has 226), which
  made CAMS look better than on the common pairs;
- fig_serie_diaria_2024.png: daily mean of the PM2.5 measured by the corrected PurpleAir sensors (39077)
  on valid days (>= 18 of 24 h) and the day-D forecast (mean of the hourly medians and 10–90% quantiles,
  04 UTC issuance) of the main model (bilstm M1 sem) and persistence, Jul–Oct 2024, with the 50 and
  125 µg/m³ thresholds.
Colours: validated categorical palette (slots 1–3) on a light surface.
Every figure carries the PurpleAir attribution required by its terms (auditoria F63).

Usage (from the root): .venv/bin/python src/figuras_resultados.py
Output: resultados/figuras/
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
import avaliacao as av  # noqa: E402

TESTE = RAIZ / "data" / "processed" / "teste_2024"
OUT = RAIZ / "resultados" / "figuras"
SUPERFICIE, TEXTO, TEXTO2, GRADE = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COR = {"bilstm_M1_sem": "#2a78d6", "bilstm_M2_era5": "#eb6834", "persistencia_-_-": "#1baf7a",
       "cams_vies_N7_-": "#52514e", "cams_bruto_-_-": "#9a9893", "observado": "#0b0b0b"}
NOME = {"bilstm_M1_sem": "Bi-LSTM M1 (principal, sem met. futura)", "bilstm_M2_era5": "Bi-LSTM M2 (ERA5, limite superior)",
        "persistencia_-_-": "Persistência", "cams_vies_N7_-": "CAMS com viés móvel (7 d)",
        "cams_bruto_-_-": "CAMS bruto"}


ATRIBUICAO = "Data from PurpleAir (purpleair.com), corrected with Barkjohn et al. (2022). Forecasts: Powered by PurpleAir."


def rodape(fig):
    """Attribution required by the PurpleAir terms (Section 4.8 and Attribution Guide)."""
    fig.text(0.01, 0.01, ATRIBUICAO, fontsize=6.5, color=TEXTO2, ha="left", va="bottom")


def estilo(ax):
    ax.set_facecolor(SUPERFICIE)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(GRADE)
    ax.tick_params(colors=TEXTO2, labelsize=9)
    ax.grid(axis="y", color=GRADE, linewidth=0.8)
    ax.set_axisbelow(True)


def media(pred, rot):
    g = pred[(pred["modelo"] + "_" + pred["versao"] + "_" + pred["variante"]) == rot]
    return av.media_sementes(g) if g["semente"].nunique() > 1 else g


def isolados(s):
    """Valid values with no valid neighbour: a line plot does not draw them."""
    return s.where(s.notna() & s.shift(1).isna() & s.shift(-1).isna())


def curvas_comuns(teste, rotulos):
    """MAE by horizon of each model, on the (sensor, t_emissao) pairs common to all of them."""
    modelos = {rot: av.filtro_principal(media(teste, rot)) for rot in rotulos}
    comum = None
    for g in modelos.values():
        e = av.erro_por_par(g)
        comum = e.index if comum is None else comum.intersection(e.index)
    curvas = []
    for rot, g in modelos.items():
        g = g.set_index(["sensor", "t_emissao"])
        g = g[g.index.isin(comum)].reset_index()
        print(f"{rot}: MAE nos {len(comum)} pares comuns = {av.erro_por_par(g).mean():.2f}")
        curvas.append(av.curva_horizonte(g).reset_index().assign(modelo=rot))
    return pd.concat(curvas, ignore_index=True), len(comum)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": SUPERFICIE})
    prev = pd.read_parquet(TESTE / "previsoes.parquet")
    prev = prev[prev["dobra"] == "teste"]

    # 1. MAE by horizon, on the pairs common to the plotted models
    fig, ax = plt.subplots(figsize=(7.2, 4.1), dpi=200)
    estilo(ax)
    estilos = {
        "bilstm_M1_sem": ("-", 2.1),
        "bilstm_M2_era5": ("-", 1.9),
        "persistencia_-_-": ("-", 1.9),
        "cams_vies_N7_-": ("--", 1.8),
        "cams_bruto_-_-": (":", 1.7),
    }
    c, n_comum = curvas_comuns(prev, list(estilos))
    for rot, (ls, lw) in estilos.items():
        d = c[c["modelo"] == rot].sort_values("h")
        ax.plot(d["h"], d["mae"], color=COR[rot], linewidth=lw, label=NOME[rot], linestyle=ls)
    ax.set_xlim(0.5, 48.5)
    ax.set_xticks([1, 12, 24, 36, 48])
    ax.set_xlabel("Horizonte (horas depois da emissão à 0h local)", color=TEXTO2, fontsize=9)
    ax.set_ylabel("MAE (µg/m³)", color=TEXTO2, fontsize=9)
    ax.set_ylim(0, None)
    ax.legend(frameon=False, fontsize=7.8, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    ax.set_title(f"Erro por horizonte na seca de 2024 (sensores do treino, {n_comum} pares comuns)", loc="left",
                 fontsize=10.2, color=TEXTO)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    rodape(fig)
    fig.savefig(OUT / "fig_mae_horizonte_2024.png")
    plt.close(fig)

    # 2. daily series, sensor 39077, day D of the 04 UTC issuance
    p = prev[prev["sensor"] == 39077].copy()
    p["t_emissao"] = pd.to_datetime(p["t_emissao"], utc=True)
    p = p[(p["t_emissao"].dt.hour == av.HORA_EMISSAO_UTC) & p["t_emissao"].dt.month.isin(av.MESES_SECA) & (p["h"] <= 24)]
    fig, ax = plt.subplots(figsize=(7.2, 4.3), dpi=200)
    estilo(ax)
    grade = pd.date_range("2024-07-01 04:00", "2024-10-31 04:00", freq="D", tz="UTC")
    g_m1 = media(p, "bilstm_M1_sem")
    d_m1 = g_m1.groupby("t_emissao").agg(
        prev=("q50", "mean"), q10=("q10", "mean"), q90=("q90", "mean"),
        y=("y", "mean"), n=("y", "count"),
    ).reindex(grade)
    valido = d_m1["n"] >= 18  # valid day: >= 18 of 24 hours
    obs = d_m1["y"].where(valido)

    g_per = media(p, "persistencia_-_-")
    d_per = g_per.groupby("t_emissao").agg(prev=("q50", "mean")).reindex(grade)

    ax.fill_between(
        grade, d_m1["q10"].where(valido), d_m1["q90"].where(valido),
        color=COR["bilstm_M1_sem"], alpha=0.18, linewidth=0,
        label="Bi-LSTM M1, faixa 10–90% (média dos quantis)",
    )
    ax.plot(grade, d_m1["prev"].where(valido), color=COR["bilstm_M1_sem"], linewidth=1.7,
            label="Bi-LSTM M1 (principal), média das medianas")
    ax.plot(grade, d_per["prev"].where(valido), color=COR["persistencia_-_-"], linewidth=1.4, alpha=0.85,
            label="Persistência (valor da 0h local)")
    ax.plot(grade, obs, color=COR["observado"], linewidth=1.6, label="Medido (PurpleAir corrigido)")
    # isolated valid days (e.g. 13/07, inside the July gap) as dots, so that no valid day is hidden
    for s, cor in ((d_m1["prev"].where(valido), COR["bilstm_M1_sem"]),
                   (d_per["prev"].where(valido), COR["persistencia_-_-"]), (obs, COR["observado"])):
        ax.plot(grade, isolados(s), linestyle="none", marker="o", markersize=2.6, color=cor)

    for q, txt in ((50, "50 µg/m³ (PI-2)"), (125, "125 µg/m³ (Atenção)")):
        ax.axhline(q, color=TEXTO2, linewidth=0.8, linestyle=":")
        ax.annotate(txt, (grade.min(), q), xytext=(2, 3), textcoords="offset points", fontsize=7.5, color=TEXTO2)
    ax.text(pd.Timestamp("2024-07-18 04:00", tz="UTC"), 205, "7 a 29/07: só 13/07\ncom dia válido no 39077",
            fontsize=7.3, color=TEXTO2, ha="center", va="center", style="italic")
    ax.set_ylabel("Média diária de PM2.5 (µg/m³)", color=TEXTO2, fontsize=9)
    ax.set_ylim(0, None)
    ax.legend(frameon=False, fontsize=7.6, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    ax.set_title("Sensor 39077, seca de 2024: medido × previsto para o próprio dia", loc="left",
                 fontsize=10.2, color=TEXTO)
    fig.autofmt_xdate()
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    rodape(fig)
    fig.savefig(OUT / "fig_serie_diaria_2024.png")
    plt.close(fig)
    print("figuras:", sorted(f.name for f in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
