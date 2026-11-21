"""Evaluates cross-validation predictions (T21) and applies the version selection rule (T26).

Reads the out-of-fold predictions in data/processed/vc/ (2021–2023 only; asserts no 2024 data),
averages seeds quantile by quantile (METHODS.md §6), computes:
- Primary metric: aggregated MAE of the median (04 UTC issuances, Jul–Oct, 3 folds together),
  plus per-fold MAE and seed dispersion (median, min, max);
- Tail error: MAE above 50 µg/m³ (Gneiting 2011);
- Selection Brier score (> 50 µg/m³): HXLR (src/saida_diaria.py) cross-fitted across the 3 folds
  (each fold's regression is fitted on the other two folds, separately for D and D+1);
- Uncertainty (SO2): pinball loss (µg/m³), Q10–Q90 coverage and mean width (overall and by 12 h block);
- Error curves by horizon (+1..+48 h): MAE, RMSE, R²;
- Diebold-Mariano (HLN + HAC) + Holm comparisons in CV (diagnostic).

Selection rule (METHODS.md §6, decided 2026-11-16):
Among configurations whose aggregated primary MAE is <= 1.02 * best_MAE, keep the one with the
lowest aggregated cross-fitted Brier score for "day > 50 µg/m³".

Usage (from the root):
    .venv/bin/python src/avaliar_vc.py
Outputs: data/processed/avaliacao_vc/{resumo_vc.csv, selecao_vc.json, dm_vc.csv,
         incerteza_vc.csv, curvas_horizonte_vc.csv}
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
import avaliacao as av  # noqa: E402
import saida_diaria as sd  # noqa: E402

VC_DIR = RAIZ / "data" / "processed" / "vc"
ENTRADAS = RAIZ / "data" / "processed" / "entradas" / "entradas_horario_2020_2023.parquet"
OUT = RAIZ / "data" / "processed" / "avaliacao_vc"

DOBRAS = ("dobra1", "dobra2", "dobra3")
VERSOES = ("M1", "M2", "M3")
VARIANTES = ("sem", "era5")
SEMENTES = (0, 1, 2)
TOLERANCIA_SELECAO = 0.02
LIMIAR_BRIER = 50.0


def calcular_hist24(caminho=ENTRADAS):
    """Mean observed PM2.5 (Barkjohn 2022) over (t - 24 h, t] with >= 18 valid hours, per sensor."""
    tab = pd.read_parquet(caminho, columns=["sensor", "time_stamp", "barkjohn2022"])
    tab["time_stamp"] = pd.to_datetime(tab["time_stamp"], utc=True)
    # Last hour of 2023 (23:00–23:59 UTC) is relabelled by end-of-hour to 2024-01-01 00:00 UTC.
    assert tab["time_stamp"].max() <= pd.Timestamp("2024-01-01 00:00:00", tz="UTC"), "Entradas contêm 2024!"
    partes = []
    for s, g in tab.groupby("sensor"):
        g = g.set_index("time_stamp").sort_index()
        g = g.reindex(pd.date_range(g.index.min(), g.index.max(), freq="h"))
        h24 = g["barkjohn2022"].rolling(24, min_periods=18).mean()
        d = pd.DataFrame({"sensor": s, "t_emissao": h24.index, "hist24": h24.to_numpy()})
        partes.append(d)
    df = pd.concat(partes, ignore_index=True)
    return df.set_index(["sensor", "t_emissao"])["hist24"]


def carregar_parquets(padroes):
    partes = []
    for p in padroes:
        df = pd.read_parquet(p)
        ts = pd.to_datetime(df["t_emissao"], utc=True)
        assert int(ts.dt.year.max()) <= 2023, f"Previsão {p.name} contém ano > 2023!"
        df["t_emissao"] = ts
        partes.append(df)
    return pd.concat(partes, ignore_index=True)


def brier_crossfit(pred_media, hist24):
    """Cross-fitted HXLR Brier score for day > 50 µg/m³ on dry-season 04 UTC issuances."""
    if pred_media["q10"].isna().all():
        return {"brier50": np.nan, "n_dias_brier": 0, "brier50_d0": np.nan, "brier50_d1": np.nan, "hxlr_ok": True}

    t = pd.to_datetime(pred_media["t_emissao"], utc=True)
    p04 = pred_media[t.dt.hour == av.HORA_EMISSAO_UTC].copy()
    tabs = {}
    for d in DOBRAS:
        sub = p04[p04["dobra"] == d]
        tb = sd.tabela_diaria(sub, hist24)
        tb["dobra"] = d
        tb["seca"] = pd.to_datetime(tb["t_emissao"], utc=True).dt.month.isin(av.MESES_SECA)
        tabs[d] = tb

    probs_all, y_all, dia_all = [], [], []
    todos_ok = True
    for d_av in DOBRAS:
        treino_all = pd.concat([tabs[d] for d in DOBRAS if d != d_av], ignore_index=True)
        av_seca = tabs[d_av][tabs[d_av]["seca"]].copy()
        for dia in (0, 1):
            tr_dia = treino_all[treino_all["dia"] == dia]
            av_dia = av_seca[av_seca["dia"] == dia]
            mod = sd.HXLR().ajustar(tr_dia)
            todos_ok = todos_ok and bool(mod.ok)
            pr = mod.prob_acima(av_dia, LIMIAR_BRIER)
            probs_all.append(pr)
            y_all.append(av_dia["ybar"].to_numpy())
            dia_all.append(np.full(len(av_dia), dia, dtype=int))

    probs = np.concatenate(probs_all)
    ybar = np.concatenate(y_all)
    dias = np.concatenate(dia_all)
    b_tot, n_tot = sd.brier(probs, ybar, LIMIAR_BRIER)
    b_d0, _ = sd.brier(probs[dias == 0], ybar[dias == 0], LIMIAR_BRIER)
    b_d1, _ = sd.brier(probs[dias == 1], ybar[dias == 1], LIMIAR_BRIER)
    return {
        "brier50": b_tot,
        "n_dias_brier": n_tot,
        "brier50_d0": b_d0,
        "brier50_d1": b_d1,
        "hxlr_ok": todos_ok,
    }


def aplicar_regra_selecao(df_sub, tolerancia=TOLERANCIA_SELECAO):
    """METHODS.md §6: within 2% of best primary MAE, choose lowest cross-fitted Brier (> 50)."""
    df_sub = df_sub.dropna(subset=["mae_principal", "brier50"]).copy()
    melhor_mae = float(df_sub["mae_principal"].min())
    teto = melhor_mae * (1.0 + tolerancia)
    elegiveis = df_sub[df_sub["mae_principal"] <= teto].sort_values(["brier50", "mae_principal"])
    vencedor = elegiveis.iloc[0]
    return {
        "melhor_mae": melhor_mae,
        "teto_2pct": teto,
        "elegiveis": [
            {
                "modelo": r["modelo"],
                "versao": r["versao"],
                "variante": r["variante"],
                "mae_principal": float(r["mae_principal"]),
                "dif_pct_melhor": float(100.0 * (r["mae_principal"] / melhor_mae - 1.0)),
                "brier50": float(r["brier50"]),
                "mae50": float(r["mae50"]),
            }
            for _, r in elegiveis.iterrows()
        ],
        "escolhido": {
            "modelo": vencedor["modelo"],
            "versao": vencedor["versao"],
            "variante": vencedor["variante"],
            "mae_principal": float(vencedor["mae_principal"]),
            "brier50": float(vencedor["brier50"]),
            "mae50": float(vencedor["mae50"]),
        },
    }


def coletar_configuracoes():
    """Returns list of (modelo, versao, variante, [parquet paths])."""
    configs = []
    adv_files = [VC_DIR / f"pred_adversarios_-_-_{d}_s0.parquet" for d in DOBRAS]
    adv_df = carregar_parquets(adv_files)
    for m in sorted(adv_df["modelo"].unique()):
        configs.append((m, "-", "-", adv_df[adv_df["modelo"] == m].copy()))

    for v in VERSOES:
        for var in VARIANTES:
            p_r = [VC_DIR / f"pred_ridge_{v}_{var}_{d}_s0.parquet" for d in DOBRAS]
            configs.append(("ridge", v, var, carregar_parquets(p_r)))

    for m in ("xgb", "bilstm"):
        for v in VERSOES:
            for var in VARIANTES:
                ps = [VC_DIR / f"pred_{m}_{v}_{var}_{d}_s{s}.parquet" for d in DOBRAS for s in SEMENTES]
                configs.append((m, v, var, carregar_parquets(ps)))
    return configs


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    hist24 = calcular_hist24()
    configs = coletar_configuracoes()

    linhas_resumo = []
    linhas_inc = []
    linhas_curva = []
    medias_por_chave = {}

    for modelo, versao, variante, df_raw in configs:
        chave = (modelo, versao, variante)
        sementes = sorted(df_raw["semente"].unique())
        maes_sem = [av.metrica_principal(df_raw[df_raw["semente"] == s])[0] for s in sementes]

        pred_med = av.media_sementes(df_raw) if len(sementes) > 1 else df_raw
        medias_por_chave[chave] = pred_med

        mae_agg, n_pares = av.metrica_principal(pred_med, perda="mae")
        mae50_agg, _ = av.metrica_principal(pred_med, perda="mae50")
        mae_dobras = {d: av.metrica_principal(pred_med[pred_med["dobra"] == d])[0] for d in DOBRAS}
        br = brier_crossfit(pred_med, hist24)

        inc_seca = None
        if not pred_med["q10"].isna().all():
            for escopo, sub in (("ano_todo", pred_med), ("seca_04utc", av.filtro_principal(pred_med))):
                tb_inc = av.incerteza(sub).reset_index().rename(columns={"index": "bloco"})
                tb_inc.insert(0, "escopo", escopo)
                tb_inc.insert(0, "variante", variante)
                tb_inc.insert(0, "versao", versao)
                tb_inc.insert(0, "modelo", modelo)
                linhas_inc.append(tb_inc)
                if escopo == "seca_04utc":
                    inc_seca = tb_inc[tb_inc["bloco"] == "total"].iloc[0]

        curva = av.curva_horizonte(pred_med).reset_index()
        curva.insert(0, "variante", variante)
        curva.insert(0, "versao", versao)
        curva.insert(0, "modelo", modelo)
        linhas_curva.append(curva)

        linhas_resumo.append({
            "modelo": modelo,
            "versao": versao,
            "variante": variante,
            "n_sementes": len(sementes),
            "n_pares": n_pares,
            "mae_principal": mae_agg,
            "mae_d1_2021": mae_dobras["dobra1"],
            "mae_d2_2022": mae_dobras["dobra2"],
            "mae_d3_2023": mae_dobras["dobra3"],
            "mae_sem_mediana": float(np.median(maes_sem)),
            "mae_sem_min": float(np.min(maes_sem)),
            "mae_sem_max": float(np.max(maes_sem)),
            "mae50": mae50_agg,
            "brier50": br["brier50"],
            "brier50_d0": br["brier50_d0"],
            "brier50_d1": br["brier50_d1"],
            "n_dias_brier": br["n_dias_brier"],
            "hxlr_ok": br["hxlr_ok"],
            "pinball_seca04": float(inc_seca["pinball"]) if inc_seca is not None else np.nan,
            "cobertura_seca04": float(inc_seca["cobertura"]) if inc_seca is not None else np.nan,
            "largura_seca04": float(inc_seca["largura"]) if inc_seca is not None else np.nan,
        })

    df_res = pd.DataFrame(linhas_resumo).sort_values("mae_principal").reset_index(drop=True)
    df_res.to_csv(OUT / "resumo_vc.csv", index=False)
    pd.concat(linhas_inc, ignore_index=True).to_csv(OUT / "incerteza_vc.csv", index=False)
    pd.concat(linhas_curva, ignore_index=True).to_csv(OUT / "curvas_horizonte_vc.csv", index=False)

    selecao = {
        "bilstm_todas": aplicar_regra_selecao(df_res[df_res["modelo"] == "bilstm"]),
        "bilstm_sem": aplicar_regra_selecao(df_res[(df_res["modelo"] == "bilstm") & (df_res["variante"] == "sem")]),
        "bilstm_era5": aplicar_regra_selecao(df_res[(df_res["modelo"] == "bilstm") & (df_res["variante"] == "era5")]),
        "xgb_todas": aplicar_regra_selecao(df_res[df_res["modelo"] == "xgb"]),
        "xgb_sem": aplicar_regra_selecao(df_res[(df_res["modelo"] == "xgb") & (df_res["variante"] == "sem")]),
        "xgb_era5": aplicar_regra_selecao(df_res[(df_res["modelo"] == "xgb") & (df_res["variante"] == "era5")]),
    }
    (OUT / "selecao_vc.json").write_text(json.dumps(selecao, indent=2, ensure_ascii=False))

    # Diebold-Mariano diagnostics in CV: chosen Bi-LSTM (overall, sem, era5) vs adversaries & stepwise M1->M2->M3
    dm_rows = []
    def add_dm(grupo, ch_a, ch_b):
        d = av.serie_diferencial(medias_por_chave[ch_a], medias_por_chave[ch_b], perda="mae")
        r = av.diebold_mariano(d)
        dm_rows.append({
            "grupo": grupo,
            "a": f"{ch_a[0]}_{ch_a[1]}_{ch_a[2]}",
            "b": f"{ch_b[0]}_{ch_b[1]}_{ch_b[2]}",
            "n_emissoes": r["n"],
            "dif_mae_a_menos_b": r["dif_media"],
            "ic95_inf": r["ic95"][0],
            "ic95_sup": r["ic95"][1],
            "estatistica_dm": r["estatistica"],
            "p_valor": r["p"],
        })

    for var in VARIANTES:
        esc_b = selecao[f"bilstm_{var}"]["escolhido"]
        ch_b = (esc_b["modelo"], esc_b["versao"], esc_b["variante"])
        esc_x = selecao[f"xgb_{var}"]["escolhido"]
        ch_x = (esc_x["modelo"], esc_x["versao"], esc_x["variante"])
        alvos = [
            ("persistencia", "-", "-"),
            ("persistencia_24h", "-", "-"),
            ("climatologia", "-", "-"),
            ("ridge", esc_b["versao"], var),
            ch_x,
        ]
        for ch_adv in alvos:
            add_dm(f"bilstm_{var}_vs_adversarios", ch_b, ch_adv)
        # stepwise ablation M1 -> M2 -> M3
        add_dm(f"ablacao_bilstm_{var}", ("bilstm", "M2", var), ("bilstm", "M1", var))
        add_dm(f"ablacao_bilstm_{var}", ("bilstm", "M3", var), ("bilstm", "M2", var))
        add_dm(f"ablacao_bilstm_{var}", ("bilstm", "M3", var), ("bilstm", "M1", var))

    # SO6 in CV: era5 vs sem for chosen versions
    esc_all = selecao["bilstm_todas"]["escolhido"]
    add_dm("so6_bilstm", ("bilstm", esc_all["versao"], "era5"), ("bilstm", esc_all["versao"], "sem"))

    df_dm = pd.DataFrame(dm_rows)
    df_dm["p_holm"] = np.nan
    for grp, idx in df_dm.groupby("grupo").groups.items():
        df_dm.loc[idx, "p_holm"] = av.holm(df_dm.loc[idx, "p_valor"].to_numpy())
    df_dm.to_csv(OUT / "dm_vc.csv", index=False)

    # Print concise summary to stdout (captured by executar.py)
    cols_show = ["modelo", "versao", "variante", "mae_principal", "mae_d1_2021", "mae_d2_2022",
                 "mae_d3_2023", "mae50", "brier50", "cobertura_seca04", "largura_seca04"]
    print("=== RESUMO VC (ordenado por MAE principal agregada) ===")
    print(df_res[cols_show].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n=== SELECAO (regra 2% MAE + menor Brier > 50) ===")
    for k, v in selecao.items():
        e = v["escolhido"]
        print(f"  {k:14s}: {e['modelo']} {e['versao']} {e['variante']} | MAE={e['mae_principal']:.4f} "
              f"(melhor={v['melhor_mae']:.4f}, teto={v['teto_2pct']:.4f}) | Brier50={e['brier50']:.5f} | elegiveis={len(v['elegiveis'])}")
    print("\n=== DIEBOLD-MARIANO NA VC (diagnostico) ===")
    print(df_dm.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
