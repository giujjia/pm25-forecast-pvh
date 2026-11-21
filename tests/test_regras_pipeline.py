"""Automated tests of the pre-training rules (item 4 of the methodological control).

Run (from the repository root):
    .venv/bin/python -m pytest -v

Written on 15/11/2026, with the FIRMS filters
and the floor passed explicitly (they are still-pending decisions of the author).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

import correcao  # noqa: E402
import features  # noqa: E402
import selecao_entradas  # noqa: E402


# 1. Temporal availability at t (PurpleAir start [T, T+1h), ERA5, FIRMS NRT, CAMS 12 UTC)

def test_purpleair_nao_vaza_hora_da_emissao():
    t_emissao = pd.Timestamp("2023-08-15 04:00:00", tz="UTC")
    # A window that includes T == t_emissao leaks 59 future minutes because average=60 marks start [T, T+1h).
    with pytest.raises(ValueError, match="Vazamento temporal"):
        features.validar_disponibilidade_janela(
            ["2023-08-15 02:00:00", "2023-08-15 03:00:00", "2023-08-15 04:00:00"],
            t_emissao=t_emissao,
            fonte="purpleair",
        )

    horas = pd.date_range("2023-08-14 00:00:00", "2023-08-15 06:00:00", freq="h", tz="UTC")
    df = pd.DataFrame({"time_stamp": horas, "val": np.arange(len(horas), dtype=float)})
    recorte = features.selecionar_janela_disponivel(df, t_emissao=t_emissao, fonte="purpleair", janela_horas=72)
    assert recorte["time_stamp"].max() == pd.Timestamp("2023-08-15 03:00:00", tz="UTC")
    assert features.CONVENCAO_PURPLEAIR == "inicio"


def test_era5_e_firms_e_cams_respeitam_disponibilidade_em_t():
    t_emissao = pd.Timestamp("2023-08-15 04:00:00", tz="UTC")

    # ERA5 accumulation (t-1h, t] ends at valid_time=t; t+1h must fail.
    features.validar_disponibilidade_janela(
        ["2023-08-15 03:00:00", "2023-08-15 04:00:00"], t_emissao=t_emissao, fonte="era5_janela"
    )
    with pytest.raises(ValueError, match="Vazamento temporal"):
        features.validar_disponibilidade_janela(
            ["2023-08-15 04:00:00", "2023-08-15 05:00:00"], t_emissao=t_emissao, fonte="era5_janela"
        )

    # FIRMS NRT with a 3 h delay: at 04 UTC only detections up to 01:00 UTC are published.
    features.validar_disponibilidade_janela(["2023-08-15 01:00:00"], t_emissao=t_emissao, fonte="firms_nrt")
    with pytest.raises(ValueError, match="Vazamento temporal"):
        features.validar_disponibilidade_janela(["2023-08-15 02:00:00"], t_emissao=t_emissao, fonte="firms_nrt")

    # CAMS: for a 04 UTC issuance, the 12 UTC run of D-1 (guaranteed at 22 UTC of D-1) passes; 00 UTC of D fails.
    features.validar_rodada_cams("2023-08-14 12:00:00", t_emissao=t_emissao)
    with pytest.raises(ValueError, match="Vazamento temporal"):
        features.validar_rodada_cams("2023-08-15 00:00:00", t_emissao=t_emissao)


# 2. Floor >= 0 after correction, valid log(1+y) and 0-100% humidity filter

def test_correcao_filtra_ur_e_aplica_piso_zero():
    df = pd.DataFrame({
        "pm2.5_cf_1_a": [0.0, 0.2, 50.0, 20.0, 10.0],
        "pm2.5_cf_1_b": [0.0, 0.2, 50.0, 20.0, 10.0],
        "pm2.5_atm_a":  [0.0, 0.2, 50.0, 20.0, 10.0],
        "pm2.5_atm_b":  [0.0, 0.2, 50.0, 20.0, 10.0],
    # Row 0 yields -2.87 before the floor; rows 3 and 4 have invalid RH
        "humidity":     [100.0, 95.0, 50.0, -5.0, 255.0],
    })
    sem_piso = correcao.corrigir(df, piso=None)
    assert len(sem_piso) == 3  # RH -5 and 255 removed
    assert (sem_piso["humidity"].between(0.0, 100.0)).all()
    assert sem_piso["barkjohn2022"].iloc[0] == pytest.approx(-2.87)
    n_trunc, n_val, frac = correcao.resumo_truncamento(sem_piso["barkjohn2022"], piso=0.0)
    assert n_trunc == 2 and n_val == 3 and frac == pytest.approx(2 / 3)

    assert (correcao.corrigir(df)["barkjohn2022"] < 0).sum() == 2  # default: no floor
    com_piso = correcao.corrigir(df, piso=0.0)
    assert len(com_piso) == 3
    for col in ("barkjohn2021", "barkjohn2022", "airnow_fsm"):
        assert (com_piso[col] >= 0.0).all()
        assert np.isfinite(np.log1p(com_piso[col])).all()


# 3. FIRMS aggregate without type != 0 and without satellite outside the decided list

def test_firms_agregado_rejeita_tipo_e_satelite_indevidos():
    df_firms = pd.DataFrame({
        "acq_date":   ["2023-08-14", "2023-08-14", "2023-08-14", "2023-08-15"],
        "acq_time":   [1730,          1730,          1735,          200],  # 02:00 UTC > 04:00 - 3h (leak!)
        "satellite":  ["N20",         "N",           "N20",         "N20"],
        "confidence": ["n",           "n",           "n",           "h"],
        "type":       [0,             0,             2,             0],
        "frp":        [12.5,          99.0,          50.0,          20.0],
    })

    res = features.agregar_firms_janela(
        df_firms,
        t_emissao="2023-08-15 04:00:00",
        janela_horas=24,
        atraso_horas=3,
        tipos_permitidos=(0,),
        satelites_permitidos=("N20",),
        confiancas_permitidas=None,
    )
    assert res["n_focos"] == 1
    assert res["frp_total"] == pytest.approx(12.5)
    assert (res["focos"]["type"] == 0).all()
    assert (res["focos"]["satellite"] == "N20").all()

    # No default: calling without deciding the filters is a usage error.
    with pytest.raises(TypeError):
        features.filtrar_firms(df_firms)
    # An empty satellite list is forbidden; asking for a type filter without the column (NRT) fails.
    with pytest.raises(ValueError):
        features.filtrar_firms(df_firms, tipos_permitidos=None, satelites_permitidos=(), confiancas_permitidas=None)
    with pytest.raises(ValueError, match="type"):
        features.filtrar_firms(df_firms.drop(columns=["type"]), tipos_permitidos=(0,),
                               satelites_permitidos=("N20",), confiancas_permitidas=None)
    # Without a type filter, NRT (without the column) passes.
    assert len(features.filtrar_firms(df_firms.drop(columns=["type"]), tipos_permitidos=None,
                                      satelites_permitidos=("N", "N20"), confiancas_permitidas=None)) == 4

    # Aggregate validation catches contamination if type != 0 or a satellite outside the list gets through.
    with pytest.raises(ValueError, match="type"):
        features.validar_firms_agregado(df_firms.iloc[[2]], tipos_permitidos=(0,), satelites_permitidos=("N20",))
    with pytest.raises(ValueError, match="satélite"):
        features.validar_firms_agregado(df_firms.iloc[[1]], tipos_permitidos=(0,), satelites_permitidos=("N20",))


# 4. Isolation of years and sensors per fold in selecao_entradas (no Cuniã nor evaluated year/2024)

@pytest.mark.parametrize("dobra,anos_esperados,ano_proibido", [
    ("dobra1", {2020}, 2021),
    ("dobra2", {2020, 2021}, 2022),
    ("dobra3", {2020, 2021, 2022}, 2023),
    ("final", {2020, 2021, 2022, 2023}, 2024),
])
def test_selecao_entradas_isola_anos_e_sensores_por_dobra(dobra, anos_esperados, ano_proibido):
    df = pd.DataFrame({
        "sensor_index": [39077, 39077, 56831, 39077, 39077, 39077, 128603, 219595],
        "time_stamp": [
            "2020-08-10 04:00:00",
            "2020-01-15 04:00:00",  # outside the dry season (Jul-Oct)
            "2021-09-10 04:00:00",
            "2022-08-10 04:00:00",
            "2023-08-10 04:00:00",
            "2024-08-10 04:00:00",  # 2024 test
            "2023-08-10 04:00:00",  # Cuniã (SO5) forbidden in selection
            "2024-08-10 04:00:00",  # new sensor forbidden
        ],
    })
    amostra = selecao_entradas.selecionar_amostra_dobra(df, dobra=dobra)
    anos_obtidos = set(pd.to_datetime(amostra["time_stamp"], utc=True).dt.year)
    assert anos_obtidos <= anos_esperados
    assert ano_proibido not in anos_obtidos
    assert 2024 not in anos_obtidos
    assert set(amostra["sensor_index"]).issubset(set(selecao_entradas.SENSORES_TREINO))
    assert 128603 not in set(amostra["sensor_index"])

    # Injecting the evaluated year, Cuniã or an embargo violation must raise an explicit error.
    with pytest.raises(ValueError, match="Vazamento temporal"):
        selecao_entradas.validar_isolamento_dobra(
            pd.DataFrame({"sensor_index": [39077], "time_stamp": [f"{ano_proibido}-08-01 04:00:00"]}),
            dobra=dobra,
        )
    with pytest.raises(ValueError, match="Vazamento espacial"):
        selecao_entradas.validar_isolamento_dobra(
            pd.DataFrame({"sensor_index": [128603], "time_stamp": ["2020-08-01 04:00:00"]}),
            dobra=dobra,
        )

    # Even inside the last training year, the last < 48 h before the evaluation year violate the embargo.
    ultimo_treino = max(anos_esperados)
    with pytest.raises(ValueError, match="embargo"):
        selecao_entradas.validar_isolamento_dobra(
            pd.DataFrame({"sensor_index": [39077], "time_stamp": [f"{ultimo_treino}-12-31 12:00:00"]}),
            dobra=dobra,
        )


# 4. Decisions of 16/11/2026 (METHODS.md §3 and §3.4)

def test_substituir_ur_so_a_partir_da_data():
    alvo = pd.DataFrame({"time_stamp": pd.to_datetime(["2022-09-30 23:00", "2022-10-01 00:00", "2022-10-01 01:00"], utc=True),
                         "humidity": [60.0, 99.0, 100.0]})
    doador = pd.DataFrame({"time_stamp": pd.to_datetime(["2022-09-30 23:00", "2022-10-01 00:00"], utc=True),
                           "humidity": [50.0, 55.0]})
    out = correcao.substituir_ur(alvo, doador, "2022-10-01")
    assert out["humidity"].iloc[0] == 60.0  # before the cutoff, its own
    assert out["humidity"].iloc[1] == 55.0  # after, the donor's
    assert np.isnan(out["humidity"].iloc[2])  # donor without data -> no humidity (drops out in QC)
    assert correcao.REGRA_UR["56831"] == ("39077", "2022-10-01")


def test_negativos_ficam_com_sinal_e_invalidacao_abaixo_do_limiar():
    df = pd.DataFrame({"pm2.5_cf_1_a": [0.0, 30.0], "pm2.5_cf_1_b": [0.0, 30.0],
                       "pm2.5_atm_a": [0.0, 30.0], "pm2.5_atm_b": [0.0, 30.0], "humidity": [100.0, 50.0]})
    padrao = correcao.corrigir(df)
    assert padrao["barkjohn2022"].iloc[0] == pytest.approx(-2.87)  # −2.87 > −4.99: stays, with sign
    assert correcao.LIMIAR_INVALIDO == -4.99
    mais_estrito = correcao.corrigir(df, invalidar_abaixo=-1.0)  # the mechanism invalidates (NaN)
    assert np.isnan(mais_estrito["barkjohn2022"].iloc[0]) and np.isfinite(mais_estrito["barkjohn2022"].iloc[1])
    z = correcao.transformar_log(padrao["barkjohn2022"])
    assert z[0] == 0.0 and np.isfinite(z).all()  # floor only in the transformation


def test_rotulo_purpleair_pelo_fim_da_hora():
    df = pd.DataFrame({"time_stamp": pd.to_datetime(["2023-08-14 03:00"], utc=True)})
    assert features.rotular_purpleair_pelo_fim(df)["time_stamp"].iloc[0] == pd.Timestamp("2023-08-14 04:00", tz="UTC")


def test_mascara_estatica_e_dias_de_falha_do_noaa20():
    focos = pd.DataFrame({"latitude": [-8.0, -8.0001, -9.5], "longitude": [-63.0, -63.0001, -64.0],
                          "acq_date": ["2023-07-18", "2023-07-18", "2023-07-20"], "satellite": ["N20", "N20", "N"]})
    cel = features.celula_mascara([-8.0], [-63.0])
    mascara = {(int(cel[0][0]), int(cel[1][0]))}
    restam = features.aplicar_mascara_estatica(focos, mascara)
    assert len(restam) == 2 and (restam["latitude"] == -9.5).any()  # only the detection in the masked cell is removed
    falhas = features.dias_falha_firms(focos, pd.date_range("2023-07-18", "2023-07-20"), satelite="N20")
    assert [str(d) for d in falhas] == ["2023-07-19", "2023-07-20"]  # no NOAA-20 on those days
    assert features.SATELITES_FIRMS_DECIDIDOS == ("N20",) and features.CONFIANCAS_FIRMS_DECIDIDAS is None


# 7. Target table (T12): local day and 2024 lock

def test_dia_local_da_hora_rotulada_pelo_fim():
    import alvo
    t = pd.Series(pd.to_datetime(["2023-08-15 05:00", "2023-08-16 04:00", "2023-08-16 05:00"], utc=True))
    d = alvo.dia_local(t)
    # 05 UTC covers 00–01 local of 15/08; 04 UTC of 16/08 covers 23–24 local of 15/08; 05 UTC of 16/08 is already 16/08
    assert list(d.dt.strftime("%Y-%m-%d")) == ["2023-08-15", "2023-08-15", "2023-08-16"]


def test_alvo_nao_abre_2024_sem_pre_registro():
    import alvo
    with pytest.raises(SystemExit, match="pré-registro"):
        alvo.construir(ate_ano=2024)


# 8. FIRMS fire windows (T18): available at t, outage as NaN, transported wind

def _horario_sintetico(horas, hora_det, valor=10.0):
    import entradas
    cols = [entradas.nome_col(a, s) for a in entradas.NOMES_ANEIS for s in range(entradas.N_SETORES)]
    h = pd.DataFrame(0.0, index=horas, columns=cols)
    h.loc[hora_det, entradas.nome_col("a0_50", 0)] = valor
    return h


def test_frp_24h_so_entra_depois_de_publicado():
    import entradas
    horas = pd.date_range("2023-08-10 01:00", "2023-08-16 00:00", freq="h", tz="UTC")
    det = pd.Timestamp("2023-08-14 05:00", tz="UTC")  # detection in (04:00, 05:00]
    f = entradas.janelas_fogo(_horario_sintetico(horas, det), [], horas)
    c = entradas.nome_col("a0_50", 0)
    assert f.loc[det + pd.Timedelta(hours=2), c] == 0.0      # t = 07 UTC: not published yet
    assert f.loc[det + pd.Timedelta(hours=3), c] == 10.0     # t = 08 UTC: published (05 + 3 h)
    assert f.loc[det + pd.Timedelta(hours=26), c] == 10.0    # last hour of the 24 h window
    assert f.loc[det + pd.Timedelta(hours=27), c] == 0.0
    d = entradas.nome_col("a0_50", 0, "_d24_48")
    assert f.loc[det + pd.Timedelta(hours=27), d] == 10.0


def test_falha_do_noaa20_vira_ausente_e_nao_zero():
    import datetime as dt
    import entradas
    horas = pd.date_range("2023-08-10 01:00", "2023-08-16 00:00", freq="h", tz="UTC")
    f = entradas.janelas_fogo(_horario_sintetico(horas, horas[100]), [dt.date(2023, 8, 13)], horas)
    c = entradas.nome_col("a0_50", 0)
    t = pd.Timestamp("2023-08-13 12:00", tz="UTC")
    assert np.isnan(f.loc[t, c]) and f.loc[t, "falha_firms_24h"] == 1
    t_ok = pd.Timestamp("2023-08-15 06:00", tz="UTC")  # window (08-14 03:00, 08-15 03:00] does not touch 13/08
    assert not np.isnan(f.loc[t_ok, c]) and f.loc[t_ok, "falha_firms_24h"] == 0


def test_frp_transportado_usa_so_vento_do_setor_para_o_sensor():
    import entradas
    horas = pd.date_range("2023-08-10 01:00", periods=2, freq="h", tz="UTC")
    cols = [entradas.nome_col(a, s) for a in entradas.NOMES_ANEIS for s in range(entradas.N_SETORES)]
    fogo = pd.DataFrame(0.0, index=horas, columns=cols)
    fogo[entradas.nome_col("a150_300", 4)] = 100.0  # fire to the south (sector 180–225°)
    # wind blowing to the north (v > 0) carries smoke from the south to the sensor; to the south, it does not
    tr = entradas.frp_transportado(fogo, np.array([0.0, 0.0]), np.array([5.0, -5.0]))
    assert tr[0] > 0 and tr[1] == 0.0


# 9. Input selection (T20b): 48 h target after t and 24 h history up to t, without joining years

def test_pares_emissao_alvo_depois_de_t_e_historico_ate_t():
    horas = pd.date_range("2021-10-25 01:00", "2021-10-31 23:00", freq="h", tz="UTC")
    tab = pd.DataFrame({"sensor": 39077, "time_stamp": horas, "barkjohn2022": np.arange(len(horas), dtype=float)})
    # a July row of the next year, right after in the row order, must not enter the window of 30/10
    extra = pd.DataFrame({"sensor": 39077, "time_stamp": pd.date_range("2022-07-01 00:00", periods=60, freq="h", tz="UTC"),
                          "barkjohn2022": 1e6})
    p = selecao_entradas.pares_emissao(pd.concat([tab, extra], ignore_index=True))
    t = pd.Timestamp("2021-10-27 04:00", tz="UTC")
    linha = p[p["time_stamp"] == t].iloc[0]
    i = horas.get_loc(t)
    assert linha["y48"] == pytest.approx(np.mean(np.arange(i + 1, i + 49)))
    assert linha["hist24"] == pytest.approx(np.mean(np.arange(i - 23, i + 1)))
    t_fim = pd.Timestamp("2021-10-31 04:00", tz="UTC")  # only 19 h left (< 36): no target
    assert np.isnan(p[p["time_stamp"] == t_fim].iloc[0]["y48"])


def test_rho_parcial_desconta_o_historico():
    rng = np.random.default_rng(0)
    z = rng.normal(size=500)
    x = z + 0.01 * rng.normal(size=500)        # candidate = history (adds nothing)
    y = z + 0.5 * rng.normal(size=500)
    assert abs(selecao_entradas.rho_parcial_spearman(x, y, z)) < 0.15


# 10. CAMS with moving bias (item 26): only the past, factor 1 in the first week of July

def _cams_sintetico():
    rodadas = pd.date_range("2024-06-29 12:00", "2024-07-31 12:00", freq="D", tz="UTC")
    linhas = [(r, r + pd.Timedelta(hours=k), 10.0) for r in rodadas for k in range(0, 97)]
    return pd.DataFrame(linhas, columns=["rodada", "valid_time", "pm_cams"])


def test_cams_vies_movel_so_usa_o_passado():
    import cams_vies_movel as cv
    cams = _cams_sintetico()
    horas = pd.date_range("2024-06-30", "2024-08-05", freq="h", tz="UTC")
    t = pd.Timestamp("2024-07-20 04:00", tz="UTC")
    medido = pd.Series(np.where(horas <= t, 20.0, 1e6), index=horas)  # future values absurd on purpose
    p = cv.prever(cams, medido, [t], n_dias=7)
    assert p["fator"].iloc[0] == pytest.approx(2.0)
    assert p["cams_vies"].iloc[0] == pytest.approx(20.0)
    assert len(p) == 48


def test_cams_vies_movel_fator_1_na_primeira_semana_e_rodada_disponivel():
    import cams_vies_movel as cv
    cams = _cams_sintetico()
    horas = pd.date_range("2024-06-30", "2024-08-05", freq="h", tz="UTC")
    medido = pd.Series(20.0, index=horas)
    t = pd.Timestamp("2024-07-05 04:00", tz="UTC")
    assert cv.prever(cams, medido, [t], n_dias=1)["fator"].iloc[0] == 1.0
    t14 = pd.Timestamp("2024-07-10 04:00", tz="UTC")  # N = 14 with incomplete window: 9 days since 01/07
    assert cv.prever(cams, medido, [t14], n_dias=14)["fator"].iloc[0] == pytest.approx(2.0)
    # the raw forecast uses the 12 UTC run of D − 1, which was available at 04 UTC of D
    bruto = cv.previsao_bruta(cams, t)
    assert bruto.index[0] == t + pd.Timedelta(hours=1)


# 11. Version selection rule (T26): within 2% of best primary MAE, lowest cross-fitted Brier > 50

def test_regra_selecao_2pct_mae_desempata_pelo_brier():
    import avaliar_vc
    df = pd.DataFrame([
        {"modelo": "bilstm", "versao": "M1", "variante": "sem", "mae_principal": 10.00, "brier50": 0.080, "mae50": 3.0},
        {"modelo": "bilstm", "versao": "M2", "variante": "sem", "mae_principal": 10.18, "brier50": 0.065, "mae50": 2.5},  # +1.8% <= 2%
        {"modelo": "bilstm", "versao": "M3", "variante": "sem", "mae_principal": 10.25, "brier50": 0.050, "mae50": 2.0},  # +2.5% > 2%
    ])
    res = avaliar_vc.aplicar_regra_selecao(df)
    assert res["escolhido"]["versao"] == "M2"
    assert len(res["elegiveis"]) == 2
