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
