"""Tests of the download planning (src/ingest): what is requested, with no network or disk.

Written on 15/11/2026 together with the ERA5 month cutoff, the FIRMS monthly files
and EAC4 (download plan, step 1).
"""
import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "ingest"))
import cams  # noqa: E402
import era5  # noqa: E402
import firms  # noqa: E402


# ERA5

def test_era5_corta_o_ano_corrente_no_mes_pedido():
    assert era5.blocos("sl", 2026, 2026, set(), ate=(2026, 10)) == [
        (2026, [1, 2, 3, 4]), (2026, [5, 6, 7, 8]), (2026, [9, 10])]
    assert era5.blocos("pl", 2026, 2026, set(), ate=(2026, 10)) == [
        (2026, [1, 2, 3, 4, 5, 6]), (2026, [7, 8, 9, 10])]


def test_era5_meses_cortados_entram_depois_em_bloco_proprio():
    feitos = {(2026, m) for m in range(1, 11)}
    assert era5.blocos("sl", 2026, 2026, feitos, ate=(2026, 12)) == [(2026, [11, 12])]


def test_era5_sem_corte_mantem_os_blocos_de_antes():
    assert era5.blocos("sl", 2025, 2025, set()) == [
        (2025, [1, 2, 3, 4]), (2025, [5, 6, 7, 8]), (2025, [9, 10, 11, 12])]


def test_era5_para_se_pedir_o_mes_corrente():
    tarefas = [("sl", 2026, [9, 10, 11, 12], None)]
    with pytest.raises(SystemExit):
        era5.checar_meses(tarefas, hoje=date(2026, 11, 15))
    era5.checar_meses([("sl", 2026, [9, 10], None)], hoje=date(2026, 11, 15))  # does not stop


def test_era5_separa_a_opcao_ate():
    assert era5.separar_ate(["2025", "2026", "--ate", "2026-10"]) == (["2025", "2026"], (2026, 10))
    assert era5.separar_ate(["2025", "2026", "pl"]) == (["2025", "2026", "pl"], None)


# FIRMS: window equal to the availability queried on 15/11/2026

JANELA = {
    "VIIRS_SNPP_SP": (date(2012, 1, 20), date(2026, 7, 31)),
    "VIIRS_NOAA20_SP": (date(2018, 4, 1), date(2026, 7, 31)),
    "VIIRS_SNPP_NRT": (date(2026, 8, 1), date(2026, 11, 2)),
    "VIIRS_NOAA20_NRT": (date(2026, 8, 1), date(2026, 11, 15)),
}


def test_firms_ano_coberto_pelo_sp_vai_num_arquivo_so():
    assert firms.plano_ano(2025, JANELA, ate=(2026, 10)) == [
        ("", "SP", date(2025, 1, 1), date(2025, 12, 31))]
    assert firms.plano_ano(2020, JANELA) == [("", "SP", date(2020, 1, 1), date(2020, 12, 31))]


def test_firms_ano_corrente_vai_por_mes_sp_e_depois_nrt():
    plano = firms.plano_ano(2026, JANELA, ate=(2026, 10))
    assert [p[0] for p in plano] == [f"2026{m:02d}_sp" for m in range(1, 8)] + [
        "202608_nrt", "202609_nrt", "202610_nrt"]
    assert plano[-1][2:] == (date(2026, 10, 1), date(2026, 10, 31))


def test_firms_mes_sem_cobertura_completa_interrompe():
    # S-NPP NRT ends on 02/11/2026: November is not complete on both satellites
    with pytest.raises(ValueError):
        firms.plano_ano(2026, JANELA, ate=(2026, 11))
    with pytest.raises(ValueError):
        firms.plano_ano(2026, JANELA)  # without --ate, the current year does not close


def test_firms_2026_e_ano_do_teste_cego():
    assert 2026 in firms.TESTE_CEGO


# CAMS EAC4

def test_eac4_2024_fica_num_arquivo_a_parte_e_pedido():
    assert cams.BLOCOS_EAC4 == [("2020-2023", "2020-01-01", "2023-12-31"),
                                ("2024", "2024-01-01", "2024-12-31")]
    req = cams.pedido_eac4("2020-01-01", "2020-12-31")
    assert req["time"] == [f"{h:02d}:00" for h in range(0, 24, 3)]
    assert req["area"] == era5.AREA


def test_cams_previsao_cobre_2020_a_2026_sem_lacunas_nem_sobreposicao():
    from datetime import timedelta
    blocos = cams.blocos_previsao()
    datas = []
    for _, ini, fim in blocos:
        d, f = date.fromisoformat(ini), date.fromisoformat(fim)
        while d <= f:
            datas.append(d)
            d += timedelta(days=1)
    assert len(datas) == len(set(datas))
    assert min(datas) == date(2019, 12, 31)
    assert max(datas) == date(2026, 10, 31)
    assert len(datas) == (date(2026, 10, 31) - date(2019, 12, 31)).days + 1
