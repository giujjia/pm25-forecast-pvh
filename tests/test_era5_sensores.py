"""Tests of the table of ERA5 variables per sensor (src/era5_sensores.py)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import era5_sensores as e5  # noqa: E402


def _conjuntos(n=240, seed=0):
    rng = np.random.default_rng(seed)
    tempo = pd.date_range("2022-08-01", periods=n, freq="h")
    lat = np.array([-8.5, -8.75, -9.0])
    lon = np.array([-64.0, -63.75])
    forma = (n, lat.size, lon.size)
    coords = {"valid_time": tempo, "latitude": lat, "longitude": lon}
    t2m = 300.0 + rng.normal(0, 2, forma)
    sl = xr.Dataset({
        "tp": (("valid_time", "latitude", "longitude"), np.zeros(forma)),
        "blh": (("valid_time", "latitude", "longitude"), 500 + rng.random(forma) * 1000),
        "u10": (("valid_time", "latitude", "longitude"), rng.normal(0, 2, forma)),
        "v10": (("valid_time", "latitude", "longitude"), rng.normal(0, 2, forma)),
        "t2m": (("valid_time", "latitude", "longitude"), t2m),
        "d2m": (("valid_time", "latitude", "longitude"), t2m - rng.random(forma) * 10),
    }, coords=coords)
    pl = xr.Dataset({
        "u": (("valid_time", "latitude", "longitude"), rng.normal(0, 4, forma)),
        "v": (("valid_time", "latitude", "longitude"), rng.normal(0, 4, forma)),
    }, coords=coords)
    return sl, pl


def test_umidade_relativa():
    assert e5.umidade_relativa(300.0, 300.0) == pytest.approx(100.0)
    # T = 30 °C, Td = 20 °C: ~55% with the IFS saturation pressure
    assert e5.umidade_relativa(303.15, 293.15) == pytest.approx(55.1, abs=0.5)
    assert e5.umidade_relativa(300.0, 301.0) == 100.0  # clipped to 100


def test_chuva_acumulada_termina_em_t():
    tp = np.zeros(60)
    tp[10] = 1.0  # rain of (9 h, 10 h], with valid_time = 10 h
    a24 = e5.chuva_acumulada(tp, 24)
    assert np.isnan(a24[:23]).all()
    assert a24[23] == 1.0  # window 0–23 h contains hour 10
    assert a24[33] == 1.0  # window 10–33 h still contains hour 10
    assert a24[34] == 0.0  # window 11–34 h no longer does
    b = e5.chuva_acumulada(np.r_[np.zeros(9), 2.0, np.zeros(20)], 3)
    assert b[8] == 0.0 and b[9] == 2.0 and b[11] == 2.0 and b[12] == 0.0


def test_horas_sem_chuva():
    tp = np.array([0, 0, 1.5, 0.3, 0, 1.0, 0, 0])
    h = e5.horas_sem_chuva(tp)
    assert np.isnan(h[:2]).all()
    assert list(h[2:]) == [0, 1, 2, 0, 1, 2]  # 0.3 mm does not count


def test_tabela_sensor_celula_regiao_e_ventilacao():
    sl, pl = _conjuntos()
    df = e5.tabela_sensor(sl, pl, lat=-8.72, lon=-63.88)
    assert df.lat_celula.iat[0] == -8.75 and df.lon_celula.iat[0] == -64.0
    c = sl.sel(latitude=-8.75, longitude=-64.0)
    np.testing.assert_allclose(df.blh, c.blh.values)
    np.testing.assert_allclose(df.ventilacao, c.blh.values * np.hypot(c.u10.values, c.v10.values))
    np.testing.assert_allclose(df.u850_reg, pl.u.mean(["latitude", "longitude"]).values)
    assert str(df.time_stamp.dt.tz) == "UTC"


def test_nenhum_valor_depende_do_futuro():
    sl, pl = _conjuntos()
    sl["tp"][:] = np.random.default_rng(1).random(sl.tp.shape) * 2
    base = e5.tabela_sensor(sl, pl, lat=-8.72, lon=-63.88)
    t0 = 150
    sl2, pl2 = sl.copy(deep=True), pl.copy(deep=True)
    for v in sl2.data_vars:
        sl2[v][t0 + 1:] = sl2[v][t0 + 1:] * 3 + 7
    for v in pl2.data_vars:
        pl2[v][t0 + 1:] = -pl2[v][t0 + 1:]
    mudado = e5.tabela_sensor(sl2, pl2, lat=-8.72, lon=-63.88)
    cols = [c for c in base.columns if c not in ("time_stamp",)]
    pd.testing.assert_frame_equal(base.loc[:t0, cols], mudado.loc[:t0, cols])
