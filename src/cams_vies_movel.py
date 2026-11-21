"""CAMS forecast adversaries: raw and with a moving multiplicative bias (item 26 of the starting point).

Decided on 16/11/2026 (doc 04 §3.5 and §3.9): for each issuance at 0h local (04 UTC of day D),
- raw CAMS: the 12 UTC run of D − 1 (guaranteed at 22 UTC; auditoria F31), at the cell nearest the
  sensor, for the hours t + 1 … t + 48;
- moving bias: the raw CAMS times the ratio between the measured PM2.5 and the CAMS forecast over the
  N previous days (N = 7; 1, 3 and 14 as sensitivity), using only what exists at t.

IMPLEMENTAÇÃO (not fixed by the text; confirmed by the author on 18/11/2026):
- "the CAMS forecast for a previous day" is the first 24 h (t' + 1 … t' + 24) of the issuance of that
  day t' = t − k days, k = 1 … N, i.e. the forecast that was operational for it. All those hours are <= t;
- the ratio is Σ measured / Σ CAMS over the hours with both values (measured signed, as in the evaluation);
  if the denominator is <= 0 or there is no pair, the factor is 1;
- the CAMS pm2p5 at valid_time v is compared with the target labelled v (end of the hour (v − 1 h, v]);
- the first 7 days of July have factor 1 for every N, so all models are compared on the same days;
  with N = 14, days 8–14 of July use the incomplete window (only the days since 1 July).

2024 is only run after the pre-registration: --ano 2024 requires --confirmo-pre-registro.
Usage: .venv/bin/python src/cams_vies_movel.py --ano 2024 --confirmo-pre-registro
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
RAW = RAIZ / "data" / "raw" / "cams"
INVENTARIO = RAIZ / "data" / "raw" / "purpleair" / "inventario.csv"
HORA_EMISSAO_UTC = 4
HORIZONTE_H = 48
N_PRINCIPAL = 7
N_SENSIBILIDADE = (1, 3, 14)
DIAS_SEM_CORRECAO = 7


def cams_na_celula(arquivos, lat, lon):
    """Long table (rodada, valid_time, pm µg/m³) of the cell nearest (lat, lon)."""
    partes = []
    for f in arquivos:
        ds = xr.open_dataset(f)
        da = ds["pm2p5"].sel(latitude=lat, longitude=lon, method="nearest")
        fator = 1e9 if str(da.attrs.get("units", "")).replace(" ", "") in ("kgm**-3", "kgm-3", "kg/m3") else 1.0
        df = da.to_dataframe().reset_index()
        df["pm_cams"] = df["pm2p5"].astype(float) * fator
        df = df.rename(columns={"forecast_reference_time": "rodada"})
        partes.append(df[["rodada", "valid_time", "pm_cams"]])
        ds.close()
    out = pd.concat(partes, ignore_index=True)
    out["rodada"] = pd.to_datetime(out["rodada"], utc=True)
    out["valid_time"] = pd.to_datetime(out["valid_time"], utc=True)
    return out.drop_duplicates(["rodada", "valid_time"])


def previsao_bruta(cams, t_emissao):
    """Raw CAMS for t + 1 … t + 48 from the 12 UTC run of D − 1 (checked available at t)."""
    rodada = (t_emissao - pd.Timedelta(days=1)).normalize() + pd.Timedelta(hours=12)
    features.validar_rodada_cams(rodada, t_emissao)
    alvo = pd.date_range(t_emissao + pd.Timedelta(hours=1), periods=HORIZONTE_H, freq="h")
    r = cams[cams["rodada"] == rodada].set_index("valid_time")["pm_cams"]
    return pd.Series(r.reindex(alvo).to_numpy(), index=alvo)


def fator_vies(cams, medido, t_emissao, n_dias, inicio_temporada):
    """Ratio Σ measured / Σ CAMS over the first 24 h of the issuances of the n previous days (only hours <= t)."""
    dia = t_emissao.normalize()
    if (dia - inicio_temporada).days < DIAS_SEM_CORRECAO:
        return 1.0
    num = den = 0.0
    for k in range(1, n_dias + 1):
        t_k = t_emissao - pd.Timedelta(days=k)
        if t_k.normalize() < inicio_temporada:
            break  # incomplete window (N = 14, days 8–14 of July)
        prev = previsao_bruta(cams, t_k).iloc[:24]
        assert prev.index.max() <= t_emissao  # nothing after t
        obs = medido.reindex(prev.index)
        ok = prev.notna() & obs.notna()
        num += float(obs[ok].sum())
        den += float(prev[ok].sum())
    return num / den if den > 0 and num == num else 1.0


def prever(cams, medido, emissoes, n_dias=N_PRINCIPAL, inicio_temporada=None):
    """Long table (t_emissao, h, cams_bruto, fator, cams_vies) for the given issuances of one sensor."""
    linhas = []
    for t in emissoes:
        ini = inicio_temporada if inicio_temporada is not None else pd.Timestamp(t.year, 7, 1, tz="UTC")
        bruto = previsao_bruta(cams, t)
        f = fator_vies(cams, medido, t, n_dias, ini)
        for h, v in enumerate(bruto.to_numpy(), start=1):
            linhas.append((t, h, v, f, v * f))
    return pd.DataFrame(linhas, columns=["t_emissao", "h", "cams_bruto", "fator", "cams_vies"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ano", type=int, required=True)
    ap.add_argument("--confirmo-pre-registro", action="store_true")
    a = ap.parse_args()
    if a.ano >= 2024 and not a.confirmo_pre_registro:
        raise SystemExit("2024 só é aberto depois do pré-registro (T27). Use --confirmo-pre-registro.")
    raise SystemExit("Execução real fica para a T29 (teste de 2024), pelo registrador; ver docstring.")


if __name__ == "__main__":
    main()
