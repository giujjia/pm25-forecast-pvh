"""Checks the coverage of the A925 (INMET) CSVs written by src/ingest/inmet.py.

Counts only, never values: hours expected in the period of the file name, hours present,
missing and repeated, and the fraction of hours with a valid value (neither empty nor -9999) in each
main variable, per year and per month. Applies equally to all years, including 2024 and 2026,
because no statistic of the values is computed.

Usage:
    python src/verificar_inmet.py
"""
import re
import sys
from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "inmet"
LINHAS_META = 8
VARIAVEIS = {  # short label -> start of the column name in the CSV
    "precip": "PRECIPITAÇÃO TOTAL",
    "pressao": "PRESSAO ATMOSFERICA AO NIVEL",
    "radiacao": "RADIACAO GLOBAL",
    "temp": "TEMPERATURA DO AR - BULBO SECO",
    "orvalho": "TEMPERATURA DO PONTO DE ORVALHO",
    "ur": "UMIDADE RELATIVA DO AR, HORARIA",
    "vento_dir": "VENTO, DIREÇÃO HORARIA",
    "vento_vel": "VENTO, VELOCIDADE HORARIA",
}


def periodo_do_nome(nome):
    m = re.search(r"_(\d{2}-\d{2}-\d{4})_A_(\d{2}-\d{2}-\d{4})", nome)
    ini = pd.Timestamp(pd.to_datetime(m.group(1), format="%d-%m-%Y"))
    fim = pd.Timestamp(pd.to_datetime(m.group(2), format="%d-%m-%Y")) + pd.Timedelta(hours=23)
    return ini, fim


def ler(p):
    df = pd.read_csv(p, sep=";", encoding="latin-1", skiprows=LINHAS_META, dtype=str)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    data = df.iloc[:, 0].str.replace("-", "/", regex=False)
    hora = df.iloc[:, 1].str.replace(" UTC", "", regex=False).str.replace(":", "", regex=False).str.zfill(4)
    df.index = pd.to_datetime(data + " " + hora, format="%Y/%m/%d %H%M")
    return df


def colunas(df):
    achadas = {}
    for curto, prefixo in VARIAVEIS.items():
        c = [x for x in df.columns if x.upper().startswith(prefixo)]
        if len(c) != 1:
            raise RuntimeError(f"coluna de {curto!r}: esperado 1, achei {c}")
        achadas[curto] = c[0]
    return achadas


def validos(serie):
    s = serie.fillna("").str.strip()
    return (s != "") & (s != "-9999") & (s != "-9999,0")


def main():
    problemas = 0
    arquivos = sorted(RAW.glob("INMET_*_A925_*.CSV"))
    print(f"{len(arquivos)} arquivos em {RAW}")
    for p in arquivos:
        ini, fim = periodo_do_nome(p.name)
        df = ler(p)
        cols = colunas(df)
        esperadas = pd.date_range(ini, fim, freq="h")
        t = df.index
        faltando = esperadas.difference(t)
        repetidas = int(t.duplicated().sum())
        fora = int(((t < ini) | (t > fim)).sum())
        ok = len(faltando) == 0 and repetidas == 0 and fora == 0
        problemas += 0 if ok else 1
        print(f"\n== {p.name}")
        print(f"  horas: {len(t)} linhas, {len(t.unique())}/{len(esperadas)} esperadas "
              f"({ini:%Y-%m-%d} a {fim:%Y-%m-%d %H}h UTC), faltando {len(faltando)}, "
              f"repetidas {repetidas}, fora do período {fora} -> {'OK' if ok else 'PROBLEMA'}")
        frac = pd.DataFrame({k: validos(df[c]) for k, c in cols.items()}, index=t)
        print("  horas com valor válido no ano (% das esperadas): "
              + ", ".join(f"{k} {100 * frac[k].sum() / len(esperadas):.1f}" for k in cols))
        mensal = frac.groupby(frac.index.month).sum()
        horas_mes = pd.Series(1, index=esperadas).groupby(esperadas.month).sum()
        tab = (100 * mensal.div(horas_mes, axis=0)).round(0).astype(int)
        tab.index = [f"{m:02d}" for m in tab.index]
        print("  % de horas válidas por mês:")
        for linha in tab.to_string().splitlines():
            print("    " + linha)
    print(f"\nRESUMO: {problemas} arquivo(s) com problema de horas")
    return problemas


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
