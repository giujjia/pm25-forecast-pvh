"""Checks the coverage of the AERONET files written by src/ingest/aeronet.py.

Counts only, never AOD values: level declared in the header, number of measurements,
first and last day, days with measurements per month, and whether the level 2.0 stamps are contained
in those of 1.5. Applies equally to 2024 and to the files in selado_2026/.

Usage:
    python src/verificar_aeronet.py
"""
from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "aeronet"


def ler(p):
    linhas = p.read_text(encoding="latin-1").splitlines()
    nivel = next((ln.split("Level")[-1].strip() for ln in linhas[:6] if "AOD Level" in ln), "?")
    i = next((k for k, ln in enumerate(linhas) if ln.startswith("AERONET_Site,")), None)
    if i is None:
        return nivel, pd.DatetimeIndex([]), linhas[:2]
    t = pd.to_datetime([ln.split(",")[1] + " " + ln.split(",")[2] for ln in linhas[i + 1:] if ln.strip()],
                       format="%d:%m:%Y %H:%M:%S")
    return nivel, pd.DatetimeIndex(t), []


def main():
    carimbos = {}
    for p in sorted(RAW.rglob("*.txt")):
        nivel, t, aviso = ler(p)
        rel = p.relative_to(RAW)
        carimbos[str(rel)] = t
        if len(t) == 0:
            print(f"\n== {rel}: nível {nivel}, sem medições ({' '.join(aviso)[:120]})")
            continue
        dias = pd.Series(t.normalize().unique())
        print(f"\n== {rel}: nível {nivel}, {len(t)} medições, {len(dias)} dias com medição, "
              f"{t.min():%Y-%m-%d} a {t.max():%Y-%m-%d}, repetidas {int(t.duplicated().sum())}")
        tab = dias.groupby([dias.dt.year, dias.dt.month]).size().unstack(fill_value=0)
        tab = tab.reindex(columns=range(1, 13), fill_value=0)
        tab.columns = [f"{m:02d}" for m in tab.columns]
        print("  dias com medição por mês:")
        for linha in tab.to_string().splitlines():
            print("    " + linha)
    print("\nNível 2.0 contido no 1.5 (mesmos carimbos):")
    for nome, t20 in carimbos.items():
        if "_aod20_" not in nome:
            continue
        t15 = carimbos.get(nome.replace("_aod20_", "_aod15_"), pd.DatetimeIndex([]))
        fora = len(t20.difference(t15))
        print(f"  {nome}: 2.0 {len(t20)}, 1.5 {len(t15)}, do 2.0 fora do 1.5 {fora}, "
              f"do 1.5 fora do 2.0 {len(t15.difference(t20))}")


if __name__ == "__main__":
    main()
