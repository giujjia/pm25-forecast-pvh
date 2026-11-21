"""Test of the dry MERRA-2 PM2.5 against the version with water at 35% RH (gap 12).

The FRM filter is weighed after conditioning at 30-40% RH (40 CFR 50, App. L; audit F33),
so the FRM-equivalent mass keeps the water held by hygroscopic particles at ~35% RH.
GMAO Office Note 22 (Collow et al., 2023), §3.1, gives the MERRA-2 formula with growth
factors, and §3.4 defines them as

    f_rh = 1 + ((r_wet / r_dry)^3 - 1) * rho_water / rho_dry,

with r from the optics tables used by MERRA-2 (Appendix B: OC v1_3, BC v1_3, SU v1_3,
SS v3_3; downloaded in E126) and the dry densities of §2 (SU 1700, OC/BC 1800, SS 2200 kg/m3).

This script:
1. computes f_rh at 30, 35 and 40% RH for each species from the tables;
2. applies them to MERRA-2 2020-2023 (2024 stays closed until the freeze) at the Porto
   Velho cell and reports the ratio PM2.5(35%) / PM2.5(dry) and the composition, in the
   whole year and in the dry season (Jul-Oct).

Two ambiguities are reported as a range, not resolved:
- The surface collection gives only total OCSMASS and BCSMASS, without the hydrophobic /
  hydrophilic split. Applying the hydrophilic factor to all of it (as §3.1 writes) is an
  upper bound; the lower bound applies no growth to carbon.
- The SU table has two radius bins; both are reported.

No measured PM2.5 is read. Usage: python src/teste_merra2_umidade.py
"""
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

RAIZ = Path(__file__).resolve().parents[1]
OPT = RAIZ / "data/raw/merra2_optics"
RHO = {"SU": 1700.0, "OC": 1800.0, "BC": 1800.0, "SS": 2200.0}
LAT, LON = -9.0, -63.75  # MERRA-2 cell nearest the city centre (-8.79, -63.85)
SO4_NH4 = 1.3756


def f_rh(especie, versao, rh, bin_):
    d = xr.open_dataset(OPT / f"optics_{especie}.{versao}.nc")
    r = d["rEff"].isel(radius=bin_)
    razao = float(r.sel(rh=rh, method="nearest") / r.sel(rh=0.0))
    return 1.0 + (razao ** 3 - 1.0) * 1000.0 / RHO[especie]


def fatores(rh):
    return {
        "SU_bin0": f_rh("SU", "v1_3", rh, 0), "SU_bin1": f_rh("SU", "v1_3", rh, 1),
        "OC_hidrofilico": f_rh("OC", "v1_3", rh, 1), "BC_hidrofilico": f_rh("BC", "v1_3", rh, 1),
        "SS_bin0": f_rh("SS", "v3_3", rh, 0), "SS_bin1": f_rh("SS", "v3_3", rh, 1),
        "SS_bin2": f_rh("SS", "v3_3", rh, 2),
    }


def merra2():
    arqs = sorted(f for a in range(2020, 2024) for f in glob.glob(str(RAIZ / f"data/raw/merra2/{a}/*.nc4")))
    cols = ["DUSMASS25", "OCSMASS", "BCSMASS", "SSSMASS25", "SO4SMASS"]
    partes = []
    for f in arqs:
        with xr.open_dataset(f) as ds:
            partes.append(ds[cols].sel(lat=LAT, lon=LON).to_dataframe()[cols])
    return pd.concat(partes).sort_index() * 1e9  # kg/m3 -> ug/m3


def main():
    print("Fatores de crescimento f_rh (Office Note 22, §3.4, com as tabelas do MERRA-2):")
    tab = pd.DataFrame({f"UR {int(rh*100)}%": fatores(rh) for rh in (0.30, 0.35, 0.40)})
    print(tab.round(3).to_string())
    f = fatores(0.35)
    ss = np.mean([f["SS_bin0"], f["SS_bin1"], f["SS_bin2"]])

    df = merra2()
    print(f"\nMERRA-2: {len(df)} horas de 2020-2023 na célula {LAT}, {LON}")
    seco = df["DUSMASS25"] + df["OCSMASS"] + df["BCSMASS"] + df["SSSMASS25"] + SO4_NH4 * df["SO4SMASS"]
    cenarios = {}
    for su_nome in ("SU_bin0", "SU_bin1"):
        sup = (df["DUSMASS25"] + f["OC_hidrofilico"] * df["OCSMASS"] + f["BC_hidrofilico"] * df["BCSMASS"]
               + ss * df["SSSMASS25"] + f[su_nome] * SO4_NH4 * df["SO4SMASS"])
        inf = (df["DUSMASS25"] + df["OCSMASS"] + df["BCSMASS"]
               + ss * df["SSSMASS25"] + f[su_nome] * SO4_NH4 * df["SO4SMASS"])
        cenarios[f"teto ({su_nome}, carbono todo hidrofílico)"] = sup / seco
        cenarios[f"piso ({su_nome}, carbono sem crescer)"] = inf / seco

    seca = df.index.month.isin([7, 8, 9, 10])
    print("\nRazão PM2.5(35%) / PM2.5(seco), mediana e percentis 10 e 90 das horas:")
    for nome, r in cenarios.items():
        for rotulo, m in (("ano todo", slice(None)), ("seca jul-out", seca)):
            x = r[m]
            print(f"  {nome:48s} {rotulo:12s} mediana {x.median():.3f}  p10 {x.quantile(.1):.3f}  p90 {x.quantile(.9):.3f}")

    comp = df.copy()
    comp["SO4SMASS"] *= SO4_NH4
    frac = comp.div(comp.sum(axis=1), axis=0)
    print("\nComposição do PM2.5 seco (fração média das horas):")
    print(pd.DataFrame({"ano todo": frac.mean(), "seca jul-out": frac[seca].mean()}).round(3).to_string())

    saida = RAIZ / "data/testes/teste_merra2_umidade.csv"
    saida.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cenarios).to_csv(saida)
    print(f"\nRazões hora a hora: {saida.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
