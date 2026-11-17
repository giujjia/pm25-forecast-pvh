"""Gap 14: NOAA-20 continuity and a fixed spatial mask of static sources.

1. Coverage (2020-2024; day counts only, no FRP): days on which NOAA-20 has no detection in the
   1000 km box while S-NPP has at least 50 (probable NOAA-20 outage), per year.
2. Mask (2020-2023 only, the training years). 400 m grid, like the FIRMS "Static Thermal Anomalies"
   layer ("summarized on a 400m grid"), which is not distributed. Two criteria:
   - M_tipo: cells with any detection of type = 2 (other static land source) in the SP v2 product
     (the product's own attribution, with auxiliary data). Since v2 labels some type 2
     detections as 0, the spatial mask also recovers those mislabelled detections;
   - M_persist: cells with detections in at least 6 distinct months, at least 2 of them in the
     rainy season (Dec-Apr), when vegetation fires are rare in Rondônia: persistence, as in GFAS
     (fixed mask built from several years; Kaiser 2012).
   For each one: number of cells, detections inside by type, and the fraction of NOAA-20 FRP the
   mask would remove, by ring (0-300 and 300-1000 km from Porto Velho), season (dry Jul–Oct, rainy
   Dec-Apr) and year. (The first version covered 0-300 km in the dry season only; the second adds the
   rest.) Lists the cells less than 100 km from the city.
Does not read PurpleAir or PM2.5.
"""
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
RAW = RAIZ / "data" / "raw" / "firms"
OUT = RAIZ / "data" / "processed" / "firms_mascara"
OUT.mkdir(parents=True, exist_ok=True)
PVH = (-8.76, -63.90)
CEL = 0.0036  # ~400 m
CHUVA = [12, 1, 2, 3, 4]
SECA = [7, 8, 9, 10]


def dist_km(lat, lon):
    la1, lo1, la2, lo2 = map(np.radians, (PVH[0], PVH[1], lat, lon))
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


print("## 1. Dias sem nenhum foco do NOAA-20 na caixa de 1000 km com >= 50 focos do S-NPP")
for ano in range(2020, 2025):
    d = pd.read_parquet(RAW / f"firms1000_{ano}.parquet", columns=["acq_date", "satellite"])
    c = d.groupby(["acq_date", "satellite"]).size().unstack(fill_value=0)
    for s in ("N", "N20"):
        if s not in c:
            c[s] = 0
    falha = c[(c.N20 == 0) & (c.N >= 50)]
    dias = pd.to_datetime(falha.index)
    print(f"{ano}: dias com dado {len(c)} | dias sem NOAA-20 com S-NPP >= 50: {len(falha)} "
          f"(na seca: {int(dias.month.isin(SECA).sum())}) {sorted(dias.strftime('%m-%d'))[:20]}")

print("\n## 2. Máscara de fontes estáticas (2020–2023)")
fr = pd.concat([pd.read_parquet(RAW / f"firms1000_{a}.parquet",
                                columns=["latitude", "longitude", "acq_date", "satellite", "type", "frp"])
                .assign(ano=a) for a in range(2020, 2024)])
fr["mes"] = pd.to_datetime(fr.acq_date).dt.month
fr["ym"] = fr.ano * 100 + fr.mes
fr["cel"] = list(zip(np.floor(fr.latitude / CEL).astype(int), np.floor(fr.longitude / CEL).astype(int)))
fr["dist"] = dist_km(fr.latitude.values, fr.longitude.values)

m_tipo = set(fr.loc[fr["type"] == 2, "cel"])
n20 = fr[fr.satellite == "N20"]
g = n20.groupby("cel").agg(meses=("ym", "nunique"),
                           meses_chuva=("mes", lambda m: m.isin(CHUVA).sum() and m[m.isin(CHUVA)].nunique()))
# meses_chuva counts distinct months of the year; persistence needs distinct (year-month) months in the rainy season:
chuva_ym = n20[n20.mes.isin(CHUVA)].groupby("cel").ym.nunique()
g["ym_chuva"] = chuva_ym.reindex(g.index).fillna(0).astype(int)
m_persist = set(g[(g.meses >= 6) & (g.ym_chuva >= 2)].index)
mascaras = {"M_tipo": m_tipo, "M_persist": m_persist, "M_uniao": m_tipo | m_persist}
print(f"células de 400 m com foco: {fr.cel.nunique()} | M_tipo: {len(m_tipo)} | M_persist: {len(m_persist)} | "
      f"interseção: {len(m_tipo & m_persist)} | união: {len(m_tipo | m_persist)}")

linhas = []
for nome, M in mascaras.items():
    dentro = fr.cel.isin(M)
    tipos = fr.loc[dentro, "type"].value_counts().to_dict()
    print(f"\n[{nome}] focos dentro (todos os satélites, 2020–2023), por type: {tipos}")
    for anel, (d0, d1) in {"0-300km": (0, 300), "300-1000km": (300, 1000)}.items():
        for epoca, meses in {"seca": SECA, "chuva": CHUVA}.items():
            alvo = n20[(n20.dist > d0) & (n20.dist <= d1) & n20.mes.isin(meses)] if d0 else \
                n20[(n20.dist <= d1) & n20.mes.isin(meses)]
            for ano, a in alvo.groupby("ano"):
                frac = a.loc[a.cel.isin(M), "frp"].sum() / a.frp.sum()
                linhas.append({"mascara": nome, "anel": anel, "epoca": epoca, "ano": ano, "focos_n20": len(a),
                               "focos_tirados": int(a.cel.isin(M).sum()), "pct_frp_tirado": round(100 * frac, 2)})
r = pd.DataFrame(linhas)
r.to_csv(OUT / "efeito_mascaras.csv", index=False)
print("\nEfeito no FRP do NOAA-20 por anel, época e ano:")
print(r.to_string(index=False))

print("\n## 3. Células mascaradas a menos de 100 km de Porto Velho")
perto = fr[fr.cel.isin(mascaras["M_uniao"]) & (fr.dist <= 100)]
t = perto.groupby("cel").agg(lat=("latitude", "mean"), lon=("longitude", "mean"), dist_km=("dist", "mean"),
                             focos=("frp", "size"), meses=("ym", "nunique"),
                             type2=("type", lambda x: int((x == 2).sum())))
t["M_tipo"] = [c in m_tipo for c in t.index]
t["M_persist"] = [c in m_persist for c in t.index]
t = t.sort_values("dist_km").round(3)
t.to_csv(OUT / "celulas_ate_100km.csv")
print(t.to_string())
pd.DataFrame(sorted(mascaras["M_uniao"]), columns=["i_lat", "i_lon"]).to_csv(OUT / "mascara_uniao_celulas.csv", index=False)
print(f"\nSaídas em {OUT}")
