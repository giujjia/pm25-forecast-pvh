"""Test of the Sentinel-5P zone centre: city centre vs each sensor (decision of 18/11/2026).

The Sentinel-5P series (E110) are means over 41 zones: 5 rings (0-50, 50-150, 150-300,
300-600, 600-1000 km) x 8 sectors of 45 deg, centred on the city centre (-8.7936, -63.8458),
plus the ERA5 box. The FIRMS rings are computed from each sensor. This script measures,
for each sensor, which fraction of the area of each sensor-centred zone falls inside the
same city-centred zone (same ring and same sector), and the same for the ring alone.

Method: Monte Carlo, 20 000 points uniform in area inside each sensor-centred zone, on the
sphere (great-circle distance and initial bearing). No data is read besides the sensor
coordinates in the PurpleAir inventory.

Usage: python src/teste_centro_s5p.py
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
CENTRO = (-8.7936, -63.8458)
ANEIS = [(0, 50), (50, 150), (150, 300), (300, 600), (600, 1000)]
R = 6371.0
N = 20000
rng = np.random.default_rng(0)


def destino(lat, lon, az, km):
    d, a = km / R, np.radians(az)
    p1, l1 = math.radians(lat), math.radians(lon)
    p2 = np.arcsin(np.sin(p1) * np.cos(d) + np.cos(p1) * np.sin(d) * np.cos(a))
    l2 = l1 + np.arctan2(np.sin(a) * np.sin(d) * np.cos(p1), np.cos(d) - np.sin(p1) * np.sin(p2))
    return np.degrees(p2), np.degrees(l2)


def dist_az(lat0, lon0, lat, lon):
    p1, p2 = math.radians(lat0), np.radians(lat)
    dl = np.radians(lon) - math.radians(lon0)
    c = np.sin(p1) * np.sin(p2) + np.cos(p1) * np.cos(p2) * np.cos(dl)
    d = R * np.arccos(np.clip(c, -1, 1))
    az = np.degrees(np.arctan2(np.sin(dl) * np.cos(p2), np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)))
    return d, az % 360


def zona_de(d, az):
    anel = np.full(d.shape, -1)
    for i, (r0, r1) in enumerate(ANEIS):
        anel[(d >= r0) & (d < r1)] = i
    return anel, (az // 45).astype(int)


def main():
    inv = pd.read_csv(RAIZ / "data/raw/purpleair/inventario.csv")
    linhas = []
    for _, s in inv.iterrows():
        dc, azc = dist_az(CENTRO[0], CENTRO[1], np.array([s.latitude]), np.array([s.longitude]))
        for i, (r0, r1) in enumerate(ANEIS):
            for j in range(8):
                rr = np.sqrt(rng.uniform(r0 ** 2, r1 ** 2, N))  # uniform in area
                aa = rng.uniform(j * 45, (j + 1) * 45, N)
                lat, lon = destino(s.latitude, s.longitude, aa, rr)
                d, az = dist_az(CENTRO[0], CENTRO[1], lat, lon)
                anel, setor = zona_de(d, az)
                linhas.append({"sensor": s.sensor_index, "dist_centro_km": float(dc[0]), "anel": f"{r0}-{r1}",
                               "setor": j, "mesma_zona": np.mean((anel == i) & (setor == j)),
                               "mesmo_anel": np.mean(anel == i)})
    res = pd.DataFrame(linhas)
    resumo = res.groupby(["sensor", "dist_centro_km", "anel"], sort=False).agg(
        mesma_zona_media=("mesma_zona", "mean"), mesma_zona_min=("mesma_zona", "min"),
        mesmo_anel=("mesmo_anel", "mean")).reset_index()
    pd.set_option("display.width", 200, "display.max_rows", 100)
    print("Fração da área da zona centrada no sensor que cai na mesma zona centrada na cidade")
    print("(média e mínimo nos 8 setores; e fração no mesmo anel)\n")
    print(resumo.round(3).to_string(index=False))
    saida = RAIZ / "data/testes/teste_centro_s5p.csv"
    saida.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(saida, index=False)
    print(f"\nTabela por setor: {saida.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
