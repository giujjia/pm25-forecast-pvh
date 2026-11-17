"""Ingestion of Sentinel-5P TROPOMI L3 (OFFL) through Google Earth Engine: daily series per zone.

SO4 (METHODS.md §4, versions M4 and M5): CO, tropospheric NO₂ and absorbing aerosol index, per
ring and sector, "last available value, in the region and in the upwind sectors, with a missing
indicator". No image is downloaded: Earth Engine reduces each day over the zones and returns
only numbers.

Zones: the 5 FIRMS rings (0-50, 50-150, 150-300, 300-600 and 600-1000 km) × 8 sectors of 45°
(sector 0 = azimuth 0-45° from north, clockwise), around the center of Porto
Velho (-8.7936; -63.8458, the same center as aeronet.py), plus the ERA5 box. Whole rings
come from the sum of the sectors. Arcs every 5° (shape error < 1.1 km).

Daily composite: mean, pixel by pixel, of the UTC day's passes (14 passes cross the area in a
day); L3 OFFL already carries ESA's quality filter. Reductions at 5,000 m, sum and count
of valid pixels, which give the zone mean and the fraction of valid pixels (zone counts in
zonas_s5p.csv).

Output (data/raw/s5p/): s5p_{co|no2|aer}_{YYYYMM}.csv (date, zone, n_passes, sum, n_valid) and
zonas_s5p.csv. 2026 goes to data/raw/s5p/selado_2026/ (the smoke of the blind-test year). Months already
written are skipped. No value is printed.

Usage:
    python src/ingest/s5p.py zonas
    python src/ingest/s5p.py baixar 2020-01 2026-10 [workers]
"""
import csv
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import ee

RAW = Path(__file__).resolve().parents[2] / "data" / "raw" / "s5p"
CENTRO = (-8.7936, -63.8458)
ANEIS = [(0, 50), (50, 150), (150, 300), (300, 600), (600, 1000)]
SETORES = 8
CAIXA = [-65.0, -10.5, -62.5, -8.0]  # W, S, E, N (same as ERA5)
ESCALA = 5000
TESTE_CEGO = 2026
COLECOES = {
    "co": ("COPERNICUS/S5P/OFFL/L3_CO", "CO_column_number_density"),
    "no2": ("COPERNICUS/S5P/OFFL/L3_NO2", "tropospheric_NO2_column_number_density"),
    "aer": ("COPERNICUS/S5P/OFFL/L3_AER_AI", "absorbing_aerosol_index"),
}
R_TERRA = 6371.0


def destino(lat, lon, az, km):
    """Point km from (lat, lon) at azimuth az degrees (sphere)."""
    d, a = km / R_TERRA, math.radians(az)
    p1, l1 = math.radians(lat), math.radians(lon)
    p2 = math.asin(math.sin(p1) * math.cos(d) + math.cos(p1) * math.sin(d) * math.cos(a))
    l2 = l1 + math.atan2(math.sin(a) * math.sin(d) * math.cos(p1), math.cos(d) - math.sin(p1) * math.sin(p2))
    return [math.degrees(l2), math.degrees(p2)]


def zonas():
    """[(id, ring_start, ring_end, sector, polygon coordinates in [lon, lat])]."""
    out = []
    for i, (r0, r1) in enumerate(ANEIS):
        for j in range(SETORES):
            a0, a1 = j * 45, (j + 1) * 45
            passos = [a0 + 5 * k for k in range(10)]  # 0 to 45 in 9 intervals of 5°
            borda = [destino(*CENTRO, a, r1) for a in passos]
            interno = [destino(*CENTRO, a, r0) for a in reversed(passos)] if r0 > 0 else [destino(*CENTRO, 0, 0)]
            out.append((f"a{i}s{j}", r0, r1, j, borda + interno + [borda[0]]))
    out.append(("caixa", None, None, None, [[CAIXA[0], CAIXA[1]], [CAIXA[2], CAIXA[1]], [CAIXA[2], CAIXA[3]],
                                            [CAIXA[0], CAIXA[3]], [CAIXA[0], CAIXA[1]]]))
    return out


def colecao_fc():
    return ee.FeatureCollection([ee.Feature(ee.Geometry.Polygon([c], None, False), {"zona": z})
                                 for z, _, _, _, c in zonas()])


def com_retentativa(f, tentativas=5):
    for k in range(tentativas):
        try:
            return f()
        except Exception as e:  # EEException, network drop, memory limits
            if k == tentativas - 1:
                raise
            time.sleep(15 * (k + 1))
            ultimo = e  # noqa: F841


def dia_fc(col, banda, fc):
    def um(d):
        d = ee.Date(d)
        c = col.filterDate(d, d.advance(1, "day")).select(banda)
        n = c.size()
        img = ee.Image(ee.Algorithms.If(n.gt(0), c.mean(), ee.Image.constant(0).selfMask().rename(banda)))
        red = img.reduceRegions(collection=fc, reducer=ee.Reducer.sum().combine(ee.Reducer.count(), sharedInputs=True),
                                scale=ESCALA, tileScale=4)
        return red.map(lambda f: f.set({"data": d.format("YYYY-MM-dd"), "n": n}))
    return um


def linhas(col, banda, fc, dias):
    lista = ee.List([d.isoformat() for d in dias])
    um = dia_fc(col, banda, fc)
    fcs = ee.FeatureCollection(lista.map(lambda s: um(ee.Date(s)))).flatten()
    feats = com_retentativa(lambda: fcs.select(["data", "zona", "n", "sum", "count"], retainGeometry=False)
                            .getInfo()["features"])
    return [(p["data"], p["zona"], p.get("n", 0), p.get("sum"), p.get("count", 0))
            for p in (f["properties"] for f in feats)]


def mes(chave, ano, m):
    nome, banda = COLECOES[chave]
    pasta = RAW / "selado_2026" if ano == TESTE_CEGO else RAW
    arq = pasta / f"s5p_{chave}_{ano}{m:02d}.csv"
    if arq.exists():
        return arq, 0, False
    d0 = date(ano, m, 1)
    n = (date(ano + (m == 12), m % 12 + 1, 1) - d0).days
    dias = [d0 + timedelta(days=i) for i in range(n)]
    col, fc = ee.ImageCollection(nome), colecao_fc()
    try:
        rows = linhas(col, banda, fc, dias)
    except Exception:  # whole month exceeded the limit: try day by day
        rows = [r for d in dias for r in linhas(col, banda, fc, [d])]
    esperado = n * len(zonas())
    if len(rows) != esperado:
        raise RuntimeError(f"{chave} {ano}-{m:02d}: {len(rows)} linhas, esperava {esperado}")
    pasta.mkdir(parents=True, exist_ok=True)
    parcial = arq.with_suffix(".csv.part")
    with open(parcial, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["data", "zona", "n_passagens", "soma", "n_validos"])
        w.writerows(sorted(rows))
    parcial.rename(arq)
    return arq, len(rows), True


def totais():
    """Number of pixels in each zone on the 5 km grid of each collection, and the zone definitions."""
    fc = colecao_fc()
    tot = {}
    for chave, (nome, banda) in COLECOES.items():
        img = ee.ImageCollection(nome).filterDate("2020-08-15", "2020-08-16").select(banda).first().unmask(0)
        red = img.reduceRegions(collection=fc, reducer=ee.Reducer.count(), scale=ESCALA, tileScale=4)
        tot[chave] = {f["properties"]["zona"]: f["properties"]["count"]
                      for f in com_retentativa(lambda: red.getInfo()["features"])}
    RAW.mkdir(parents=True, exist_ok=True)
    with open(RAW / "zonas_s5p.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["zona", "anel_km_ini", "anel_km_fim", "setor_45graus", "centro_lat", "centro_lon",
                    "pixels_5km_co", "pixels_5km_no2", "pixels_5km_aer"])
        for z, r0, r1, s, _ in zonas():
            w.writerow([z, r0, r1, s, CENTRO[0], CENTRO[1]] + [tot[c][z] for c in COLECOES])
    print(f"zonas_s5p.csv gravado ({len(zonas())} zonas)", flush=True)


def iniciar():
    projeto = json.loads((Path.home() / ".config/earthengine/credentials").read_text()).get("project")
    ee.Initialize(project=projeto)


def baixar(ini, fim, trabalhadores):
    iniciar()
    a, b = (tuple(int(x) for x in s.split("-")) for s in (ini, fim))
    meses = [(y, m) for y in range(a[0], b[0] + 1) for m in range(1, 13) if a <= (y, m) <= b]
    tarefas = [(c, y, m) for (y, m) in meses for c in COLECOES]
    print(f"{len(tarefas)} coleção×mês, {trabalhadores} em paralelo", flush=True)
    t0, feitos, erros = time.perf_counter(), 0, 0
    with ThreadPoolExecutor(trabalhadores) as ex:
        futuros = {ex.submit(mes, c, y, m): (c, y, m) for c, y, m in tarefas}
        for fu in as_completed(futuros):
            c, y, m = futuros[fu]
            try:
                _, n, novo = fu.result()
                feitos += 1
            except Exception as e:
                erros += 1
                print(f"{c} {y}-{m:02d}: ERRO {str(e)[:300]}", flush=True)
                continue
            if novo and feitos % 6 == 0 or feitos == len(tarefas):
                print(f"{c} {y}-{m:02d}: ok | {feitos}/{len(tarefas)} em {(time.perf_counter() - t0) / 60:.0f} min",
                      flush=True)
    print(f"fim: {feitos} gravados, {erros} com erro (rodar de novo retoma só os que faltam)", flush=True)
    if erros:
        sys.exit(1)


if __name__ == "__main__":
    if sys.argv[1] == "zonas":
        iniciar()
        totais()
    else:
        baixar(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 4)
