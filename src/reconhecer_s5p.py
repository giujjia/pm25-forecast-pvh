"""Survey of Sentinel-5P L3 on Google Earth Engine (download plan, section 5.6).

Uses the author's Earth Engine credential (~/.config/earthengine/credentials, with the Google
Cloud project saved in it) without reading or printing the token. Metadata only: period of each
collection, number of images on one day over the ERA5 area, and band names. No values.

Usage:
    python src/reconhecer_s5p.py
"""
import json
from pathlib import Path

import ee

COLECOES = ["COPERNICUS/S5P/OFFL/L3_CO", "COPERNICUS/S5P/OFFL/L3_NO2", "COPERNICUS/S5P/OFFL/L3_AER_AI"]
AREA = [-65.0, -10.5, -62.5, -8.0]  # W, S, E, N (same as ERA5)


def main():
    projeto = json.loads((Path.home() / ".config/earthengine/credentials").read_text()).get("project")
    ee.Initialize(project=projeto)
    print(f"Earth Engine iniciado com o projeto {projeto}", flush=True)
    area = ee.Geometry.Rectangle(AREA)
    for c in COLECOES:
        col = ee.ImageCollection(c)
        ini = ee.Date(col.aggregate_min("system:time_start")).format("YYYY-MM-dd").getInfo()
        fim = ee.Date(col.aggregate_max("system:time_start")).format("YYYY-MM-dd").getInfo()
        n = col.filterDate("2020-08-15", "2020-08-16").filterBounds(area).size().getInfo()
        bandas = col.first().bandNames().getInfo()
        print(f"{c}: de {ini} a {fim}; imagens em 15/08/2020 na área: {n}; bandas: {bandas}", flush=True)


if __name__ == "__main__":
    main()
