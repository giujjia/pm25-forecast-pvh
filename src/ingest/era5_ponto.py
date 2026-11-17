"""ERA5 by point (CDS dataset reanalysis-era5-single-levels-timeseries).

Alternative to gridded ERA5 for the surface variables, when the CDS queue is
stalled. Writes to data/raw/era5_ponto/.

Usage:
    python src/ingest/era5_ponto.py teste            # 1 point, 1 month, 1 variable
"""
import sys
from pathlib import Path

import cdsapi

RAW = Path(__file__).resolve().parents[2] / "data" / "raw" / "era5_ponto"
CONJUNTO = "reanalysis-era5-single-levels-timeseries"


def teste():
    alvo = RAW / "teste" / "teste_39077_202008_blh.csv"
    alvo.parent.mkdir(parents=True, exist_ok=True)
    pedido = {"variable": ["boundary_layer_height"],
              "location": {"latitude": -8.716443, "longitude": -63.87924},  # sensor 39077
              "date": ["2020-08-01/2020-08-31"],
              "data_format": "csv"}
    print("pedido:", pedido, flush=True)
    cdsapi.Client().retrieve(CONJUNTO, pedido, str(alvo))
    print(f"ok → {alvo} ({alvo.stat().st_size} bytes)")
    print(alvo.read_bytes()[:600])


if __name__ == "__main__":
    {"teste": teste}[sys.argv[1]]()
