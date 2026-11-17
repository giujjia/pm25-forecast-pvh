"""Checks whether the PurpleAir hourly download (2020–2024) is complete.

Counts and dates only; no PM2.5 value is read or printed (2024 is the test year).

For each sensor:
  1. blocks: does each 14-day block of the grid, within the sensor's lifetime, exist in horario_partes/?
  2. hours per year in the hourly data × days with data per year in the daily means (<sensor>_diario.parquet):
     if the hourly data has far fewer hours than days×24, hourly data is missing;
  3. days that exist in the daily data but have no hour in the hourly data.

Usage:
    python src/verificar_purpleair.py
"""
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "purpleair"
SENSORES = [39077, 56831, 219595, 219981, 225327, 128603]
INI, FIM = datetime(2020, 1, 1, tzinfo=timezone.utc), datetime(2025, 1, 1, tzinfo=timezone.utc)
PASSO = 14 * 86400


def main():
    inv = pd.read_csv(RAW / "inventario.csv").set_index("sensor_index")
    problemas = 0
    for s in SENSORES:
        h = pd.read_parquet(RAW / f"{s}_horario.parquet", columns=["time_stamp"])
        d = pd.read_parquet(RAW / f"{s}_diario.parquet", columns=["time_stamp"])
        d["time_stamp"] = pd.to_datetime(d["time_stamp"], utc=True)
        d = d[(d.time_stamp >= INI) & (d.time_stamp < FIM)]

    # 1. grid blocks within the sensor's lifetime
        vida0 = pd.Timestamp(inv.at[s, "date_created"]).timestamp()
        vida1 = pd.Timestamp(inv.at[s, "last_seen"]).timestamp()
        esperados = [t0 for t0 in range(int(INI.timestamp()), int(FIM.timestamp()), PASSO)
                     if min(t0 + PASSO, FIM.timestamp()) > vida0 and t0 < vida1]
        pasta = RAW / "horario_partes" / str(s)
        faltando = [t0 for t0 in esperados
                    if not (pasta / f"{datetime.fromtimestamp(t0, timezone.utc):%Y%m%d}.parquet").exists()]

    # 2. hours per year × days of the daily data
        ha = h.groupby(h.time_stamp.dt.year).size()
        da = d.groupby(d.time_stamp.dt.year).size()
        tab = pd.DataFrame({"horas_horario": ha, "dias_diario": da}).fillna(0).astype(int)
        tab["horas_esperadas_max"] = tab["dias_diario"] * 24
        tab["razao"] = (tab["horas_horario"] / tab["horas_esperadas_max"]).round(3)

    # 3. daily-data days with no hour in the hourly data
        dias_h = set(h.time_stamp.dt.floor("D"))
        sem_hora = sorted(t for t in d.time_stamp.dt.floor("D") if t not in dias_h)

        print(f"\n=== {s}: {len(esperados)} blocos esperados, {len(faltando)} faltando; "
              f"{len(h)} horas no arquivo horário")
        print(tab.to_string())
        print(f"dias com média diária mas sem nenhuma hora: {len(sem_hora)}"
              + (f" (primeiros: {[str(x.date()) for x in sem_hora[:5]]})" if sem_hora else ""))
        problemas += len(faltando) + len(sem_hora)
    print(f"\nRESUMO: {problemas} problemas (blocos faltando + dias sem hora)")


if __name__ == "__main__":
    main()
