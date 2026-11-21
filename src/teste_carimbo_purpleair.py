"""Does the time_stamp of the PurpleAir API hourly mean mark the start or the end of the hour?

Requests only humidity (not PM2.5) of sensor 39077 over 48 h of 2023, with average=0
(raw, ~2 min) and average=60, and compares the API hourly series with two means
of the raw data: [h, h+1h) labelled h (start) and (h-1h, h] labelled h (end).
Writes to data/raw/purpleair/testes/, separate from the project data.
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ingest.purpleair import RAW, get  # noqa: E402

SENSOR, INI, FIM, CAMPO = 39077, "2023-08-15", "2023-08-17", "humidity"
t0 = int(datetime.fromisoformat(INI).replace(tzinfo=timezone.utc).timestamp())
t1 = int(datetime.fromisoformat(FIM).replace(tzinfo=timezone.utc).timestamp())
OUT = RAW / "testes"
OUT.mkdir(parents=True, exist_ok=True)

series = {}
for avg in (0, 60):
    js = get(f"/sensors/{SENSOR}/history", fields=CAMPO, average=avg,
             start_timestamp=t0, end_timestamp=t1)
    df = pd.DataFrame(js["data"], columns=js["fields"]).sort_values("time_stamp")
    df["time_stamp"] = pd.to_datetime(df["time_stamp"], unit="s", utc=True)
    arq = OUT / f"{SENSOR}_{CAMPO}_{INI}_{FIM}_average{avg}.csv"
    df.to_csv(arq, index=False)
    print(f"average={avg}: {len(df)} linhas, {df.time_stamp.min()} → {df.time_stamp.max()} → {arq.name}")
    series[avg] = df.set_index("time_stamp")[CAMPO].astype(float)

raw, api60 = series[0], series[60]
print(f"intervalo mediano do bruto: {raw.index.to_series().diff().median()}")
print(f"minutos dos carimbos horários: {sorted(set(api60.index.minute))}")
inicio = raw.resample("1h", label="left", closed="left").mean()  # [h, h+1h) → h
fim = raw.resample("1h", label="right", closed="right").mean()  # (h-1h, h] → h
cmp = pd.DataFrame({"api60": api60, "media_inicio": inicio, "media_fim": fim}).dropna()
cmp["erro_inicio"] = (cmp.api60 - cmp.media_inicio).abs()
cmp["erro_fim"] = (cmp.api60 - cmp.media_fim).abs()
cmp.round(3).to_csv(OUT / f"{SENSOR}_{CAMPO}_{INI}_{FIM}_comparacao.csv")
print(f"\n{len(cmp)} horas comparadas")
print(f"erro absoluto médio contra a média [h, h+1h) (início): {cmp.erro_inicio.mean():.3f}")
print(f"erro absoluto médio contra a média (h-1h, h] (fim):    {cmp.erro_fim.mean():.3f}")
print(f"horas em que 'início' ajusta melhor: {(cmp.erro_inicio < cmp.erro_fim).sum()}; "
      f"'fim' melhor: {(cmp.erro_fim < cmp.erro_inicio).sum()}")
print("\nprimeiras 12 horas:")
print(cmp.head(12).round(2).to_string())
