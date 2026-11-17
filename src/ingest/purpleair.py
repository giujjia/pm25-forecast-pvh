"""PurpleAir ingestion: sensor inventory and hourly history.

Usage:
    python src/ingest/purpleair.py inventario
    python src/ingest/purpleair.py cobertura [ids...]   # daily means, cheap
    python src/ingest/purpleair.py historico 39077 [other ids...]   # hourly 2020–2024, resumable
    python src/ingest/purpleair.py trecho 39077 temperature 2020-08-15 2020-08-16   # cheap test
"""
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()
API = "https://api.purpleair.com/v1"
# Which .env variable holds the key: PURPLEAIR_READ_KEY (default) or another, chosen by
# PURPLEAIR_CHAVE (e.g. PURPLEAIR_CHAVE=PURPLEAIR_READ_KEY2 for the second account).
# The key is never printed; only the variable name.
VAR_CHAVE = os.environ.get("PURPLEAIR_CHAVE", "PURPLEAIR_READ_KEY")
HEADERS = {"X-API-Key": os.environ[VAR_CHAVE]}
RAW = Path(__file__).resolve().parents[2] / "data" / "raw" / "purpleair"

# Porto Velho airshed (same BBOX as METHODS.md)
BBOX = dict(nwlat=-8.0, nwlng=-65.0, selat=-10.5, selng=-62.5)

META_FIELDS = "name,model,location_type,latitude,longitude,date_created,last_seen,channel_flags"
# 5 fields (decision of 11/11/2026, checked with a test run): temperature
# enters no correction nor the model, and can be requested alone later
# (confirmed, in °F). Saves ~92 thousand points, leaving a balance for 2025–2026.
HIST_FIELDS = "pm2.5_cf_1_a,pm2.5_cf_1_b,pm2.5_atm_a,pm2.5_atm_b,humidity"
# Maximum window per API call, in days, per averaging period (minutes)
CHUNK_DAYS = {60: 14, 1440: 365}


def get(path, tentativas=4, **params):
    """GET with retries on network error, 429 (rate limit) and 5xx."""
    for n in range(1, tentativas + 1):
        try:
            r = requests.get(f"{API}{path}", headers=HEADERS, params=params, timeout=60)
        except (requests.ConnectionError, requests.Timeout) as e:
            motivo = str(e)[:200]
        else:
            if r.status_code == 429 or r.status_code >= 500:
                motivo = f"HTTP {r.status_code}: {r.text[:200]}"
            else:
                r.raise_for_status()  # other 4xx: request error, retrying is pointless
                return r.json()
        if n == tentativas:
            raise RuntimeError(f"{path}: {motivo}")
        print(f"  erro ({motivo}); nova tentativa {n + 1} em {30 * n} s", flush=True)
        time.sleep(30 * n)


def historico_horario(sensor_id, inicio="2020-01-01", fim="2025-01-01", fields=HIST_FIELDS, saida=None):
    """Hourly history in 14-day blocks, each block written as soon as it arrives.

    Blocks go to data/raw/purpleair/horario_partes/<sensor>/<YYYYMMDD>.parquet,
    on a fixed grid starting at inicio; a new run skips those that already exist
    (the points of an already paid block are not spent again). Blocks with no data are
    also written (empty), so they are not requested again. At the end, joins everything into
    saida (default: <sensor>_horario.parquet), over the interval [inicio, fim).
    Prints only row and date counts, never PM2.5 values.
    """
    meta = get(f"/sensors/{sensor_id}", fields="date_created,last_seen")["sensor"]
    vida0, vida1 = int(meta["date_created"]), int(meta["last_seen"])
    t_ini = int(datetime.fromisoformat(inicio).replace(tzinfo=timezone.utc).timestamp())
    t_fim = int(datetime.fromisoformat(fim).replace(tzinfo=timezone.utc).timestamp())
    step = CHUNK_DAYS[60] * 86400
    # Parts of a sealed period (e.g. selado_2026/) stay inside the sealed folder.
    base = RAW / saida.split("/")[0] / "partes" if saida and saida.startswith("selado_") else RAW / "horario_partes"
    pasta = base / str(sensor_id)
    pasta.mkdir(parents=True, exist_ok=True)
    pedidos = pulados = 0
    for t0 in range(t_ini, t_fim, step):
        t1 = min(t0 + step, t_fim)
        if t1 <= vida0 or t0 >= vida1:
            continue  # outside the sensor's lifetime: nothing to request
        arq = pasta / f"{datetime.fromtimestamp(t0, timezone.utc):%Y%m%d}.parquet"
        if arq.exists():
            pulados += 1
            continue
        js = get(f"/sensors/{sensor_id}/history", fields=fields, average=60,
                 start_timestamp=t0, end_timestamp=t1)
        pd.DataFrame(js["data"], columns=js["fields"]).to_parquet(arq, index=False)
        pedidos += 1
        print(f"{sensor_id} {arq.stem}: {len(js['data'])} linhas", flush=True)
        time.sleep(1)  # respects the rate limit
    partes = [pd.read_parquet(p) for p in sorted(pasta.glob("*.parquet"))]
    df = pd.concat([p for p in partes if len(p)]).drop_duplicates("time_stamp").sort_values("time_stamp")
    df = df[(df["time_stamp"] >= t_ini) & (df["time_stamp"] < t_fim)]
    df["time_stamp"] = pd.to_datetime(df["time_stamp"], unit="s", utc=True)
    out = RAW / (saida or f"{sensor_id}_horario.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"{sensor_id}: {pedidos} blocos pedidos, {pulados} já existiam; {len(df)} horas, "
          f"{df.time_stamp.min()} → {df.time_stamp.max()} → {out}", flush=True)
    return df


def inventario():
    js = get("/sensors", fields=META_FIELDS, max_age=0, **BBOX)
    df = pd.DataFrame(js["data"], columns=js["fields"])
    for c in ("date_created", "last_seen"):
        df[c] = pd.to_datetime(df[c], unit="s", utc=True)
    df["location_type"] = df["location_type"].map({0: "externo", 1: "interno"})
    RAW.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW / "inventario.csv", index=False)
    print(df.sort_values("date_created").to_string(index=False))


def historico(sensor_id, average=60, fields=HIST_FIELDS):
    meta = get(f"/sensors/{sensor_id}", fields="date_created,last_seen")["sensor"]
    start, end = int(meta["date_created"]), int(meta["last_seen"])
    step = CHUNK_DAYS[average] * 86400
    partes = []
    for t0 in range(start, end, step):
        js = get(f"/sensors/{sensor_id}/history", fields=fields,
                 average=average, start_timestamp=t0, end_timestamp=min(t0 + step, end))
        if js["data"]:
            partes.append(pd.DataFrame(js["data"], columns=js["fields"]))
        time.sleep(1)  # respects the rate limit
    df = pd.concat(partes).drop_duplicates("time_stamp").sort_values("time_stamp")
    df["time_stamp"] = pd.to_datetime(df["time_stamp"], unit="s", utc=True)
    RAW.mkdir(parents=True, exist_ok=True)
    nome = "horario" if average == 60 else "diario"
    out = RAW / f"{sensor_id}_{nome}.parquet"
    df.to_parquet(out, index=False)
    print(f"{sensor_id}: {len(df)} linhas, {df.time_stamp.min()} → {df.time_stamp.max()} → {out}")
    return df


ANO_CEGO = 2026  # blind test (METHODS.md §6): no PM2.5 statistic for this year


def saldo():
    """Remaining points of the account in use (/organization endpoint, free)."""
    js = get("/organization")
    print(f"variável da chave: {VAR_CHAVE}; pontos restantes: {js.get('remaining_points')}")


def cobertura(ids):
    """Days with data per sensor and year, using daily means (cheap in points)."""
    linhas = []
    for sid in ids:
        df = historico(sid, average=1440, fields="pm2.5_cf_1_a,pm2.5_cf_1_b,"
                       "pm2.5_atm_a,pm2.5_atm_b,humidity")
        df["ano"] = df.time_stamp.dt.year
        seca = df.time_stamp.dt.month.between(7, 10)
        linhas.append(pd.DataFrame({
            "sensor": sid,
            "dias": df.groupby("ano").size(),
            "dias_seca_jul_out": df[seca].groupby("ano").size(),
    # 2026 is the blind test: day counts only, no PM2.5 value.
            "pm25_cf1_max": df[df.ano < ANO_CEGO].groupby("ano")["pm2.5_cf_1_a"].max().round(0),
        }).fillna(0))
    tab = pd.concat(linhas).reset_index(names="ano")
    tab.to_csv(RAW / "cobertura.csv", index=False)
    print(tab.pivot(index="sensor", columns="ano", values="dias_seca_jul_out").fillna(0).astype(int))


def trecho(sensor_id, campos, inicio, fim, average=60):
    """Requests only the given fields over a short interval (cheap test in points).

    Writes to data/raw/purpleair/testes/, separate from the project data.
    """
    t0 = int(datetime.fromisoformat(inicio).replace(tzinfo=timezone.utc).timestamp())
    t1 = int(datetime.fromisoformat(fim).replace(tzinfo=timezone.utc).timestamp())
    js = get(f"/sensors/{sensor_id}/history", fields=campos, average=average,
             start_timestamp=t0, end_timestamp=t1)
    df = pd.DataFrame(js["data"], columns=js["fields"]).sort_values("time_stamp")
    df["time_stamp"] = pd.to_datetime(df["time_stamp"], unit="s", utc=True)
    out = RAW / "testes" / f"{sensor_id}_{campos.replace(',', '+')}_{inicio}_{fim}_avg{average}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"campos devolvidos pela API: {js['fields']}")
    print(f"{len(df)} linhas → {out}")
    print(df.to_string(index=False))
    return out


if __name__ == "__main__":
    cmd, *args = sys.argv[1:] or ["inventario"]
    if cmd == "inventario":
        inventario()
    elif cmd == "trecho":  # trecho <sensor> <fields> <start YYYY-MM-DD> <end YYYY-MM-DD> [average]
        trecho(int(args[0]), args[1], args[2], args[3], int(args[4]) if len(args) > 4 else 60)
    elif cmd == "cobertura":
        ids = args or pd.read_csv(RAW / "inventario.csv")["sensor_index"].tolist()
        cobertura(ids)
    elif cmd == "historico":  # hourly 2020–2024, in resumable blocks
        for sid in args:
            historico_horario(int(sid))
    elif cmd == "saldo":
        saldo()
    elif cmd == "periodo":  # periodo <start YYYY-MM-DD> <end YYYY-MM-DD> <sensor> [...]
        # 2025 goes to <sensor>_horario_2025.parquet. 2026 (blind test) goes to the
        # selado_2026/ folder, which no analysis opens before the final model is frozen.
        ini, fim, *sids = args
        for sid in sids:
            nome = (f"selado_2026/{sid}_horario_{ini}_{fim}.parquet" if ini[:4] == str(ANO_CEGO)
                    else f"{sid}_horario_{ini}_{fim}.parquet")
            historico_horario(int(sid), inicio=ini, fim=fim, saida=nome)
