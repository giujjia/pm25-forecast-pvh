"""ERA5 ingestion (Copernicus CDS): hourly meteorology over the Porto Velho region.

ERA5 Single Levels: PBLH, 10 m wind, 2 m temperature and dew point,
total precipitation. ERA5 Pressure Levels: 850 hPa wind.
The key is in ~/.cdsapirc.

Usage:
    python src/ingest/era5.py teste 2024-08-15         # one day, prints a summary
    python src/ingest/era5.py historico 2020 2024          # multi-month blocks, both datasets
    python src/ingest/era5.py historico 2020 2024 pl       # 850 hPa only
    python src/ingest/era5.py custo 2020 2026 t925 --ate 2026-10   # estimates, requests nothing
    python src/ingest/era5.py historico 2020 2026 t925 --ate 2026-10  # 925 hPa temperature
    python src/ingest/era5.py historico 2025 2026 --ate 2026-10   # current year: up to the last closed month

Without --ate, the run stops if any block includes the current month or a future one.
"""
import sys
import zipfile
from datetime import date
from pathlib import Path

import json

import xarray as xr

from cds_fila import cliente as cliente_cds  # same key as ~/.cdsapirc

RAW = Path(__file__).resolve().parents[2] / "data" / "raw" / "era5"

# north, west, south, east: the region of METHODS.md §4 (contains all sensors)
AREA = [-8.0, -65.0, -10.5, -62.5]
HORAS = [f"{h:02d}:00" for h in range(24)]

CONJUNTOS = {
    "sl": ("reanalysis-era5-single-levels", {
        "variable": ["boundary_layer_height", "10m_u_component_of_wind",
                     "10m_v_component_of_wind", "2m_temperature",
                     "2m_dewpoint_temperature", "total_precipitation"],
    }),
    "pl": ("reanalysis-era5-pressure-levels", {
        "variable": ["u_component_of_wind", "v_component_of_wind"],
        "pressure_level": ["850"],
    }),
# 925 hPa temperature (author's decision, 16/11/2026): stability below the boundary
# layer top, for the stability analysis (METHODS.md §5).
    "t925": ("reanalysis-era5-pressure-levels", {
        "variable": ["temperature"],
        "pressure_level": ["925"],
    }),
}


def _normal(req):
    """Canonical form of a request, to compare with those in the queue."""
    return json.dumps({k: (sorted(v) if isinstance(v, list) and k != "area" else v)
                       for k, v in req.items()}, sort_keys=True)


def remoto_existente(c, nome, req):
    """Identical request already completed, running or queued, if any (in that order).

    Avoids creating duplicate requests when a run is interrupted and resumed:
    the old request is adopted instead of becoming an orphan, and a
    result that is already ready is downloaded without a new request.
    """
    alvo = _normal(req)
    jobs = c.get_jobs(limit=100, status=["successful", "running", "accepted"])
    lista = jobs.json.get("jobs", []) if hasattr(jobs, "json") else []
    achados = {}
    for j in lista:
        if j.get("processID") != nome or j.get("status") in achados:
            continue
        r = c.get_remote(j["jobID"])
        if _normal(r.request) == alvo:
            achados[j.get("status")] = r
    for status in ("successful", "running", "accepted"):  # a ready result takes priority
        if status in achados:
            return achados[status]
    return None


def montar_pedido(chave, ano, meses, dias):
    nome, extra = CONJUNTOS[chave]
    req = {
        "product_type": ["reanalysis"],
        "year": [str(ano)], "month": [f"{m:02d}" for m in meses],
        "day": [f"{d:02d}" for d in dias],
        "time": HORAS, "area": AREA,
        "data_format": "netcdf", "download_format": "unarchived",
        **extra,
    }
    return nome, req


def pedir(chave, ano, meses, dias, alvo, so_prontos=False):
    """Makes (or adopts) the request and returns a single .nc (the CDS may send a zip).

    With so_prontos=True, only downloads if an identical result is already ready; sends no
    new request and returns None if there is none.
    """
    nome, req = montar_pedido(chave, ano, meses, dias)
    bruto = alvo.with_suffix(".download")  # only exists if the download finished
    parcial = alvo.with_suffix(".download.part")  # transfer in progress
    if not bruto.exists():  # reuses a complete download from a previous run
        c = cliente_cds("cds")
        r = remoto_existente(c, nome, req)
        if so_prontos and (r is None or r.status != "successful"):
            return None
        if r is not None:
            print(f"{alvo.name}: adotado o pedido {r.request_id} ({r.status})", flush=True)
        else:
            r = c.submit(nome, req)
            print(f"{alvo.name}: pedido {r.request_id} criado", flush=True)
        r.download(str(parcial))  # wait for the CDS to finish and download
        parcial.rename(bruto)
    if zipfile.is_zipfile(bruto):
        pasta = alvo.with_suffix("")
        with zipfile.ZipFile(bruto) as z:
            z.extractall(pasta)
        ds = xr.merge([xr.open_dataset(p) for p in sorted(pasta.glob("*.nc"))],
                      compat="override")
        ds.to_netcdf(alvo)
        for p in pasta.glob("*.nc"):
            p.unlink()
        pasta.rmdir()
        bruto.unlink()
    else:
        bruto.rename(alvo)
    return alvo


def teste(dia):
    d = date.fromisoformat(dia)
    RAW.mkdir(parents=True, exist_ok=True)
    for chave in CONJUNTOS:
        alvo = pedir(chave, d.year, [d.month], [d.day], RAW / f"teste_{chave}_{d}.nc")
        ds = xr.open_dataset(alvo)
        print(f"\n== {chave}: {alvo.name}")
        print(ds)


# Months per request: the largest block under the CDS size limit
# (estimate_costs on 09/11/2026: surface 26,784 per month, limit 121,000;
# 850 hPa 8,928 per month, limit 60,000).
MESES_POR_PEDIDO = {"sl": 4, "pl": 6, "t925": 12}  # t925: 1 variable, half the cost of pl


def ja_baixados(chave):
    """Months (year, month) already present in era5_<key>_YYYYMM[-MM].nc files."""
    feitos = set()
    for p in RAW.glob(f"era5_{chave}_*.nc"):
        tag = p.stem.split("_")[-1]  # YYYYMM or YYYYMM-MM
        ano, m1 = int(tag[:4]), int(tag[4:6])
        m2 = int(tag.split("-")[1]) if "-" in tag else m1
        feitos.update((ano, m) for m in range(m1, m2 + 1))
    return feitos


def blocos(chave, ano_ini, ano_fim, feitos, ate=None):
    """Blocks (year, months) still missing for a dataset, without touching disk or network.

    feitos: months (year, month) already downloaded. ate = (year, month): last month requested;
    later months are left out. Used for the current year, when ERA5 only
    exists up to ~5 days ago: without the cutoff, the Sep–Dec block would ask for months that
    do not exist yet. The cut months come later in their own blocks.
    """
    n = MESES_POR_PEDIDO[chave]
    saida = []
    for ano in range(ano_ini, ano_fim + 1):
        for m0 in range(1, 13, n):
            meses = [m for m in range(m0, m0 + n)
                     if (ano, m) not in feitos and (ate is None or (ano, m) <= ate)]
            if meses:
                saida.append((ano, meses))
    return saida


def tarefas_pendentes(ano_ini, ano_fim, conjuntos, ate=None):
    tarefas = []
    for chave in conjuntos:  # in the order requested (e.g. "pl,sl" does 850 hPa first)
        for ano, meses in blocos(chave, ano_ini, ano_fim, ja_baixados(chave), ate):
            tag = f"{ano}{meses[0]:02d}-{meses[-1]:02d}"
            tarefas.append((chave, ano, meses, RAW / f"era5_{chave}_{tag}.nc"))
    return tarefas


def checar_meses(tarefas, hoje=None):
    """Stops the run if any block includes the current month or a future month (missing --ate)."""
    hoje = hoje or date.today()
    ruins = sorted({(ano, m) for _, ano, meses, _ in tarefas for m in meses
                    if (ano, m) >= (hoje.year, hoje.month)})
    if ruins:
        print(f"PARADA: os blocos incluem meses ainda incompletos ou futuros {ruins}; "
              f"use --ate AAAA-MM com o último mês fechado.", flush=True)
        sys.exit(2)


def prontos(ano_ini, ano_fim, conjuntos=("pl", "sl"), ate=None):
    """Downloads only the results already ready on the CDS for missing blocks. Sends no request."""
    RAW.mkdir(parents=True, exist_ok=True)
    tarefas = tarefas_pendentes(ano_ini, ano_fim, conjuntos, ate)
    checar_meses(tarefas)
    print(f"{len(tarefas)} blocos faltando; procurando resultados prontos", flush=True)
    for chave, ano, meses, alvo in tarefas:
        if pedir(chave, ano, meses, range(1, 32), alvo, so_prontos=True):
            print(f"{alvo.name}: baixado de resultado pronto", flush=True)


def historico(ano_ini, ano_fim, conjuntos=("sl", "pl"), ate=None):
    """Multi-month blocks per request, sent one at a time.

    Months already downloaded (including in old monthly files) are not requested
    again, and an identical request already in the queue is adopted (see pedir()).
    """
    RAW.mkdir(parents=True, exist_ok=True)
    tarefas = tarefas_pendentes(ano_ini, ano_fim, conjuntos, ate)
    checar_meses(tarefas)
    print(f"{len(tarefas)} pedidos, um de cada vez", flush=True)

    # Policy of ECMWF (notice of 22/10/2026): "Submit requests
    # sequentially, rather than in parallel or in large batches"; accounts that
    # send requests in parallel or automatically resubmit after a queue refusal
    # were deactivated. Hence: one request at a time, and a queue refusal STOPS
    # the run (manual review), with no automatic resubmission.
    for n, (chave, ano, meses, alvo) in enumerate(tarefas, start=1):
        if alvo.exists():  # downloaded by another run while this one was running
            print(f"[{n}/{len(tarefas)}] {alvo.name}: já existe, pulado", flush=True)
            continue
        print(f"[{n}/{len(tarefas)}] {alvo.name}: início", flush=True)
        try:
            pedir(chave, ano, meses, range(1, 32), alvo)
        except Exception as e:
            motivo = str(e).replace("\n", " | ")[:400]
            print(f"[{n}/{len(tarefas)}] {alvo.name} PARADA: {motivo}", flush=True)
            print("Execução interrompida para revisão manual (sem reenvio automático).", flush=True)
            sys.exit(2)
        print(f"[{n}/{len(tarefas)}] {alvo.name} ok", flush=True)

def separar_ate(args):
    """Strips the --ate YYYY-MM option from the arguments and returns (rest, (year, month) or None)."""
    if "--ate" not in args:
        return args, None
    i = args.index("--ate")
    ano, mes = (int(x) for x in args[i + 1].split("-"))
    return args[:i] + args[i + 2:], (ano, mes)


if __name__ == "__main__":
    cmd, *args = sys.argv[1:] or ["teste"]
    args, ate = separar_ate(args)
    if cmd == "teste":
        teste(args[0] if args else "2024-08-15")
    elif cmd == "prontos":  # prontos 2020 2024 [pl,sl] [--ate YYYY-MM]: only downloads what is already ready
        conj = tuple(args[2].split(",")) if len(args) > 2 else ("pl", "sl")
        prontos(int(args[0]), int(args[1]), conj, ate)
    elif cmd == "custo":  # custo 2020 2026 t925 [--ate YYYY-MM]: estimates the blocks, requests nothing
        c = cliente_cds("cds")
        for chave, ano, meses, alvo in tarefas_pendentes(int(args[0]), int(args[1]), tuple(args[2].split(",")), ate):
            nome, req = montar_pedido(chave, ano, meses, range(1, 32))
            print(f"{alvo.name}: {c.estimate_costs(nome, req)}", flush=True)
    elif cmd == "historico":
        conj = tuple(args[2].split(",")) if len(args) > 2 else ("sl", "pl")
        historico(int(args[0]), int(args[1]), conj, ate)
