"""CAMS forecast ingestion (Copernicus ADS): forecast PM2.5 over the Porto Velho region.

"Raw CAMS forecast" baseline (METHODS.md §5):
2024 dry season only, 12 UTC run of the day before issuance.

Lead times: issuance at local hour h (0–23) of day D is at 04+h UTC; the 12 UTC run
of D−1 is 16+h hours earlier. For target t+k (k = 1..48), the
lead time is 16 + h + k, from 17 to 87 h. Hours 0 to 96 are downloaded, with margin.

Runs: from 30/06/2024 (issuances of 01/07) to 31/10/2024.

Same discipline as era5.py (ECMWF policy): one request at a
time, an identical existing request is adopted, a refusal STOPS the run without resubmission.
The key is in ~/.cdsapirc and is never printed.

EAC4 reanalysis (SO3, METHODS.md §8), added on 15/11/2026: surface PM2.5 every 3
h, from 2020 to 2024, over the same area, in data/raw/cams_eac4/. Same queue discipline.
For 2024, only the PM2.5 coverage is checked until the freeze (METHODS.md §8).

Usage:
    python src/ingest/cams.py custo                  # estimates the block cost, requests nothing
    python src/ingest/cams.py historico              # downloads the missing blocks, one at a time
    python src/ingest/cams.py previsao-custo         # whole years 2020–2023, 2025 and 2026 (to Oct)
    python src/ingest/cams.py previsao-historico
    python src/ingest/cams.py eac4-custo             # the same for the EAC4 reanalysis
    python src/ingest/cams.py eac4-historico
"""
import sys
import zipfile
from pathlib import Path

import xarray as xr

from cds_fila import cliente
from era5 import AREA, _normal

RAW = Path(__file__).resolve().parents[2] / "data" / "raw" / "cams"
CONJUNTO = "cams-global-atmospheric-composition-forecasts"
ANTECEDENCIAS = [str(h) for h in range(0, 97)]

# Two two-month blocks; the first starts on 30/06 (run used for the issuances
# of 01/07). Cost estimated on 12/11/2026: ~3,000 per month, limit
# of 10,000 per request; 2-month blocks (~6,100) face the queue 2 times, not 4.
BLOCOS = [
    ("202407-08", "2024-06-30", "2024-08-31"),
    ("202409-10", "2024-09-01", "2024-10-31"),
]


# Whole years (decided by the author on 16/11/2026): 2020–2023, 2025 and 2026 up to October, in
# 2-month blocks of runs, like those of 2024. The first block of 2020 and that of 2025 start
# on 31/12 of the previous year (run used for the issuances of 01/01). The 2026 forecast PM2.5 goes
# to data/raw/cams/selado_2026/ (blind test), with no value printed.
ANOS_PREVISAO = (2020, 2021, 2022, 2023, 2025, 2026)
ATE_2026 = 10  # last month of 2026
TESTE_CEGO = 2026


def blocos_previsao():
    from calendar import monthrange
    out = []
    for ano in ANOS_PREVISAO:
        ultimo = ATE_2026 if ano == TESTE_CEGO else 12
        for m in range(1, ultimo + 1, 2):
            m2 = min(m + 1, ultimo)
            ini = f"{ano - 1}-12-31" if m == 1 and ano - 1 not in ANOS_PREVISAO else f"{ano}-{m:02d}-01"
            fim = f"{ano}-{m2:02d}-{monthrange(ano, m2)[1]:02d}"
            pasta = RAW / "selado_2026" if ano == TESTE_CEGO else RAW
            out.append((pasta / f"cams_pm25_{ano}{m:02d}-{m2:02d}.nc", ini, fim))
    return out


RAW_EAC4 = RAW.parent / "cams_eac4"
CONJUNTO_EAC4 = "cams-global-reanalysis-eac4"
HORAS_EAC4 = [f"{h:02d}:00" for h in range(0, 24, 3)]  # ADS catalog
# Cost on 15/11/2026: ~2,920 per year, limit of 100,000 per request; the 5 years
# cost 14,616 and would fit in a single request. There are two requests so that the 2024 PM2.5 stays
# in a separate file, only opened after the freeze (METHODS.md §8).
BLOCOS_EAC4 = [
    ("2020-2023", "2020-01-01", "2023-12-31"),
    ("2024", "2024-01-01", "2024-12-31"),
]


def pedido_eac4(ini, fim):
    return {
        "variable": ["particulate_matter_2.5um"],
        "date": [f"{ini}/{fim}"],
        "time": HORAS_EAC4,
        "area": AREA,
        "data_format": "netcdf_zip",
    }


def montar_pedido(ini, fim):
    return {
        "variable": ["particulate_matter_2.5um"],
        "date": [f"{ini}/{fim}"],
        "time": ["12:00"],
        "leadtime_hour": ANTECEDENCIAS,
        "type": ["forecast"],
        "area": AREA,
        "data_format": "netcdf_zip",
    }


def remoto_existente(c, req, conjunto=CONJUNTO):
    """Identical request already completed, running or queued (in that priority order)."""
    alvo = _normal(req)
    jobs = c.get_jobs(limit=100, status=["successful", "running", "accepted"])
    lista = jobs.json.get("jobs", []) if hasattr(jobs, "json") else []
    achados = {}
    for j in lista:
        if j.get("processID") != conjunto or j.get("status") in achados:
            continue
        r = c.get_remote(j["jobID"])
        if _normal(r.request) == alvo:
            achados[j.get("status")] = r
    for status in ("successful", "running", "accepted"):
        if status in achados:
            return achados[status]
    return None


def custo():
    """Cost estimate of each block on the ADS. Creates no request."""
    c = cliente("ads")
    for tag, ini, fim in BLOCOS:
        est = c.estimate_costs(CONJUNTO, montar_pedido(ini, fim))
        print(f"cams_pm25_{tag}: {est}", flush=True)


def pedir(tag, ini, fim):
    return baixar(RAW / f"cams_pm25_{tag}.nc", CONJUNTO, montar_pedido(ini, fim))


def baixar(alvo, conjunto, req):
    """Makes (or adopts) the ADS request and writes a single .nc to alvo."""
    bruto = alvo.with_suffix(".download")
    parcial = alvo.with_suffix(".download.part")
    if not bruto.exists():
        c = cliente("ads")
        r = remoto_existente(c, req, conjunto)
        if r is not None:
            print(f"{alvo.name}: adotado o pedido {r.request_id} ({r.status})", flush=True)
        else:
            r = c.submit(conjunto, req)
            print(f"{alvo.name}: pedido {r.request_id} criado", flush=True)
        r.download(str(parcial))  # wait for the ADS to finish and download
        parcial.rename(bruto)
    if zipfile.is_zipfile(bruto):
        pasta = alvo.with_suffix("")
        with zipfile.ZipFile(bruto) as z:
            z.extractall(pasta)
        ncs = sorted(pasta.glob("*.nc"))
        ds = xr.merge([xr.open_dataset(p) for p in ncs], compat="override")
        ds.load().to_netcdf(alvo)
        for p in ncs:
            p.unlink()
        pasta.rmdir()
        bruto.unlink()
    else:
        bruto.rename(alvo)
    return alvo


def _lote(tarefas, conjunto, montar):
    """tarefas: [(alvo, ini, fim)] still missing. One request at a time; a refusal STOPS without resubmission."""
    print(f"{len(tarefas)} pedidos, um de cada vez", flush=True)
    for n, (alvo, ini, fim) in enumerate(tarefas, start=1):
        print(f"[{n}/{len(tarefas)}] {alvo.stem} ({ini} a {fim}): início", flush=True)
        try:
            baixar(alvo, conjunto, montar(ini, fim))
        except Exception as e:
            motivo = str(e).replace("\n", " | ")[:400]
            print(f"[{n}/{len(tarefas)}] {alvo.stem} PARADA: {motivo}", flush=True)
            print("Execução interrompida para revisão manual (sem reenvio automático).", flush=True)
            sys.exit(2)
        print(f"[{n}/{len(tarefas)}] {alvo.stem} ok", flush=True)


def historico():
    RAW.mkdir(parents=True, exist_ok=True)
    tarefas = [(RAW / f"cams_pm25_{t}.nc", i, f) for t, i, f in BLOCOS]
    _lote([x for x in tarefas if not x[0].exists()], CONJUNTO, montar_pedido)


def previsao_custo():
    """Cost estimate of the whole-year blocks. Creates no request."""
    c = cliente("ads")
    for alvo, ini, fim in blocos_previsao():
        est = c.estimate_costs(CONJUNTO, montar_pedido(ini, fim))
        print(f"{alvo.stem} ({ini} a {fim}): {est}", flush=True)


def previsao_historico():
    tarefas = blocos_previsao()
    for alvo, _, _ in tarefas:
        alvo.parent.mkdir(parents=True, exist_ok=True)
    _lote([x for x in tarefas if not x[0].exists()], CONJUNTO, montar_pedido)


def eac4_custo():
    """Cost estimate of each EAC4 block. Creates no request."""
    c = cliente("ads")
    for tag, ini, fim in BLOCOS_EAC4:
        est = c.estimate_costs(CONJUNTO_EAC4, pedido_eac4(ini, fim))
        print(f"cams_eac4_pm25_{tag}: {est}", flush=True)


def eac4_historico():
    RAW_EAC4.mkdir(parents=True, exist_ok=True)
    tarefas = [(RAW_EAC4 / f"cams_eac4_pm25_{t}.nc", i, f) for t, i, f in BLOCOS_EAC4]
    _lote([x for x in tarefas if not x[0].exists()], CONJUNTO_EAC4, pedido_eac4)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "custo"
    {"custo": custo, "historico": historico,
     "previsao-custo": previsao_custo, "previsao-historico": previsao_historico,
     "eac4-custo": eac4_custo, "eac4-historico": eac4_historico}[cmd]()
