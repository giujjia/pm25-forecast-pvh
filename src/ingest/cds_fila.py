"""Lists (and, if asked, cancels) the account's requests on the Copernicus CDS/ADS.

Used to diagnose the queue: the CDS refuses new requests when there are too many
in the queue per dataset, and requests from interrupted runs are still there.
The key comes from ~/.cdsapirc and is never printed.

Usage:
    python src/ingest/cds_fila.py listar [cds|ads]
    python src/ingest/cds_fila.py resumo cds            # all requests, with what each asked for
    python src/ingest/cds_fila.py detalhar cds <request_id> [...]
    python src/ingest/cds_fila.py cancelar cds <request_id> [<request_id> ...]
"""
import json
import re
import sys
from pathlib import Path

from ecmwf.datastores import Client

URLS = {"cds": "https://cds.climate.copernicus.eu/api",
        "ads": "https://ads.atmosphere.copernicus.eu/api"}


def cliente(qual):
    cfg = (Path.home() / ".cdsapirc").read_text()
    chave = re.search(r"key:\s*(\S+)", cfg).group(1)
    return Client(url=URLS[qual], key=chave)


def listar(qual):
    c = cliente(qual)
    jobs = c.get_jobs(limit=100, sortby="-created", status=["accepted", "running"])
    dados = jobs.json if hasattr(jobs, "json") else jobs
    lista = dados.get("jobs", []) if isinstance(dados, dict) else dados
    print(f"{len(lista)} pedidos ativos (accepted/running) em {qual}")
    for j in lista:
        print(json.dumps({k: j.get(k) for k in ("jobID", "processID", "status", "created", "started", "updated")},
                         ensure_ascii=False))


def detalhar(qual, ids):
    """Status, dataset and error message of specific requests (including refused ones)."""
    c = cliente(qual)
    for rid in ids:
        try:
            r = c.get_remote(rid)
            info = r.json if hasattr(r, "json") else {}
            print(json.dumps({"jobID": rid, "processID": info.get("processID"), "status": info.get("status"),
                              "created": info.get("created"), "updated": info.get("updated")}, ensure_ascii=False))
        # get_results() waits for the request to finish: only called once it has
        # finished, to read the refusal reason (waiting on a queued request hangs).
            if info.get("status") in ("failed", "rejected", "dismissed"):
                try:
                    r.get_results()
                except Exception as e:  # the refusal reason comes in the exception
                    print("   motivo:", str(e)[:400].replace("\n", " | "))
        except Exception as e:
            print(json.dumps({"jobID": rid, "erro": str(e)[:300]}, ensure_ascii=False))


def resumo(qual):
    """All requests of the account: status, dates and, for the useful ones, what was requested.

    For 'successful', 'accepted' and 'running' requests, shows dataset, year and months
    (and level, if any), to cross-check with the files already downloaded.
    """
    c = cliente(qual)
    jobs = c.get_jobs(limit=500, sortby="-created")
    lista = jobs.json.get("jobs", []) if hasattr(jobs, "json") else []
    cont = {}
    for j in lista:
        cont[j.get("status")] = cont.get(j.get("status"), 0) + 1
    print(f"{len(lista)} pedidos na conta: {cont}")
    for j in lista:
        if j.get("status") not in ("successful", "accepted", "running"):
            continue
        r = c.get_remote(j["jobID"])
        req = r.request or {}
        nivel = ",".join(req.get("pressure_level", [])) or "sup"
        print(json.dumps({"jobID": j["jobID"], "status": j.get("status"),
                          "created": j.get("created"), "finished": j.get("finished"),
                          "conjunto": j.get("processID"), "nivel": nivel,
                          "ano": req.get("year"), "meses": req.get("month"),
                          "dias": len(req.get("day", [])), "area": req.get("area")},
                         ensure_ascii=False))


def cancelar(qual, ids):
    c = cliente(qual)
    print(json.dumps(c.delete(*ids), ensure_ascii=False, default=str)[:2000])


if __name__ == "__main__":
    cmd, qual, *resto = sys.argv[1:] + ["cds"][:max(0, 2 - len(sys.argv[1:]))]
    if cmd == "listar":
        listar(qual)
    elif cmd == "resumo":
        resumo(qual)
    elif cmd == "detalhar":
        detalhar(qual, resto)
    elif cmd == "cancelar":
        cancelar(qual, resto)
