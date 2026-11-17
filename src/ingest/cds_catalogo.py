"""Describes a Copernicus CDS/ADS dataset (accepted variables and parameters). Read-only.

Usage:
    python src/ingest/cds_catalogo.py cds reanalysis-era5-single-levels-timeseries
"""
import json
import sys

from cds_fila import cliente

qual, conjunto = sys.argv[1], sys.argv[2]
c = cliente(qual)
proc = c.get_process(conjunto)
info = proc.json if hasattr(proc, "json") else proc
print("título:", info.get("title"))
for nome, spec in (info.get("inputs") or {}).items():
    schema = spec.get("schema", {})
    opcoes = schema.get("items", {}).get("enum") or schema.get("enum")
    print(f"- {nome}: {json.dumps(opcoes, ensure_ascii=False) if opcoes else json.dumps(schema, ensure_ascii=False)[:300]}")
