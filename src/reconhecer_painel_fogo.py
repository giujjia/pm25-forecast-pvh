"""Survey of the Painel do Fogo (Censipam): what the API and the WFS provide, and since when.

Public routes, no account (Painel do Fogo FAQ, question 27; API Swagger):
  - REST API (FastAPI): https://panorama.sipam.gov.br/painel-do-fogo/api/v1 ; the specification
    is at .../api/openapi.json. The event routes only return active or under-observation events;
    the detection routes accept a date.
  - GeoServer WFS: https://panorama.sipam.gov.br/geoserver/painel_do_fogo/wfs . The layers
    tb_evento, tb_evento_extinto, tb_deteccao and mv_indicadores_queimadas_archive hold the
    historical series.

Coverage only: record counts per year, never values (area, FRP etc.). The box is the ERA5 area.
The BBOX filter carries the explicit CRS ('EPSG:4674'); without it GeoServer reads the box in
lat/lon order and returns zero.

Usage:
    python src/reconhecer_painel_fogo.py
"""
import re
from datetime import date

import requests

API = "https://panorama.sipam.gov.br/painel-do-fogo/api"
WFS = "https://panorama.sipam.gov.br/geoserver/painel_do_fogo/wfs"
CAIXA = "BBOX({g},-65,-10.5,-62.5,-8,'EPSG:4674')"  # W, S, E, N (same as ERA5)
ANOS = range(2020, 2027)
CAMADAS = {  # layer -> (date field, geometry field)
    "tb_evento": ("dt_minima", "geom"),
    "tb_evento_extinto": ("dt_minima", "geom"),
    "tb_deteccao": ("dt_aquisicao", "geom"),
    "mv_indicadores_queimadas_archive": ("dt_passagem", "geom_acumulada"),
}

S = requests.Session()
S.headers["User-Agent"] = "Mozilla/5.0 (X11; Linux x86_64)"


def wfs(**p):
    base = {"service": "WFS", "version": "2.0.0", "request": "GetFeature"}
    return S.get(WFS, params={**base, **p}, timeout=300)


def hits(camada, cql=None):
    p = {"typeNames": f"painel_do_fogo:{camada}", "resultType": "hits"}
    if cql:
        p["CQL_FILTER"] = cql
    m = re.findall(r'numberMatched="([^"]+)"', wfs(**p).text)
    return int(m[0]) if m else None


def no_ano(campo, a):
    return f"{campo} >= {a}-01-01T00:00:00Z AND {campo} < {a + 1}-01-01T00:00:00Z"


def api():
    print("== API")
    print("ping:", S.get(f"{API}/v1/diagnosticos/ping", timeout=60).text.strip())
    spec = S.get(f"{API}/openapi.json", timeout=60).json()
    print(f"{spec['info']['title']} (versão {spec['info']['version']}); "
          f"esquemas de segurança: {list(spec.get('components', {}).get('securitySchemes', {})) or 'nenhum'}")
    for rota, ops in spec["paths"].items():
        print(f"  GET {rota.removeprefix('/painel-do-fogo/api')}: {ops['get'].get('summary')}")
    print("Focos por data (responde 200 com dado, 404 sem):")
    for rota, par, v in (("focos/firms", "horas", 24), ("focos/censipam", "horas", 24), ("focos/goes", "minutos", 60)):
        def tem(d):
            return S.get(f"{API}/v1/{rota}", params={"data": d.isoformat(), par: v}, timeout=180).status_code == 200
        lin = {d: tem(d) for d in (date(2020, 8, 15), date(2023, 8, 15), date(2025, 8, 15), date(2026, 8, 15))}
        txt = ", ".join(f"{d}: {'sim' if t else 'não'}" for d, t in lin.items())
        a, b = max((d for d, t in lin.items() if not t), default=None), min((d for d, t in lin.items() if t), default=None)
        if a and b and a < b:  # binary search for the first day with data
            while (b - a).days > 1:
                m = date.fromordinal(a.toordinal() + (b - a).days // 2)
                a, b = (a, m) if tem(m) else (m, b)
            txt += f"; primeiro dia com dado: {b}"
        print(f"  {rota}: {txt}")


def wfs_camadas():
    print("\n== WFS")
    caps = S.get(WFS, params={"service": "WFS", "version": "2.0.0", "request": "GetCapabilities"}, timeout=120).text
    print("taxas:", re.findall(r"<ows:Fees>([^<]*)", caps), "restrições:", re.findall(r"<ows:AccessConstraints>([^<]*)", caps))
    print("camadas:", ", ".join(n.split(":")[1] for n in re.findall(r"<Name>(painel_do_fogo:[^<]+)</Name>", caps)))
    for camada, (campo, geo) in CAMADAS.items():
        d = S.get(WFS, params={"service": "WFS", "version": "2.0.0", "request": "DescribeFeatureType",
                               "typeNames": f"painel_do_fogo:{camada}"}, timeout=120).text
        campos = [n for n, _ in re.findall(r'name="([^"]+)"[^>]*type="([^"]+)"', d) if n != camada]
        extremos = []
        for ordem in ("A", "D"):
            r = wfs(typeNames=f"painel_do_fogo:{camada}", count=1, sortBy=f"{campo} {ordem}",
                    propertyName=campo, outputFormat="application/json")
            extremos.append(r.json()["features"][0]["properties"][campo][:10])
        print(f"\n{camada}: {hits(camada)} registros em toda a área do painel; {campo} de {extremos[0]} a {extremos[1]}")
        print(f"  atributos: {', '.join(campos)}")
        cx = CAIXA.format(g=geo)
        print("  por ano, na caixa de Porto Velho: "
              + " ".join(f"{a}: {hits(camada, cx + ' AND ' + no_ano(campo, a))}" for a in ANOS), flush=True)
    cx = CAIXA.format(g="geom")
    print("\ntb_evento na caixa, por id_status_evento: "
          + " ".join(f"{s}: {hits('tb_evento', f'{cx} AND id_status_evento = {s}')}" for s in range(0, 6)))
    r = wfs(typeNames="painel_do_fogo:tb_deteccao", count=5000, propertyName="dt_aquisicao",
            CQL_FILTER=f"{cx} AND {no_ano('dt_aquisicao', 2020)}", outputFormat="application/json")
    print(f"página de teste (count=5000, tb_deteccao, caixa, 2020): {len(r.json()['features'])} registros, "
          f"{len(r.content) / 1e6:.2f} MB")


if __name__ == "__main__":
    api()
    wfs_camadas()
