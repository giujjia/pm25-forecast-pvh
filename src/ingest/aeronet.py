"""AERONET ingestion: aerosol optical depth (AOD) version 3, sites near Porto Velho.

Source: the public AERONET (NASA) web service, no account. Usage policy: the data
are public domain; contact the site's principal investigator (PI), cite the AERONET key
papers and acknowledge the PI. The PI's name is in the header of the downloaded files.

Commands:
  inventario: reads the yearly site lists (Site_Lists_V3, one per year and level) and shows the
      sites within RAIO_KM of Porto Velho, with the months that have data in each year. Coverage only.
  baixar: AOD of all points (AVG=10) of the given sites, at levels 1.5 and 2.0, from 2020 to
      the end date. 2020–2025 go to data/raw/aeronet/; 2026 goes to
      data/raw/aeronet/selado_2026/ (2026 AOD shows the smoke of the blind-test year).
      Files unaltered, as the service delivers them.

Nothing here prints AOD values.

Usage:
    python src/ingest/aeronet.py inventario 2020 2026
    python src/ingest/aeronet.py baixar 2026-11-15 Porto_Velho_UNIR Ji_Parana_SE Rio_Branco
"""
import hashlib
import math
import sys
import time
from pathlib import Path

import requests

BASE = "https://aeronet.gsfc.nasa.gov"
LISTA = BASE + "/Site_Lists_V3/aeronet_locations_v3_{ano}_lev{nivel}.txt"
SERVICO = BASE + "/cgi-bin/print_web_data_v3"
PVH = (-8.7936, -63.8458)  # INMET station A925, city center
RAIO_KM = 500
NIVEIS = ("15", "20")
TESTE_CEGO = 2026
RAW = Path(__file__).resolve().parents[2] / "data" / "raw" / "aeronet"
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"}
MESES = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]


def km(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin((p2 - p1) / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return 2 * 6371 * math.asin(math.sqrt(a))


def inventario(ano_ini, ano_fim):
    s = requests.Session()
    s.headers.update(UA)
    achados = {}
    for ano in range(ano_ini, ano_fim + 1):
        for nivel in NIVEIS:
            r = s.get(LISTA.format(ano=ano, nivel=nivel), timeout=120)
            r.raise_for_status()
            linhas = r.text.strip().splitlines()
            cab = linhas[1].split(",")
            for ln in linhas[2:]:
                c = ln.split(",")
                nome, lon, lat = c[0], float(c[1]), float(c[2])
                d = km(PVH[0], PVH[1], lat, lon)
                if d <= RAIO_KM:
                    meses = [m for m, v in zip(cab[4:], c[4:]) if v.strip() == "1"]
                    achados.setdefault((nome, round(d)), {})[(ano, nivel)] = meses
            print(f"{ano} nível {nivel[0]}.{nivel[1]}: lista com {len(linhas) - 2} sítios", flush=True)
    print(f"\nSítios a até {RAIO_KM} km de Porto Velho com dado em {ano_ini}–{ano_fim}:")
    for (nome, d), anos in sorted(achados.items(), key=lambda x: x[0][1]):
        print(f"\n{nome} ({d} km)")
        for (ano, nivel), meses in sorted(anos.items()):
            print(f"  {ano} nível {nivel[0]}.{nivel[1]}: {len(meses)} meses "
                  f"({' '.join(m.title() for m in meses)})")
    if not achados:
        print("  nenhum")


def baixar(data_fim, sitios):
    a2, m2, d2 = (int(x) for x in data_fim.split("-"))
    s = requests.Session()
    s.headers.update(UA)
    for pasta, (ini, fim) in {RAW: ((2020, 1, 1), (2025, 12, 31)),
                              RAW / "selado_2026": ((TESTE_CEGO, 1, 1), (a2, m2, d2))}.items():
        pasta.mkdir(parents=True, exist_ok=True)
        for sitio in sitios:
            for nivel in NIVEIS:
                alvo = pasta / f"{sitio}_aod{nivel}_todos_{ini[0]}{ini[1]:02d}{ini[2]:02d}-{fim[0]}{fim[1]:02d}{fim[2]:02d}.txt"
                if alvo.exists():
                    print(f"{alvo.name}: já existe, pulado", flush=True)
                    continue
                params = {"site": sitio, "year": ini[0], "month": ini[1], "day": ini[2],
                          "year2": fim[0], "month2": fim[1], "day2": fim[2],
                          f"AOD{nivel}": 1, "AVG": 10, "if_no_html": 1}
                r = s.get(SERVICO, params=params, timeout=600)
                r.raise_for_status()
                corpo = r.content
                alvo.write_bytes(corpo)
                linhas = corpo.decode("latin-1").splitlines()
                n_dados = max(0, len(linhas) - 7)  # 6-line header + column names
                pi = next((ln for ln in linhas[:8] if "PI=" in ln or "Contact" in ln), "")
                print(f"{alvo.relative_to(RAW.parent)}: {len(corpo)} bytes, ~{n_dados} linhas de dado, "
                      f"SHA-256 {hashlib.sha256(corpo).hexdigest()}"
                      + (f" | {pi.strip()[:200]}" if pi else ""), flush=True)
                time.sleep(2)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "inventario":
        inventario(int(sys.argv[2]), int(sys.argv[3]))
    elif cmd == "baixar":
        baixar(sys.argv[2], sys.argv[3:])
