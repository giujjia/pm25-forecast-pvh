"""NASA FIRMS (VIIRS) ingestion: fire detections and FRP.

Downloads through the area API, in blocks of up to 5 days (API limit), VIIRS on S-NPP
and NOAA-20. "SP" product (standard processing) for the history; "NRT" (near real
time) only in the months SP does not cover yet (SP comes out 2 to 3 months later).

Two crops (METHODS.md §4):
  - "300": 300 km around all sensors (rings 0–50, 50–150 and 150–300 km);
  - "1000": 1000 km, for the 300–600 and 600–1000 km rings, added on 12/11/2026 after a
    literature review. Separate files: firms1000_<...>.parquet.
The distance from each detection to each sensor is computed later, when assembling the inputs.

Files (see plano_ano):
  - <prefix>_<year>.parquet: whole year in SP (like 2020–2024);
  - <prefix>_<YYYYMM>_sp.parquet or _nrt.parquet: one month, when SP does not cover the whole year.
    When the SP of a month comes out, that month's NRT file goes to nrt_substituido/.

Blind-test year (TESTE_CEGO): the log shows no detection counts or FRP, only coverage.

Usage:
    python src/ingest/firms.py disponibilidade          # period covered by each product
    python src/ingest/firms.py teste 2024-08-15         # one day, only prints
    python src/ingest/firms.py historico 2020 2024      # whole years, one parquet per year (300 km)
    python src/ingest/firms.py historico 2020 2024 1000 # the same with the 1000 km crop
    python src/ingest/firms.py historico 2025 2026 300 --ate 2026-10   # current year up to the last closed month
"""
import calendar
import io
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()
KEY = os.environ.get("FIRMS_MAP_KEY", "")  # checked at use; importing does not require the key (tests)
API = "https://firms.modaps.eosdis.nasa.gov/api"
RAW = Path(__file__).resolve().parents[2] / "data" / "raw" / "firms"

# west, south, east, north: 300 km (~2.75°) beyond the extreme sensors
# (lat −8.05 to −8.75; lon −63.49 to −63.91).
AREAS = {
    "300": ("-66.8,-11.6,-60.6,-5.2", "firms"),
# 1000 km ≈ 9.0° of latitude and 9.1° of longitude at 8.4° S (cos 8.4° = 0.989), rounded outward
    "1000": ("-73.0,-17.8,-54.4,1.0", "firms1000"),
}
AREA = AREAS["300"][0]
PRODUTOS = {
    "SP": ["VIIRS_SNPP_SP", "VIIRS_NOAA20_SP"],
    "NRT": ["VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT"],
}
FONTES = PRODUTOS["SP"]
BLOCO = 5  # days per call (API maximum)
TESTE_CEGO = {2026}  # METHODS.md §6: for 2026 only coverage is checked


def _chave():
    if not KEY:
        raise SystemExit("FIRMS_MAP_KEY ausente no .env")
    return KEY


def disponibilidade(imprimir=True):
    r = requests.get(f"{API}/data_availability/csv/{_chave()}/all", timeout=60)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    if imprimir:
        print(df.to_string(index=False))
    return df


def janelas(disp):
    """{data_id: (min_date, max_date)} from the availability table."""
    return {r.data_id: (date.fromisoformat(str(r.min_date)), date.fromisoformat(str(r.max_date)))
            for r in disp.itertuples()}


def _cobre(janela, produto, ini, fim):
    """Do all satellites of the product cover [ini, fim]?"""
    return all(f in janela and janela[f][0] <= ini and janela[f][1] >= fim for f in PRODUTOS[produto])


def plano_ano(ano, janela, ate=None):
    """Files of a year: list of (suffix, product, start, end), without touching the network.

    Whole year in a single file (suffix "") when SP of both satellites covers the year
    and ate = (year, month) does not cut it. Otherwise, one file per month up to ate: SP if both
    satellites cover the whole month in SP, NRT if they cover it in NRT. A month without
    complete coverage in either product stops with ValueError (use --ate before it).
    """
    ini_ano, fim_ano = date(ano, 1, 1), date(ano, 12, 31)
    corta = ate is not None and ate < (ano, 12)
    if not corta and _cobre(janela, "SP", ini_ano, fim_ano):
        return [("", "SP", ini_ano, fim_ano)]
    ultimo = ate[1] if corta else 12
    plano = []
    for mes in range(1, ultimo + 1):
        ini, fim = date(ano, mes, 1), date(ano, mes, calendar.monthrange(ano, mes)[1])
        for produto in ("SP", "NRT"):
            if _cobre(janela, produto, ini, fim):
                plano.append((f"{ano}{mes:02d}_{produto.lower()}", produto, ini, fim))
                break
        else:
            raise ValueError(f"{ano}-{mes:02d}: nem SP nem NRT cobrem o mês inteiro nos dois "
                             f"satélites; use --ate com um mês anterior")
    return plano


def baixar(fonte, inicio, dias=BLOCO, area=AREA):
    url = f"{API}/area/csv/{_chave()}/{fonte}/{area}/{dias}/{inicio:%Y-%m-%d}"
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    if not r.text.strip() or r.text.lstrip().startswith(("Invalid", "Error")):
        raise RuntimeError(f"{fonte} {inicio}: {r.text[:200]}")
    df = pd.read_csv(io.StringIO(r.text))
    df["fonte"] = fonte
    return df


def baixar_periodo(fontes, ini, fim, area):
    partes = []
    d = ini
    while d <= fim:
        n = min(BLOCO, (fim - d).days + 1)
        for fonte in fontes:
            partes.append(baixar(fonte, d, dias=n, area=area))
            time.sleep(0.5)
        d += timedelta(days=n)
    return pd.concat(partes, ignore_index=True)


def relatar(out, df, ano, produto, ini, fim):
    versoes = sorted(map(str, df["version"].dropna().unique())) if "version" in df else []
    if ano in TESTE_CEGO:
        print(f"{out.name}: gravado ({produto}, {ini} a {fim}); versões {versoes}; "
              f"contagens omitidas (ano do teste cego)", flush=True)
    else:
        print(f"{out.name}: {len(df)} focos ({produto}, {ini} a {fim}); versões {versoes}", flush=True)


def historico(ano_ini, ano_fim, raio="300", ate=None):
    area, prefixo = AREAS[raio]
    RAW.mkdir(parents=True, exist_ok=True)
    janela = janelas(disponibilidade(imprimir=False))
    for ano in range(ano_ini, ano_fim + 1):
        for sufixo, produto, ini, fim in plano_ano(ano, janela, ate):
            out = RAW / (f"{prefixo}_{sufixo}.parquet" if sufixo else f"{prefixo}_{ano}.parquet")
            if out.exists():
                print(f"{out.name} já existe, pulando", flush=True)
                continue
            df = baixar_periodo(PRODUTOS[produto], ini, fim, area)
            df.to_parquet(out, index=False)
            relatar(out, df, ano, produto, ini, fim)
            nrt = RAW / f"{prefixo}_{sufixo[:6]}_nrt.parquet"
            if produto == "SP" and sufixo and nrt.exists():
                (RAW / "nrt_substituido").mkdir(exist_ok=True)
                nrt.rename(RAW / "nrt_substituido" / nrt.name)
                print(f"{nrt.name}: substituído pelo SP, movido para nrt_substituido/", flush=True)


def teste(dia):
    d = date.fromisoformat(dia)
    for fonte in FONTES:
        df = baixar(fonte, d, dias=1)
        print(f"{fonte} {d}: {len(df)} focos, FRP total {df['frp'].sum():.0f} MW")
        if len(df):
            print(df.head(3).to_string(index=False))


if __name__ == "__main__":
    cmd, *args = sys.argv[1:] or ["disponibilidade"]
    ate = None
    if "--ate" in args:
        i = args.index("--ate")
        ate = tuple(int(x) for x in args[i + 1].split("-"))
        args = args[:i] + args[i + 2:]
    if cmd == "disponibilidade":
        disponibilidade()
    elif cmd == "teste":
        teste(args[0] if args else "2024-08-15")
    elif cmd == "historico":
        historico(int(args[0]), int(args[1]), *(args[2:3] or ["300"]), ate=ate)
