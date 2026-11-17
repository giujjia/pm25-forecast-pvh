"""Ingestion of MERRA-2 tavg1_2d_aer_Nx (M2T1NXAER v5.12.4) cropped to the ERA5 area.

SO3 (METHODS.md §8): MERRA-2 PM2.5 is reconstructed from the aerosol species with the GMAO
formula (METHODS.md, MERRA-2 section), from 2020 to 2024. For 2024, only the PM2.5 coverage is checked
until the freeze (METHODS.md §8).

Route: the local GES DISC OPeNDAP was retired (HTTP 410). Access is through the
Earthdata Cloud OPeNDAP, with the author's Earthdata account in ~/.netrc (never read or printed
by this script; requests applies it on its own on the redirect to urs.earthdata.nasa.gov).
File names (MERRA2_400, MERRA2_401 in reprocessed months) and addresses come from the
NASA CMR catalog, collection C1276812830-GES_DISC.

Crop: 0.5° × 0.625° grid; latitudes -10.5 to -8.0 (indices 159 to 164) and longitudes -65.0
to -62.5 (indices 184 to 188), 24 hours per file. Variables: the five of the formula (BCSMASS,
OCSMASS, SO4SMASS, DUSMASS25, SSSMASS25) and the total aerosol optical thickness (TOTEXTTAU), to compare
with GOES. Each OPeNDAP response is written as it arrived (netCDF4), in
data/raw/merra2/{year}/, with a manifest in data/raw/merra2/manifesto_merra2.csv. Days already
written are skipped. No value is printed.

Usage:
    python src/ingest/merra2.py baixar 2020-01-01 2024-12-31 [workers]
"""
import csv
import hashlib
import io
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote

import h5py

sys.path.insert(0, str(Path(__file__).resolve().parent))
from faixa import sessao_com_retentativas  # noqa: E402

RAW = Path(__file__).resolve().parents[2] / "data" / "raw" / "merra2"
MANIFESTO = RAW / "manifesto_merra2.csv"
COLECAO = "C1276812830-GES_DISC"
CMR = "https://cmr.earthdata.nasa.gov/search/granules.json"
OPENDAP = "https://opendap.earthdata.nasa.gov/collections/{c}/granules/{g}"
VARS = ["BCSMASS", "OCSMASS", "SO4SMASS", "DUSMASS25", "SSSMASS25", "TOTEXTTAU"]
LAT, LON = (159, 164), (184, 188)  # inclusive
_trava = threading.Lock()


def granulos(ini, fim):
    """[(day YYYYMMDD, granule id)] from the CMR, paginated. The CMR is public, no login."""
    s = sessao_com_retentativas()
    out, pagina = [], 1
    while True:
        r = s.get(CMR, params={"collection_concept_id": COLECAO, "page_size": 2000, "page_num": pagina,
                               "temporal": f"{ini}T00:00:00Z,{fim}T23:59:59Z", "sort_key": "start_date"},
                  timeout=120)
        r.raise_for_status()
        e = r.json()["feed"]["entry"]
        out += [(g["title"].split(".")[-2], g["title"]) for g in e]
        if len(e) < 2000:
            return out
        pagina += 1


def restricao():
    t = "[0:1:23]"
    la, lo = f"[{LAT[0]}:1:{LAT[1]}]", f"[{LON[0]}:1:{LON[1]}]"
    partes = [f"/{v}{t}{la}{lo}" for v in VARS] + [f"/time{t}", f"/lat{la}", f"/lon{lo}"]
    return ";".join(partes)


def alvo(dia, titulo):
    nome = titulo.split(":")[1].replace(".nc4", ".recorte_pvh.nc4")
    return RAW / dia[:4] / nome


def baixar_dia(dia, titulo):
    s = sessao_com_retentativas()  # trust_env: applies ~/.netrc at the Earthdata login
    t0 = time.perf_counter()
    url = OPENDAP.format(c=COLECAO, g=quote(titulo, safe="")) + ".dap.nc4"
    r = s.get(url, params={"dap4.ce": restricao()}, timeout=300)
    if r.status_code != 200 or r.content[:4] != b"\x89HDF":
        raise RuntimeError(f"HTTP {r.status_code}, {len(r.content)} bytes: {r.text[:200]!r}")
    with h5py.File(io.BytesIO(r.content), "r") as f:
        formas = {v: f[v].shape for v in VARS}
    esperado = (24, LAT[1] - LAT[0] + 1, LON[1] - LON[0] + 1)
    if any(fm != esperado for fm in formas.values()):
        raise RuntimeError(f"formas inesperadas: {formas}")
    destino = alvo(dia, titulo)
    destino.parent.mkdir(parents=True, exist_ok=True)
    parcial = destino.with_suffix(".part")
    parcial.write_bytes(r.content)
    parcial.rename(destino)
    with _trava:
        novo = not MANIFESTO.exists()
        with open(MANIFESTO, "a", newline="") as f:
            w = csv.writer(f)
            if novo:
                w.writerow(["dia", "granulo", "bytes", "segundos", "arquivo", "sha256"])
            w.writerow([dia, titulo, len(r.content), f"{time.perf_counter() - t0:.1f}",
                        str(destino.relative_to(RAW.parent)), hashlib.sha256(r.content).hexdigest()])
    return len(r.content)


def baixar(ini, fim, trabalhadores=4):
    lista = granulos(ini, fim)
    dias = [d for d, _ in lista]
    print(f"CMR: {len(lista)} granulos de {ini} a {fim} "
          f"({len(set(dias))} dias distintos; prefixos {sorted({t.split(':')[1][:10] for _, t in lista})})",
          flush=True)
    faltam = [(d, t) for d, t in lista if not alvo(d, t).exists()]
    print(f"{len(faltam)} a baixar, {trabalhadores} em paralelo", flush=True)
    t0, feitos, total, erros = time.perf_counter(), 0, 0, 0
    with ThreadPoolExecutor(trabalhadores) as ex:
        futuros = {ex.submit(baixar_dia, d, t): d for d, t in faltam}
        for fu in as_completed(futuros):
            d = futuros[fu]
            try:
                total += fu.result()
                feitos += 1
            except Exception as e:
                erros += 1
                print(f"{d}: ERRO {str(e)[:300]}", flush=True)
                continue
            if feitos % 50 == 0 or feitos == len(faltam):
                print(f"{feitos}/{len(faltam)} em {(time.perf_counter() - t0) / 60:.0f} min, "
                      f"{total / 1e6:.0f} MB", flush=True)
    print(f"fim: {feitos} dias gravados, {erros} com erro (rodar de novo retoma só os que faltam)", flush=True)
    if erros:
        sys.exit(1)


if __name__ == "__main__":
    baixar(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 4)
