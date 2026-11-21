"""Feasibility test of the archived GFS (download plan, step 5): measures bytes and time.

Two routes, one cycle each, without keeping the batch:

  aws: GFS 0.25° file on AWS (noaa-gfs-bdp-pds), which starts on 01/01/2021 and has a 1 h step
       from 26/02/2021. Reads the .idx index of each step and downloads only the GRIB
       messages of the method's fields, by byte ranges. Cycle 15/08/2023 18 UTC, f000–f060.
  gdex: NSF NCAR GDEX d084001 (GFS 0.25°, 3 h step, since 2015), the only one of the two that
       covers 2020. There is no .idx; the test walks the GRIB2 message headers by ranges
       (only sections 0 to 4 of each message) to find the fields. One file: 15/08/2020 18 UTC,
       f003. The cycle estimate multiplies by the 21 steps from f000 to f060.

The fields are the ERA5 ones in the method (METHODS.md §2), plus the 925 hPa temperature. Without
eccodes in the environment, the test does not decode the grids: it checks that each piece starts
with "GRIB", edition 2, and has the size the index indicates. The messages of one step (f006 on
AWS) stay in data/raw/gfs_teste/ for decoding, once eccodes is available.

Usage:
    python src/teste_gfs.py aws
    python src/teste_gfs.py gdex
"""
import struct
import sys
import time
from pathlib import Path

import requests

AWS = "https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{d}/{h}/atmos/gfs.t{h}z.pgrb2.0p25.f{f:03d}"
GDEX = "https://data.gdex.ucar.edu/d084001/{a}/{d}/gfs.0p25.{d}{h}.f{f:03d}.grib2"
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"}
SAIDA = Path(__file__).resolve().parents[1] / "data" / "raw" / "gfs_teste"

# name in the .idx -> (discipline, category, number, surface type, surface value)
CAMPOS = {
    "HPBL:surface": (0, 3, 196, 1, None),
    "UGRD:10 m above ground": (0, 2, 2, 103, 10),
    "VGRD:10 m above ground": (0, 2, 3, 103, 10),
    "TMP:2 m above ground": (0, 0, 0, 103, 2),
    "DPT:2 m above ground": (0, 0, 6, 103, 2),
    "APCP:surface": (0, 1, 8, 1, None),
    "UGRD:850 mb": (0, 2, 2, 100, 85000),
    "VGRD:850 mb": (0, 2, 3, 100, 85000),
    "TMP:925 mb": (0, 0, 0, 100, 92500),
}


def mb(n):
    return f"{n / 1e6:.1f} MB"


def teste_aws(d="20230815", h="18", passos=range(0, 61)):
    s = requests.Session()
    s.headers.update(UA)
    SAIDA.mkdir(parents=True, exist_ok=True)
    tot_bytes = tot_idx = tot_arq = n_msg = 0
    t0 = time.perf_counter()
    for f in passos:
        url = AWS.format(d=d, h=h, f=f)
        idx = s.get(url + ".idx", timeout=60)
        idx.raise_for_status()
        tot_idx += len(idx.content)
        linhas = [ln.split(":") for ln in idx.text.strip().splitlines()]
        tam = int(s.head(url, timeout=60).headers["Content-Length"])
        tot_arq += tam
        achados = []
        for i, c in enumerate(linhas):
            chave = f"{c[3]}:{c[4]}"
            if chave in CAMPOS:
                ini = int(c[1])
                fim = int(linhas[i + 1][1]) - 1 if i + 1 < len(linhas) else tam - 1
                achados.append((chave, c[5], ini, fim))
        guardar = []
        for chave, prazo, ini, fim in achados:
            r = s.get(url, headers={"Range": f"bytes={ini}-{fim}"}, timeout=120)
            assert r.status_code == 206, r.status_code
            b = r.content
            ok = b[:4] == b"GRIB" and b[7] == 2 and struct.unpack(">Q", b[8:16])[0] == len(b)
            if not ok:
                raise RuntimeError(f"f{f:03d} {chave}: pedaço não é uma mensagem GRIB2 inteira")
            tot_bytes += len(b)
            n_msg += 1
            if f == 6:
                guardar.append(b)
        if guardar:
            (SAIDA / f"gfs_aws_{d}{h}_f006_campos.grib2").write_bytes(b"".join(guardar))
        faltam = sorted(set(CAMPOS) - {a[0] for a in achados})
        print(f"f{f:03d}: {len(achados)} mensagens ({', '.join(sorted({a[0].split(':')[0] for a in achados}))})"
              + (f"; ausentes: {faltam}" if faltam else "")
              + (f"; APCP: {[a[1] for a in achados if a[0] == 'APCP:surface']}" if any(a[0] == 'APCP:surface' for a in achados) else ""),
              flush=True)
    dt = time.perf_counter() - t0
    print(f"\nAWS {d} {h} UTC, {len(passos)} passos: {n_msg} mensagens, {mb(tot_bytes)} transferidos "
          f"(+ {mb(tot_idx)} de índices), de {mb(tot_arq)} dos arquivos inteiros "
          f"({100 * tot_bytes / tot_arq:.2f}%), em {dt:.0f} s")
    print(f"Estimativa: 1 rodada por dia em 2021–2026 (~2 140 ciclos): ~{tot_bytes * 2140 / 1e12:.2f} TB "
          f"e ~{dt * 2140 / 3600:.0f} h sequenciais; 4 rodadas por dia, 4 vezes isso")


def cabecalhos(s, url, tam, ler=320):
    """Walks the GRIB2 messages reading only the start of each. Returns the list and the count."""
    pos = pedidos = transferidos = 0
    msgs = []
    while pos < tam:
        r = s.get(url, headers={"Range": f"bytes={pos}-{min(pos + ler, tam) - 1}"}, timeout=60)
        assert r.status_code == 206, r.status_code
        b = r.content
        pedidos += 1
        transferidos += len(b)
        if b[:4] != b"GRIB":
            raise RuntimeError(f"byte {pos}: esperado 'GRIB'")
        disc, total = b[6], struct.unpack(">Q", b[8:16])[0]
        p, cat = 16, None
        while p + 5 <= len(b):
            n, sec = struct.unpack(">I", b[p:p + 4])[0], b[p + 4]
            if sec == 4:
                if p + 28 > len(b):
                    break
                cat, num, sup = b[p + 9], b[p + 10], b[p + 22]
                escala = b[p + 23]
                val = struct.unpack(">I", b[p + 24:p + 28])[0]
                val = val / 10 ** escala if escala not in (0, 255) else val
                msgs.append((pos, total, disc, cat, num, sup, val))
                break
            p += n
        if cat is None:
            raise RuntimeError(f"byte {pos}: seção 4 não coube em {ler} bytes")
        pos += total
    return msgs, pedidos, transferidos


def bate(m, alvo):
    _, _, disc, cat, num, sup, val = m
    d, c, n, s_, v = alvo
    return (disc, cat, num, sup) == (d, c, n, s_) and (v is None or abs(val - v) < 1e-6)


def teste_gdex(a="2020", d="20200815", h="18", f=3):
    s = requests.Session()
    s.headers.update(UA)
    url = GDEX.format(a=a, d=d, h=h, f=f)
    tam = int(s.head(url, timeout=60, allow_redirects=True).headers["Content-Length"])
    t0 = time.perf_counter()
    msgs, pedidos, transf = cabecalhos(s, url, tam)
    t_cab = time.perf_counter() - t0
    achados = {k: [m for m in msgs if bate(m, v)] for k, v in CAMPOS.items()}
    t1 = time.perf_counter()
    baixado = 0
    for k, ms in achados.items():
        for pos, total, *_ in ms:
            r = s.get(url, headers={"Range": f"bytes={pos}-{pos + total - 1}"}, timeout=120)
            assert r.status_code == 206 and r.content[:4] == b"GRIB" and len(r.content) == total
            baixado += total
    t_campos = time.perf_counter() - t1
    print(f"GDEX {d} {h} UTC f{f:03d}: arquivo {mb(tam)}, {len(msgs)} mensagens")
    for k, ms in achados.items():
        print(f"  {k}: {len(ms)} mensagem(ns)")
    print(f"  cabeçalhos: {pedidos} pedidos, {mb(transf)}, {t_cab:.0f} s")
    print(f"  campos: {mb(baixado)} em {t_campos:.0f} s")
    passos, ciclos = 21, 366
    seg = (t_cab + t_campos) * passos
    print(f"Estimativa por ciclo (f000–f060 a cada 3 h, {passos} arquivos): ~{seg / 60:.0f} min e "
          f"{mb((transf + baixado) * passos)}; 2020 com 1 rodada por dia ({ciclos} ciclos): "
          f"~{seg * ciclos / 3600:.0f} h sequenciais. Arquivos inteiros, para comparar: "
          f"{tam * passos * ciclos / 1e12:.1f} TB")


if __name__ == "__main__":
    {"aws": teste_aws, "gdex": teste_gdex}[sys.argv[1]]()
