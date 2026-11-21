"""Feasibility test of GOES (download plan, step 5): can only the crop be read?

ABI-L2-AODF product (AOD, full disk, every 10 min) in the public AWS buckets
(noaa-goes16 until 07/04/2025; noaa-goes19 afterwards). Opens the remote netCDF4 with h5py over
byte-range reads (ingest/faixa.py), with a block cache, reads the x/y coordinates and the
projection, finds the rows and columns of the ERA5 area and reads only that crop of AOD and DQF.
Measures bytes transferred, requests and time, against the file size.

Coverage only: prints the number of pixels in the crop and how many have valid AOD, never values.

Usage:
    python src/teste_goes.py goes16 2023 227 15    # year, day of year, UTC hour
"""
import io
import math
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent / "ingest"))
from faixa import ArquivoRemoto  # noqa: E402

AREA = (-8.0, -65.0, -10.5, -62.5)  # N, W, S, E (same as ERA5)
BUCKET = "https://noaa-{sat}.s3.amazonaws.com"


class Blocos(io.RawIOBase):
    """Cache of fixed blocks over an ArquivoRemoto: each block is requested only once."""

    def __init__(self, cru, bloco=1 << 18):
        self.cru, self.bloco, self.cache, self.pos = cru, bloco, {}, 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.cru.tamanho + off}[whence]
        return self.pos

    def _bloco(self, i):
        if i not in self.cache:
            self.cru.seek(i * self.bloco)
            buf = bytearray(min(self.bloco, self.cru.tamanho - i * self.bloco))
            self.cru.readinto(buf)
            self.cache[i] = bytes(buf)
        return self.cache[i]

    def readinto(self, b):
        n = min(len(b), self.cru.tamanho - self.pos)
        if n <= 0:
            return 0
        out, p = bytearray(), self.pos
        while len(out) < n:
            i, off = divmod(p, self.bloco)
            pedaco = self._bloco(i)[off:off + n - len(out)]
            out += pedaco
            p += len(pedaco)
        b[:n] = out
        self.pos += n
        return n


def arquivos_da_hora(sat, ano, dia, hora):
    url = BUCKET.format(sat=sat) + f"/?list-type=2&prefix=ABI-L2-AODF/{ano}/{dia:03d}/{hora:02d}/"
    t = requests.get(url, timeout=60).text
    chaves = [k.split("</Key>")[0] for k in t.split("<Key>")[1:]]
    tamanhos = [int(s.split("</Size>")[0]) for s in t.split("<Size>")[1:]]
    return list(zip(chaves, tamanhos))


def escala(ds):
    v = ds[()].astype("f8")
    return v * ds.attrs.get("scale_factor", 1.0) + ds.attrs.get("add_offset", 0.0)


def latlon_para_xy(lat, lon, p):
    """GOES-R PUG formulas (vol. 4, §4.2.8): scan angles x, y in radians."""
    req, rpol = p["semi_major_axis"], p["semi_minor_axis"]
    H = p["perspective_point_height"] + req
    l0 = math.radians(p["longitude_of_projection_origin"])
    phi, lam = math.radians(lat), math.radians(lon)
    e2 = (req ** 2 - rpol ** 2) / req ** 2
    phic = math.atan((rpol ** 2 / req ** 2) * math.tan(phi))
    rc = rpol / math.sqrt(1 - e2 * math.cos(phic) ** 2)
    sx = H - rc * math.cos(phic) * math.cos(lam - l0)
    sy = -rc * math.cos(phic) * math.sin(lam - l0)
    sz = rc * math.sin(phic)
    return math.asin(-sy / math.sqrt(sx ** 2 + sy ** 2 + sz ** 2)), math.atan(sz / sx)


def main(sat, ano, dia, hora):
    lista = arquivos_da_hora(sat, ano, dia, hora)
    print(f"{sat} {ano}/{dia:03d} {hora:02d} UTC: {len(lista)} arquivos na hora "
          f"({', '.join(f'{s / 1e6:.1f} MB' for _, s in lista)})")
    chave, tam = lista[0]
    url = BUCKET.format(sat=sat) + "/" + chave
    t0 = time.perf_counter()
    cru = ArquivoRemoto(url)
    f = h5py.File(Blocos(cru), "r")
    proj = {k: float(np.ravel(v)[0]) for k, v in f["goes_imager_projection"].attrs.items()
            if isinstance(v, np.ndarray) and v.dtype.kind == "f"}
    x, y = escala(f["x"]), escala(f["y"])
    cantos = [latlon_para_xy(la, lo, proj) for la in (AREA[0], AREA[2]) for lo in (AREA[1], AREA[3])]
    xs, ys = [c[0] for c in cantos], [c[1] for c in cantos]
    c0, c1 = int(np.searchsorted(x, min(xs))), int(np.searchsorted(x, max(xs)))
    yd = y[::-1]  # y decreases with the row
    l1 = len(y) - int(np.searchsorted(yd, min(ys)))
    l0 = len(y) - int(np.searchsorted(yd, max(ys)))
    t_meta = time.perf_counter() - t0
    b_meta = cru.transferidos
    aod = f["AOD"][l0:l1, c0:c1]
    dqf = f["DQF"][l0:l1, c0:c1]
    dt = time.perf_counter() - t0
    preenchido = aod != f["AOD"].attrs.get("_FillValue", np.array([-1]))[0]
    d = f["AOD"]
    print(f"arquivo: {chave.split('/')[-1]} ({tam / 1e6:.1f} MB); AOD {d.shape}, blocos HDF5 {d.chunks}, "
          f"compressão {d.compression}")
    print(f"recorte da área {AREA}: linhas {l0}–{l1}, colunas {c0}–{c1} ({aod.size} pixels); "
          f"com AOD preenchido: {int(preenchido.sum())}; DQF presente: {dqf.shape == aod.shape}")
    print(f"transferidos: {cru.transferidos / 1e6:.2f} MB em {cru.pedidos} pedidos "
          f"({100 * cru.transferidos / tam:.1f}% do arquivo); metadados e coordenadas {b_meta / 1e6:.2f} MB; "
          f"{t_meta:.1f} s até o recorte, {dt:.1f} s no total")
    por_ano = 12 * 365  # one image per daytime hour
    print(f"Estimativa, 1 imagem por hora de dia (~{por_ano} por ano, ~{por_ano * 7} em 2020–2026): "
          f"~{cru.transferidos * por_ano * 7 / 1e9:.0f} GB transferidos e ~{dt * por_ano * 7 / 3600:.0f} h "
          f"sequenciais; 6 imagens por hora, 6 vezes isso. Arquivos inteiros: "
          f"~{tam * por_ano * 7 / 1e12:.2f} TB por imagem/hora")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
