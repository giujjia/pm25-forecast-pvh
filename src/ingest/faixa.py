"""Remote file read by byte ranges (HTTP Range), counting what was transferred.

Used to read only the needed part of large files: the index of a zip and one station's member
(INMET), or the crop of a netCDF4/HDF5 (GOES test). Open with
io.BufferedReader(ArquivoRemoto(url), buffer_size) and pass to zipfile or h5py.
"""
import io

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

UA = "Mozilla/5.0 (X11; Linux x86_64)"  # generic, no personal data


def sessao_com_retentativas(total=5, pool=4):
    """HTTP session that retries GET and HEAD on connection drops and on 429/5xx,
    with increasing wait (2, 4, 8, 16, 32 s). Added on 16/11/2026 (S3 closes
    idle connections and a single error took down the whole day)."""
    s = requests.Session()
    s.headers["User-Agent"] = UA
    r = Retry(total=total, connect=total, read=total, status=total, backoff_factor=2,
              status_forcelist=(429, 500, 502, 503, 504), allowed_methods=("GET", "HEAD"),
              raise_on_status=False)
    s.mount("https://", HTTPAdapter(max_retries=r, pool_maxsize=pool))
    return s


class ArquivoRemoto(io.RawIOBase):
    def __init__(self, url, sessao=None, timeout=120):
        self.url, self.timeout = url, timeout
        self.s = sessao or requests.Session()
        self.s.headers.setdefault("User-Agent", UA)
        r = self.s.head(url, allow_redirects=True, timeout=timeout)
        r.raise_for_status()
        self.cabecalhos = dict(r.headers)
        self.tamanho = int(r.headers["Content-Length"])
        self.pos = 0
        self.transferidos = 0  # bytes received in the response bodies
        self.pedidos = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.tamanho + off}[whence]
        return self.pos

    def readinto(self, b):
        n = min(len(b), self.tamanho - self.pos)
        if n <= 0:
            return 0
        r = self.s.get(self.url, headers={"Range": f"bytes={self.pos}-{self.pos + n - 1}"},
                       timeout=self.timeout)
        if r.status_code != 206:
            raise IOError(f"{self.url}: esperado 206 (faixa), veio {r.status_code}")
        d = r.content
        b[:len(d)] = d
        self.pos += len(d)
        self.transferidos += len(d)
        self.pedidos += 1
        return len(d)


def abrir(url, buffer=1 << 16, sessao=None):
    """Returns (buffered file, raw ArquivoRemoto, to read the counter)."""
    cru = ArquivoRemoto(url, sessao)
    return io.BufferedReader(cru, buffer), cru
