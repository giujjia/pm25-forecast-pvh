"""Survey of the public AWS buckets used in the GFS and GOES tests (step 5).

Listings and HEAD requests only, no data downloaded:
  - GFS (noaa-gfs-bdp-pds): first day of the archive; first day with the 1 h step (f001)
    in the 18 UTC run, by binary search; and whether 2020 exists.
  - GOES ABI-L2-AODF (noaa-goes16, noaa-goes19): years present in each bucket; number of
    files at 15 UTC on days of 2025 around the GOES-16 to GOES-19 switch.

Usage:
    python src/reconhecer_aws.py
"""
import re
from datetime import date, timedelta

import requests

S = requests.Session()
S.headers["User-Agent"] = "Mozilla/5.0 (X11; Linux x86_64)"
GFS = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
GOES = "https://noaa-{sat}.s3.amazonaws.com"


def prefixos(base, prefixo):
    t = S.get(f"{base}/?list-type=2&prefix={prefixo}&delimiter=/", timeout=60).text
    return [p for p in re.findall(r"<Prefix>([^<]+)</Prefix>", t) if p != prefixo]


def chaves(base, prefixo):
    t = S.get(f"{base}/?list-type=2&prefix={prefixo}&max-keys=1000", timeout=60).text
    return re.findall(r"<Key>([^<]+)</Key>", t)


def existe(url):
    return S.head(url, timeout=60).status_code == 200


def gfs_f001(d):
    ds = d.strftime("%Y%m%d")
    return any(existe(f"{GFS}/gfs.{ds}/18/{sub}gfs.t18z.pgrb2.0p25.f001") for sub in ("", "atmos/"))


def main():
    print("== GFS na AWS (noaa-gfs-bdp-pds)")
    print(f"prefixos gfs.2020*: {len(prefixos(GFS, 'gfs.2020'))}")
    dias = prefixos(GFS, "gfs.2021010")
    print(f"primeiros dias de 2021: {[p.strip('/') for p in dias[:3]]}")
    passos = sorted({k.split('.f')[-1] for k in chaves(GFS, 'gfs.20210101/18/gfs.t18z.pgrb2.0p25.f0')
                     if not k.endswith('.idx')})
    print(f"20210101 18 UTC, passos f0xx: {passos}")
    a, b = date(2021, 1, 1), date(2021, 3, 1)
    assert not gfs_f001(a) and gfs_f001(b)
    while (b - a).days > 1:
        m = a + timedelta(days=(b - a).days // 2)
        a, b = (a, m) if gfs_f001(m) else (m, b)
    print(f"passo de 1 h (f001 às 18 UTC): ausente em {a}, presente a partir de {b}")

    print("\n== GOES ABI-L2-AODF")
    for sat in ("goes16", "goes19"):
        anos = [p.split("/")[1] for p in prefixos(GOES.format(sat=sat), "ABI-L2-AODF/")]
        print(f"{sat}: anos {anos}")
    for dj in (90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 120, 200):
        n = {sat: len(chaves(GOES.format(sat=sat), f"ABI-L2-AODF/2025/{dj:03d}/15/")) for sat in ("goes16", "goes19")}
        dia = date(2025, 1, 1) + timedelta(days=dj - 1)
        print(f"2025/{dj:03d} ({dia}): arquivos às 15 UTC goes16 {n['goes16']}, goes19 {n['goes19']}")


if __name__ == "__main__":
    main()
