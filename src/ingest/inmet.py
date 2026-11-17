"""INMET ingestion: automatic station A925 (Porto Velho), hourly data.

Source: the yearly historical-data files on the INMET portal, one zip per year with all
automatic stations, public and without an account. Instead of downloading the whole zip (80 to 104 MB),
it reads only the index and the A925 member by byte ranges (faixa.py). zipfile checks the
member's CRC-32. The CSV is written unaltered to data/raw/inmet/, under its original name.

Each write enters data/raw/inmet/manifesto_inmet.csv: URL, Last-Modified and size of the
zip, member name, size and CRC, SHA-256 of the written CSV and bytes transferred. The current
year's zip is updated by INMET (the member name carries the end date, e.g. 31-10-2026), so
a new version writes a new file and does not overwrite the previous one.

File times are UTC (column "Hora UTC" or "HORA (UTC)").

Blind-test year (TESTE_CEGO): listar shows only the header; no command prints values.

Usage:
    python src/ingest/inmet.py listar 2020 2026    # A925 member in each zip and the header
    python src/ingest/inmet.py baixar 2020 2026    # writes the A925 CSVs
"""
import csv
import hashlib
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from faixa import abrir  # noqa: E402

URL = "https://portal.inmet.gov.br/uploads/dadoshistoricos/{ano}.zip"
ESTACAO = "A925"
RAW = Path(__file__).resolve().parents[2] / "data" / "raw" / "inmet"
MANIFESTO = RAW / "manifesto_inmet.csv"
TESTE_CEGO = {2026}
LINHAS_CABECALHO = 9  # 8 lines of station metadata + the line with column names


def agora():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def membro_estacao(z, ano):
    nomes = [n for n in z.namelist() if f"_{ESTACAO}_" in n.upper()]
    if len(nomes) != 1:
        raise RuntimeError(f"{ano}: esperado 1 arquivo da {ESTACAO} no zip, achei {nomes}")
    return nomes[0]


def listar(ano_ini, ano_fim):
    total = 0
    for ano in range(ano_ini, ano_fim + 1):
        f, cru = abrir(URL.format(ano=ano))
        z = zipfile.ZipFile(f)
        nome = membro_estacao(z, ano)
        info = z.getinfo(nome)
        print(f"{ano}: zip {cru.tamanho} bytes, Last-Modified {cru.cabecalhos.get('Last-Modified')}, "
              f"{len(z.namelist())} arquivos | {nome} | {info.file_size} bytes, CRC {info.CRC:08x}")
        with z.open(nome) as m:
            for _ in range(LINHAS_CABECALHO):  # metadata and column names only, no values
                print("    " + m.readline().decode("latin-1").rstrip())
        total += cru.transferidos
        print(f"    transferidos: {cru.transferidos} bytes em {cru.pedidos} pedidos", flush=True)
    print(f"total transferido: {total} bytes")


def ja_gravados():
    if not MANIFESTO.exists():
        return {}
    with open(MANIFESTO, newline="") as fh:
        return {(r["membro"], r["crc32"]): r for r in csv.DictReader(fh)}


def baixar(ano_ini, ano_fim):
    RAW.mkdir(parents=True, exist_ok=True)
    feitos = ja_gravados()
    novo = not MANIFESTO.exists()
    with open(MANIFESTO, "a", newline="") as fh:
        w = csv.writer(fh)
        if novo:
            w.writerow(["utc", "ano", "url", "zip_last_modified", "zip_bytes", "membro", "crc32",
                        "bytes", "arquivo", "sha256", "transferidos"])
        for ano in range(ano_ini, ano_fim + 1):
            url = URL.format(ano=ano)
            f, cru = abrir(url)
            z = zipfile.ZipFile(f)
            nome = membro_estacao(z, ano)
            info = z.getinfo(nome)
            crc = f"{info.CRC:08x}"
            destino = RAW / Path(nome).name
            if (nome, crc) in feitos and destino.exists():
                print(f"{ano}: {destino.name} já gravado (mesmo CRC), pulado", flush=True)
                continue
            if destino.exists():
                raise RuntimeError(f"{destino.name} existe com outro conteúdo; não sobrescrevo")
            dados = z.read(nome)  # zipfile checks the CRC-32
            destino.write_bytes(dados)
            sha = hashlib.sha256(dados).hexdigest()
            w.writerow([agora(), ano, url, cru.cabecalhos.get("Last-Modified"), cru.tamanho, nome,
                        crc, len(dados), destino.name, sha, cru.transferidos])
            fh.flush()
            print(f"{ano}: {destino.name} gravado | {len(dados)} bytes, CRC {crc} conferido, "
                  f"SHA-256 {sha} | transferidos {cru.transferidos} de {cru.tamanho}", flush=True)


if __name__ == "__main__":
    cmd, a, b = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    {"listar": listar, "baixar": baixar}[cmd](a, b)
