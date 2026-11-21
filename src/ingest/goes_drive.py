"""Synchronizes GOES-16/19 AOD daily crops downloaded on Colab from Google Drive to data/raw/goes/.

Reads the three Colab output folders (saida_A: 2020–2021, saida_B: 2022–2023, saida_C: 2024–oct/2026),
downloads missing .nc files via the `gdrive` CLI, verifies every file's SHA-256 against the Drive
manifests (manifesto_goes.csv of each batch), and writes the consolidated data/raw/goes/manifesto_goes.csv.
Files already present locally with matching SHA-256 are skipped. No AOD values are ever read or printed.

Usage:
    python src/ingest/goes_drive.py [workers]
"""
import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

RAW = Path(__file__).resolve().parents[2] / "data" / "raw" / "goes"
MANIFESTO = RAW / "manifesto_goes.csv"

MANIFESTOS_DRIVE = {
    "A": "161bQzCDmXSh9F-JEdo4aApjmV9rdHG3j",
    "B": "1iDMCPHylO8opTE7SQzJGrYoJUyPNwwSq",
    "C": "1nw2W-GQlAuFnxKDHV3ne-lqc5fpCgwEi",
}

PASTAS_DRIVE = {
    "2020": "18KRnsp_0MsXosb2iFzKSSf3bzGFUc-tZ",
    "2021": "1M14lOjLRY4HQGSjX-eSPPjGWjl95Apn_",
    "2022": "1oWoNYKH_Ysf8qp0k-AmVkxXxP1DG22nw",
    "2023": "1MQ4u1AXKrlgTiwZ-2UAkKM3AgtTgBamp",
    "2024": "1lhYI2FXSEF6mfK7Nz5UKiF4vZOrBsunw",
    "2025": "1WSLC4iMnky3Zy2uNlE_kpetkBGOaeLde",
    "selado_2026": "1A0cCTUtZQO0GKUfsOut5HlwJkwfL-rER",
}

_trava = threading.Lock()


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def gdrive_json(*args):
    r = subprocess.run(["gdrive", *args, "--json"], capture_output=True, text=True, check=True)
    return json.loads(r.stdout)


def carregar_manifestos_drive(tmp: Path):
    """Downloads the three batch manifests and returns (rows_by_relpath, fieldnames)."""
    por_rel = {}
    campos = None
    for faixa, fid in MANIFESTOS_DRIVE.items():
        dst = tmp / f"manifesto_{faixa}.csv"
        subprocess.run(["gdrive", "readonly", "download", fid, "--out", str(dst)],
                       capture_output=True, text=True, check=True)
        with open(dst, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if campos is None:
                campos = reader.fieldnames
            for row in reader:
                rel = row["arquivo"].replace("\\", "/")
                if rel.startswith("goes/"):
                    rel = rel[len("goes/"):]
                por_rel[rel] = row
    return por_rel, campos


def listar_arquivos_drive():
    """Returns [(subpasta, name, file_id, size_bytes)]."""
    itens = []
    for sub, fid in PASTAS_DRIVE.items():
        lista = gdrive_json("readonly", "ls", fid, "--max", "0")
        nc = [x for x in lista if x["name"].endswith(".nc")]
        print(f"Drive {sub}: {len(nc)} arquivos .nc", flush=True)
        for x in nc:
            itens.append((sub, x["name"], x["id"], int(x.get("size", 0))))
    return itens


def baixar_um(sub: str, nome: str, fid: str, sha_esperado: str):
    destino = RAW / sub / nome
    destino.parent.mkdir(parents=True, exist_ok=True)
    if destino.exists():
        if sha256(destino) == sha_esperado:
            return "pulado", destino.stat().st_size
        # Corrupted or mismatched local file: re-download
    parcial = destino.with_suffix(".nc.part")
    ultimo_erro = None
    for tentativa in range(1, 5):
        try:
            if parcial.exists():
                parcial.unlink()
            r = subprocess.run(["gdrive", "readonly", "download", fid, "--out", str(parcial)],
                               capture_output=True, text=True, timeout=180)
            if r.returncode != 0:
                raise RuntimeError(r.stderr.strip() or r.stdout.strip() or f"exit {r.returncode}")
            got = sha256(parcial)
            if got != sha_esperado:
                raise RuntimeError(f"SHA-256 divergente (esperado {sha_esperado[:12]}, obtido {got[:12]})")
            parcial.rename(destino)
            return "baixado", destino.stat().st_size
        except Exception as e:
            ultimo_erro = str(e)
            if parcial.exists():
                parcial.unlink()
            time.sleep(min(15, 2 * tentativa))
    return f"erro: {ultimo_erro}", 0


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as td:
        por_rel, campos = carregar_manifestos_drive(Path(td))
    print(f"Manifestos do Drive carregados: {len(por_rel)} dias únicos", flush=True)

    itens = listar_arquivos_drive()
    print(f"Total listado no Drive: {len(itens)} arquivos | workers={workers}", flush=True)
    if len(itens) < len(por_rel):
        raise RuntimeError(f"Listagem incompleta do Drive: {len(itens)} < {len(por_rel)} no manifesto")

    tarefas = []
    sem_manifesto = []
    for sub, nome, fid, sz in itens:
        rel = f"{sub}/{nome}"
        row = por_rel.get(rel)
        if row is None:
            sem_manifesto.append(rel)
            continue
        tarefas.append((sub, nome, fid, row["sha256"]))

    if sem_manifesto:
        print(f"AVISO: {len(sem_manifesto)} arquivos no Drive fora do manifesto: {sem_manifesto[:5]}", flush=True)

    baixados = pulados = erros = 0
    bytes_baixados = 0
    falhas = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(baixar_um, sub, nome, fid, sha): f"{sub}/{nome}"
                for sub, nome, fid, sha in tarefas}
        for i, fut in enumerate(as_completed(futs), 1):
            rel = futs[fut]
            status, sz = fut.result()
            if status == "baixado":
                baixados += 1
                bytes_baixados += sz
            elif status == "pulado":
                pulados += 1
            else:
                erros += 1
                falhas.append((rel, status))
                print(f"FALHA {rel}: {status}", flush=True)
            if i % 50 == 0 or i == len(tarefas):
                dt = time.perf_counter() - t0
                mb_s = (bytes_baixados / 1e6) / max(dt, 1e-6)
                print(f"[{i}/{len(tarefas)}] baixados={baixados} pulados={pulados} erros={erros} "
                      f"({bytes_baixados/1e9:.2f} GB, {mb_s:.2f} MB/s, {dt/60:.1f} min)", flush=True)

    # Consolidate manifest only with files verified on disk
    linhas = sorted(por_rel.values(), key=lambda r: r["dia"])
    presentes = []
    faltantes = []
    for row in linhas:
        rel = row["arquivo"].replace("\\", "/")
        if rel.startswith("goes/"):
            rel = rel[len("goes/"):]
        if (RAW / rel).exists():
            presentes.append(row)
        else:
            faltantes.append(rel)

    RAW.mkdir(parents=True, exist_ok=True)
    tmp_man = MANIFESTO.with_suffix(".csv.part")
    with open(tmp_man, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        w.writerows(presentes)
    tmp_man.rename(MANIFESTO)

    print(f"FIM: baixados={baixados}, pulados={pulados}, erros={erros}, "
          f"no manifesto consolidado={len(presentes)}/{len(linhas)}, faltantes={len(faltantes)}", flush=True)
    if erros > 0 or faltantes:
        sys.exit(1)


if __name__ == "__main__":
    main()
