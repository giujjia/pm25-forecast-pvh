"""Orquestra um bloco da grade da T25 (rede M1-M3 x variantes x sementes) numa dobra.

Roda uma rede por vez chamando src/validacao_cruzada.py e, ao fim de cada uma,
copia pred_*.parquet/.json para a pasta de cópia (o Google Drive no Colab) e
acrescenta uma linha em progresso.csv nessa pasta (configuração, início, fim,
duração, dispositivo, sha256). Redes cujo parquet e json já estão na pasta de
cópia são puladas, o que permite retomar um bloco depois de a sessão cair.

Uso (sempre pelo registrador executar.py):
  python src/grade_vc.py --dobra dobra1 --copia /content/drive/MyDrive/.../volta/vc
  [--versoes M1 M2 M3] [--variantes sem era5] [--sementes 0 1 2] [--seco]
"""
import argparse
import csv
import hashlib
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
VC = RAIZ / "data" / "processed" / "vc"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def agora():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dobra", required=True, choices=["dobra1", "dobra2", "dobra3"])
    ap.add_argument("--copia", required=True, type=Path)
    ap.add_argument("--versoes", nargs="+", default=["M1", "M2", "M3"])
    ap.add_argument("--variantes", nargs="+", default=["sem", "era5"])
    ap.add_argument("--sementes", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--seco", action="store_true", help="só lista o que rodaria")
    a = ap.parse_args()

    a.copia.mkdir(parents=True, exist_ok=True)
    prog = a.copia / "progresso.csv"
    novo = not prog.exists()
    falhas = 0
    for versao in a.versoes:
        for variante in a.variantes:
            for semente in a.sementes:
                nome = f"pred_bilstm_{versao}_{variante}_{a.dobra}_s{semente}"
                destinos = [a.copia / f"{nome}.parquet", a.copia / f"{nome}.json"]
                if all(d.exists() for d in destinos):
                    print(f"[pula] {nome}: já está na cópia", flush=True)
                    continue
                print(f"[roda] {nome}", flush=True)
                if a.seco:
                    continue
                ini, t0 = agora(), time.time()
                cmd = [sys.executable, "src/validacao_cruzada.py", "bilstm", "--dobra", a.dobra,
                       "--versao", versao, "--variante", variante, "--semente", str(semente)]
                proc = subprocess.run(cmd, cwd=RAIZ, capture_output=True, text=True)
                sys.stdout.write(proc.stdout)
                sys.stdout.write(proc.stderr)
                dispositivo = next((l.split(":", 1)[1].strip() for l in proc.stdout.splitlines()
                                    if l.startswith("dispositivo:")), "?")
                dur = round(time.time() - t0)
                if proc.returncode != 0:
                    falhas += 1
                    print(f"[falha] {nome}: código {proc.returncode}", flush=True)
                    continue
                hashes = []
                for ext in ("parquet", "json"):
                    orig = VC / f"{nome}.{ext}"
                    shutil.copy2(orig, a.copia / orig.name)
                    hashes.append(sha256(orig))
                with open(prog, "a", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    if novo:
                        w.writerow(["rede", "dobra", "versao", "variante", "semente", "inicio_utc",
                                    "fim_utc", "duracao_s", "dispositivo", "sha256_parquet", "sha256_json"])
                        novo = False
                    w.writerow([nome, a.dobra, versao, variante, semente, ini, agora(), dur,
                                dispositivo, *hashes])
                print(f"[ok] {nome}: {dur} s, {dispositivo}, copiado", flush=True)
    sys.exit(1 if falhas else 0)


if __name__ == "__main__":
    main()
