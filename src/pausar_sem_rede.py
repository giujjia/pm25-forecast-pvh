"""Network watchdog: freezes download batches when the internet drops and resumes them when it returns.

Without it, each outage makes a batch run through all its items failing with connection errors
(observed on 16/11/2026 at 18h UTC: ~1,100 failed items within minutes, no data
lost, but no progress and the queue wasted). With the watchdog, the src/ingest/{gfs,goes,merra2,s5p}.py
processes get SIGSTOP on the drop and SIGCONT on the return, and resume where they were.

Network considered down: in 2 consecutive checks (every 10 s), either the name resolution of
s3.amazonaws.com or the TCP connection to 1.1.1.1:443 fails. Considered back: 2 consecutive
checks with both working. CDS and ADS requests have their own retries
(up to 500, every 120 s) and are not touched.

After the return, connections left open during the pause may be dead; the sessions with
retries (faixa.sessao_com_retentativas) redo the request. If the laptop is
suspended (lid closed), everything freezes the same way and this watchdog only checks on waking.

Usage (leave running in the background, through executar.py):
    python src/pausar_sem_rede.py
"""
import os
import re
import signal
import socket
import time
from datetime import datetime, timezone

ALVO = re.compile(r"^(\S*python\S*) -u src/ingest/(gfs|goes|merra2|s5p)\.py ")
INTERVALO = 10


def agora():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def rede_ok():
    try:
        socket.getaddrinfo("s3.amazonaws.com", 443)
        socket.create_connection(("1.1.1.1", 443), timeout=4).close()
        return True
    except OSError:
        return False


def processos():
    """PIDs of the batches (python itself, not the executar.py wrapping it)."""
    out = []
    for p in os.listdir("/proc"):
        if p.isdigit() and int(p) != os.getpid():
            try:
                cmd = open(f"/proc/{p}/cmdline", "rb").read().replace(b"\0", b" ").decode().strip()
            except OSError:
                continue
            if ALVO.match(cmd + " "):
                out.append(int(p))
    return out


def enviar(sinal, pids):
    for p in pids:
        try:
            os.kill(p, sinal)
        except ProcessLookupError:
            pass


def main():
    pausado, falhas, acertos, parados = False, 0, 0, set()
    print(f"{agora()} | vigia iniciado, {len(processos())} lotes", flush=True)
    while True:
        if rede_ok():
            falhas, acertos = 0, acertos + 1
        else:
            falhas, acertos = falhas + 1, 0
        if not pausado and falhas >= 2:
            parados = set(processos())
            enviar(signal.SIGSTOP, parados)
            pausado = True
            print(f"{agora()} | rede fora do ar: {len(parados)} lotes congelados", flush=True)
        elif pausado:
            novos = set(processos()) - parados  # batch launched during the outage
            enviar(signal.SIGSTOP, novos)
            parados |= novos
            if acertos >= 2:
                enviar(signal.SIGCONT, parados)
                print(f"{agora()} | rede de volta: {len(parados)} lotes retomados", flush=True)
                pausado, parados = False, set()
        time.sleep(INTERVALO)


if __name__ == "__main__":
    main()
