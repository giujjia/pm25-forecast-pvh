"""Times the Bi-LSTM of METHODS.md §5 on synthetic data, with no project data.

Measures milliseconds per training batch (a full AdamW step) and per evaluation batch. Runs on
the CPU (via the runner) or on a GPU (Colab), to decide where to train the network
(the earlier benchmark had no saved output).

Same network as the earlier benchmark scripts: Bi-LSTM encoder,
bridge to the initial state, hour-by-hour LSTM decoder with additive attention
(Bahdanau, the one in METHODS.md §5), not fed its own prediction, and 3 quantiles with
q10 ≤ q50 ≤ q90. Additions: number of layers and units as arguments, GPU, and the option
--atencao luong, in which the decoder runs the 48 h in a single call and attention comes
afterwards. This is a different architecture (METHODS.md fixes Bahdanau): it is included only to
measure the cost of the hour-by-hour loop, in case the author wants to decide on it.

Usage:
    python src/bench_rede.py --threads 4 --camadas 1 --unidades 64 --batch 256
    python src/bench_rede.py --device cuda --camadas 2 --unidades 128
Prints one JSON line.
"""
import argparse
import json
import platform
import time

import torch
import torch.nn as nn

N_ENC, N_DEC = 24, 18  # encoder and decoder inputs (as in the earlier benchmark)
T_IN, T_OUT = 72, 48


class Rede(nn.Module):
    def __init__(self, camadas, H, atencao, dropout=0.2):
        super().__init__()
        self.atencao = atencao
        self.enc = nn.LSTM(N_ENC, H, num_layers=camadas, batch_first=True, bidirectional=True,
                           dropout=dropout if camadas > 1 else 0.0)
        self.ponte_h = nn.Linear(2 * H, H)
        self.ponte_c = nn.Linear(2 * H, H)
        self.W_k = nn.Linear(2 * H, H, bias=False)
        self.W_q = nn.Linear(H, H, bias=False)
        if atencao == "bahdanau":
            self.dec = nn.LSTMCell(N_DEC + 2 * H, H)
            self.v = nn.Linear(H, 1, bias=False)
        else:
            self.dec = nn.LSTM(N_DEC, H, batch_first=True)
        self.out = nn.Linear(3 * H, 3)
        self.drop = nn.Dropout(dropout)

    @staticmethod
    def quantis(raw):
        q50 = raw[..., 1]
        return torch.stack([q50 - nn.functional.softplus(raw[..., 0]), q50,
                            q50 + nn.functional.softplus(raw[..., 2])], -1)

    def forward(self, x_enc, x_dec):
        enc_out, (h, c) = self.enc(x_enc)
        h = torch.tanh(self.ponte_h(torch.cat([h[-2], h[-1]], -1)))  # last layer, 2 directions
        c = torch.tanh(self.ponte_c(torch.cat([c[-2], c[-1]], -1)))
        keys = self.W_k(enc_out)
        if self.atencao == "bahdanau":
            saidas = []
            for t in range(T_OUT):
                alpha = torch.softmax(self.v(torch.tanh(keys + self.W_q(h).unsqueeze(1))), dim=1)
                ctx = (alpha * enc_out).sum(1)
                h, c = self.dec(torch.cat([x_dec[:, t], ctx], -1), (h, c))
                saidas.append(self.out(self.drop(torch.cat([h, ctx], -1))))
            raw = torch.stack(saidas, 1)
        else:
            dec_out, _ = self.dec(x_dec, (h.unsqueeze(0), c.unsqueeze(0)))
            alpha = torch.softmax(self.W_q(dec_out) @ keys.transpose(1, 2), dim=-1)  # (B, 48, 72)
            ctx = alpha @ enc_out
            raw = self.out(self.drop(torch.cat([dec_out, ctx], -1)))
        return self.quantis(raw)  # (B, 48, 3)


def pinball(pred, y, q):
    e = y.unsqueeze(-1) - pred
    return torch.maximum(q * e, (q - 1) * e).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--camadas", type=int, default=1)
    ap.add_argument("--unidades", type=int, default=64)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--atencao", choices=["bahdanau", "luong"], default="bahdanau")
    ap.add_argument("--n", type=int, default=15, help="batches medidos (depois de 3 de aquecimento)")
    ap.add_argument("--rotulo", default="")
    a = ap.parse_args()

    torch.manual_seed(0)
    torch.set_num_threads(a.threads)
    dev = torch.device(a.device)
    sinc = torch.cuda.synchronize if dev.type == "cuda" else (lambda: None)
    modelo = Rede(a.camadas, a.unidades, a.atencao).to(dev)
    opt = torch.optim.AdamW(modelo.parameters(), lr=1e-3, weight_decay=1e-4)
    q = torch.tensor([0.1, 0.5, 0.9], device=dev).view(1, 1, 3)
    x_enc = torch.randn(a.batch, T_IN, N_ENC, device=dev)
    x_dec = torch.randn(a.batch, T_OUT, N_DEC, device=dev)
    y = torch.randn(a.batch, T_OUT, device=dev)

    def passo():
        opt.zero_grad()
        pinball(modelo(x_enc, x_dec), y, q).backward()
        opt.step()

    modelo.train()
    for _ in range(3):
        passo()
    sinc()
    t0 = time.perf_counter()
    for _ in range(a.n):
        passo()
    sinc()
    t_treino = (time.perf_counter() - t0) / a.n

    modelo.eval()
    with torch.no_grad():
        modelo(x_enc, x_dec)
        sinc()
        t0 = time.perf_counter()
        for _ in range(a.n):
            modelo(x_enc, x_dec)
        sinc()
        t_aval = (time.perf_counter() - t0) / a.n

    nome = torch.cuda.get_device_name(dev) if dev.type == "cuda" else platform.processor() or platform.machine()
    print(json.dumps({
        "rotulo": a.rotulo, "device": a.device, "nome": nome, "torch": torch.__version__,
        "threads": torch.get_num_threads(), "camadas": a.camadas, "unidades": a.unidades,
        "batch": a.batch, "atencao": a.atencao,
        "parametros": sum(p.numel() for p in modelo.parameters()),
        "treino_ms": round(t_treino * 1000, 1), "aval_ms": round(t_aval * 1000, 1), "n": a.n,
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
