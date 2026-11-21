"""Bi-LSTM encoder–decoder with temporal attention and quantile output (T24, doc 04 §3.5).

- Encoder: Bi-LSTM over the 72 h up to t (bidirectional only inside the window, which ends at t).
- Decoder: LSTM cell stepping t + 1 … t + 48; at each hour it receives the future inputs of that hour
  (calendar, and ERA5 in the "era5" variant) and the attention context (Bahdanau et al., 2015) over the
  encoder states. It does not receive its own previous prediction.
- Output: Q10, Q50, Q90 per hour, built as Q50 ± softplus so that Q10 <= Q50 <= Q90.
- Loss: pinball on log(1 + max(y, 0)), mean over the 3 quantiles and the available hours (missing
  target hours are masked, never interpolated).
- Initial hyperparameters of doc 04 §3.6 (fixed in the first delivery): 1 Bi-LSTM layer, 64 units per
  direction, decoder 64, dropout 0.2, AdamW lr 1e-3, weight decay 1e-4, batch 256, up to 100 epochs,
  early stopping with patience 10.
- Device: GPU (CUDA) when available, otherwise CPU. The batch permutation is drawn on the CPU with the
  seed, so the batch order is the same on both; floating-point results are not bit-identical across
  devices (doc 04 §7).
"""
import time

import numpy as np
import torch
from torch import nn

QUANTIS = torch.tensor([0.1, 0.5, 0.9])
DISPOSITIVO = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HP = dict(unidades_enc=64, unidades_dec=64, camadas=1, dropout=0.2, lr=1e-3, weight_decay=1e-4,
          batch=256, epocas_max=100, paciencia=10)


class Atencao(nn.Module):
    def __init__(self, d_enc, d_dec, d_att=64):
        super().__init__()
        self.We = nn.Linear(d_enc, d_att, bias=False)
        self.Wd = nn.Linear(d_dec, d_att)
        self.v = nn.Linear(d_att, 1, bias=False)

    def forward(self, H, He, s):
        # H: [B, L, d_enc], He = We(H) precomputed, s: [B, d_dec]
        e = self.v(torch.tanh(He + self.Wd(s)[:, None, :])).squeeze(-1)  # [B, L]
        a = torch.softmax(e, dim=1)
        return torch.bmm(a[:, None, :], H).squeeze(1), a


class BiLSTMQuantis(nn.Module):
    def __init__(self, f_enc, f_dec, hp=HP, horizonte=48):
        super().__init__()
        ue, ud = hp["unidades_enc"], hp["unidades_dec"]
        self.horizonte = horizonte
        self.enc = nn.LSTM(f_enc, ue, num_layers=hp["camadas"], batch_first=True, bidirectional=True,
                           dropout=hp["dropout"] if hp["camadas"] > 1 else 0.0)
        self.ini_h = nn.Linear(2 * ue, ud)
        self.ini_c = nn.Linear(2 * ue, ud)
        self.att = Atencao(2 * ue, ud)
        self.dec = nn.LSTMCell(f_dec + 2 * ue, ud)
        self.drop = nn.Dropout(hp["dropout"])
        self.saida = nn.Linear(ud + 2 * ue, 3)

    def forward(self, x_enc, x_dec, devolver_atencao=False):
        H, (hn, cn) = self.enc(x_enc)
        H = self.drop(H)
        fim = torch.cat([hn[-2], hn[-1]], dim=1)
        fimc = torch.cat([cn[-2], cn[-1]], dim=1)
        s, c = torch.tanh(self.ini_h(fim)), torch.tanh(self.ini_c(fimc))
        He = self.att.We(H)
        saidas, pesos = [], []
        for k in range(self.horizonte):
            ctx, a = self.att(H, He, s)
            s, c = self.dec(torch.cat([x_dec[:, k], ctx], dim=1), (s, c))
            o = self.saida(self.drop(torch.cat([s, ctx], dim=1)))
            q50 = o[:, 1]
            q10 = q50 - nn.functional.softplus(o[:, 0])
            q90 = q50 + nn.functional.softplus(o[:, 2])
            saidas.append(torch.stack([q10, q50, q90], dim=1))
            if devolver_atencao:
                pesos.append(a)
        out = torch.stack(saidas, dim=1)  # [B, 48, 3]
        return (out, torch.stack(pesos, dim=1)) if devolver_atencao else out


def pinball(pred, y, mascara):
    """pred [B, 48, 3], y [B, 48] (log), mascara [B, 48]: mean over quantiles and valid hours."""
    q = QUANTIS.to(pred.device)
    e = y[..., None] - pred
    l = torch.maximum(q * e, (q - 1) * e)  # [B, 48, 3]
    l = l.mean(-1) * mascara
    return l.sum() / mascara.sum().clamp(min=1)


def janelas(serie, idx, janela=72, horizonte=48):
    """Encoder [N, 72, F_enc], decoder [N, 48, F_dec], target log [N, 48], mask [N, 48]."""
    off_e = np.arange(-janela + 1, 1)
    off_d = np.arange(1, horizonte + 1)
    xe = serie.enc[idx[:, None] + off_e[None, :]]
    xd = serie.dec[idx[:, None] + off_d[None, :]]
    y = serie.y[idx[:, None] + off_d[None, :]]
    m = ~np.isnan(y)
    yl = np.log1p(np.maximum(np.nan_to_num(y), 0.0))
    return xe, xd, yl.astype(np.float32), m.astype(np.float32)


class Janelas:
    """Lazy windows over the concatenated hourly arrays of the sensors (built batch by batch)."""

    def __init__(self, conj, idx_por_sensor, janela=72, horizonte=48, dispositivo=None):
        dev = dispositivo or DISPOSITIVO
        enc, dec, y, glob, desl = [], [], [], [], 0
        for s, idx in idx_por_sensor.items():
            se = conj.series[s]
            enc.append(se.enc); dec.append(se.dec); y.append(se.y)
            glob.append(np.asarray(idx) + desl)
            desl += len(se.y)
        self.enc = torch.from_numpy(np.concatenate(enc).astype(np.float32)).to(dev)
        self.dec = torch.from_numpy(np.concatenate(dec).astype(np.float32)).to(dev)
        yy = np.concatenate(y)
        self.m = torch.from_numpy((~np.isnan(yy)).astype(np.float32)).to(dev)
        self.y = torch.from_numpy(np.log1p(np.maximum(np.nan_to_num(yy), 0.0)).astype(np.float32)).to(dev)
        self.idx = torch.from_numpy(np.concatenate(glob) if glob else np.zeros(0, int)).long().to(dev)
        self.off_e = torch.arange(-janela + 1, 1, device=dev)
        self.off_d = torch.arange(1, horizonte + 1, device=dev)

    def __len__(self):
        return len(self.idx)

    def lote(self, sel):
        i = self.idx[sel]
        je = i[:, None] + self.off_e[None, :]
        jd = i[:, None] + self.off_d[None, :]
        return self.enc[je], self.dec[jd], self.y[jd], self.m[jd]


def _perda_conjunto(modelo, dados, batch=1024):
    modelo.eval()
    tot, n = 0.0, 0.0
    with torch.no_grad():
        for k in range(0, len(dados), batch):
            xe, xd, y, m = dados.lote(torch.arange(k, min(k + batch, len(dados)), device=dados.idx.device))
            tot += float(pinball(modelo(xe, xd), y, m)) * float(m.sum())
            n += float(m.sum())
    return tot / max(n, 1)


def treinar(conj, idx_treino, idx_aval=None, semente=0, epocas=None, hp=HP, log=print):
    """Trains with early stopping on idx_aval (folds) or for a fixed number of epochs (re-training).

    Returns (model, best epoch, history).
    """
    torch.manual_seed(semente)
    torch.cuda.manual_seed_all(semente)
    np.random.seed(semente)
    tr = Janelas(conj, idx_treino)
    av = Janelas(conj, idx_aval) if idx_aval is not None else None
    modelo = BiLSTMQuantis(tr.enc.shape[-1], tr.dec.shape[-1], hp).to(DISPOSITIVO)
    opt = torch.optim.AdamW(modelo.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    g = torch.Generator().manual_seed(semente)
    n = len(tr)
    melhor, melhor_ep, estado, hist, sem_melhora = np.inf, 0, None, [], 0
    n_ep = epocas if epocas is not None else hp["epocas_max"]
    for ep in range(1, n_ep + 1):
        t0 = time.time()
        modelo.train()
        perm = torch.randperm(n, generator=g).to(DISPOSITIVO)
        tot = 0.0
        for k in range(0, n, hp["batch"]):
            b = perm[k:k + hp["batch"]]
            xe, xd, y, m = tr.lote(b)
            loss = pinball(modelo(xe, xd), y, m)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(b)
        perda_tr = tot / n
        perda_av = _perda_conjunto(modelo, av) if av is not None else np.nan
        hist.append((ep, perda_tr, perda_av, time.time() - t0))
        log(f"  época {ep}: treino {perda_tr:.4f} aval {perda_av:.4f} ({time.time() - t0:.0f} s)")
        if av is not None:
            if perda_av < melhor - 1e-5:
                melhor, melhor_ep, sem_melhora = perda_av, ep, 0
                estado = {k: v.detach().clone() for k, v in modelo.state_dict().items()}
            else:
                sem_melhora += 1
                if sem_melhora >= hp["paciencia"]:
                    break
    if estado is not None:
        modelo.load_state_dict(estado)
    else:
        melhor_ep = n_ep
    return modelo, melhor_ep, hist


def prever(modelo, conj, idx_por_sensor, batch=1024):
    """dict sensor -> [N, 48, 3] quantiles in µg/m³."""
    modelo.eval()
    out = {}
    with torch.no_grad():
        for s, idx in idx_por_sensor.items():
            if not len(idx):
                continue
            xe, xd, _, _ = janelas(conj.series[s], idx)
            dev = next(modelo.parameters()).device
            xe = torch.from_numpy(xe.astype(np.float32)).to(dev)
            xd = torch.from_numpy(xd.astype(np.float32)).to(dev)
            q = torch.cat([modelo(xe[k:k + batch], xd[k:k + batch]) for k in range(0, len(xe), batch)])
            out[s] = np.expm1(q.cpu().numpy())
    return out
