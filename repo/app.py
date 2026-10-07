"""Local demo server for the trained FAENet MD17-aspirin model.

  python app.py [port]     # then open http://127.0.0.1:8765
"""
import json
import math
import re
import sys
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch

from faenet import FAENet, random_orthogonal

ROOT = Path(__file__).parent
DEV = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = next(p for p in [ROOT / "runs/aspirin_canon_grad_50k.pt", ROOT / "runs/aspirin_canon_grad.pt"] if p.exists())
# confidence meter partner: an independently trained model; their force disagreement flags unfamiliar geometry
PARTNER = next(p for p in [ROOT / "runs/al_random.pt", ROOT / "runs/aspirin_sfa.pt"] if p.exists())
SYM = {1: "H", 6: "C", 7: "N", 8: "O"}
MASS = {1: 1.008, 6: 12.011, 7: 14.007, 8: 15.999}
KB = 0.0019872  # kcal/mol/K
ACC = 4.184e-4  # (kcal/mol/A/amu) -> A/fs^2

ck = torch.load(CKPT, map_location=DEV)
model = FAENet(**ck["cfg"], frame=ck["frame"], forces=ck["forces"]).to(DEV).eval()
model.load_state_dict(ck["model"])
MU, SD = ck["mu"], ck["sd"]
pk = torch.load(PARTNER, map_location=DEV)
partner = FAENet(**pk["cfg"], frame="full" if pk["frame"] == "sfa" else pk["frame"], forces=pk["forces"]).to(DEV).eval()
partner.load_state_dict(pk["model"])
LOCK = threading.Lock()  # ponytail: one global model lock, fine for a single-user demo

d = np.load(ROOT / "data/md17_aspirin.npz")
Z = torch.from_numpy(d["z"].astype(np.int64)).to(DEV)
R_ALL, E_ALL, F_ALL = d["R"], d["E"][:, 0], d["F"]
torch.manual_seed(0)  # same permutation as train_lj.py -> first n_used were train/val, rest is unseen
HELD_OUT = torch.randperm(len(E_ALL))[ck.get("n_used", 2000):].numpy()
BATCH = torch.zeros(len(Z), dtype=torch.long, device=DEV)
MASSES = torch.tensor([MASS[int(a)] for a in Z], device=DEV)[:, None]
# Bond list from a reference aspirin geometry (covalent radii); "broken" = any bond stretched >1.4x.
# Catches lone atoms *and* whole fragments splitting off, which a nearest-neighbour test misses.
COV = torch.tensor([{1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66}[int(a)] for a in Z], device=DEV)
_ref = torch.tensor(R_ALL[HELD_OUT[0]], dtype=torch.float32, device=DEV)
BI, BJ = torch.triu_indices(len(Z), len(Z), 1, device=DEV)
_keep = (_ref[BI] - _ref[BJ]).norm(dim=-1) < 1.2 * (COV[BI] + COV[BJ])
BI, BJ = BI[_keep], BJ[_keep]
BL0 = (_ref[BI] - _ref[BJ]).norm(dim=-1)


def disagreement(pos, F):
    """Mean |F_model - F_partner| in kcal/mol/A."""
    with LOCK:
        Fp = partner(pos.clone(), Z, BATCH)[1].detach() * pk["sd"]
    return (F - Fp).abs().mean().item()


def predict(pos):
    """pos [N,3] float tensor on DEV -> (energy kcal/mol, forces [N,3] kcal/mol/A, ms)."""
    with LOCK:
        if DEV == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        E, F = model(pos.clone(), Z, BATCH)
        if DEV == "cuda":
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t) * 1000
    return E.item() * SD + MU, (F.detach() * SD), ms


def pack(pos, E, F, ms, **extra):
    return {"symbols": [SYM[int(a)] for a in Z], "pos": pos.tolist(), "E": E, "F": F.tolist(), "ms": ms, **extra}


def sample(_):
    i = int(np.random.choice(HELD_OUT))
    pos = torch.tensor(R_ALL[i], dtype=torch.float32, device=DEV)
    predict(pos)  # warm-up so the reported latency is steady-state
    E, F, ms = predict(pos)
    Ft = torch.tensor(F_ALL[i], device=DEV)
    return pack(pos, E, F, ms, idx=i, E_true=float(E_ALL[i]), F_true=F_ALL[i].tolist(),
                F_mae=(F - Ft).abs().mean().item())


def rotate(body):
    """Random rotation/reflection + translation: energy must not change, forces must rotate with the molecule."""
    pos = torch.tensor(body["pos"], device=DEV)
    E0, F0, _ = predict(pos)
    R = random_orthogonal(torch.float32).to(DEV)
    pos2 = pos @ R + torch.randn(3, device=DEV)
    E, F, ms = predict(pos2)
    extra = {}
    if "F_true" in body:  # true forces are equivariant too, so we can carry them along
        extra["F_true"] = (torch.tensor(body["F_true"], device=DEV) @ R).tolist()
        extra["E_true"] = body.get("E_true")
    return pack(pos2, E, F, ms, dE=abs(E - E0), dF=(F0 @ R - F).abs().max().item(), **extra)


def jiggle(body):
    pos = torch.tensor(body["pos"], device=DEV)
    pos = pos + float(body.get("amount", 0.05)) * torch.randn_like(pos)
    return pack(pos, *predict(pos))


def intact(pos):
    return ((pos[BI] - pos[BJ]).norm(dim=-1) / BL0).max().item() < 1.4


def md(body):
    """Velocity-Verlet NVE dynamics driven by the model's (energy-conserving) forces."""
    pos = torch.tensor(body["pos"], device=DEV)
    if not intact(pos):
        raise ValueError("this structure already has a broken bond; press New conformation")
    steps, dt, T = min(int(body.get("steps", 400)), 2000), float(body.get("dt", 0.5)), float(body.get("T", 300))
    v = torch.randn_like(pos) * torch.sqrt(KB * T / MASSES) * math.sqrt(ACC)  # A/fs
    v -= (MASSES * v).sum(0) / MASSES.sum()  # zero net momentum
    E, F, _ = predict(pos)
    frames, energies, broke_at = [], [], None
    t0 = time.perf_counter()
    for s in range(steps):
        v = v + 0.5 * dt * F / MASSES * ACC
        pos = pos + dt * v
        E, F, _ = predict(pos)
        v = v + 0.5 * dt * F / MASSES * ACC
        ke = 0.5 * (MASSES * v ** 2).sum().item() / ACC
        if not intact(pos):  # left the training distribution; drop this frame so the page keeps an intact molecule
            broke_at = s * dt
            break
        if s % 4 == 0:
            frames.append(pos.tolist())
            energies.append([s * dt, E, ke, disagreement(pos, F)])
    wall = time.perf_counter() - t0
    return {"symbols": [SYM[int(a)] for a in Z], "frames": frames, "energies": energies,
            "ms_per_step": wall / (s + 1) * 1000, "steps": s + 1, "dt": dt, "broke_at": broke_at}


def step(body):
    """Live MD for the grab-and-pull view: a few velocity-Verlet steps with a Berendsen thermostat and an
    optional spring pulling one atom toward the mouse. Client holds the state (pos, v); server is stateless."""
    pos = torch.tensor(body["pos"], device=DEV)
    if not intact(pos):
        raise ValueError("this structure already has a broken bond; press New conformation")
    T, dt, k = float(body.get("T", 300)), 0.5, float(body.get("k", 10.0))
    v = torch.tensor(body["v"], device=DEV) if body.get("v") else \
        torch.randn_like(pos) * torch.sqrt(KB * T / MASSES) * math.sqrt(ACC)
    pull = body.get("pull")

    def forces(p):
        E, F, _ = predict(p)
        if pull:  # spring on one atom (kcal/mol/A^2), reaction spread over all atoms: deforms/rotates, no drift
            F = F.clone()
            f = k * (torch.tensor(pull["target"], device=DEV) - p[pull["i"]])
            F[pull["i"]] += f
            F -= f / len(Z)
        return E, F

    E, F = forces(pos)
    broken, t0 = False, time.perf_counter()
    for _ in range(min(int(body.get("steps", 10)), 50)):
        v_half = v + 0.5 * dt * F / MASSES * ACC
        new = pos + dt * v_half
        if not intact(new):  # refuse the step; the page keeps the last intact geometry
            broken = True
            break
        pos = new
        E, F = forces(pos)
        v = v_half + 0.5 * dt * F / MASSES * ACC
        ke = 0.5 * (MASSES * v ** 2).sum() / ACC
        T_now = (2 * ke / (3 * len(Z) * KB)).item()
        v = v * math.sqrt(min(max(1 + dt / 50 * (T / max(T_now, 1) - 1), 0.8), 1.25))  # Berendsen, tau = 50 fs
    ms = (time.perf_counter() - t0) * 1000
    ke = (0.5 * (MASSES * v ** 2).sum() / ACC).item()
    return {"pos": pos.tolist(), "v": v.tolist(), "E": E, "T": 2 * ke / (3 * len(Z) * KB),
            "u": disagreement(pos, predict(pos)[1]), "broken": broken, "ms": ms}


def runs(_):
    out = {}
    for log in sorted((ROOT / "runs").glob("*.log")):
        rows = re.findall(r"epoch\s+(\d+).*?val E MAE ([\d.]+)\s+F MAE ([\d.]+)", log.read_text())
        if "hung" in log.name or not rows:
            continue
        frames = re.findall(r"^\s+(none|sfa|canon|soft|full)\s+([\d.]+)\s+([\d.]+)\s+([\de.+-]+)$", log.read_text(), re.M)
        out[log.stem] = {"curve": [[int(e), float(a), float(b)] for e, a, b in rows],
                         "frames": [[f, float(a), float(b), float(c)] for f, a, b, c in frames]}
    out["model"] = {"name": CKPT.stem, "partner": PARTNER.stem, "u50": U50, "u99": U99, "frame": ck["frame"], "forces": ck["forces"], "epoch": ck["epoch"], "val_E": ck["val_E"],
                    "val_F": ck["val_F"], "params": sum(p.numel() for p in model.parameters()), "device": DEV,
                    "gpu": torch.cuda.get_device_name() if DEV == "cuda" else "CPU", "held_out": len(HELD_OUT)}
    return out


_u = [disagreement(p, predict(p)[1]) for p in
      (torch.tensor(R_ALL[i], dtype=torch.float32, device=DEV) for i in HELD_OUT[1:129])]
U50, U99 = float(np.median(_u)), float(np.quantile(_u, 0.99))  # "normal" disagreement on unseen MD17 frames

API = {"/api/step": step, "/api/sample": sample, "/api/rotate": rotate, "/api/jiggle": jiggle, "/api/md": md, "/api/runs": runs}


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=str(ROOT / "web"), **k)

    def _api(self, body):
        try:
            res, code = API[self.path.split("?")[0]](body), 200
        except Exception as e:  # surface errors to the page instead of a dead socket
            res, code = {"error": str(e)}, 500
        data = json.dumps(res).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.split("?")[0] in API:
            return self._api({})
        super().do_GET()

    def do_POST(self):
        if self.path not in API:
            return self.send_error(404)
        n = int(self.headers.get("Content-Length", 0))
        if n > 1_000_000:
            return self.send_error(413)
        self._api(json.loads(self.rfile.read(n) or b"{}"))

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")  # always serve the latest page after edits
        super().end_headers()

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    print(f"FAENet demo on http://127.0.0.1:{port}  ({DEV}, {len(HELD_OUT)} held-out conformations)", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
