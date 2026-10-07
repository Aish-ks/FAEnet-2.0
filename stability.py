"""MD stability benchmark: how long does a trained model keep MD17 molecules intact?

Runs R replicas of NVE velocity-Verlet MD in one batch from unseen MD17 frames and reports
  - broken: replicas where any bond stretched > 1.4x its reference length
  - mean time to break (ps, censored at the run length)
  - total-energy drift and the energy kicks at canon-frame sign flips

  python stability.py runs/aspirin_canon_grad_50k.pt --T 300 500
"""
import argparse
import math
import time

import numpy as np
import torch

from faenet import FAENet, pca_frames

DEV = "cuda" if torch.cuda.is_available() else "cpu"
KB, ACC = 0.0019872, 4.184e-4  # kcal/mol/K ; (kcal/mol/A/amu) -> A/fs^2
MASS = {1: 1.008, 6: 12.011, 7: 14.007, 8: 15.999}
COV = {1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66}


def bond_list(Z, ref):
    """Covalent bonds of a reference geometry: (i, j, length)."""
    cov = torch.tensor([COV[int(a)] for a in Z], device=ref.device)
    i, j = torch.triu_indices(len(Z), len(Z), 1, device=ref.device)
    d = (ref[i] - ref[j]).norm(dim=-1)
    keep = d < 1.2 * (cov[i] + cov[j])
    return i[keep], j[keep], d[keep]


def load_model(path, frame=None):
    """Our checkpoints or the official baseline's ("kind": "official"). `path:frame` overrides the frame."""
    path, _, f = path.partition(":")
    frame = f or frame
    ck = torch.load(path, map_location=DEV)
    if ck.get("kind") == "official":
        import os
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "baseline"))
        from official import Official
        m = Official(**ck["cfg"]).to(DEV).eval()
        m.load_state_dict(ck["model"])
        m.frame = frame or ck["frame"]
        return m, ck
    m = FAENet(**ck["cfg"], frame=frame or ck["frame"], forces=ck["forces"]).to(DEV).eval()
    m.load_state_dict(ck["model"])
    return m, ck


def run(model, sd, R0, Z, T, steps, dt=0.5, seed=0):
    """R0 [R,n,3] start geometries. Returns dict of stability metrics."""
    R, n = R0.shape[:2]
    batch = torch.arange(R, device=DEV).repeat_interleave(n)
    z = Z.repeat(R)
    mass = torch.tensor([MASS[int(a)] for a in Z], device=DEV).repeat(R)[:, None]
    bi, bj, l0 = bond_list(Z, R0[0])
    torch.manual_seed(seed)
    pos = R0.reshape(-1, 3).clone()
    v = torch.randn_like(pos) * torch.sqrt(KB * T / mass) * math.sqrt(ACC)
    F = model(pos.clone(), z, batch)[1].detach() * sd
    broke = torch.full((R,), -1, device=DEV)
    U_prev = pca_frames(pos, batch, R, "canon")[2][:, 0]
    e_prev, e0, drift, kicks, flips = None, None, torch.zeros(R, device=DEV), [], 0
    t0 = time.time()
    for s in range(steps):
        v = v + 0.5 * dt * F / mass * ACC
        pos = pos + dt * v
        E, F = model(pos.clone(), z, batch)
        F = F.detach() * sd
        v = v + 0.5 * dt * F / mass * ACC
        alive = broke < 0
        P = pos.view(R, n, 3)
        stretch = ((P[:, bi] - P[:, bj]).norm(dim=-1) / l0).max(1).values
        broke = torch.where(alive & (stretch > 1.4), s, broke)
        etot = E.detach() * sd + torch.zeros(R, device=DEV).index_add(0, batch, 0.5 * (mass * v ** 2).sum(1) / ACC)
        if e0 is None:
            e0 = etot
        drift = torch.where(alive, torch.maximum(drift, (etot - e0).abs()), drift)
        U = pca_frames(pos, batch, R, "canon")[2][:, 0]
        flip = ((U * U_prev).sum(1) < 0).any(1) & alive  # canon would have flipped an axis this step
        if e_prev is not None and flip.any():
            kicks.append((etot - e_prev).abs()[flip])
            flips += int(flip.sum())
        U_prev, e_prev = U, etot
    ttb = torch.where(broke >= 0, broke, steps).float() * dt / 1000
    kick = torch.cat(kicks).median().item() if kicks else 0.0
    return {"T": T, "broken": int((broke >= 0).sum()), "R": R, "ps": steps * dt / 1000, "mean_ttb_ps": ttb.mean().item(),
            "drift_med": drift.median().item(), "flips": flips / R, "kick_med": kick, "wall_s": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", nargs="+")
    ap.add_argument("--data", default="data/md17_aspirin.npz")
    ap.add_argument("--T", type=float, nargs="+", default=[300, 500])
    ap.add_argument("--replicas", type=int, default=16)
    ap.add_argument("--ps", type=float, default=3.0)
    ap.add_argument("--frame", help="override the checkpoint's frame (e.g. soft)")
    a = ap.parse_args()
    d = np.load(a.data)
    Z = torch.from_numpy(d["z"].astype(np.int64)).to(DEV)
    torch.manual_seed(0)
    start = torch.randperm(len(d["E"]))[-a.replicas:].numpy()  # tail of the train_lj permutation: never trained on
    R0 = torch.tensor(d["R"][start], dtype=torch.float32, device=DEV)
    print(f"{'model':38s} {'T':>5s} {'broken':>8s} {'mean t-break':>13s} {'|dEtot| med':>12s} {'flips':>6s} {'kick med':>9s}")
    for path in a.ckpt:
        model, ck = load_model(path, a.frame)
        name = f"{path.partition(':')[0].split('/')[-1]} [{model.frame}]"
        for T in a.T:
            r = run(model, ck["sd"], R0, Z, T, int(a.ps * 2000))
            print(f"{name:38s} {T:5.0f} {r['broken']:3d}/{r['R']:<4d} {r['mean_ttb_ps']:9.2f} ps {r['drift_med']:9.2f}    "
                  f"{r['flips']:6.1f} {r['kick_med']:9.3f}   ({r['wall_s']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
