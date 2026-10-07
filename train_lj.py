"""Train FAENet on synthetic 2-species Lennard-Jones clusters or MD17 (energy + forces).

  python train_lj.py --frame sfa --forces direct      # paper setup
  python train_lj.py --frame canon --forces grad      # upgrades: 1-frame exact invariance + conservative forces
  python train_lj.py --data data/md17_aspirin.npz --epochs 300   # MD17 (paper Sec. 5), 1000 train samples
"""
import argparse
import time

import numpy as np
import torch

from faenet import FAENet, symmetry_error

DEV = "cuda" if torch.cuda.is_available() else "cpu"
EPS = torch.tensor([[1.0, 1.5], [1.5, 0.7]])  # pair well depths by species


def lj_cluster(n):
    # jittered cubic-lattice sites -> no overlapping atoms, so LJ stays finite
    grid = torch.stack(torch.meshgrid(*[torch.arange(3.0)] * 3, indexing="ij"), -1).view(-1, 3)
    pos = (grid[torch.randperm(27)[:n]] * 1.2 + 0.05 * torch.randn(n, 3)).requires_grad_()
    s = torch.randint(0, 2, (n,))
    d = torch.cdist(pos, pos) + torch.eye(n)
    E = (4 * EPS[s][:, s] * (d ** -12 - d ** -6)).triu(1).sum()
    F = -torch.autograd.grad(E, pos)[0]
    return pos.detach(), s + 1, E.detach(), F


def load_md17(path, n, train_idx=None):
    """First n of a seed-0 permutation, or train_idx + the 1000 val frames that a 1k run uses (perm[1000:2000])."""
    # ponytail: original MD17 npz (quantum-machine.org); units kcal/mol and kcal/mol/A
    d = np.load(path)
    perm = torch.randperm(len(d["E"])).numpy()
    idx = perm[:n] if train_idx is None else np.concatenate([train_idx, perm[1000:2000]])
    z = torch.from_numpy(d["z"].astype(np.int64))
    R, E, F = (torch.from_numpy(d[k][idx]).float() for k in ("R", "E", "F"))
    return [(R[i], z, E[i, 0], F[i]) for i in range(len(idx))]


def collate(items):
    pos, z, E, F = zip(*items)
    batch = torch.repeat_interleave(torch.arange(len(pos)), torch.tensor([len(p) for p in pos]))
    return tuple(x.to(DEV) for x in (torch.cat(pos), torch.cat(z), batch, torch.stack(E), torch.cat(F)))


def evaluate(model, data, mu, sd, frame):
    model.eval()
    model.frame, train_frame = frame, model.frame
    eE = eF = 0.0
    for b in data:
        pos, z, batch, E, F = b
        pE, pF = model(pos, z, batch)
        eE += (pE.detach() * sd + mu - E).abs().sum().item()
        eF += (pF.detach() * sd - F).abs().mean().item() * len(E)
    model.frame = train_frame
    n = sum(len(b[3]) for b in data)
    return eE / n, eF / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", default="sfa", choices=["sfa", "full", "canon", "soft", "none"])
    ap.add_argument("--forces", default="direct", choices=["direct", "grad"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--n_train", type=int, default=2000)
    ap.add_argument("--data", default="lj", help="'lj' or path to an MD17 .npz")
    ap.add_argument("--out", default="faenet.pt")
    ap.add_argument("--resume", help="checkpoint to continue from (same data/flags)")
    ap.add_argument("--init", help="start from these weights (and their mu/sd) with a fresh schedule, e.g. to fine-tune")
    ap.add_argument("--indices", help=".npy of MD17 train indices (val = the 1k run's val set)")
    ap.add_argument("--lr", type=float, default=1e-3)
    a = ap.parse_args()
    torch.manual_seed(0)
    torch.set_num_threads(4)  # leave the rest of the CPU for the user

    md17 = a.data != "lj"
    if a.indices:
        idx = np.load(a.indices)
        a.n_train = len(idx)
        data = load_md17(a.data, None, idx)
    elif md17:
        data = load_md17(a.data, a.n_train + 1000)
    else:
        data = [lj_cluster(int(torch.randint(6, 14, ()))) for _ in range(a.n_train + 300)]
    train, val = data[:a.n_train], data[a.n_train:]
    Es = torch.stack([d[2] for d in train])
    mu, sd = Es.mean().to(DEV), Es.std().to(DEV)  # forces share the energy scale sd
    val_b = [collate(val[i:i + 64]) for i in range(0, len(val), 64)]

    cfg = dict(hidden=128, layers=4, cutoff=5.0, max_z=10) if md17 else dict(hidden=64, layers=3, cutoff=3.0, max_z=3)
    model = FAENet(**cfg, frame=a.frame, forces=a.forces).to(DEV)
    best, start = float("inf"), 0
    if a.init:
        ck = torch.load(a.init, map_location=DEV)
        # strict=False: a direct-force checkpoint has a force head that a --forces grad model drops
        skipped = model.load_state_dict(ck["model"], strict=False)
        mu, sd = torch.tensor(ck["mu"], device=DEV), torch.tensor(ck["sd"], device=DEV)  # keep the output scale it learned
        print(f"initialised from {a.init} (skipped {skipped.unexpected_keys}, new {skipped.missing_keys})", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    if a.resume:
        ck = torch.load(a.resume, map_location=DEV)
        model.load_state_dict(ck["model"])
        if "opt" in ck:  # older checkpoints have no optimizer state; Adam moments re-warm in a few steps
            opt.load_state_dict(ck["opt"])
        best, start = ck["val_F"], ck["epoch"]
        for _ in range(start):
            sched.step()  # fast-forward the LR schedule
        print(f"resumed from {a.resume} at epoch {start} (val F MAE {best:.3f})", flush=True)
    print(f"device: {DEV}")
    print(f"baseline (predict mean) energy MAE: {(Es - Es.mean()).abs().mean():.3f}")

    gstep = 0
    for ep in range(start, a.epochs):
        model.train()
        t0 = time.time()
        for i in torch.randperm(len(train)).split(32):
            pos, z, batch, E, F = collate([train[j] for j in i])
            pE, pF = model(pos, z, batch)
            loss = (pE - (E - mu) / sd).abs().mean() + 10 * (pF - F / sd).abs().mean()
            if a.init and gstep < 300:  # warm up: a fresh Adam's first steps knock a trained model's energy off
                for g in opt.param_groups:
                    g["lr"] = sched.get_last_lr()[0] * (gstep + 1) / 300
            gstep += 1
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
        sched.step()
        if ep % 5 == 4 or ep == a.epochs - 1:
            eE, eF = evaluate(model, val_b, mu, sd, a.frame)
            print(f"epoch {ep + 1:3d}  loss {loss.item():.3f}  val E MAE {eE:.3f}  F MAE {eF:.3f}  ({time.time() - t0:.1f}s/epoch)", flush=True)
            if eF < best:  # keep the best-on-val checkpoint; mu/sd are needed to un-normalise predictions
                best = eF
                torch.save({"model": model.state_dict(), "cfg": cfg, "frame": a.frame, "forces": a.forces,
                            "mu": mu.item(), "sd": sd.item(), "epoch": ep + 1, "n_used": len(data), "opt": opt.state_dict(), "val_E": eE, "val_F": eF}, a.out)

    print("\nsame weights, different inference frames (E MAE, F MAE, |dE| under random E(3) move):")
    for frame in ["none", "sfa", "canon", "soft", "full"]:
        eE, eF = evaluate(model, val_b, mu, sd, frame)
        model.frame, tf = frame, model.frame
        model.eval()
        dE, _ = symmetry_error(model, *val_b[0][:3])
        model.frame = tf
        print(f"  {frame:5s}  {eE:.3f}  {eF:.3f}  {dE * sd:.1e}")


if __name__ == "__main__":
    main()
