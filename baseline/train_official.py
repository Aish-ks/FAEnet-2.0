"""Train the authors' FAENet (official package) on exactly our MD17 data split, with their published recipe.

  python baseline/train_official.py --n_train 1000 --epochs 500 --out runs/official_1k.pt

Recipe = paper Table 7, QM7-X column (molecules, energy + forces):
  AdamW lr 2e-4, batch 100, ReduceLROnPlateau, linear warmup, loss = 1 * E_mae + 100 * F_l2mae + 15 * EC_l2mae
  where EC (paper Eq. 9) pulls the direct force head toward the energy gradient; SFA frames during training.
Same as ours: data split (seed-0 permutation), energy normalisation (mu/sd), epoch budget, evaluation code.
Deviation: warmup = min(3000 steps, 5% of all steps), since the paper's 3000 assumes 1.4M steps.
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from official import PAPER_QM7X, Official  # noqa: E402
from train_lj import DEV, collate, evaluate, load_md17  # noqa: E402


def l2mae(a, b):
    return (a - b).norm(dim=-1).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/md17_aspirin.npz")
    ap.add_argument("--n_train", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--accum", type=int, default=4, help="split each batch into this many micro-batches (6GB GPU)")
    ap.add_argument("--out", default="runs/official_1k.pt")
    ap.add_argument("--resume")
    a = ap.parse_args()
    torch.manual_seed(0)
    torch.set_num_threads(4)

    data = load_md17(a.data, a.n_train + 1000)  # same frames as our runs: train perm[:n], val perm[n:n+1000]
    train, val = data[:a.n_train], data[a.n_train:]
    Es = torch.stack([d[2] for d in train])
    mu, sd = Es.mean().to(DEV), Es.std().to(DEV)
    val_b = [collate(val[i:i + 64]) for i in range(0, len(val), 64)]

    model = Official(**PAPER_QM7X).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
    plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=10)
    steps_per_epoch = -(-a.n_train // a.batch)
    warmup = min(3000, int(0.05 * steps_per_epoch * a.epochs))
    best, start, gstep = float("inf"), 0, 0
    if a.resume:
        ck = torch.load(a.resume, map_location=DEV)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        plateau.load_state_dict(ck["plateau"])
        best, start = ck["val_F"], ck["epoch"]
        gstep = start * steps_per_epoch
        print(f"resumed from {a.resume} at epoch {start}", flush=True)
    print(f"official FAENet: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params, "
          f"{steps_per_epoch} steps/epoch, warmup {warmup} steps", flush=True)

    for ep in range(start, a.epochs):
        model.train()
        model.frame = "sfa"
        t0 = time.time()
        for i in torch.randperm(len(train)).split(a.batch):
            if gstep < warmup:
                for g in opt.param_groups:
                    g["lr"] = 2e-4 * (gstep + 1) / warmup
            gstep += 1
            opt.zero_grad()
            # gradient accumulation: same update as one batch of `batch`, in micro-batches that fit 6GB
            for k in torch.arange(len(i)).tensor_split(a.accum):
                items = [train[j] for j in i[k]]
                pos, z, batch, E, F = collate(items)
                pE, pF = model(pos, z, batch)
                grad_target = model.last["forces_grad_target"]
                loss = (pE - (E - mu) / sd).abs().mean() + 100 * l2mae(pF, F / sd) + 15 * l2mae(pF, grad_target)
                (loss * len(items) / len(i)).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
        eE, eF = evaluate(model, val_b, mu, sd, "sfa")
        if gstep >= warmup:
            plateau.step(eF)
        if ep % 5 == 4 or ep == a.epochs - 1:
            print(f"epoch {ep + 1:3d}  loss {loss.item():.3f}  val E MAE {eE:.3f}  F MAE {eF:.3f}  "
                  f"lr {opt.param_groups[0]['lr']:.1e}  ({time.time() - t0:.1f}s/epoch)", flush=True)
        if eF < best:
            best = eF
            torch.save({"kind": "official", "cfg": PAPER_QM7X, "frame": "sfa", "forces": "direct+EC",
                        "model": model.state_dict(), "opt": opt.state_dict(), "plateau": plateau.state_dict(),
                        "mu": mu.item(), "sd": sd.item(), "epoch": ep + 1, "n_used": len(data),
                        "val_E": eE, "val_F": eF}, a.out)

    print("\nsame weights, different inference frames (E MAE, F MAE):")
    for frame in ["sfa", "full", "det"]:
        eE, eF = evaluate(model, val_b, mu, sd, frame)
        print(f"  {frame:5s}  {eE:.3f}  {eF:.3f}")


if __name__ == "__main__":
    main()
