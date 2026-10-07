"""Active learning on the MD17 pool: pick the frames two independently trained models disagree on most.

  python active.py runs/aspirin_canon_grad.pt runs/aspirin_sfa.pt --add 4000
  -> runs/al_active.npy, runs/al_random.npy   (seed 1k frames + 4000 chosen / random)
  python train_lj.py --data data/md17_aspirin.npz --indices runs/al_active.npy ...
  python active.py --eval runs/al_active.pt runs/al_random.pt   # MAE on 2000 never-used frames

Uncertainty = mean |F_a - F_b| over atoms (ensemble disagreement). We tested the "free" alternative, energy spread
over the 8 PCA frames of one model: rank correlation with true force error was -0.05 vs +0.35 for the ensemble.
"""
import argparse

import numpy as np
import torch

from stability import DEV, load_model


def disagreement(ma, sda, mb, sdb, R, Z, chunk=64):
    n, out = len(Z), []
    for i in range(0, len(R), chunk):
        P = torch.tensor(R[i:i + chunk], dtype=torch.float32, device=DEV)
        C = len(P)
        b = torch.arange(C, device=DEV).repeat_interleave(n)
        z = Z.repeat(C)
        p = P.reshape(-1, 3)
        Fa = ma(p.clone(), z, b)[1].detach() * sda
        Fb = mb(p.clone(), z, b)[1].detach() * sdb
        out.append((Fa - Fb).abs().view(C, n, 3).mean((1, 2)).cpu())
    return torch.cat(out).numpy()


def test_mae(path, d, idx, Z, chunk=64):
    m, ck = load_model(path)
    n, eE, eF = len(Z), 0.0, 0.0
    for i in range(0, len(idx), chunk):
        j = idx[i:i + chunk]
        C = len(j)
        b = torch.arange(C, device=DEV).repeat_interleave(n)
        E, F = m(torch.tensor(d["R"][j], dtype=torch.float32, device=DEV).reshape(-1, 3), Z.repeat(C), b)
        eE += (E.detach().double() * ck["sd"] + ck["mu"] - torch.tensor(d["E"][j, 0], device=DEV)).abs().sum().item()
        eF += (F.detach() * ck["sd"] - torch.tensor(d["F"][j], device=DEV).reshape(-1, 3)).abs().mean().item() * C
    return eE / len(idx), eF / len(idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", nargs="+", help="two models to score the pool with, or with --eval any models to test")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--data", default="data/md17_aspirin.npz")
    ap.add_argument("--pool", type=int, default=30000)
    ap.add_argument("--add", type=int, default=4000)
    a = ap.parse_args()
    d = np.load(a.data)
    Z = torch.from_numpy(d["z"].astype(np.int64)).to(DEV)
    torch.manual_seed(0)
    perm = torch.randperm(len(d["E"])).numpy()  # same permutation as train_lj.py
    if a.eval:
        test = perm[-2016:-16]  # never in any train/val/pool; the last 16 are stability.py's starts
        for path in a.ckpt:
            print(f"{path:40s} test E MAE {{:.3f}}  F MAE {{:.3f}}".format(*test_mae(path, d, test, Z)), flush=True)
        return
    seed = perm[:1000]  # what the 1k models trained on; perm[1000:2000] is val
    # pool from 2000..150000 only: the tail of perm is kept for stability.py and test sets
    rng = np.random.default_rng(0)
    pool = rng.choice(perm[2000:150000], a.pool, replace=False)
    ma, cka = load_model(a.ckpt[0])
    mb, ckb = load_model(a.ckpt[1], "full" if torch.load(a.ckpt[1])["frame"] == "sfa" else None)
    u = disagreement(ma, cka["sd"], mb, ckb["sd"], d["R"][pool], Z)
    top = pool[np.argsort(-u)[:a.add]]
    rand = rng.choice(pool, a.add, replace=False)
    np.save("runs/al_active.npy", np.concatenate([seed, top]))
    np.save("runs/al_random.npy", np.concatenate([seed, rand]))
    print(f"pool {a.pool}: disagreement median {np.median(u):.2f}, chosen median {np.median(np.sort(u)[-a.add:]):.2f} kcal/mol/A")
    print("wrote runs/al_active.npy and runs/al_random.npy")


if __name__ == "__main__":
    main()
