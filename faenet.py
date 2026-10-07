"""FAENet (Duval et al., ICML 2023, arXiv:2305.05577) in plain PyTorch, plus three upgrades.

Paper-faithful parts:
  - PCA frames F(X) = {(U, t) | U = [±u1, ±u2, ±u3]}, 8 frames for E(3), 4 for SE(3)   (Eq. 2)
  - Full FA (average over all frames) and Stochastic FA (one random frame per pass)      (Eq. 1, 5)
  - Edge embedding e_ij = σ(MLP(r_ij || RBF(d_ij)))                                     (Eq. 7)
  - Interaction h_i += MLP(Σ_j h_j ⊙ σ(W[e_ij || h_i || h_j]))                           (Eq. 8)
  - Jumping connections, weighted-sum energy head, direct force head rotated back by U^T

Upgrades over the paper (all optional flags):
  1. frame="canon": fix eigenvector signs with the third moment (skewness) along each axis ->
     ONE frame, exactly invariant for generic structures, 8x cheaper than Full FA at inference.
  2. forces="grad": F = -dE/dx through the frame itself -> energy-conserving forces
     (the paper uses a direct head and only nudges it with a fine-tuning loss, Eq. 9).
  3. Smooth cosine cutoff envelope on messages -> energy is continuous when atoms cross the
     cutoff (paper uses a hard adjacency, Eq. 6), needed for stable MD with grad forces.
  4. frame="soft": weighted frames. Each of the 8 sign choices gets weight prod_k sigmoid(s_k * skew_k / tau),
     so when an axis' skewness passes through 0 the output blends between the two signs instead of
     jumping (canon's sign flip kicks ~0.6 kcal/mol into MD). Frames with weight < 1e-4 are pruned,
     so the cost is ~1 frame except near ambiguous axes.
"""
import itertools
import math

import torch
from torch import nn

SIGNS = torch.tensor(list(itertools.product([1.0, -1.0], repeat=3)))  # 8 x 3


def radius_graph(pos, batch, cutoff):
    # ponytail: dense O(N^2) pairs, fine for molecules/small batches; swap in a cell list for >10k atoms
    d = torch.cdist(pos, pos)
    m = (batch[:, None] == batch[None]) & (d < cutoff)
    m.fill_diagonal_(False)
    dst, src = m.nonzero().T  # message src(j) -> dst(i)
    return src, dst


def seg_sum(x, index, size):
    return torch.zeros(size, *x.shape[1:], dtype=x.dtype, device=x.device).index_add(0, index, x)


def pca_frames(pos, batch, B, mode="full", se3=False, tau=0.05):
    """Returns centroid t [B,3], centered X [N,3], frames U [B,F,3,3] (columns = axes), weights W [B,F]."""
    n = seg_sum(torch.ones_like(pos[:, 0]), batch, B)
    t = seg_sum(pos, batch, B) / n[:, None]
    X = pos - t[batch]
    one = pos.new_ones(B, 1)
    if mode == "none":  # No-FA ablation: raw coordinates
        return torch.zeros_like(t), pos, torch.eye(3, dtype=pos.dtype, device=pos.device).expand(B, 1, 3, 3), one
    cov = seg_sum(X[:, :, None] * X[:, None, :], batch, B) / n[:, None, None]
    lam, V = torch.linalg.eigh(cov)
    lam, V = lam.flip(-1), V.flip(-1)  # descending eigenvalues

    if mode == "canon":
        # ponytail: skewness sign is discontinuous where a moment ~ 0 (symmetric molecules);
        # upgrade path is falling back to Full FA over the ambiguous axes only.
        m3 = seg_sum(torch.einsum("ni,nij->nj", X, V[batch]) ** 3, batch, B)
        s = torch.where(m3 >= 0, 1.0, -1.0).to(pos.dtype)
        U = V * s[:, None, :]
        if se3:  # proper rotation: third axis is forced by the first two
            U = U * torch.stack([torch.ones_like(s[:, 0])] * 2 + [torch.linalg.det(U).sign()], -1)[:, None, :]
        return t, X, U[:, None], one

    U = V[:, None] * SIGNS.to(pos)[None, :, None, :]  # B,8,3,3
    if mode == "soft":
        assert not se3  # ponytail: E(3) only; SE(3) would tie the 3rd sign to det
        skew = seg_sum(torch.einsum("ni,nij->nj", X, V[batch]) ** 3, batch, B) / n[:, None] / lam.clamp_min(1e-6) ** 1.5
        return t, X, U, torch.sigmoid(SIGNS.to(pos)[None] * skew[:, None] / tau).prod(-1)
    if se3:  # keep the 4 frames with det = +1 (exactly 4 per graph, order preserved)
        U = U[torch.linalg.det(U) > 0].view(B, 4, 3, 3)
    if mode == "sfa":
        return t, X, U[torch.arange(B), torch.randint(U.shape[1], (B,), device=pos.device)][:, None], one
    return t, X, U, pos.new_full((B, U.shape[1]), 1 / U.shape[1])


def mlp(*dims, last_act=True):
    layers = []
    for i, (a, b) in enumerate(zip(dims, dims[1:])):
        layers.append(nn.Linear(a, b))
        if last_act or i < len(dims) - 2:
            layers.append(nn.SiLU())
    return nn.Sequential(*layers)


class FAENet(nn.Module):
    def __init__(self, hidden=128, layers=4, rbf=32, cutoff=5.0, max_z=100,
                 frame="sfa", forces="direct", se3=False, smooth_cutoff=True, tau=0.05):
        super().__init__()
        self.tau = tau
        assert frame in ("sfa", "full", "canon", "soft", "none") and forces in ("direct", "grad", None)
        self.cutoff, self.frame, self.forces, self.se3, self.smooth = cutoff, frame, forces, se3, smooth_cutoff
        H = hidden
        # ponytail: Z embedding only; the paper also concatenates period/group/physics-table embeddings
        # (Appendix B.1). Add them if you train on OC20 where they help.
        self.emb = nn.Sequential(nn.Embedding(max_z, H), mlp(H, H, H))
        self.register_buffer("mu", torch.linspace(0, cutoff, rbf))
        self.gamma = 0.5 / (cutoff / (rbf - 1)) ** 2
        self.edge = mlp(3 + rbf, H, H)
        self.filt = nn.ModuleList(mlp(3 * H, H) for _ in range(layers))
        self.upd = nn.ModuleList(mlp(H, H, H) for _ in range(layers))
        # ponytail: LayerNorm instead of GraphNorm (Cai et al.); swap if deep models get unstable
        self.norm = nn.ModuleList(nn.LayerNorm(H) for _ in range(layers))
        self.jump = nn.Linear(H * (layers + 1), H)
        self.energy = mlp(H, H, 1, last_act=False)
        self.alpha = nn.Linear(H, 1)
        self.force = mlp(H, H, H, 3, last_act=False) if forces == "direct" else None

    def phi(self, p, z, src, dst, batch, B):
        """Backbone on canonical coordinates p. Returns energy [B], canonical-frame forces [N,3] or None."""
        r = p[dst] - p[src]
        d = r.norm(dim=-1, keepdim=True)
        e = self.edge(torch.cat([r, torch.exp(-self.gamma * (d - self.mu) ** 2)], -1))
        if self.smooth:
            e = e * 0.5 * (torch.cos(math.pi * d / self.cutoff) + 1)
        h = self.emb(z)
        hs = [h]
        for filt, upd, norm in zip(self.filt, self.upd, self.norm):
            f = filt(torch.cat([e, h[dst], h[src]], -1))
            h = norm(h + upd(seg_sum(h[src] * f, dst, len(h))))
            hs.append(h)
        h = self.jump(torch.cat(hs, -1))
        E = seg_sum(self.alpha(h) * self.energy(h), batch, B).squeeze(-1)
        return E, (self.force(h) if self.force is not None else None)

    def forward(self, pos, z, batch):
        B = int(batch.max()) + 1
        if self.forces == "grad" and not pos.requires_grad:
            pos = pos.requires_grad_()
        t, X, U, W = pca_frames(pos, batch, B, self.frame, self.se3, self.tau)
        # every kept (graph, frame) pair becomes its own copy of the graph -> one batched phi call
        gb, fk = (W > 1e-4).nonzero().T
        n = torch.bincount(batch, minlength=B)
        ptr = n.cumsum(0) - n
        cnt = n[gb]
        pair = torch.repeat_interleave(torch.arange(len(gb), device=pos.device), cnt)
        atom = ptr[gb][pair] + torch.arange(len(pair), device=pos.device) - (cnt.cumsum(0) - cnt)[pair]
        Up = U[gb, fk][pair]
        p = torch.einsum("ni,nij->nj", X[atom], Up)
        # ponytail: dense radius graph over all copies, (frames*N)^2; reuse the base edge list if full FA gets big
        src, dst = radius_graph(p.detach(), pair, self.cutoff)
        Ep, Fp = self.phi(p, z[atom], src, dst, pair, len(gb))
        w = W[gb, fk]
        w = w / seg_sum(w, gb, B)[gb]  # renormalise after pruning
        E = seg_sum(w * Ep, gb, B)
        if self.forces == "grad":
            # ponytail: backprops through eigh, which blows up for (near-)degenerate eigenvalues
            F = -torch.autograd.grad(E.sum(), pos, create_graph=self.training)[0]
        elif Fp is not None:  # rho_2: rotate back with U^T, then weighted average
            F = seg_sum(w[pair, None] * torch.einsum("nj,nkj->nk", Fp, Up), atom, len(pos))
        else:
            F = None
        return E, F


def random_orthogonal(dtype, proper=False):
    Q, R = torch.linalg.qr(torch.randn(3, 3, dtype=dtype))
    Q = Q * torch.sign(torch.diagonal(R))
    if proper and torch.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    if not proper and torch.linalg.det(Q) > 0:
        Q[:, 0] = -Q[:, 0]  # include a reflection to exercise the full E(3)
    return Q


def symmetry_error(model, pos, z, batch, proper=False):
    """Max |E(gX) - E(X)| and |F(gX) - F(X)R| for a random g in E(3) (SE(3) if proper)."""
    R = random_orthogonal(pos.dtype, proper).to(pos.device)
    shift = torch.randn(3, dtype=pos.dtype, device=pos.device)
    E1, F1 = model(pos.clone(), z, batch)
    E2, F2 = model(pos.clone() @ R + shift, z, batch)
    return (E1 - E2).abs().max().item(), (F1 @ R - F2).abs().max().item()


if __name__ == "__main__":
    torch.manual_seed(0)
    sizes = [5, 9, 12]
    batch = torch.repeat_interleave(torch.arange(3), torch.tensor(sizes))
    pos = torch.randn(len(batch), 3, dtype=torch.float64) * 1.5
    z = torch.randint(1, 10, (len(batch),))

    for frame, forces, se3 in [("full", "direct", False), ("canon", "direct", False),
                               ("full", "grad", False), ("canon", "grad", False),
                               ("soft", "direct", False), ("soft", "grad", False),
                               ("full", "direct", True), ("canon", "direct", True)]:
        model = FAENet(hidden=32, layers=2, frame=frame, forces=forces, se3=se3).double().eval()
        eE, eF = symmetry_error(model, pos, z, batch, proper=se3)
        print(f"{frame:5s} forces={forces:6s} se3={se3!s:5s}  |dE|={eE:.1e}  |dF|={eF:.1e}")
        assert eE < 1e-8 and eF < 1e-8, "symmetry broken"

    model = FAENet(hidden=32, layers=2, frame="sfa").double().eval()
    eE, eF = symmetry_error(model, pos, z, batch)
    print(f"sfa   (untrained, approximate by design)   |dE|={eE:.1e}  |dF|={eF:.1e}")
    model.frame = "none"
    eE, eF = symmetry_error(model, pos, z, batch)
    print(f"none  (no symmetry at all)                 |dE|={eE:.1e}  |dF|={eF:.1e}")
    print("all exact-symmetry checks passed")

    # continuity: move one atom of an asymmetric molecule along a line until the skewness of the first PCA
    # axis crosses 0. canon's sign choice flips there and its energy jumps; soft's must not.
    def skew0(x):
        X = x - x.mean(0)
        lam, V = torch.linalg.eigh(X.T @ X / len(X))
        return ((X @ V[:, -1]) ** 3).mean() / lam[-1] ** 1.5

    torch.manual_seed(1)
    mol = torch.randn(10, 3, dtype=torch.float64) * 1.5
    zz, bb = torch.randint(1, 10, (10,)), torch.zeros(10, dtype=torch.long)
    nudge = torch.zeros_like(mol)
    nudge[0] = torch.tensor([1.0, 0.5, -0.3], dtype=torch.float64)
    ts = torch.linspace(-4, 4, 4001, dtype=torch.float64)
    sk = torch.stack([skew0(mol + t * nudge) for t in ts])
    k = int((sk[:-1].sign() != sk[1:].sign()).nonzero()[0])
    lo, hi = ts[k], ts[k + 1]
    for _ in range(60):  # bisect to the crossing
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if skew0(mol + mid * nudge).sign() == skew0(mol + lo * nudge).sign() else (lo, mid)
    jumps = {}
    for frame in ("canon", "soft"):
        model = FAENet(hidden=32, layers=2, frame=frame, forces=None).double().eval()
        Es = torch.stack([model(mol + t * nudge, zz, bb)[0] for t in lo + torch.linspace(-1e-6, 1e-6, 41, dtype=torch.float64)])
        jumps[frame] = Es.flatten().diff().abs().max().item()
        print(f"{frame:5s} largest energy step across the skew = 0 crossing: {jumps[frame]:.1e}")
    assert jumps["soft"] < 1e-3 * jumps["canon"], "soft frames are not continuous"
    print("soft frames are continuous where canon jumps")
