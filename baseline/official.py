"""The authors' FAENet (pip package `faenet` 0.1.3, faenet.readthedocs.io) behind our interface:
    model(pos [N,3], z [N], batch [N]) -> (energy [B], forces [N,3])   (normalised units, like ours)

Their package and our faenet.py share the module name `faenet`, so the package is loaded as
`faenet_official`. Frames come from their own frame_averaging_3D, one molecule at a time, as in their
data transform. Their model_forward runs the network per frame and rotates forces back.
"""
import importlib
import os
import sys

import torch
from torch import nn

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)


def _patch_mendeleev():
    """mendeleev 1.3 hands pandas 3 a *compiled* SQL object it can't execute; pass the original query.
    Patched only inside mendeleev.fetch (their element-property embeddings read from it)."""
    import types

    import mendeleev.fetch as mf
    import pandas as pd

    def read_sql_query(sql, con, **kw):
        return pd.read_sql_query(getattr(sql, "statement", sql), con, **kw)

    mf.pd = types.SimpleNamespace(**{k: getattr(pd, k) for k in dir(pd) if not k.startswith("__")})
    mf.pd.read_sql_query = read_sql_query

    # mendeleev >= 1.0 dropped `atomic_volume` (molar volume, cm^3/mol), 1 of the 12 properties faenet embeds.
    # Rebuild it as atomic_weight / density. Py3.14 can't install the paper-era mendeleev (0.12-0.14).
    fetch_table = mf.fetch_table

    def fetch_table_with_volume(table, **kw):
        df = fetch_table(table, **kw)
        if table == "elements" and "atomic_volume" not in df:
            df["atomic_volume"] = df["atomic_weight"] / df["density"]
        return df

    mf.fetch_table = fetch_table_with_volume


def _load_official():
    _patch_mendeleev()
    ours = sys.modules.pop("faenet", None)
    saved = list(sys.path)
    sys.path[:] = [_here] + [p for p in saved if os.path.abspath(p or ".") != _root]  # hide our faenet.py
    try:
        pkg = importlib.import_module("faenet")
        from faenet.frame_averaging import frame_averaging_3D
        from faenet.fa_forward import model_forward
    finally:
        sys.path[:] = saved
        sys.modules["faenet_official"] = sys.modules.pop("faenet")
        if ours is not None:
            sys.modules["faenet"] = ours
    assert "site-packages" in pkg.__file__, pkg.__file__
    return pkg, frame_averaging_3D, model_forward


official, frame_averaging_3D, model_forward = _load_official()
from torch_geometric.data import Batch, Data  # noqa: E402


def dense_preprocess(data, cutoff, max_num_neighbors):
    """Their base_preprocess, minus the torch_cluster dependency. Aspirin has 20 neighbours max,
    below their max_num_neighbors (40), so the edge set is identical."""
    pos, batch = data.pos, data.batch
    d = torch.cdist(pos, pos)
    m = (batch[:, None] == batch[None]) & (d < cutoff)
    m.fill_diagonal_(False)
    i, j = m.nonzero().T
    edge_index = torch.stack([j, i])  # their convention: rel_pos = pos[edge_index[0]] - pos[edge_index[1]]
    rel_pos = pos[edge_index[0]] - pos[edge_index[1]]
    return data.atomic_numbers.long(), batch, edge_index, rel_pos, rel_pos.norm(dim=-1)


# Paper Table 7, QM7-X column (molecules, energy + forces): the closest of their four configs to MD17.
PAPER_QM7X = dict(cutoff=5.0, hidden_channels=500, num_filters=400, num_gaussians=50, num_interactions=5,
                  max_num_neighbors=40, pg_hidden_channels=32, tag_hidden_channels=0,
                  force_decoder_type="mlp", force_decoder_model_config={"mlp": {"hidden_channels": 256}})


class Official(nn.Module):
    def __init__(self, fa_method="stochastic", **cfg):
        super().__init__()
        self.fa_method = fa_method  # "stochastic" (SFA, paper default), "all" (Full FA), "det"
        self.net = official.FAENet(preprocess=dense_preprocess, regress_forces="direct_with_gradient_target", **cfg)
        self.last = {}

    @property
    def frame(self):  # lets our evaluate()/stability code switch inference frames like for our model
        return {"stochastic": "sfa", "all": "full", "det": "det"}[self.fa_method]

    @frame.setter
    def frame(self, f):
        self.fa_method = {"sfa": "stochastic", "full": "all", "det": "det", "canon": "det"}.get(f, f)

    def forward(self, pos, z, batch):
        B = int(batch.max()) + 1
        n = torch.bincount(batch, minlength=B).tolist()
        # their FA is a CPU data transform applied before batching: same here, then move to the GPU
        frames = [frame_averaging_3D(p, None, self.fa_method) for p in pos.detach().cpu().split(n)]
        k = len(frames[0][0])
        data = Batch(pos=pos, atomic_numbers=z, batch=batch, natoms=torch.tensor(n, device=pos.device))
        data.fa_pos = [torch.cat([f[0][i] for f in frames]).to(pos) for i in range(k)]
        data.fa_rot = [torch.cat([f[2][i] for f in frames]).to(pos) for i in range(k)]
        mode = "train" if self.training else "inference"
        # train mode needs grad (Eq. 9 gradient target); inference uses the direct head only, so no graph
        with torch.enable_grad() if self.training else torch.no_grad():
            preds = model_forward(data, self.net, "3D", mode=mode, crystal_task=False)
        self.last = preds
        return preds["energy"].view(-1), preds["forces"]
