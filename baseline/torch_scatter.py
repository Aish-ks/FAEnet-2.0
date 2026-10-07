"""Stand-in for the compiled torch_scatter extension (no wheel for torch 2.11 / py3.14).
The official faenet package only calls `scatter(src, index, dim, reduce=...)`; PyG ships a pure-torch version."""
from torch_geometric.utils import scatter  # noqa: F401
