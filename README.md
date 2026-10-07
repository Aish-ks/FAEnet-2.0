# FAENet on MD17 aspirin: exact single-frame symmetry and energy-conserving forces

A re-implementation of FAENet (Duval et al., ICML 2023, [arXiv:2305.05577](https://arxiv.org/abs/2305.05577)) with three changes:
- a canonical single frame (exact E(3) invariance from one pass);
- forces computed as −∇E;
- a smooth cutoff.

It is compared head-to-head against the authors' official implementation on the same data. Results and method are in **[WRITEUP.md](WRITEUP.md)**.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install faenet==0.1.3 --no-deps          # official baseline only; its declared deps are incomplete
mkdir -p data && curl -L -o data/md17_aspirin.npz http://www.quantum-machine.org/gdml/data/npz/md17_aspirin.npz
python faenet.py                             # symmetry + continuity self-checks
```

## Run

```bash
python app.py                                # interactive demo on http://127.0.0.1:8765
python train_lj.py --data data/md17_aspirin.npz --n_train 1000 --epochs 500 --frame canon --forces grad --out runs/aspirin_canon_grad.pt
python baseline/train_official.py --n_train 1000 --epochs 500 --out runs/official_1k.pt
python active.py --eval runs/official_1k.pt runs/aspirin_canon_grad.pt
python stability.py runs/official_1k.pt runs/aspirin_canon_grad.pt --T 300 500
```

## Layout

| Path | What |
|---|---|
| `faenet.py` | Model, frames, self-checks |
| `stability.py`, `active.py` | Simulation stability benchmark, test-set evaluation and active learning |
| `baseline/` | Official FAENet behind our interface |
| `train_lj.py`, `baseline/train_official.py` | Training scripts |
| `runs/` | Training logs, results and the demo checkpoints |
| `app.py`, `web/` | Interactive demo |
