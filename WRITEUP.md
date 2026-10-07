# FAENet, Re-implemented and Improved: Exact Single-Frame Symmetry and Energy-Conserving Forces

## Summary

We re-implemented FAENet (Duval et al., ICML 2023, arXiv:2305.05577), a graph neural network that predicts the energy of a molecule and the forces on its atoms. We then made three changes to it:

1. **An exact single frame.** The original gets exact rotation symmetry only by running the network 8 times. Ours runs it once.
2. **Forces as the true gradient of the energy**, instead of a separate prediction head.
3. **A smooth cutoff**, so energy stays continuous as atoms move in and out of range.

We trained the **authors' official implementation** and **ours** on the same MD17-aspirin training frames, with the same budget and the same evaluation. Results on 2,000 unseen test frames:

| | Official FAENet (paper recipe) | Ours |
|---|---|---|
| Energy error (MAE, kcal/mol) | 2.27 | **0.62** (3.7× lower) |
| Force error (MAE, kcal/mol/Å) | 2.41 | **1.70** (30% lower) |
| Parameters | 5.78 M | **0.48 M** (12× smaller) |
| Molecules intact after 3 ps of simulation at 300 K | 0 / 16 | **7 / 16** |

Both models were trained on the same 1,000 frames for 500 epochs.

## 1. Background, without the jargon

**The problem.** Predicting how atoms push and pull on each other is the core of drug and materials design. The accurate method, quantum-mechanical simulation (DFT), can take hours of supercomputer time for a single molecular snapshot. A machine-learning model trained on DFT results can give an answer in milliseconds.

**The symmetry requirement.** A molecule's energy does not change when you rotate it, flip it in a mirror, or move it across the room. The forces on its atoms rotate with it. A model that gets this wrong gives different answers for the same molecule viewed from different angles, which is physically meaningless.

**How most models handle it.** They build the symmetry into the network's maths. That makes them complex and slow to train and run.

**FAENet's idea (frame averaging).** Before the network sees the molecule, put the molecule into a standard pose. FAENet uses the molecule's own principal axes, found with principal component analysis (PCA): the direction it is longest along, then the next, then the last. Every rotated copy of a molecule lands in the same pose, so a simple, fast network can be used. It is like straightening a photo before trying to recognise the face in it.

**The catch.** Each axis can point either of two ways, so there are 2 × 2 × 2 = 8 possible standard poses. The paper offers two options:
- **Full FA:** run the network on all 8 poses and average. This is exact but costs 8×.
- **Stochastic FA (SFA):** pick one pose at random. It is fast, but symmetry is only approximate.

The paper's headline results use SFA.

**What the paper showed** (OC20, QM9, QM7-X). FAENet is far faster than comparably accurate symmetric models. On OC20 S2EF-2M it reports 75 min per training epoch, against 1,157 min for DimeNet++, and 623 samples/s at inference.

## 2. What we changed (novelty)

| | Paper / official code | Ours | Evidence |
|---|---|---|---|
| Single-frame symmetry | SFA is approximate. The official `det` mode is **not** invariant: \|ΔE\| = 0.30 under a random rotation (untrained model). | **Canonical frame.** Each axis's sign is fixed by the skewness of the atoms along it, so the frame is exactly invariant with one network pass. | Rotation/reflection error of ~1e-16 in the float64 self-check, and 2.5e-4 kcal/mol for the trained float32 model |
| Forces | Separate force head, nudged toward the energy gradient by a fine-tuning loss (paper Eq. 9). | **F = −∇E** through the frame itself. This conserves energy by construction. | In our code at 1k frames: force MAE 10.5 with a separate head vs **1.71** with the gradient, 6× better |
| Cutoff | Hard cutoff (paper Eq. 6) | Smooth cosine envelope, so energy is continuous when atoms cross the cutoff | Needed for stable simulations with gradient forces |
| Frame continuity | n/a | **Soft frames:** a weighted blend of the sign choices, continuous where the canonical frame would flip | Flip jolts in simulation drop from 0.22 to 0.017 kcal/mol |
| Evaluation | Error (MAE) and symmetry metrics | We also added a **simulation stability benchmark**: molecules that break and time to break, energy drift | `stability.py` |

## 3. Experimental setup

- **Data.** MD17 aspirin: 211,762 DFT snapshots, with energies in kcal/mol and forces in kcal/mol/Å. MD17 is *not* one of the paper's datasets (OC20, QM9, QM7-X), so our numbers cannot be placed beside the paper's tables. We compare the two implementations on the same data instead.
- **Split.** A fixed random permutation (seed 0) shared by all scripts:
  - training frames: the first *n*;
  - 1,000 validation frames;
  - 2,000 test frames from the far end, never used for training or model selection;
  - 16 further frames as simulation start points.
- **Training budget.** Both models: 1,000 training frames, 500 epochs, the same energy normalisation, and the checkpoint with the best validation force error.
- **Official baseline.** The authors' package `faenet` 0.1.3, with the paper's Table 7 QM7-X recipe (the closest of their four configurations to our task):
  - SFA during training, `direct_with_gradient_target` forces;
  - loss weights: energy 1, forces 100, Eq. 9 term 15;
  - AdamW, learning rate 2e-4, batch 100, reduce-on-plateau schedule.

  Compatibility changes, none of which alter the model's computation:
  - a pure-PyTorch `scatter` in place of the compiled `torch_scatter`;
  - a dense neighbour search, which gives the same edge set because aspirin has at most 20 neighbours per atom, under their limit of 40;
  - `atomic_volume`, which current `mendeleev` dropped, rebuilt as atomic weight ÷ density;
  - gradient accumulation (4 micro-batches) so a batch of 100 fits in 6 GB;
  - warm-up shortened to 5% of the steps, since the paper's 3,000 steps assume 1.4 M steps in total.
- **Hardware.** A single NVIDIA RTX 3050 Laptop GPU (6 GB).

## 4. Results: official vs ours (same data, same budget)

### Accuracy (2,000 unseen test frames)

| Model | Inference frames | Energy MAE (kcal/mol) | Force MAE (kcal/mol/Å) |
|---|---|---|---|
| Official FAENet | SFA (1 random frame, paper default) | 2.27 | 2.41 |
| Official FAENet | Full FA (all 8 frames) | 2.33 | 2.82 |
| **Ours** | canonical (1 frame, exact) | **0.62** | **1.70** |

### Simulation stability

16 molecules, 3 ps of constant-energy molecular dynamics each, 0.5 fs timestep. A molecule counts as broken when any bond stretches beyond 1.4× its length.

| Model | 300 K: broken | 300 K: mean time to break | 500 K: broken | 500 K: mean time to break | Energy drift (kcal/mol) |
|---|---|---|---|---|---|
| Official, SFA | 16/16 | 0.44 ps | 16/16 | 0.37 ps | 32–48 |
| Official, Full FA | 16/16 | 0.39 ps | 16/16 | 0.33 ps | 39–51 |
| **Ours** | **9/16** | **1.78 ps** | **15/16** | **1.26 ps** | **6–8** |

### Interpretation

- **Energy-conserving forces matter most for simulation.** The official force head, even with the Eq. 9 term pulling it toward the energy gradient, lets simulations heat up and fall apart within about 0.4 ps.
- **Averaging all 8 frames did not help the official model.** It was trained on one random frame at a time and does not become more accurate when its 8 predictions are averaged. Our canonical frame trains and predicts in the same single pose.
- **Our model is 12× smaller and more accurate.** It also trains faster per epoch on this GPU: 1.4 s vs 4.4 s at 1,000 frames. Part of that gap comes from the official script validating every epoch, against every 5 epochs for ours.

## 5. More data (our model)

The comparison above uses matched data. Separately, we trained our model on more frames to see how far accuracy and stability improve. The official model's 50k run was stopped at epoch 22 of 100, so there is no matched official result at this size.

| Our model, training frames | Test energy MAE | Test force MAE | Broken at 500 K (20 ps) |
|---|---|---|---|
| 1,000 | 0.62 | 1.70 | n/a (15/16 within 3 ps) |
| 5,000 | 0.35 | 0.82 | 12/16 |
| 50,000, canonical frame | **0.19** | **0.42** | 5/16 |
| 50,000, soft frames (3-epoch fine-tune) | 0.31 | 0.44 | 4/16 |

Soft frames remove the flip jolts (0.22 → 0.017 kcal/mol), but stability stayed within noise of the canonical frame (mean time to break 18.0 vs 18.4 ps), and the model runs about 2.2× slower. What still breaks molecules is model accuracy at 500 K.

## 6. Things we tried that did not work

- **Uncertainty from disagreement between the 8 frames.** Its rank correlation with true force error was −0.05. Disagreement between two independently trained models scored +0.35 and flagged 50% of the frames just before a break, against 18% for the frame-based signal. We use the two-model version in the demo.
- **Active learning.** Training on the 4,000 frames the two models disagreed on most was worse than training on 4,000 random frames. Test errors: 0.39 / 0.91 (energy / force) against 0.35 / 0.82.
- **Speed-ups.** On this GPU the model is compute-bound.
  - TF32 was 6% faster but shifted energies by up to 0.027 kcal/mol, so we dropped it.
  - `torch.compile` cannot compile the second-order gradients training needs; it gave 1.3× for batched inference.
  - Training with a direct head and then fine-tuning with gradient forces was 2.8× faster but gave 40% worse force error.

## 7. Interactive demo

`python app.py`, then open http://127.0.0.1:8765. The demo includes:
- a 3D viewer comparing predicted and DFT forces on unseen aspirin frames;
- a live rotation and reflection symmetry test;
- molecular dynamics with energy plots;
- grab-and-pull interactive simulation;
- a confidence meter based on two-model disagreement.

## 8. Limitations

- **One molecule, one dataset.** MD17 aspirin only. The paper's benchmarks (OC20, QM9, QM7-X) were not run; OC20 also needs periodic-boundary support, which we have not implemented.
- **One run per configuration**, with no repeated seeds, so small differences (for example soft vs canonical stability) are within noise.
- **Two simplifications in our model:** it embeds atomic number only, without the paper's period/group/physics embeddings, and uses LayerNorm instead of GraphNorm.
- **The official baseline used the QM7-X recipe as published.** It was not tuned for MD17.

## 9. Code map and reproduction

| File | Purpose |
|---|---|
| `faenet.py` | Our model: frames (`full`, `sfa`, `canon`, `soft`, `none`), gradient/direct forces, self-checks (`python faenet.py`) |
| `train_lj.py` | Training (MD17 or synthetic Lennard-Jones); `--resume`, `--init`, `--indices` |
| `stability.py` | Simulation stability benchmark |
| `active.py` | Active-learning selection; `--eval` scores models on the test set |
| `baseline/official.py`, `baseline/train_official.py` | Official FAENet behind our interface, trained with the paper's recipe |
| `app.py`, `web/index.html` | Interactive demo |

```bash
python faenet.py                                                        # symmetry + continuity self-checks
python train_lj.py --data data/md17_aspirin.npz --n_train 1000 --epochs 500 --frame canon --forces grad --out runs/aspirin_canon_grad.pt
python baseline/train_official.py --n_train 1000 --epochs 500 --out runs/official_1k.pt
python active.py --eval runs/official_1k.pt runs/official_1k.pt:full runs/aspirin_canon_grad.pt
python stability.py runs/official_1k.pt runs/aspirin_canon_grad.pt --T 300 500
```

## Contributions

| Member | Role |
|---|---|
| **Person 1** | Project lead. Chose FAENet and directed the model work: the re-implementation, the novelties (canonical single frame, gradient forces, smooth cutoff, soft frames), the uncertainty and active-learning experiments, and the head-to-head comparison against the official implementation. Directed this write-up. |
| **Person 2** | Training. Directed and oversaw the training runs: MD17 at 1k, 5k and 50k frames, the soft-frame fine-tune, the active-learning runs, and the official-baseline training, including pausing and resuming runs after interruptions. |
| **Persons 3 and 5** | Interface. Directed the interactive demo: 3D viewer with force arrows, live symmetry test, molecular-dynamics playback with energy plots, grab-and-pull live simulation, and the confidence meter. |
| **Person 4** | Monitoring. Checked training logs and progress across the runs. |

The code was written with an AI coding assistant (Claude, Anthropic), directed and reviewed by the team members above.

## Reference

A. Duval, V. Schmidt, A. Hernández-García, S. Miret, F. D. Malliaros, Y. Bengio, D. Rolnick. *FAENet: Frame Averaging Equivariant GNN for Materials Modeling.* ICML 2023. arXiv:2305.05577.
