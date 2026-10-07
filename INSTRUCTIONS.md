# Person 2: training

You are **step 2 of 5** (order: person 1 → 2 → 4 → 3 → 5).

## Your files

- `train_lj.py`
- `baseline/train_official.py`
- `runs/aspirin_canon_grad.pt`
- `runs/aspirin_canon_grad_50k.pt`
- `runs/al_random.pt`
- `runs/al_active.npy`
- `runs/al_random.npy`

They are already laid out at their repo paths inside `repo/`.

## Steps

1. Accept the GitHub collaborator invite from person 1.
2. Make sure git uses **your** GitHub identity, so the commits count on your profile:
   ```bash
   git config --global user.name  "Your Name"
   git config --global user.email "the-email-on-your-github-account"
   ```
3. Add your files and push:
   ```bash
   git clone https://github.com/OWNER/REPO.git faenet-md17   # skip if you already have it
   cd faenet-md17 && git pull
   cp -r /path/to/person2/repo/. .
   git add -A
   git commit -m "Add training scripts (ours + official recipe) and trained checkpoints"
   git push
   ```
   If the push is rejected because someone pushed first, run `git pull --rebase`, then `git push` again.
4. Tell person 4.
