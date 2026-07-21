# Reproduction environments

The local audit environment is ignored at `audit-venv/`. It currently uses Python
3.11.5 because Python 3.10 was unavailable locally, `tabpfn==2.0.8`,
`pyarrow==18.1.0`, PyTorch 2.5.1, pandas 2.2.3, NumPy 1.26.4, SciPy 1.17.1, and
scikit-learn 1.6.1. It has no CUDA device and is suitable only for inspection and
small compatibility tests.

The full GPU environment must be resolved on the approved compute host from the
repository `requirements.txt`, then captured before execution with:

```bash
bash reproduction/scripts/capture_environment.sh
```

That generated `environment.txt` is ignored because it can include host-specific
details. It should accompany the final untracked result bundle, not a Git commit.
