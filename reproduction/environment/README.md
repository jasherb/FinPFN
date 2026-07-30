# Reproduction environments

Two pinned environments are provided:

- `requirements-cpu-lock.txt` supports data inspection, Ridge, LightGBM,
  evaluation, reporting, and release verification.
- `requirements-gpu-cu121-lock.txt` adds the exact PyTorch, CUDA, TabPFN, and
  training-support versions used for released-checkpoint inference.

The verified GPU stack was Python 3.10, PyTorch 2.5.1+cu121, TabPFN 2.0.8,
NumPy 2.2.6, pandas 2.3.3, SciPy 1.15.3, scikit-learn 1.6.1,
LightGBM 4.7.0, and PyArrow 18.1.0.

Create separate environments when practical:

```bash
python3.10 -m venv .venv-cpu
. .venv-cpu/bin/activate
python -m pip install --upgrade pip
python -m pip install -r reproduction/environment/requirements-cpu-lock.txt
```

For a CUDA 12.1-compatible host:

```bash
python3.10 -m venv .venv-gpu
. .venv-gpu/bin/activate
python -m pip install --upgrade pip
python -m pip install -r reproduction/environment/requirements-gpu-cu121-lock.txt
```

The CUDA environment requires an NVIDIA driver compatible with CUDA 12.1.
`capture_environment.sh` writes a compact, ignored record containing package
versions and the visible GPU model. It deliberately excludes usernames,
hostnames, filesystem paths, scheduler details, kernel fingerprints, and
credentials.
