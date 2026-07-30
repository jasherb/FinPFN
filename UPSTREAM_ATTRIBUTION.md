# Upstream attribution

This repository began as a clone of the official [FinPFN repository](https://github.com/wangy8989/FinPFN). The original `finpfn.ipynb`, files under
`scripts/`, and `LICENSE` are upstream FinPFN work. The upstream implementation
itself credits
[finetune_tabpfn_v2](https://github.com/LennartPurucker/finetune_tabpfn_v2) as
its code base.

The independent reproduction and model-risk audit is contained in `reproduction/`. It adds:

- deterministic data, split, and checksum audits;
- validation-only Ridge and LightGBM baselines;
- released-checkpoint inference wrappers for FinPFN and vanilla TabPFN;
- common-universe IC, portfolio, turnover, and transaction-cost evaluation;
- uncertainty, tail-selection, rank-stability, and cross-market diagnostics;
- public reports and reproducible figure generation.

The root `README.md`, `Makefile`, `UPSTREAM_ATTRIBUTION.md`, `.gitignore`,
`requirements.txt`, and `pyproject.toml` are release packaging maintained for
this audit. They do not replace the upstream research implementation.

No endorsement by the FinPFN or TabPFN authors is implied. The original BSD 3-Clause [LICENSE](LICENSE) is preserved without modification.

## Citation

If this repository is useful, cite the original FinPFN paper:

```bibtex
@article{wang2026finpfn,
  title={Meta-learning for return prediction in shifting market regimes},
  author={Wang, Yicheng and Lera, Sandro Claudio},
  journal={Journal of Financial Markets},
  volume={79},
  pages={101042},
  year={2026},
  publisher={Elsevier},
  doi={10.1016/j.finmar.2025.101042}
}
```

FinPFN builds on the TabPFN ecosystem. Refer to the [official TabPFN repository](https://github.com/PriorLabs/TabPFN) for its current citation and model-specific terms.
