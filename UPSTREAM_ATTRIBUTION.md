# Upstream attribution

This repository began as a clone of the official [FinPFN repository](https://github.com/wangy8989/FinPFN). The original training code, notebook, configuration, and model-loading implementation outside `reproduction/` are upstream FinPFN work. The upstream FinPFN implementation itself credits [finetune_tabpfn_v2](https://github.com/LennartPurucker/finetune_tabpfn_v2) as its code base.

The independent reproduction and model-risk audit is contained in `reproduction/`. It adds:

- deterministic data, split, and checksum audits;
- validation-only Ridge and LightGBM baselines;
- released-checkpoint inference wrappers for FinPFN and vanilla TabPFN;
- common-universe IC, portfolio, turnover, and transaction-cost evaluation;
- uncertainty, tail-selection, rank-stability, and cross-market diagnostics;
- public reports and reproducible figure generation.

No endorsement by the FinPFN or TabPFN authors is implied. The original BSD 3-Clause [LICENSE](LICENSE) is preserved without modification.

## Citation

If this repository is useful, cite the original FinPFN paper:

```bibtex
@article{wang2025finpfn,
  title={Meta-learning for return prediction in shifting market regimes},
  author={Wang, Yicheng and Lera, Sandro Claudio},
  journal={Journal of Financial Markets},
  pages={101042},
  year={2025},
  publisher={Elsevier},
  doi={10.1016/j.finmar.2025.101042}
}
```

FinPFN builds on the TabPFN ecosystem. Refer to the [official TabPFN repository](https://github.com/PriorLabs/TabPFN) for its current citation and model-specific terms.
