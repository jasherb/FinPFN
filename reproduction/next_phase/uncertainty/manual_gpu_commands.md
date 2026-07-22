# Phase 2 人工 GPU 命令

以下命令只供研究者在获准的计算服务器上手动运行。Codex 不执行 SSH 或服务器命令。

前提：包含 `reproduction/next_phase/` 的本地改动已经由研究者批准提交并推送，服务器随后手动拉取该提交。当前仓库分支是 `master`，因此服务器使用 `git pull origin master`。

```bash
cd ~/jason/FinPFN
git pull origin master
mamba activate finpfn

python - <<'PY'
import tabpfn
import torch
assert tabpfn.__version__ == "2.0.8", tabpfn.__version__
assert torch.cuda.is_available()
print("torch", torch.__version__)
print("tabpfn", tabpfn.__version__)
print("visible GPUs", torch.cuda.device_count())
PY
```

确认物理 GPU 2 空闲后，在 tmux 中运行；`CUDA_VISIBLE_DEVICES=2` 会把物理 GPU 2 映射为进程内的 `cuda:0`，脚本使用 `--device cuda` 是正确的。

```bash
mkdir -p reproduction/next_phase/uncertainty/logs
tmux new -s csi_uncertainty

CUDA_VISIBLE_DEVICES=2 \
bash reproduction/next_phase/uncertainty/scripts/run_csi_uncertainty_validation.sh \
  2>&1 | tee reproduction/next_phase/uncertainty/logs/csi_uncertainty_validation.log
```

用 `Ctrl-b d` 离开 tmux。重新进入：

```bash
tmux attach -t csi_uncertainty
```

两个模型顺序执行，总预计约 45–70 分钟，使用一张 A100 80GB 和四个 CPU workers。任何目标 parquet/metadata 已存在时，runner 会拒绝覆盖。若第一个模型成功、第二个失败，只运行对应的单模型 wrapper；不要删除成功文件：

```bash
CUDA_VISIBLE_DEVICES=2 \
bash reproduction/next_phase/uncertainty/scripts/run_csi_finpfn_uncertainty_validation.sh \
  2>&1 | tee reproduction/next_phase/uncertainty/logs/csi_finpfn_uncertainty_validation.log
```

完成后在服务器上只做 schema/finite/checksum 校验：

```bash
python reproduction/next_phase/uncertainty/validate_uncertainty_artifacts.py \
  --predictions \
    reproduction/next_phase/uncertainty/artifacts/validation/csi500_tabpfn_seed42_validation_notebook_with_replacement_members.parquet \
    reproduction/next_phase/uncertainty/artifacts/validation/csi500_finpfn_seed42_validation_notebook_with_replacement_members.parquet \
  --expected-dates 242 \
  | tee reproduction/next_phase/uncertainty/logs/validate_uncertainty_artifacts.log
```

需要取回本地的文件：

- `reproduction/next_phase/uncertainty/artifacts/validation/*.parquet`
- `reproduction/next_phase/uncertainty/artifacts/validation/*.metadata.json`
- `reproduction/next_phase/uncertainty/logs/csi_uncertainty_validation.log`
- `reproduction/next_phase/uncertainty/logs/validate_uncertainty_artifacts.log`

此时不要运行测试期成员推断。测试期只能在 validation 完成、Phase 3 配置冻结以后运行一次。
