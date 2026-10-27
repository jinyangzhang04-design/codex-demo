# 第一周实验复现

以下命令在项目根目录执行。原始数据不会提交到 Git，下载脚本会按固定来源和 SHA256 校验值重新获取。

```bash
python -m venv .venv
.venv/bin/pip install -e .
.venv/bin/python scripts/download_datasets.py
.venv/bin/python scripts/prepare_datasets.py
.venv/bin/python scripts/audit_split_leakage.py
.venv/bin/python scripts/train_baseline.py --seeds 0,1,42
.venv/bin/python scripts/train_single_task_mlp.py --seeds 0,1,42
```

运行预训练和微调还需要 CPU 版 PyTorch：

```bash
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python scripts/pretrain_smiles_encoder.py --epochs 8
.venv/bin/python scripts/finetune_smiles_encoder.py --epochs 20 --split-mode global
```

Windows 将 `.venv/bin/python` 替换为 `.venv\\Scripts\\python.exe`。正式结果以 `reports/` 下的 JSON 为准，运行时同时记录当前 Git 提交号。
