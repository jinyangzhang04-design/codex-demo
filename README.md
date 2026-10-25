# 分子垂类大模型驱动的多任务成药性预测与虚拟筛选

当前版本提供可运行的分子标准化、Morgan 指纹与物化描述符特征、随机森林多任务基线，以及批量虚拟筛选 CLI。

## 使用

```bash
pip install -e .
drugscreen train --input data/train.csv --model artifacts/baseline.joblib
drugscreen screen --input data/candidates.csv --model artifacts/baseline.joblib --output results.csv
```

训练数据需要包含 `smiles` 列和一个或多个数值任务列。后续将加入分类任务、scaffold split、GNN/Transformer 模型和 API 服务。

## Web 页面

```bash
pip install -e '.[web]'
uvicorn drugscreen.web:app --host 0.0.0.0 --port 8000
```

打开 `http://localhost:8000`，输入每行一个 SMILES 即可筛选。
