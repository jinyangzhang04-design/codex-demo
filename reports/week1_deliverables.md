# 基于 AIDD 垂类大模型的小分子药物分子性质预测与候选分子筛选

## 第一周成果报告

日期：2026-10-27

## 1. 项目目标

项目目标是使用分子预训练表征和多任务学习，预测小分子的药物相关性质，并根据预测性质进行候选分子优选。

两个月项目周期内，“AIDD 垂类大模型”具体落地为公开预训练分子编码器的迁移学习，并保留一个本地 SMILES 预训练模型作为可复现基线。

## 2. 数据建设

本周获取并统一处理了四个公开分子数据集：

| 数据集 | 任务 | 唯一分子数 |
|---|---|---:|
| ESOL | 水溶解度回归 | 1,117 |
| FreeSolv | 溶剂化自由能回归 | 642 |
| BBBP | 血脑屏障渗透性分类 | 1,975 |
| ClinTox CT_TOX | 临床试验毒性分类 | 1,461 |

合并后共有 4,526 个唯一分子。处理流程包括 SMILES 合法性检查、canonical SMILES 标准化、分子去重、缺失标签保留和 SHA256 文件校验。

任务之间共同分子数量有限：ESOL 与 FreeSolv 为 340 个，BBBP 与 ClinTox 为 202 个，四个任务同时有标签的分子只有 4 个。因此多任务学习的研究重点是跨任务表征迁移，而不是同一分子的完整多性质监督。

## 3. 数据划分与泄漏审计

- 单任务模型使用任务独立的 Bemis-Murcko scaffold split；
- 多任务主实验使用联合 scaffold split；
- 所有任务的单任务训练集与测试集 scaffold 重叠数为 0；
- 联合 scaffold split 的训练集与测试集分子重叠数为 0；
- 分类阈值只使用 validation 集选择，测试集只用于最终报告。

联合 scaffold split 下 FreeSolv 的测试标签只有 5 个，因此该结果仅作为探索性结果，不能作为稳定结论。

## 4. 基线实验

### ECFP + Random Forest

使用 3 个随机种子，在任务独立 scaffold split 上训练。

| 任务 | 结果 |
|---|---|
| ESOL | RMSE 0.757±0.001，MAE 0.612±0.001，R² 0.884±0.000 |
| FreeSolv | RMSE 3.591±0.051，MAE 2.341±0.021，R² 0.567±0.012 |
| BBBP | ROC-AUC 0.893±0.001，PR-AUC 0.942±0.002 |
| ClinTox | ROC-AUC 0.880±0.006，PR-AUC 0.268±0.007，F1 0.283±0.078 |

### ECFP + 单任务 MLP

| 任务 | 结果 |
|---|---|
| ESOL | RMSE 1.727±0.009 |
| FreeSolv | RMSE 4.766±0.308 |
| BBBP | ROC-AUC 0.840±0.017，PR-AUC 0.929±0.009 |
| ClinTox | ROC-AUC 0.492±0.041，PR-AUC 0.079±0.019 |

当前 Random Forest 是更强、更稳定的传统基线。后续主模型必须证明预训练分子表征带来的增益。

## 5. 多任务与预训练实验

已经实现了缺失标签掩码、共享编码器、任务专属预测头、验证集早停和分类阈值选择。

另外使用联合 scaffold 训练分子中的 3,621 个唯一 SMILES 做了 8 轮掩码语言模型预训练，模型为 64 维、2 层、4 头 Transformer，预训练损失从 3.185 降到 2.194。

本地预训练模型微调结果低于 Random Forest：ESOL RMSE 1.499（联合 split），BBBP ROC-AUC 0.790，ClinTox PR-AUC 0.069。这个结果说明小规模本地预训练可以验证流程，但不足以替代大规模公开分子基础模型。

## 6. 文献与主模型选择

已经通过官方仓库或 DeepChem 官方教程核验了 MoleculeNet、ChemBERTa、ChemBERTa-2、GROVER、MoLFormer 和 Uni-Mol 的标题、预训练方式及公开代码/权重入口。

第二周优先尝试 ChemBERTa，原因是它直接使用 SMILES 掩码语言建模，已有公开模型名 `seyonec/ChemBERTa-zinc-base-v1`，并且有 DeepChem 官方迁移学习教程。GROVER 作为备用主模型，MoLFormer 和 Uni-Mol作为扩展对照。

## 7. 第一周结论

第一周已经完成数据、实验协议、基线、泄漏审计、复现说明和核心文献来源核验。当前最可靠的结果是 ECFP + Random Forest 基线；当前最需要验证的问题是公开预训练分子编码器能否在 scaffold split 和稀疏标签条件下超过该基线。

## 8. 第二周计划

1. 下载并验证 ChemBERTa 或 GROVER 公开权重；
2. 实现冻结编码器、轻量微调和全量微调三个版本；
3. 在联合 scaffold split 上运行多任务主实验；
4. 对比 Random Forest、单任务 MLP、本地 SMILES Transformer 和公开预训练模型；
5. 保存统一结果表和训练配置。

## 9. 复现入口

完整命令见 `docs/reproduce_week1.md`。主要脚本为：

```bash
python scripts/download_datasets.py
python scripts/prepare_datasets.py
python scripts/audit_split_leakage.py
python scripts/train_baseline.py --seeds 0,1,42
python scripts/train_single_task_mlp.py --seeds 0,1,42
```
