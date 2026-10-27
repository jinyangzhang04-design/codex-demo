# 实验协议（v0.1）

1. 使用 RDKit 将 SMILES 规范化为 canonical SMILES；非法分子丢弃并记录原因。
2. 按 canonical SMILES 去重；同一分子多条标签先保留原始记录并在任务层聚合。
3. 每个任务保留缺失标签掩码，不以整行删除代替缺失处理。
4. 每个任务按 Bemis-Murcko scaffold 独立划分 train/validation/test；固定随机种子，并保存划分文件。多任务模型另保留一份联合 scaffold 划分，用于检查跨任务分子重叠造成的影响。
5. 回归任务报告 RMSE、MAE、R²；分类任务报告 ROC-AUC、PR-AUC、F1。类别极不平衡时以 PR-AUC 为主要指标。
6. 至少运行 3 个随机种子，报告均值、标准差和测试集样本数。
7. 额外记录模型参数、数据版本、代码提交号和运行配置，确保结果可复现。

## 当前数据版本

首轮数据来自 DeepChem 官方仓库中的 ESOL、BBBP、ClinTox 文件，以及 MobleyLab FreeSolv 的 `database.txt`。下载脚本会重新获取这些来源，处理脚本会生成 `data/processed/joint_multitask.csv`、任务划分和 `reports/data_audit.json`。
