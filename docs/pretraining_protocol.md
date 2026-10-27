# SMILES 预训练与微调协议

## 预训练

- 只使用联合 scaffold 划分中的训练分子，避免将测试分子的结构用于自监督预训练；
- 使用 94 个 SMILES token 的词表，最长序列 96；
- 随机遮盖约 15% 的非特殊 token，训练 Transformer 做掩码 token 预测；
- 当前模型为 64 维 token 表征、2 层 Transformer、4 个注意力头；
- 保存预训练损失、词表、模型配置和随机种子。

## 下游微调

- 共享 SMILES 编码器后接多任务预测头；
- 回归任务使用训练集均值和标准差归一化；
- 分类与回归任务均使用缺失标签掩码；
- 使用各任务 validation split 做早停和分类阈值选择；
- 多任务主结果使用联合 scaffold split；各任务独立 scaffold split 只用于单任务基线和补充分析。

当前结果见 `reports/smiles_pretraining.json` 和 `reports/smiles_finetuned_metrics.json`。该模型是本地可复现的预训练基线，不等同于大规模通用分子基础模型。
