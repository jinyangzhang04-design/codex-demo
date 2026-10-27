# 文献核对矩阵

第一周只记录可核验的论文和数据说明，不把模型名称直接当作创新点。每篇文献补齐：数据集、切分方式、模型权重、任务类型、指标、是否处理缺失标签、是否报告不确定性、可复现代码。

## 需要核对的方向

| 方向 | 检索关键词 | 需要回答的问题 |
|---|---|---|
| 多任务分子性质预测 | multi-task molecular property prediction scaffold split | 多任务在稀疏标签下何时有效？ |
| SMILES 预训练 | molecular language model SMILES pretraining | 预训练语料、词表和下游微调如何设置？ |
| 图分子基础模型 | pretrained molecular graph encoder ADMET | 冻结、全量微调和参数高效微调的差异是什么？ |
| ADMET 可靠性 | ADMET uncertainty calibration OOD | 如何报告校准、适用域和分布外误差？ |
| 虚拟筛选评价 | virtual screening enrichment factor top-k | 哪些指标能证明筛选优于随机？ |

## 当前不作结论的项目

- “分子垂类大模型”是否构成方法创新，需要结合具体预训练模型和已有工作核对；
- “多任务”是否优于单任务，需要在同一数据版本、同一 scaffold 规则和多个随机种子下比较；
- “候选分子筛选”只有在有活性标签或明确靶点时，才能写成靶点虚拟筛选。
