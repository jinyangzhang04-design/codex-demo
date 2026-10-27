# 文献核对矩阵

第一周只记录可核验的论文和数据说明，不把模型名称直接当作创新点。每篇文献补齐：数据集、切分方式、模型权重、任务类型、指标、是否处理缺失标签、是否报告不确定性、可复现代码。

## 已建立的候选阅读清单

以下条目用于本周阅读，具体卷期、页码、DOI 和代码地址需要在可联网环境中逐条核验后再写入论文参考文献。

| 代表工作 | 主题 | 当前状态 |
|---|---|---|
| Wu et al., *MoleculeNet: a benchmark for molecular machine learning*, Chemical Science, 2018 | 分子性质基准与数据切分 | 题名和期刊已确认，DOI 10.1039/C7SC02664A 待复核 |
| Rong et al., *Self-Supervised Graph Transformer on Large-Scale Molecular Data*, NeurIPS 2020 | GROVER 图分子预训练 | 题名和年份已确认，代码与权重地址待核验 |
| Chithrananda et al., *ChemBERTa: Large-Scale Self-Supervised Pretraining for Molecular Property Prediction* | SMILES 预训练 | 预印本版本和权重地址待核验 |
| Ross et al., *Large-Scale Chemical Language Models for Drug Discovery*, Nature Machine Intelligence, 2022 | MolFormer 类化学语言模型 | 题名和期刊待核验 |
| Zhou et al., *Uni-Mol: A Universal 3D Molecular Representation Learning Framework* | 3D 分子预训练 | 会议版本、模型许可和推理资源待核验 |
| Bemis and Murcko, *The Properties of Known Drugs. 1. Molecular Frameworks*, Journal of Medicinal Chemistry, 1996 | scaffold 定义 | 用于说明 Bemis-Murcko scaffold split，DOI 待复核 |
| Lakshminarayanan et al., *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles*, NeurIPS 2017 | 不确定性估计 | 用于校准和不确定性实验设计，版本待核验 |

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
