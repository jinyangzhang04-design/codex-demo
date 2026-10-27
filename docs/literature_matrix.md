# 文献核对矩阵

第一周只记录可核验的论文和数据说明，不把模型名称直接当作创新点。每篇文献补齐：数据集、切分方式、模型权重、任务类型、指标、是否处理缺失标签、是否报告不确定性、可复现代码。

## 已核验的核心阅读清单

以下条目的标题、预训练方式和官方代码/权重入口已通过官方仓库或 DeepChem 官方教程核对。出版信息写入论文前仍应在本地联网环境中导出 BibTeX，避免手工录入格式错误。

| 代表工作 | 主题 | 核验来源 |
|---|---|---|
| Wu et al., *MoleculeNet: a benchmark for molecular machine learning*, Chemical Science, 2018 | 分子性质基准与数据切分 | DeepChem 官方 MoleculeNet 教程；DOI 10.1039/C7SC02664A |
| Rong et al., *Self-Supervised Graph Transformer on Large-Scale Molecular Data* | GROVER 图分子预训练 | [Tencent AI Lab 官方仓库](https://github.com/tencent-ailab/grover)，提供预训练权重和下游评估 |
| Chithrananda et al., *ChemBERTa: Large-Scale Self-Supervised Pretraining for Molecular Property Prediction* | SMILES 掩码语言模型 | [DeepChem 官方教程](https://github.com/deepchem/deepchem/blob/master/examples/tutorials/Transfer_Learning_With_ChemBERTa_Transformers.ipynb)；[官方辅助仓库](https://github.com/seyonechithrananda/bert-loves-chemistry)；arXiv:2010.09885 |
| Chithrananda et al., *ChemBERTa-2: Towards Chemical Foundation Models* | 化学基础模型扩展 | [官方辅助仓库](https://github.com/seyonechithrananda/bert-loves-chemistry)；arXiv:2209.01712；公开模型 `seyonec/ChemBERTa-zinc-base-v1` |
| Ross et al., *Large-Scale Chemical Language Representations Capture Molecular Structure and Properties*, Nature Machine Intelligence, 2022 | MoLFormer 化学语言模型 | [IBM 官方仓库](https://github.com/IBM/molformer)；DOI 10.1038/s42256-022-00580-7；arXiv:2106.09553 |
| Zhou et al., *Uni-Mol: A Universal 3D Molecular Representation Learning Framework* | 3D 分子预训练 | [Uni-Mol 官方仓库](https://github.com/dptech-corp/Uni-Mol)；ICLR 2023 OpenReview 页面 |

## 补充文献（第二周核验）

Bemis-Murcko scaffold 原始论文和 Deep Ensembles 不确定性论文已列入检索方向，出版信息暂不写入正式参考文献；第二周在可联网环境中导出 BibTeX 后再补充。

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
