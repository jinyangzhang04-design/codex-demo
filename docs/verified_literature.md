# 第一周已核验的模型来源

第一周优先核验了与主模型选择直接相关的来源。核验使用官方 GitHub 仓库或 DeepChem 官方教程完成；外部 DOI 页面在云环境中返回 403，因此最终论文引用仍应在本地网络中导出 BibTeX。

## 选择结论

第二周优先尝试 **ChemBERTa**，原因是它直接以 SMILES 掩码语言建模为预训练目标，公开了 Hugging Face 模型名，并有 DeepChem 官方迁移学习教程。若权重下载或许可证不适合项目，再切换到 GROVER；MoLFormer 和 Uni-Mol 作为扩展对照，不作为两个月内的首个实现目标。

## 官方来源

- MoleculeNet：DeepChem 官方教程确认 Wu et al. 2018 基准论文和数据集范围。
- ChemBERTa：DeepChem 官方教程确认 SMILES 预训练流程、ClinTox/BBBP 下游示例和模型名 `seyonec/ChemBERTa-zinc-base-v1`。
- ChemBERTa-2：官方辅助仓库给出 arXiv:2209.01712 和模型加载方式。
- GROVER：Tencent AI Lab 官方仓库提供预训练、微调、预测、指纹生成、评估流程和 BBBP、ClinTox、ESOL、FreeSolv 权重入口。
- MoLFormer：IBM 官方仓库确认论文 DOI、SMILES 掩码语言建模和公开预训练模型包。
- Uni-Mol：官方仓库确认 3D 分子表征框架、ICLR 2023 OpenReview 来源和下游性质预测用途。
