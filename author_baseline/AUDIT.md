# 作者盲识别核心审计

审计对象为 `vendor/zhouph0313_DNA`，固定提交 `1ac47fce3bb9526633e38a7863612d7dc5db3a40`。

## 可以原样复用的核心

- `models.py`：`CNN1DClassifier`、`LSTMClassifier`、`TransformerClassifier`、`ResNet1DClassifier`。
- 四个模型均接收 `(B,4,L)` one-hot；mask 为 `(B,L)`，1 表示有效位置。
- Transformer 参数为 `d_model=128`、4 heads、2 layers、ReLU、可学习位置编码和 mask 加权平均池化。
- `dataset.py` / `dataset_adaptive.py`：HDF5 中 `(L,4)` 序列转为 `(4,L)`，同时返回 mask。
- `vote6.1copyerror.py`：逐 read softmax 后，对 `group_size × num_copies` 的概率直接求算术平均，再取 argmax。
- `bchDNA4.py`、`conveDNA4.py`、`ldpcDNA4.py`、`polarDNA4.py`：作者最新版数据生成代码。

## 上游缺失，不能自行宣称为“原代码基线”的部分

- 仓库没有模型权重、HDF5 数据集或 `dataset_index*.csv`。
- 最新提交没有统一的四模型训练入口；`models.py` 只包含模型定义。
- `dataset.py` 明确将 `code_rate / code_length` 标为未来扩展，非 `code_type` 任务当前返回占位标签 0。
- 仓库中的 `cnnparameter.py` 与 `lstmparameter.py` 是早期的五文件 LDPC 参数分类脚本，使用硬编码作者桌面路径，并非最新版四码型统一码率/码长训练流程。
- 仓库没有与最新版 `models.py` 对应的 ResNet 训练脚本。
- 根目录没有 LICENSE 文件。

因此，在取得作者当时使用的训练入口、数据索引、标签映射和/或模型权重之前，可以开展“基于作者盲识别核心的增量功能验证”，但不能称为“原论文基线复现”。

## 外层适配原则

- `author_baseline/original_models.py` 从 vendor 的 `models.py` 动态加载类，不复制或改写网络。
- `author_baseline/recognizer.py` 只负责 batch 化调用、恢复 `[M,q,C]` 和调用作者软投票。
- `author_baseline/cascade.py` 在原模型外执行 `no_ecc` 和 `unknown_ecc` 判决；只有已知 ECC 才继续调用码率和码长模型。
- `author_baseline/cli.py run-author-vote` 直接以子进程运行作者原始投票脚本，并把原硬编码输入改为外层 CLI 参数。
- 每次运行前用 SHA-256 清单验证 vendor 未发生改动。
