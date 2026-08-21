# 作者盲识别核心适配层

本目录只负责外围适配。作者源码固定在 `vendor/zhouph0313_DNA`，其中
`models.py` 保持原样；输入仍为 4 维 one-hot，Mask-Padding 和 softmax
软投票语义不变。

本实验应称为“基于作者盲识别核心的增量功能验证”，不称为“原论文基线复现”。

## 校验作者源码和用户提供的权重

```powershell
python -m author_baseline.cli verify-vendor
python -m author_baseline.cli inspect-weights `
  --weights-root artifacts/model_weights `
  --device cpu
```

`inspect-weights` 会计算 SHA-256，并把全部 12 个权重以 `strict=True` 加载到
作者原始 CNN、LSTM、Transformer、ResNet 定义中。当前主实验使用：

```text
results/results/type/transformer_model_f10.6033.pt
```

文件名中的 F1 仅作为已有权重的元数据，不作为本轮实验测得或复现的指标。

码率、码长权重已登记但本轮不接入级联；因此结果中的 `code_rate` 和
`code_length` 仍为 `null`。CNN、LSTM、ResNet 仅保留为可选稳健性验证。

## 开放集层次适配

`HierarchicalAuthorAdapter` 的调用顺序为：

1. 外部 No-ECC 检测器对逐 read 概率做软平均；低于阈值时输出 `no_ecc`。
2. 对其余输入调用作者原始码型 Transformer，保留逐 read logits 和原软投票。
3. 在作者模型之外对同一 logits 计算能量分数；超过阈值时输出
   `unknown_ecc`。
4. 其余样本输出 BCH、Convolutional、LDPC 或 Polar。

闭集流程和新级联流程必须复用同一份作者模型 logits。喷泉码只进入最终测试，
不得参与 No-ECC 训练、验证或任何阈值校准。

## 当前运行所缺数据

`results/results` 只包含权重，没有归档 reads、HDF5、NPZ 或索引 CSV。
完整增量验证还需要：

- 四类已知 ECC 的验证集和测试集；
- 独立的 No-ECC 验证集和测试集；
- 仅用于最终未知测试的纯 LT/XOR 喷泉码；
- 外部 No-ECC 检测器权重，或可用于训练该检测器的数据。

这些数据应按 `[M,q,4,L]` one-hot 和 `[M,q,L]` mask 提供，或提供能够稳定
生成该格式的原始序列与分组索引。
