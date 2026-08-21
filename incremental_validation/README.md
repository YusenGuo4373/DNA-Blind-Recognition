# 基于作者盲识别核心的增量功能验证

本实验不称为“原论文基线复现”。作者 `models.py`、4 维 one-hot、Mask-Padding 和 softmax 概率平均投票保持不变；新增功能全部位于模型外。

论文建议表述：**在保持作者盲识别网络、输入表示及软投票机制不变的条件下，验证外接层次化开放集判决模块的有效性。**

## 共享 logits 对照

每个测试归档只保存一次作者 Transformer 的逐 read 码型 logits。随后同一份 logits 同时进入：

- 闭集流程：无条件执行原软投票，强制输出 BCH、Convolutional、LDPC 或 Polar。
- 新级联流程：先根据外部 ECC 概率输出 `no_ecc`，再根据作者 logits 的外部能量阈值输出 `unknown_ecc`，其余样本执行相同的原软投票。

因此两组结果之间不存在网络权重、输入表示或 read 抽样差异。

`collector.py` 中的 `collect_shared_logits` 同时调用外部 4 通道 No-ECC detector 和作者原始 Transformer，并把两者输出保存到同一 NPZ。作者码型模型对每个归档只调用一次。

共享 logits 文件为 NPZ，必须包含：

- `categories`: `[N]` 字符串；
- `presence_probabilities`: `[N,M,q]`；
- `type_logits`: `[N,M,q,4]`；
- `archive_ids`: 可选 `[N]`。

## 阈值校准

校准文件只能包含四种已知 ECC 和 No-ECC，出现 Fountain/LT 会立即报错：

```powershell
python -m incremental_validation.cli calibrate `
  --validation-logits outputs/validation_shared_logits.npz `
  --output outputs/incremental/thresholds.json
```

`tau1` 按验证集宏 F1 选择；`tau2` 只使用已知 ECC 能量，并接受至少 95% 的已知验证归档。

## 同 logits 比较

```powershell
python -m incremental_validation.cli compare `
  --test-logits outputs/test_shared_logits.npz `
  --thresholds outputs/incremental/thresholds.json `
  --output outputs/incremental/comparison
```

输出同时包含闭集与级联的逐归档预测、No-ECC/喷泉码误判为已知码型的比例、已知 ECC 接受率、已知码型宏 F1 变化，以及六类端到端混淆矩阵。码率和码长固定返回空值。
## 模拟先导测试

当真实归档数据暂不可用时，可运行自建数据先导测试：

```powershell
python -m incremental_validation.simulation `
  --output outputs/incremental_simulation_pilot_seed42 `
  --seed 42 `
  --device cuda
```

该命令生成合法的四类 ECC、两类 No-ECC 和纯 LT/XOR Fountain 数据，转换为
`[M,q,4,400]` one-hot 与 mask。Fountain 只在最终测试阶段构造。作者
Transformer 与外部 No-ECC 检测器分别运行，闭集和级联比较复用完全相同的
作者模型 logits。

这只是模拟先导测试：自建编码分布可能与作者权重的原始训练分布不同，因此结果
用于验证流程和发现失效模式，不能直接作为论文正式结果。
