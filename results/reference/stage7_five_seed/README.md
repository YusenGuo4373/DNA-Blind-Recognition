# Stage-7 默认同分布大规模独立重复性验证

本实验定位为：**冻结作者盲识别核心和既有外部门控，在默认同分布IDS条件下进行的大规模独立重复性与统计稳定性验证。** 不应表述为跨测序平台或跨模拟器泛化证明。

## 冻结协议

- 作者 Transformer、`models.py`、ECC-presence CNN、Stage-5 proxy detector 和全部阈值均未训练、微调、拟合或校准。
- 默认信道由正式 Stage-5/6 运行记录共同核实为 `p_ins=p_del=p_sub=0.05`，`M=20`、`q=50`、`Lmax=400`。
- 固定阈值：presence `0.464293801413849`；energy `-1.526681987616622`；proxy `0.826897548106894`。
- 统计单位是 archive，不把同一 molecule 的 reads 当作独立统计样本。
- `code_rate=null`，`code_length=null`。

## 数据规模

- seeds：46, 47, 48, 49, 50
- 7 类 × 100 archives × 20 molecules × 50 reads × 5 seeds
- 3,500 archives、70,000 个全局唯一参考分子、3,500,000 reads
- HEDGES 为无固定引物的纯内码；HEDGES/DNA-Aeon 未进入开发、拟合或校准。

## 主要结果（五 seed 合并）

| 方法 | 已知接受率 | 合并未知召回 | 未知 forced-known | 未知误判 BCH | 已知 macro-F1 变化 |
|---|---:|---:|---:|---:|---:|
| Energy-only | 98.100% | 7.600% | 92.400% | 91.100% | -0.0088 |
| Proxy-only | 97.900% | 90.500% | 9.500% | 8.400% | -0.0156 |

三态协同：未知 `unknown+uncertain` 风险覆盖 90.500%；已知 uncertain 2.400%；未知直接输出已知码型 9.500%；全部 archive 人工复核比例 25.057%。

Stage-6 小规模基准为 proxy-only 已知接受率 97.333%、未知召回率 90.000%。

## 预注册标准

- proxy-only：`{"mean_known_acceptance_ge_97": true, "worst_known_acceptance_ge_95": true, "pooled_unknown_recall_ge_85": true, "worst_unknown_recall_ge_80": true, "unknown_forced_known_le_15": true, "unknown_BCH_le_10": true, "macro_f1_drop_le_002": true}`
- 三态协同：`{"unknown_risk_coverage_ge_90": true, "known_uncertain_le_5": true, "unknown_direct_known_le_10": true}`
- proxy-only 全部通过：`True`
- 三态协同全部通过：`True`

## 统计与审计

- 95% CI 使用分层 cluster bootstrap：先重采 seed，再在每个 seed/真实类别内以 archive 为单位重采；没有以 read 为独立单位计算 CI。
- checkpoint 与配置运行前后全部不变：`True`。
- Stage-7 五 seed 参考分子零重叠：`True`；与本地已保存旧参考 FASTA 零重叠：`True`。
- 限制：作者 checkpoint 未附其原始训练参考序列，因此无法逐序列核对那一外部数据集；新数据使用独立 SHA256 namespace，且本地可访问的旧参考数据均已核查。
- Stage-6 的 Sequence-only、Embedding-only、Logits-only 及两两融合只保存了超参数/阈值，没有保存分类器系数；本轮遵守“不重新拟合”，故 A-F 标记为不可执行。全融合 G 使用已序列化 Stage-5 detector 正常评测。
- 全目录 pytest 受不完整的 `ECC_round5_staging_20260813` 副本缺少两个模块而在收集阶段阻塞；仓库主 `tests/` 完整测试结果单独记录。

## 输出说明

- 每个正式 seed 的 `shards/` 含每 archive 的 `per_read_predictions.csv.gz`、归档预测、冻结特征和 SHA256 完成标记，可断点续跑。
- 汇总指标见 `aggregate_metrics.json`；置信区间见 `bootstrap_confidence_intervals.json`；数据与模型审计见 `experiment_manifest.json`、`checkpoint_hash_audit.json` 和 `data_independence_audit.json`。
