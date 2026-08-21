# 公开发布前检查清单

当前目录是内部整理版本，并非可以直接公开的最终GitHub仓库。发布前需要完成：

- [ ] 确认论文最终标题、作者顺序、联系作者和机构信息。
- [ ] 在 `CITATION.cff` 中填写作者、仓库地址、版本和论文DOI。
- [ ] 由作者选择项目许可证，并用正式 `LICENSE` 替换
      `LICENSE_PENDING.md`。
- [ ] 确认 `vendor/zhouph0313_DNA` 上游源码的再分发许可。当前固定提交没有
      LICENSE，因此本整理目录只保存来源、提交号和SHA-256清单，没有复制源码。
- [ ] 从一个全新的clone中取得上游源码并运行 `verify-vendor`。
- [x] 已从发布包移除包含本地绝对路径、且不属于论文主实验链的跨模拟器
      `dnaterrasim_frozen_test` 脚本及其专用测试。
- [ ] 统一LDPC码率、码长模型的公开评估入口。权重已经整理，但当前开放集级联的
      `build_primary_type_recognizer` 只连接码型Transformer，不能单独证明参数识别
      结果可复现。
- [ ] 将论文中所有表格和图片逐项映射到生成脚本、配置文件和参考结果。
- [ ] 准备小型、可快速执行的toy/smoke数据示例，并在CPU环境验证。
- [ ] 将完整参考序列、数据划分manifest和逐read预测存入Mendeley Data、Zenodo
      或其他带DOI的数据仓库。
- [ ] 在 `DATA_AVAILABILITY.md` 中填写GitHub Release和数据集DOI。
- [ ] 核对完整数据仓库中的SHA-256与本目录 `manifests/artifact_sha256.json`。
- [ ] 确认匿名审稿阶段的仓库策略不会暴露作者身份。
- [ ] 建立与投稿版本一致的Git tag和GitHub Release。

## 当前已整理

- 候选集码型识别适配层和开放集实验代码；
- 四种网络在码型、LDPC码率和LDPC码长任务上的12个权重；
- coded/uncoded discriminator及其阈值；
- post-hoc OOD detector、特征定义和固定配置；
- IDS错误率、coverage、消融和五个独立测试数据种子的紧凑结果；
- 逐archive预测、混淆矩阵和分层cluster bootstrap置信区间；
- 外部HEDGES和DNA-Aeon实现的来源、固定提交与适配器。

## 当前未复制

- 五个正式测试数据种子的完整逐read shards和参考序列；
- 大型训练、验证与校准数据；
- 第三方无许可证源码；
- LaTeX临时文件、PPT、审稿材料和实验中间版本。
