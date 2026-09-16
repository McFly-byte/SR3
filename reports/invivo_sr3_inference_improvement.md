# 活体数据 SR3 推理改进记录

> 日期：2026-09-16
> 范围：`scripts/infer_healthy_rats_sr.py` 推理管线改进（不涉及模型重训）
> 模型：`healthy_phantom_i300000_ema`（checkpoint `I300000_E2522_ema_gen.pth`，未改动）
> 数据：4 只健康大鼠（R001–R004），16 个 CSI 扫描，320 个样本

---

## 1. 问题：基线推理结果差在哪

### 1.1 基线设置

- 单种子（seed=0）、DDIM 50 步、EMA 权重
- 纯模型输出，**无任何后处理**
- 输出目录：`D:\LMC\data\invivo_zlx\zlx_healthy_rats_data\inference\`

### 1.2 观察到的问题（有视觉和数字证据）

**问题一：SR 输出满是盐胡椒散斑**

4/4 只大鼠的 SR 图都是密密麻麻的随机噪点，原生 9×9 数据里的中心热区在散斑中被淹没。bicubic 虽然模糊但至少干净，SR 是"又糊又花"。

基线示例（R001 scan57 slice2 Glc）：
- `inference/R001/scan_057/slice_02_Glc_comparison.png`

**问题二：前向一致性比 bicubic 还差**

把 SR 图用前向投影算子降回 9×9，跟实测 9×9 数据比，SR 的相对 L1 误差（0.535）比 bicubic（0.405）还高 32%。说明模型加的"细节"不被实测数据支持。

**问题三：梯度比虚高**

SR/bicubic 梯度比约 2.0×，看起来像"恢复了 2 倍细节"，实际全是散斑贡献的假梯度。

### 1.3 基线数字（320 样本平均）

| 代谢物 | bicubic 前向 rel-L1 | 基线 SR 前向 rel-L1 | SR/bicubic 梯度比 |
|--------|---------------------|---------------------|-------------------|
| Glx | 0.391 | 0.506 | 2.03 |
| Glc | 0.314 | 0.470 | 2.20 |
| Lac | 0.458 | 0.584 | 1.99 |
| Lipid | 0.459 | 0.582 | 2.08 |
| **全部** | **0.405** | **0.535** | **2.08** |

---

## 2. 根因分析

### 2.1 域不匹配（主因）

模型在合成体模数据上训练，训练时原生采集矩阵是 16–32；真实活体只有 **9×9**，SNR 也低得多。模型没见过这种输入，扩散采样时会在高频区域 hallucinate 出不存在的纹理。

### 2.2 缺少数据一致性约束

代码库里已经有 `core/mrsi_physics.py::refine_native_data_consistency()` 函数，`model.py::test()` 也内置了 DC 开关，但推理脚本里显式设了 `{"enabled": False}`，也没有往 batch 里塞 `LR_NATIVE` / `LR_MATRIX`。等于有刹车不用。

### 2.3 单种子随机噪声

扩散模型采样是随机过程，单种子输出的方差大。对 OOD 输入，这种方差表现为散斑。

---

## 3. 改进内容（两项，均不需重训）

### 3.1 多种子平均（Multi-seed averaging）

对同一个输入，用 N 个不同种子独立跑扩散采样，然后对输出取平均。

- 原理：散斑是零均值随机噪声，平均后互相抵消；真实结构在不同种子间一致，被保留。
- 默认 N=4（种子 0,1,2,3），可通过 `--n_seeds` 调整。
- 代价：推理时间 ×N。

### 3.2 数据一致性后处理（Data Consistency, DC）

多种子平均后，用梯度下降把 SR 图往"前向投影 = 实测 9×9"的方向拉：

```
loss = Charbonnier(forward(SR) - native_9x9) + anchor_weight * ||SR - SR_initial||²
```

- 前向算子：`mrsi_native_forward_batch`（中心 k 空间裁剪 + Hamming 窗 + 正确幅值缩放）
- 锚点项：防止 DC 把 SR 拉得偏离模型输出太远
- Charbonnier 损失：比 L2 更抗噪，不会强迫拟合噪声尖峰
- 默认参数：20 次 Adam 迭代，lr=0.02，anchor_weight=0.05
- 可通过 `--dc` 开关，`--dc_iters` / `--dc_lr` / `--dc_anchor` 调参

### 3.3 为什么先平均再 DC，而不是反过来

平均先把随机散斑压下去，DC 再在干净的平均图上做物理约束。如果先 DC 再平均，每个种子的散斑会被 DC 部分"固化"，平均效果打折。

---

## 4. 代码改动

只改了一个文件：`scripts/infer_healthy_rats_sr.py`

| 改动位置 | 内容 |
|----------|------|
| import | 新增 `center_pad_native`, `refine_native_data_consistency` |
| `_build_model_opt()` | 新增 `output_root` 参数；模型内部 DC 保持关闭（我们手动在平均后做） |
| `process_scan()` | 新增 `n_seeds`, `dc_enabled`, `dc_iters`, `dc_lr`, `dc_anchor` 参数；单种子推理改为多种子循环 + 平均 + 可选 DC |
| 指标 / NPZ | 新增 `n_seeds`, `dc_enabled`, `dc_applied`, `dc_iters`, `dc_lr`, `dc_anchor` 字段，方便回溯 |
| `main()` | 新增命令行参数 `--n_seeds`, `--dc`, `--dc_iters`, `--dc_lr`, `--dc_anchor`, `--out_name`；summary 新增 `_method` 字段 |

**向后兼容**：不加任何新参数时（`--n_seeds 1`，不加 `--dc`），行为与原脚本完全一致。

---

## 5. 复现命令

```powershell
$PY = "D:\code_software\Anaconda\envs\msr_mrsi\python.exe"
cd D:\LMC\projects\Image-Super-Resolution-via-Iterative-Refinement

# 基线（与原脚本等价，输出到 inference/）
& $PY scripts\infer_healthy_rats_sr.py --rats R001,R002,R003,R004 --out_name inference

# 改进版（4 种子平均 + DC，输出到 inference_improved/）
& $PY scripts\infer_healthy_rats_sr.py --rats R001,R002,R003,R004 `
    --n_seeds 4 --dc --dc_iters 20 --dc_anchor 0.05 `
    --out_name inference_improved
```

> 注意：Python 环境用 `msr_mrsi`（有 pydicom），不是 `PRAC_MRSI`。

---

## 6. 改进结果

### 6.1 数字对比（320 样本平均）

| 代谢物 | bicubic rel-L1 | 基线 SR rel-L1 | 改进后 SR rel-L1 | 基线梯度比 | 改进后梯度比 |
|--------|---------------|----------------|-----------------|-----------|-------------|
| Glx | 0.391 | 0.506 | **0.226** | 2.03 | 1.47 |
| Glc | 0.314 | 0.470 | **0.123** | 2.20 | 1.53 |
| Lac | 0.458 | 0.584 | **0.308** | 1.99 | 1.44 |
| Lipid | 0.459 | 0.582 | **0.306** | 2.08 | 1.42 |
| **全部** | **0.405** | **0.535** | **0.241** | **2.08** | **1.47** |

**关键变化**：
- SR 前向 rel-L1：0.535 → 0.241，下降 **55%**
- SR 从"比 bicubic 差 32%"变成"比 bicubic 好 **41%**"
- 梯度比：2.08 → 1.47，虚高的假细节被压制

### 6.2 视觉对比

同一样本（R001 scan57 slice2 Glc）：

| | 基线 | 改进后 |
|---|---|---|
| SR 图 | 盐胡椒散斑覆盖全图，中心热区模糊 | 干净平滑的椭圆热区，结构清晰 |
| 前向残差 | 中心蓝色（低估）+ 周边红色（高估），幅度大 | 残差接近 0，幅度大幅缩小 |
| SR-bicubic 差值 | 全图随机噪点 | 结构内增强（红）、结构外去模糊（蓝） |

改进后示例：
- `inference_improved/R001/scan_057/slice_02_Glc_comparison.png`
- `inference_improved/R003/scan_036/slice_02_Glc_comparison.png`
- `inference_improved/R004/scan_060/slice_02_Glc_comparison.png`

4/4 只大鼠、4/4 代谢物均观察到一致改善。

---

## 7. 局限

1. **前向算子是简化的**：用的是通用 Hamming 窗 k 空间截断，不是真实 Bruker 采集算子。DC 的"物理一致性"是近似的。
2. **没有 HR 金标准**：活体数据没有配对的高分辨率 MRSI，所以只能用前向一致性和视觉判断，不能算 PSNR/SSIM。
3. **DC 参数未系统调优**：20 次迭代 / anchor=0.05 是经验值，可能还有提升空间。
4. **多种子增加推理时间**：4 种子约为单种子的 4 倍时间（A6000 上全量 320 样本约 30 分钟）。
5. **模型本身没变**：这是推理时的后处理改进，不是模型能力的根本提升。域不匹配的问题仍然存在，只是被 DC 掩盖了。
6. **T1/FLAIR 仍是 T2 顶替**：活体没有单独的 T1/FLAIR 扫描，条件输入有信息缺失。

---

## 8. 后续建议（与重训相关）

用户计划重新生成一批仿真数据，重训时建议：

1. **训练退化范围覆盖 9×9**：当前训练 block∈[16,32]，真实活体是 9×9。把退化范围扩到 [8,32] 甚至 [6,32]，让模型见过小矩阵输入。（`core/improved_degradation.py` 已支持可变 block，`data/improved_waterfilm_dataset.py` 已实现运行时重退化）
2. **加入活体风格数据增强**：低 SNR、9×9 小矩阵、T2-only 条件，缩小域差。
3. **训练时就开 DC loss**：`model.py` 已支持 `native_data_consistency`，训练配置里打开可以让模型自己学会输出物理一致的结果，减少推理时对后处理的依赖。
4. **重训后用同一套推理脚本（带 DC）复评**：对比"重训模型 + DC" vs "旧模型 + DC" vs "重训模型无 DC"，分离模型改进和后处理改进的贡献。
5. **真实 Bruker 前向算子**：如果能拿到 Bruker 采集的梯度编码和滤波参数，替换 `mrsi_native_forward_batch` 里的 Hamming 近似，DC 会更准确。

---

## 9. 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `scripts/infer_healthy_rats_sr.py` | 修改 | 新增多种子平均 + DC 后处理 + 可配置输出目录 |
| `reports/invivo_sr3_inference_improvement.md` | 新增 | 本文档 |
| `core/mrsi_physics.py` | 未改 | DC 函数已存在，本次只是调用它 |
| `model/model.py` | 未改 | 内置 DC 开关已存在，本次在推理脚本中手动调用 |
| 模型权重 | 未改 | `I300000_E2522_ema_gen.pth` 保持原样 |
