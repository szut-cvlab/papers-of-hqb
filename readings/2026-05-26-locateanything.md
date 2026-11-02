# LocateAnything: Fast and High-Quality Vision-Language Grounding with Parallel Box Decoding

- **来源**: https://arxiv.org/abs/2605.27365
- **本地PDF**: `../raw/2026-05-26-locateanything.pdf`
- **本地LaTeX**: `../raw/2026-05-26-locateanything-latex/`
- **参考链接**: https://research.nvidia.com/labs/lpr/locate-anything/（项目主页）；https://github.com/NVlabs/Eagle/tree/main/Embodied（官方代码）
- **公众号/知乎推文**: [LOCATEANYTHING：通过并行框解码实现快速且高质量的视觉语言定位，论文解读与实战指南](https://mp.weixin.qq.com/s?__biz=Mzk5MDY3NDk3Ng==&mid=2247485062&idx=1&sn=163f33b49b15631e8e81fc03374d7e2a)（微信公众号“遇见CV啊”；发布日期未显示）；[告别“逐字蹦坐标”！LOCATEANYTHING：并行框解码让VLM定位速度提升2.5倍，精度再创新高](https://mp.weixin.qq.com/s?__biz=MzkzMTIyMjQ2MA==&mid=2247499124&idx=1&sn=ee962cc965a2c844413312bb4acefce5)（微信公众号“AI智韵”；发布日期未显示）；知乎未检索到
- **日期**: 2026-05-26
- **标签**: `locateanything`, `parallel-box-decoding`, `vision-language-grounding`, `object-detection`, `gui-grounding`, `ocr`, `eccv2026`
- **研究方向**: Multimodal Learning → Vision-Language Grounding → 快速检测与定位
- **作者**: Shihao Wang, Shilong Liu, Yuanguo Kuang, Xinyu Wei, Yangzhou Liu, Zhiqi Li, Yunze Man, Guo Chen, Andrew Tao, Guilin Liu, Jan Kautz, Lei Zhang, Zhiding Yu（NVIDIA；香港理工大学；普林斯顿大学；南京大学；UIUC）
- **提交日期**: 2026-05-26 (v1)；2026-05-27 (v2)
- **会议/期刊**: ECCV 2026（官方仓库 2026/06 标注接收）
- **代码**: https://github.com/NVlabs/Eagle/tree/main/Embodied
- **项目主页**: https://research.nvidia.com/labs/lpr/locate-anything/

---

## 摘要

**问题**: 视觉语言模型（VLM）通常把视觉定位/检测建模为“坐标 token 生成”：把二维边界框序列化成多个一维 token，逐个自回归解码。这种 token-by-token 解码与框几何的耦合结构不匹配，且严格串行生成造成实际推理瓶颈。

**方案**: 提出 **LocateAnything**，基于 **Parallel Box Decoding（PBD，并行框解码）** 的统一生成式定位与检测框架。PBD 把边界框、点等几何元素作为**原子单元**在单步内解码，保留框内几何一致性，同时释放大量并行性。

**关键创新/贡献**：
- **创新点1（PBD）**: 以固定长度 6 的 block 为原子单位，一个 block 同时输出 `<box>`、四个量化坐标、`</box>`；块间因果、块内双向注意力。
- **创新点2（NTP+MTP 联合训练）**: 同一份 GT 同时构造 token 级 NTP 序列与 block 级 MTP 序列，共享视觉/文本上下文但用异构 attention mask 隔离，既保留因果推理又学并行框预测。
- **创新点3（按需推理）**: Fast（MTP）、Slow（NTP）、Hybrid（默认并行，格式异常/空间歧义时局部回退 NTP）三种模式。
- **创新点4（数据引擎）**: **LocateAnything-Data**，12M 唯一图像、138M 自然语言查询、785M 标注框，覆盖检测、GUI、指代、OCR、文档布局、点定位六类任务。

**验证**:
- LVIS mean F1 **50.7**（Rex-Omni 46.9），COCO mean F1 **54.7**（Rex-Omni 52.9），Hybrid 吞吐 **12.7 BPS**（Rex-Omni 5.0，Qwen3-VL 1.1）（Table 1）。
- 密集检测：Dense200 mean F1 **58.7**、VisDrone **39.9**（Rex-Omni 58.3 / 35.8）（Table 2）。
- GUI Grounding：ScreenSpot-Pro 平均 **60.3**（GUI-Owl-32B 58.0，Qwen3-VL-30B-A3B 53.7）（Table 3）。
- 文档/OCR：DocLayNet **76.8**、M6Doc **70.1**、TotalText **43.3**（Table 4）。
- 指代：HumanRef **78.7**、RefCOCOg val **76.7** / test **77.6**（Table 5）。

---

## 1. 引言

**背景 & 动机**:
- VLM 正成为交互式/具身系统的通用骨干；要在真实世界行动，VLM 必须高质量、低延迟地把自然语言意图定位到图像中的实体（物体、UI 元素、区域）。
- 现有生成式定位方法（Pix2Seq、Kosmos、Rex-Omni 等）把坐标表示为**文本数字**（“1024”拆成“1/0/2/4”）或**量化 token**（x1→y1→x2→y2），本质都是把 2D 几何对象序列化成 1D 流，推理时必须逐 token 生成。
- 这带来两个问题：
  1. **速度瓶颈**：每个框要 4–8 个 token，100 个目标要数百步解码；
  2. **几何失配**：四个坐标强耦合（x1<x2、y1<y2），逐 token 独立学习/解码容易生成畸形框、坐标偏移。
- 通用 Multi-Token Prediction（MTP）虽能并行预测多个 token，但通常**结构无关**：随机选位置/随机分块，会跨框、跨类别学习虚假组合，导致精度下降。

**本文思路**:
- 不做“任意 token 块”的 MTP，而是把**整个边界框作为原子块**：一个 block 恰好包含一个几何单元，块内 token 双向交互，块间保持因果。
- 用 NTP + MTP 双范式联合训练：NTP 保留语言因果推理，MTP 学框对齐并行预测。
- 推理时默认并行，检测到格式错误/空间歧义就局部回退 NTP，兼顾吞吐与鲁棒性。
- 配套 138M 查询的大规模数据，覆盖通用检测、GUI、指代、OCR、文档、点定位。

**贡献**:
1. LocateAnything：把 MTP 用于 VLM 检测/grounding 的早期探索，通过 PBD 做框对齐解码，同时提升吞吐与精度。
2. Hybrid 解码策略：检测不可靠的并行块，只对问题块做 NTP 重解码，降低最坏情况失败。
3. 在布局 grounding、长尾检测、GUI grounding 等任务上推进速度-精度前沿，最高 **2.5×** 吞吐提升。

---

## 2. 相关工作

### 2.1 VLM 中的视觉检测与定位

- 传统方法依赖任务专用头（DETR、Grounding DINO、Faster R-CNN、Far3D 等）。
- 近期 VLM（Qwen-VL 系列、InternVL、Shikra）把定位建模为自回归 token 生成；但存在结构幻觉与高延迟。
- 缓解方向：
  - Rex-Omni 用 point-based 预测；
  - PaDT、Groma 用 visual reference token 直接指向图像 patch；
  - Pink、ViP-LLaVA、Griffon、DnU、PAM 用 visual prompt / 多粒度特征增强指代理解；
  - LLMDet 调数据分布提升召回；WeDetect 把检测视为并行检索；
  - CoT4Det、Vision-R1、UniVG-R1、GW-VLM 用 CoT/强化学习减少 grounding 错误。

### 2.2 通过 MTP 与 Diffusion LLM 的并行解码

- MTP（Medusa、EAGLE 等）一次预测多个未来 token，常配合 speculative decoding 加速。
- Diffusion Language Model（LLaDA、Dream、DiffuCoder）把生成建模为离散去噪，支持双向上下文与非自回归解码。
- 混合半自回归（Block Diffusion、SDLM、Fast-dLLM v2）按固定大小 token block 并行解码，保持块间因果以兼容 KV cache。
- 这些方法大多**结构无关**；LocateAnything 的区别是：block 与边界框严格对齐，一个 block 就是一个几何原子，避免任意分块造成的跨框/跨类虚假模式。

---

## 3. 方法

### 3.1 模型架构与输出形式

**架构**:
- **视觉编码器**：Moon-ViT，native resolution 提取视觉 token $Z=\mathrm{Encoder}(\mathcal{I})$，保留高分辨率空间细节。
- **语言解码器**：Qwen2.5。
- **连接**：MLP projector。
- 视觉 token 与文本查询 $\mathcal{E}$ 送入 LM，直接输出 block 级预测。

**Block 化输出**:
- 连续坐标归一化到 $[0,1000]$，离散化为 token（沿用 Pix2Seq/Rex-Omni 式量化）。
- 输出被组织为 block 序列 $\mathbf{B}=(b_1,\dots,b_N)$，联合概率：

$$
P(\mathbf{B}\mid Z,\mathcal{E})=\prod_{i=1}^{N}P(b_i\mid b_{<i},Z,\mathcal{E}).
$$

- 每个 block 是**固定长度 $L=6$ 的原子单元**：一个框 + 两个结构 token（`<box>`、`</box>`）。空位用 `<null>` padding，保证并行解码时张量形状统一。

**四种 block**:
1. **Semantic Block**：编码语言身份/类别短语；若表达超过一个 block 容量，则拆成多个连续 block。
2. **Box Block**：四个量化坐标，表示边界框。
3. **Negative Block**：显式表示“查询目标不存在”，用于拒识。
4. **End Block**：表示生成结束。

### 3.2 训练设计：NTP + MTP 双范式

直接把输出并行化会破坏模型原有的因果推理。为此作者用**双形式训练**，同一份 GT 同时构造两种表示：

$$
x_{\text{all}}=x_{\text{vis}}\oplus x_{\text{q}}\oplus x_{\text{ntp}}\oplus x_{\text{blk}},
$$

- $x_{\text{vis}}$、$x_{\text{q}}$：共享的视觉与文本查询上下文；
- $x_{\text{ntp}}$：标准 NTP token 序列；
- $x_{\text{blk}}$：block 级 MTP 序列，由 $x_{\text{ntp}}$ 从左到右按 block 规则切分/填充得到。

**Block MTP 构造**:
- 每个 block 保留第一个 token 作为预测上下文，其余 token 替换为 `[mask]`；
- 模型在单步内同时预测 block 内所有被 mask 的 token；
- 若 block size=1，MTP 退化为标准 NTP。

**异构 Attention Mask**:
- **NTP 因果**：共享上下文与 $x_{\text{ntp}}$ 用标准 causal mask，且不能看 $x_{\text{blk}}$，防止泄漏；与推理时 KV cache 一致。
- **块间因果**：$x_{\text{blk}}$ 中不同 block 严格因果，当前 block 可看共享上下文和所有已提交 block，不能看未来 block；学习框间依赖，减少重复/漏检。
- **块内双向**：同一 block 内 token 全连接，捕捉坐标间的几何依赖，并在一个功能单元内同时解出。

**目标**:

$$
\mathcal{L}=\mathcal{L}_{\mathrm{ntp}}+\mathcal{L}_{\mathrm{mtp}}.
$$

**系统实现**:
- **Stream Packing**：把变长样本在线打包成约 36,864 token 的密集序列，best-fit buffering，>95%  packing 效率。
- **MagiAttention**：原生支持 NTP 因果 + 块间因果 + 块内双向的异构 mask，避免传统 FlashAttention 只支持统一 causal/full 的限制。

### 3.3 按需推理：三种模式

并行解码在复杂场景有两种失败：
1. **Format Irregularity**：类别边界处犹豫，block 内混入结构 token 与坐标 token，语法畸形；
2. **Spatial Ambiguity**：密集网格排列时，MTP 把边界模糊成两个物体之间的中间坐标，IoU 低。

**NTP fallback 机制**:
- 监测语法完整性与空间置信度；
- 触发条件（同时满足）：
  - top-1 坐标 token 概率 **< 0.7**；
  - top-5 坐标 token 在 [0,1000] 空间中的 max-min 差 **> 80**。
- 触发后丢弃当前问题 block，回退到最后一个已验证前缀，用 NTP 自回归重解码该 block；完成后切回 MTP。

**三种模式**:
1. **Slow Mode（NTP）**：逐 token 生成，最稳定，适合高精度标注/离线评测。
2. **Fast Mode（MTP）**：并行预测完整 block，吞吐最高；丢弃 `<null>` padding，把已提交 token 写入 KV cache。
3. **Hybrid Mode**：默认 MTP，检测到格式错误/空间歧义时局部回退 NTP；默认评测模式。

**推理细节**:
- KV cache 在每个 MTP step 后截断到已提交 token，驱逐 mask token 与重复 anchor，保持与训练时因果前缀一致。
- 生成：nucleus sampling，temperature 0.7、top-p 0.9、repetition penalty 1.1、block size 6、max new tokens 8192、BF16、batch size 1。

### 3.4 LocateAnything-Data

**规模**:
- **12M** 唯一图像；
- **138M** 自然语言查询；
- **785M** 标注框；
- 附录：总查询超过 **139M**，负样本超过 **22M**。

**六类任务（按查询占比）**:
1. **General object detection**：66.9% 查询、83.1% 框；基础密集框监督。
2. **GUI grounding**：16.5%；面向具身智能体、GUI 导航。
3. **Natural language referring**：7.3%；复杂语言意图→特定区域。
4. **Text localization**：3.6%；文字区域紧致定位。
5. **Document and scene layout grounding**：3.5%；文档/场景结构理解。
6. **Point-based localization**：2.2%；点级细粒度定位。

**数据引擎**:
- **有 GT 框的检测数据**：以每个框的类别为 prompt，让 Qwen3-VL 合成属性/空间关系/推理线索的 object-centric 查询；再用 Molmo 预测候选点，只保留落在 GT 框内的点作为可靠监督。
- **无标注图像**：Qwen3-VL 直接生成多样查询 → Molmo 预测点 → SAM 3 生成框，或直接让 Rex-Omni 出框 → Qwen3-VL 后验校验。
- **负样本**：构造图中不存在的目标查询，分配 Negative Block，让模型学会拒识；各域负样本比例按数据统计设置。
- **GroundCUA 增强**：用 Qwen3-VL 从 appearance/spatial/functional 三个角度把短 UI 标签扩写成自然语言查询。

---

## 4. 实验

### 4.1 训练与评测设置

**训练**:
- **Base VLM 预训练**（前两阶段）：
  - Stage 1：caption/knowledge 数据，视觉概念初始化；
  - Stage 2：加入数学/代码、科学、图表、OCR、VQA、纯文本指令、计数等，**不含检测/框数据**。
- **LocateAnything SFT**（后两阶段）：
  - Stage 3：138M 查询混合数据，全参数解冻，最大序列 25,600，lr $4\times10^{-5}$，cosine；
  - Stage 4：通用数据比例降到 20%，增加 MOT20Det、SKU110K 等多目标图像，lr 衰减到 $1\times10^{-5}$。
- COCO/LVIS 评测短边 resize 到 840；其他 benchmark 用原分辨率。
- 吞吐在单张 H100、batch size 1 上测 BPS（boxes per second）。

**对比方法**:
- 专用检测器：DETR、Deformable-DETR、Grounding DINO、DocLayout-YOLO、PaddleOCRv5；
- 通用 VLM：Qwen3-VL、DeepSeek-VL2、OVIS2.5、MiMo-VL、SEED1.5-VL；
- VLM 检测专家：Rex-Omni；
- GUI 专用：GUI-Owl、MAI-UI、UI-Venus、ScaleCUA 等。

**指标**:
- 框任务：F1@IoU 0.5 / 0.95 / mean（mIoU）。
- 点任务：预测点落在 GT mask/框内即正确，报 F1@Point。

### 4.2 主结果

**通用检测（Table 1）**:

| 方法 | 吞吐 BPS | 零样本 LVIS | LVIS mean F1 | 零样本 COCO | COCO mean F1 |
|---|---:|---|---:|---|---:|
| Grounding DINO-Swin-T | — | Yes | 38.8 | Yes | 56.6 |
| DINO-Swin-L（闭集） | — | — | — | — | **62.1** |
| Qwen3-VL-4B | 1.1 | — | 43.5 | — | 46.1 |
| SEED1.5-VL | — | Yes | 46.7 | Yes | 51.4 |
| Rex-Omni-3B | 5.0 | Yes | 46.9 | Yes | 52.9 |
| **LocateAnything-3B** | **12.7** | Yes | **50.7** | Yes | **54.7** |

- 相比 Rex-Omni：LVIS mean +3.8、COCO mean +1.8，吞吐 2.5×；相比 Qwen3-VL，吞吐约 11.5×。
- 注意 COCO mean F1 仍低于闭集 DINO-Swin-L 的 62.1；后者是专用检测器、非生成式、非零样本口径，不能直接等同。

**密集检测（Table 2）**:

| 方法 | Dense200 F1@0.5 | Dense200 F1@0.95 | Dense200 mean | VisDrone F1@0.5 | VisDrone F1@0.95 | VisDrone mean |
|---|---:|---:|---:|---:|---:|---:|
| Rex-Omni-3B | **78.4** | 10.3 | 58.3 | 61.6 | 1.5 | 35.8 |
| **LocateAnything-3B** | 74.0 | **18.5** | **58.7** | **63.0** | **3.2** | **39.9** |

- 平均 F1 更高，主要来自高 IoU 阈值：Dense200 F1@0.95 从 10.3 提升到 18.5，VisDrone 从 1.5 到 3.2。
- 说明 PBD 的框内几何一致性对高精度定位有帮助，而不仅是速度。

**GUI Grounding（Table 3）**:
- ScreenSpot-Pro 平均 **60.3**，超过 GUI-Owl-32B（58.0）、UI-Venus-1.5-2B（57.7）、MAI-UI-2B（57.4）、Qwen3-VL-30B-A3B（53.7）。
- 细类优势：Icon 类提升明显（Dev Icon 50.3、Creative Icon 46.9、Sci. Icon 58.2、Office Icon 69.8）。

**文档布局与 OCR（Table 4）**:

| 方法 | DocLayNet mean | M6Doc mean | TotalText mean |
|---|---:|---:|---:|
| DocLayout-YOLO | **81.1** | — | — |
| PaddleOCRv5 | — | — | 25.7 |
| Rex-Omni-3B | 70.7 | 55.6 | 40.6 |
| **LocateAnything-3B** | 76.8 | **70.1** | **43.3** |

- DocLayNet 低于专用 DocLayout-YOLO（76.8 vs 81.1），但远高于 Rex-Omni。
- M6Doc 70.1、TotalText 43.3 为通用模型最佳。

**指代理解（Table 5）**:

| 方法 | HumanRef mean | RefCOCOg val mean | RefCOCOg test mean |
|---|---:|---:|---:|
| DeepSeek-VL2-Small | 64.7 | **81.4** | **81.6** |
| SEED1.5-VL | **81.6** | 71.9 | 73.2 |
| Rex-Omni-3B | **79.9** | 73.6 | 74.3 |
| **LocateAnything-3B** | 78.7 | **76.7** | **77.6** |

- RefCOCOg 明显优于 Rex-Omni（val +3.1、test +3.3），但 HumanRef 略低于 Rex-Omni/SEED。
- 说明 PBD 在通用指代上强，但“拒绝不存在目标/HumanRef”仍有提升空间。

### 4.3 消融

Table 6（COCO，平均 IoU 阈值）：

**(a) 坐标表示**:

| 方法 | 吞吐 BPS | R | P | F1 |
|---|---:|---:|---:|---:|
| Textual | 1.3 | 45.7 | 52.3 | 49.1 |
| Quantized | 3.9 | 48.2 | 52.2 | 50.1 |
| **PBD Slow** | 3.9 | **49.4** | **55.2** | **52.1** |
| PBD Fast | **16.9** | 45.6 | 54.6 | 49.6 |
| PBD Hybrid | 13.2 | 48.7 | 54.8 | 51.6 |

- 即使同为 NTP，PBD 的框对齐表示也比文本/量化表示高 **1.0–3.0 F1**，说明“框作为一个单元”本身就是更好的监督。
- Fast 最快但精度下降；Hybrid 保留大部分速度，精度接近 Slow。

**(b) MTP 形式**:

| 方法 | 吞吐 | F1 |
|---|---:|---:|
| SDLM-B4 | 5.2 | 46.5 |
| SDLM-B6 | 5.5 | 46.1 |
| SDLM-B8 | 6.7 | 45.8 |
| Block Diff-B6 | 4.7 | 44.8 |
| **PBD Fast** | **16.9** | **49.6** |

- 结构无关 MTP 随 block 增大只小幅加速、精度持续下降；PBD 框对齐 block 同时取得 16.9 BPS 与 49.6 F1。

**(c) 解码模式与损失**:
- 仅 NTP loss + Slow：50.1 F1；
- 仅 MTP loss + Fast：47.2 F1；
- 联合训练 + Slow：**52.1**；
- 联合训练 + Fast：49.6 @16.9 BPS；
- 联合训练 + Hybrid：**51.6 @13.2 BPS**。
- 联合训练把 Slow 上限从 50.1 提到 52.1，说明 NTP 与 MTP 互补。

**框排序**:
- X-Y Corner Order（按左上角 x 再 y）F1 最高，设为默认；Center Distance、Area、Random 都更低。

**吞吐随目标数**:
- 目标从 20 增到 300，NTP 延迟严重增长；PBD 生成时间增长很小，吞吐从 12 BPS 升到约 25 BPS，取得 2–6× 加速。

### 4.4 定性结果

- **组合式 grounding**：属性/部件/空间/推理查询均能对齐。
- **大实例数鲁棒**：从稀疏到拥挤，框保持结构化；Stage 4 多目标训练增强密集场景。
- **杂乱场景**：遮挡、重复纹理、网格布局下框仍紧凑分离；Hybrid 通过回退稳定难例。

---

## 5. 结论

**总结**:
- LocateAnything 用 PBD 把几何元素提升为原子单元，使训练监督与坐标耦合结构对齐。
- 138M 查询 + 按需推理机制，在多种定位任务上取得 SOTA 精度，同时最高 **2.5×** 加速。
- 为延迟敏感的具身机器人、交互式智能体提供可部署的实时视觉感知路线。

**局限**（作者明确）:
- 目前主要用监督微调（SFT）；强化学习是下一步，用于优化块级解码策略、降低回退频率、改善密集/长尾难例的探索。
- Hybrid 的回退会带来额外开销；Fast 模式在复杂场景精度下降。
- 数据规模大但依赖合成/自动标注引擎，存在噪声与领域偏差风险。
- 高 IoU 定位仍受量化坐标精度限制。

---

## 附录 A. 训练与推理配置

- **四阶段训练**：
  1. Stage 1：caption/knowledge，视觉概念初始化；
  2. Stage 2：全部多模态数据（不含检测/框）；
  3. Stage 3：138M 查询，全参数解冻，seq 25,600，lr 4e-5；
  4. Stage 4：通用数据 20%，增加 MOT20Det/SKU110K，seq 25,600，lr 1e-5。
- **Stream Packing**：约 36,864 token/包，best-fit buffering，>95% 利用率。
- **MagiAttention**：处理 NTP causal + block-causal + intra-block bidirectional 的异构 mask。
- **推理**：temperature 0.7、top-p 0.9、repetition penalty 1.1、block size 6、max new tokens 8192、BF16、batch 1。
- **吞吐**：单张 H100、COCO 上测 BPS；COCO/LVIS 短边 840，其他原分辨率。

## 附录 B. LocateAnything-Data 构建

- 开源检测/grounding 数据统一清洗、格式归一化。
- GroundCUA 用 Qwen3-VL 从 appearance/spatial/functional 三维度扩写 UI 查询。
- 指代数据聚合 Flickr30k Entities、gRefCOCO、RefCOCO、HumanPart、HumanRef。
- 检测数据聚合 OpenImages、Objects365、Unsplash。
- 有 GT 框：Qwen3-VL 生成查询 → Molmo 预测点 → 保留框内点。
- 无标注：Qwen3-VL 生成查询 → Molmo 点 → SAM 3 框 / Rex-Omni 框 → Qwen3-VL 后验校验。
- 负样本：构造不存在目标查询，分配 Negative Block；总负样本 >22M。
- 统计：>139M 查询；每查询目标数长尾分布，多数少目标、少量多目标。

## 附录 C. 额外实验

- **Pointing**：COCO F1@Point 83.9、Dense200 87.6、HumanRef 84.7、RefCOCOg test 91.0。
- **Backbone 泛化**：同样 PBD 设计用于 Qwen3-VL-4B，COCO F1 50.8→52.2（Slow）、52.0（Hybrid），BPS 2.8→9.4（Hybrid），说明 PBD 不依赖特定 backbone。
- **模式分析**：Slow 峰值 F1 55.1（COCO）、79.8（DocLayNet）；Fast 15.3 BPS；Hybrid 12.7 BPS。
- **定性对比**：REC、密集检测、OCR 三组，PBD 在框紧致性/分离度上优于 Qwen3-VL 与 Rex-Omni。

---

## 本文涉及到的模型具体结构

| 模块 | 结构 | 说明 |
|---|---|---|
| 视觉编码器 | Moon-ViT | native resolution，保留细粒度空间细节 |
| 语言解码器 | Qwen2.5 | 生成 block 序列 |
| 连接 | MLP projector | 视觉 token → LM |
| 输出单元 | block，长度 L=6 | `<box>` + 4 坐标 + `</box>`，空位 `<null>` |
| Block 类型 | Semantic / Box / Negative / End | 类别短语、框、拒识、结束 |
| 训练 | NTP + block MTP 联合 | 异构 mask：NTP causal、块间 causal、块内 bidirectional |
| 推理 | Fast / Slow / Hybrid | Hybrid 检测格式/空间歧义后局部 NTP 回退 |
| 模型规模 | 3B | LocateAnything-3B |

---

## 实验的全部流程

### 数据准备
1. 收集/清洗开源检测、grounding、GUI、OCR、文档、指代数据。
2. 有 GT 框数据：Qwen3-VL 生成 object-centric 查询，Molmo 预测点，保留框内点。
3. 无标注图像：Qwen3-VL 生成查询，Molmo 点 → SAM 3/Rex-Omni 框，Qwen3-VL 校验。
4. 构造负样本（不存在目标），分配 Negative Block。
5. 统一为 block 序列；坐标归一化 [0,1000]，按 X-Y Corner Order 排序。

### 训练
6. Base VLM Stage 1：caption/knowledge；Stage 2：全多模态数据（不含检测）。
7. LocateAnything Stage 3：138M 查询，全参数解冻，seq 25,600，lr 4e-5。
8. Stage 4：通用数据降至 20%，增加密集多目标数据，lr 1e-5。
9. 同一 GT 构造 NTP 与 block MTP 两条序列；异构 mask 隔离；损失 $L_{ntp}+L_{mtp}$。
10. Stream packing + MagiAttention 支持长序列/异构 mask 高效训练。

### 推理与评测
11. 默认 Hybrid：MTP 并行出块；top-1 坐标概率 <0.7 且 top-5 极差 >80，或格式异常 → 局部 NTP 回退。
12. 输出 block 反序列化为框/点；`<null>` 丢弃，已提交 token 写 KV cache。
13. 通用检测：COCO/LVIS，F1@IoU 0.5/0.95/mean，BPS。
14. 密集：Dense200/VisDrone；GUI：ScreenSpot-Pro；文档/OCR：DocLayNet/M6Doc/TotalText；指代：HumanRef/RefCOCOg。
15. 消融：坐标表示、MTP 形式、解码模式/损失、框排序、吞吐随目标数。
16. 附录：pointing、backbone 泛化、三模式详细结果、定性对比。

---

## 个人评价

**核心价值**: LocateAnything 把“并行解码”从语言模型技巧变成**几何原生的输出表示**：不是随便切 token block，而是让 block 的边界与边界框的边界重合。这样既减少解码步数，又给模型一个强结构先验（四个坐标必须一起、且满足几何关系）。对遥感/无人机密集小目标，这一点尤其关键：高 IoU 指标（F1@0.95）提升远大于平均 F1，说明它改善的是框的紧致性，而不只是召回。

**可借鉴点**:
- **原子化输出**：把领域结构（框、多边形、点、航道、地块边界）作为原子 block，而不是拆成 token 流；可迁移到遥感旋转框、建筑物轮廓、道路中心线。
- **NTP+MTP 联合训练**：不要直接丢掉自回归因果能力；用双序列 + 异构 mask 同时保留语言推理与并行结构化输出。
- **Hybrid 回退**：用置信度/极差触发局部 NTP，比全局慢速或全局快速更实用；适合生产环境。
- **数据引擎**：用强 VLM 生成查询 + 点模型 + SAM 分割 + 后验校验，可低成本扩充领域 grounding 数据；负样本显式建模“不存在”。
- **高 IoU 指标**：遥感检测不能只看 mAP/F1@0.5；F1@0.95 才能反映框的几何质量。
- **注意口径**：COCO mean F1 54.7 仍低于闭集 DINO-Swin-L 62.1，不能宣称全面超过专用检测器；它的优势是零样本/开放词汇 + 速度。

**实验设计亮点**:
- 同时报告吞吐 BPS 与 F1@0.5/0.95/mean，速度-精度权衡清晰。
- 消融把“坐标表示”和“MTP 形式”分开，证明收益来自框对齐而非单纯并行。
- 与 Rex-Omni（最接近的 VLM 检测专家）、Qwen3-VL（NTP 代表）、专用检测器/OCR/GUI 模型多线对比。
- 附录补充 pointing、backbone 泛化、三模式与定性，证据较完整。
- 诚实报告 HumanRef 低于 Rex-Omni、DocLayNet 低于 DocLayout-YOLO、COCO 低于闭集 DINO。

---

## 疑问解答 (Q&A)

### Q1: PBD 与普通 MTP/Block Diffusion 的本质区别是什么？

**答**：普通 MTP 把 token 流按固定大小/随机位置切块，块边界与语义结构无关，会跨框、跨类别学习虚假组合。PBD 把 block 边界与**边界框边界**对齐：一个 block 恰好是一个 `<box> + 4 坐标 + </box>` 的原子单元，块内双向注意力专门建模坐标几何耦合。因此 PBD 不是“更快的 MTP”，而是“结构对齐的 MTP”。

### Q2: 为什么 block 长度是 6？

**答**：一个框需要 4 个量化坐标，加上 `<box>` 与 `</box>` 两个结构 token，共 6 个位置。空位用 `<null>` padding，保证所有 block 等长、可并行张量化。语义 block 如果超过一个 block 容量，会拆成多个连续 block；负样本/结束 block 也复用同一长度。

### Q3: NTP 与 MTP 双序列训练会不会互相泄漏？

**答**：不会。作者用异构 attention mask 把两条流隔离：NTP 流与共享上下文是标准 causal，且不能看 MTP 流；MTP 流块间 causal、块内 bidirectional，但只能看共享上下文和已提交块，不能看 NTP 流。两条流共享视觉/文本上下文，但答案表示互相不可见。

### Q4: Hybrid 模式什么时候回退？回退粒度是什么？

**答**：当当前 block 出现格式异常（结构 token 与坐标 token 混排），或空间歧义（top-1 坐标概率 <0.7 且 top-5 坐标极差 >80）时，丢弃该 block，回退到最后一个已验证前缀，用 NTP 逐 token 重解码**这个 block**，完成后切回 MTP。回退是局部的，不是整段重来。

### Q5: 12.7 BPS 与 15.3 BPS 分别是什么口径？

**答**：12.7 BPS 是默认 **Hybrid** 模式（Table 1），15.3 BPS 是 **Fast** 模式（附录模式分析）。Fast 最快但复杂场景精度下降；Hybrid 保留大部分速度且更稳。论文主表用 Hybrid，引用时不要混用。

### Q6: 数据引擎如何保证自动标注质量？

**答**：三条防线：
1. 有 GT 框时，Molmo 预测点只保留落在 GT 框内的，作为可靠正样本；
2. 无标注图像由 Qwen3-VL 生成查询，SAM 3/Rex-Omni 出框后再由 Qwen3-VL 后验校验，过滤不一致；
3. 显式构造负样本，让模型学会拒绝不存在的目标。
但仍存在合成数据噪声与领域偏差，论文未给出人工抽检精度。

### Q7: 为什么 COCO mean F1 54.7 仍低于 DINO-Swin-L 62.1？

**答**：口径不同。DINO-Swin-L 是闭集专用检测器，在 COCO 上监督训练/微调，输出连续框回归；LocateAnything 是零样本生成式 VLM，用离散坐标 token 输出，且要兼顾 GUI/OCR/指代/点定位。它的优势是开放词汇、统一接口、速度与高 IoU 框质量，不是全面取代专用检测器。

### Q8: 主要局限是什么？

**答**：作者明确：目前只做 SFT，未用强化学习优化块级解码策略；Hybrid 回退有额外开销，Fast 在复杂场景精度下降；量化坐标限制极高 IoU 精度；自动标注数据存在噪声/偏差。此外，3B 模型在 HumanRef 上仍低于 Rex-Omni/SEED，拒识能力不是最强。

---

## 相关推文（微信公众号/知乎）

- [LOCATEANYTHING：通过并行框解码实现快速且高质量的视觉语言定位，论文解读与实战指南](https://mp.weixin.qq.com/s?__biz=Mzk5MDY3NDk3Ng==&mid=2247485062&idx=1&sn=163f33b49b15631e8e81fc03374d7e2a) — 微信公众号“遇见CV啊”，发布日期未显示。
- [告别“逐字蹦坐标”！LOCATEANYTHING：并行框解码让VLM定位速度提升2.5倍，精度再创新高](https://mp.weixin.qq.com/s?__biz=MzkzMTIyMjQ2MA==&mid=2247499124&idx=1&sn=ee962cc965a2c844413312bb4acefce5) — 微信公众号“AI智韵”，发布日期未显示。
- 知乎：未检索到可验证的专门解读。

> 推文仅作传播线索，方法与数字均以 arXiv 原文和官方代码为准；推文中出现的“10 倍/2.5 倍”分别对应 Qwen3-VL 与 Rex-Omni 不同基线，不能混用。

---

## 关联笔记

- [WeDetect: Fast Open-Vocabulary Object Detection as Retrieval](/readings/2025-12-13-wedetect)：同样关注检测效率，但 WeDetect 走“双塔检索”，LocateAnything 走“生成式并行框解码”，可对照两条效率路线。
- [Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond](/readings/2023-08-24-qwenvl)：Qwen-VL 用坐标 token 自回归 + 位置感知 adapter；LocateAnything 用 Qwen2.5 + PBD，是同一“生成式定位”范式的并行化改进。
- [InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks](/readings/2023-12-21-internvl)：InternVL 用 6B 视觉编码器 + QLLaMA 做通用对齐；LocateAnything 用 3B 模型专注定位速度-精度，可对照“通用理解”与“专用定位”的取舍。
- [DETR: End-to-End Object Detection with Transformers](/readings/2020-05-26-detr)：DETR 用集合预测取消 NMS；LocateAnything 用 block 化生成取消逐 token 串行，都是对检测输出结构的重设计。
