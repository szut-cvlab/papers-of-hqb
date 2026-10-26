# Attention Is All You Need

- **来源**: https://arxiv.org/abs/1706.03762
- **本地PDF**: `../raw/2017-06-12-transformer.pdf`
- **本地LaTeX**: `../raw/2017-06-12-transformer-latex/`
- **参考链接**: https://github.com/tensorflow/tensor2tensor (官方开源实现，本笔记用于对照验证超参)
- **日期**: 2017-06-12
- **标签**: `transformer`, `self-attention`, `multi-head-attention`, `machine-translation`, `NLP`, `architecture`
- **研究方向**: Deep Learning → Sequence Modeling / NLP → Neural Machine Translation Architecture
- **作者**: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin (Google Brain / Google Research / University of Toronto)
- **提交日期**: 2017-06-12 (v1), 2023-08-02 (v7)
- **会议/期刊**: NIPS 2017 (Neural Information Processing Systems)
- **代码**: https://github.com/tensorflow/tensor2tensor
- **项目主页**: 无（论文未设独立项目主页，代码即 tensor2tensor 仓库）

---

## 摘要

**问题**: 主流的序列转换（sequence transduction，序列到序列映射）模型都基于复杂的循环或卷积网络构建的 encoder-decoder，最好的模型还要在 encoder 与 decoder 之间加注意力机制。循环模型沿符号位置逐步计算隐状态 $h_t = f(h_{t-1}, x_t)$，这种**固有的顺序性**（inherently sequential nature）排除了训练样本内的并行化；在序列较长时尤为致命，因为显存限制会约束跨样本的 batch 规模。

**方案**: 提出 Transformer —— 一个**完全基于注意力机制**、彻底抛弃循环与卷积的网络架构，用纯注意力来刻画输入与输出之间的全局依赖（global dependencies）。

**关键创新/贡献**：
- **创新点1（纯注意力架构）**: 第一个完全不依赖序列对齐 RNN 或卷积的转换模型，encoder/decoder 全部由多头自注意力（Multi-Head Self-Attention）+ 逐位置前馈网络（Position-wise Feed-Forward Network）堆叠而成。
- **创新点2（Scaled Dot-Product Attention）**: 在点积注意力上引入 $1/\sqrt{d_k}$ 缩放因子，解决 $d_k$ 较大时点积幅值过大把 softmax 推入梯度极小区域的问题。
- **创新点3（Multi-Head Attention）**: 将 $d_{model}$ 维的 Q/K/V 用 $h$ 组不同 learned linear projections 投影到低维（$d_k=d_v=d_{model}/h=64$），并行做 $h$ 次注意力再拼接投影，让模型能**联合关注来自不同表示子空间、不同位置的信息**，同时抵消"注意力加权平均"带来的分辨率损失。
- **创新点4（可并行 + 训练成本骤降）**: 路径长度为 $O(1)$、顺序操作数 $O(1)$，训练可完全并行，SOTA 质量的训练成本只有此前最优模型的极小一部分。

**验证**：在 WMT 2014 英德（EN-DE）与英法（EN-FR）两个机器翻译任务上评测。
- EN-DE：**28.4 BLEU**（big 模型），比此前最好的结果（含集成模型）高出 **2.0 BLEU 以上**（Table 2）。
- EN-FR：**41.8 BLEU**（big 模型），**8 块 P100 GPU 训练 3.5 天**（Table 2 与 Section 6.1）。
- base 模型训练成本 $3.3\cdot10^{18}$ FLOPs，big 模型 $2.3\cdot10^{19}$ FLOPs，而对比的 GNMT+RL 集成需 $1.8\cdot10^{20}$（EN-DE）（Table 2）。
- 泛化性验证：英语成分句法分析（English constituency parsing），WSJ 仅训练时 **F1 91.3**，半监督时 **92.7**（Table 4）。

---

## 1. Introduction

**背景 & 动机**:
- RNN、特别是 LSTM 与 GRU，已成为序列建模与转换（语言建模、机器翻译）的事实标准。
- 循环模型沿输入/输出符号位置展开计算，生成隐状态序列 $h_t$，依赖前一时刻隐状态 $h_{t-1}$。**这种顺序性排除了训练样本内的并行化**，在长序列时成为关键瓶颈（显存限制导致无法跨样本增大 batch）。
- 已有工作用 factorization tricks（因式分解技巧）与 conditional computation（条件计算）提升效率，但**顺序计算的根本约束依然存在**。
- 注意力机制已成为各类序列建模模型的组成部分，可以在不考虑输入输出距离的情况下建模依赖；但除少数例外，注意力几乎总是**与循环网络配合使用**。

**本文贡献**:
1. 提出 Transformer：摒弃循环、完全依赖注意力机制来刻画输入与输出的全局依赖。
2. 并行化程度显著提高，**在 8 块 P100 GPU 上训练仅 12 小时**即可达到翻译质量的新 SOTA（引言原文表述）。
3. 在 EN-DE / EN-FR 上取得 SOTA，且训练成本远低于已有模型。
4. 证明架构可泛化到机器翻译之外的任务（英语成分句法分析）。

---

## 2. Background

**背景 & 动机**（用 CNN 替代 RNN 的路线及其缺陷）:
- 降低顺序计算量这一目标，也是 Extended Neural GPU、ByteNet、ConvS2S 的出发点，它们都以卷积神经网络为基本构件，可对所有输入输出位置并行计算隐表示。
- 但这些模型中，关联任意两个输入/输出位置所需的操作数随位置间距离增长：**ConvS2S 线性增长，ByteNet 对数增长**，因此**远距离依赖更难学习**。
- Transformer 把这一代价降到**常数级操作数**，代价是注意力加权平均导致的"有效分辨率下降"（reduced effective resolution），本文用 Multi-Head Attention 来对抗这一效应。

**自注意力（Self-Attention）的既有工作**:
- 自注意力（也称 intra-attention）是把**单个序列的不同位置**关联起来以计算该序列表示的注意力机制，已在阅读理解、抽象式摘要、文本蕴含、任务无关句子表示等任务上成功应用。
- End-to-end memory networks 基于循环注意力机制（而非序列对齐的循环），在简单语言问答与语言建模上表现良好。
- **本文定位**：据作者所知，Transformer 是**第一个完全依靠自注意力**计算输入输出表示、不使用序列对齐 RNN 或卷积的转换模型。

---

## 3. Model Architecture

**核心思想**: 保留主流神经序列转换模型的 encoder-decoder 结构 —— encoder 把输入符号表示序列 $(x_1,...,x_n)$ 映射为连续表示序列 $\mathbf{z}=(z_1,...,z_n)$；给定 $\mathbf{z}$，decoder 逐元素生成输出序列 $(y_1,...,y_m)$。每一步模型是**自回归**（auto-regressive）的，生成下一个符号时把此前已生成的符号作为额外输入。Transformer 用**堆叠的自注意力 + 逐点全连接层**实现这一整体架构。

### 3.1 Encoder and Decoder Stacks

**Encoder**：由 $N=6$ 个相同层堆叠而成。每层两个子层：
1. 多头自注意力机制；
2. 简单的逐位置全连接前馈网络。

每个子层外都包一层**残差连接**（residual connection），后接 **layer normalization**，即子层输出为：

$$
\mathrm{LayerNorm}(x + \mathrm{Sublayer}(x))
$$

为方便残差连接，模型中所有子层及 embedding 层的输出维度统一为 $d_{model}=512$。

**Decoder**：同样 $N=6$ 层，除 encoder 的两个子层外，额外插入第三个子层，对 encoder 栈输出做多头注意力（encoder-decoder attention）。同样使用残差 + LayerNorm。
- **关键修改**：decoder 的自注意力子层被 masked（掩码），防止位置关注到后续位置；结合输出 embedding 偏移一个位置，保证位置 $i$ 的预测只能依赖于 $<i$ 的已知输出。

### 3.2 Attention

**核心思想**：注意力函数可描述为把一个 query 和一组 key-value 对映射为一个输出，其中 query、keys、values、output 都是向量。输出是 values 的加权和，权重由 query 与对应 key 的**兼容函数**（compatibility function）计算得到。

#### 3.2.1 Scaled Dot-Product Attention

输入为维度 $d_k$ 的 queries 与 keys、维度 $d_v$ 的 values。计算 query 与所有 keys 的点积，各除以 $\sqrt{d_k}$，再经 softmax 得到 values 的权重。实践中把 queries 打包成矩阵 $Q$，keys/values 打包为 $K$、$V$：

$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}(\frac{QK^T}{\sqrt{d_k}})V
$$

**与另两类注意力的对比**：
- 最常用的两种注意力是 **additive attention**（加性注意力，用单层隐藏层前馈网络计算兼容函数）与 **dot-product (multiplicative) attention**（点积/乘性注意力）。
- 点积注意力与上式的唯一差别就是缩放因子 $1/\sqrt{d_k}$。二者理论复杂度相近，但点积注意力**在实践中更快、更省空间**，因为可以用高度优化的矩阵乘法代码实现。

**为什么要缩放**（原文 Section 3.2.1 的论证）:
- $d_k$ 较小时两种机制表现相近；$d_k$ 较大时，**不缩放的点积注意力会被加性注意力超过**。
- 作者怀疑原因是：$d_k$ 大时点积幅值变得很大，把 softmax 推入**梯度极小的区域**。
- 原文脚注给出的严格说明：假设 $q$ 与 $k$ 的各分量是均值 0、方差 1 的独立随机变量，则点积 $q\cdot k = \sum_{i=1}^{d_k} q_i k_i$ 的均值为 0、**方差为 $d_k$**。因此用 $1/\sqrt{d_k}$ 缩放以抵消该效应。

#### 3.2.2 Multi-Head Attention

**核心思想**：与其用 $d_{model}$ 维的 keys/values/queries 做**单次**注意力，作者发现更好的做法是：用 $h$ 组不同的、可学习的线性投影，把 queries/keys/values 分别投影到 $d_k$、$d_k$、$d_v$ 维，在这 $h$ 组投影版本上**并行**执行注意力，得到 $d_v$ 维输出，再拼接并做一次投影：

$$
\mathrm{MultiHead}(Q, K, V) = \mathrm{Concat}(\mathrm{head_1}, ..., \mathrm{head_h})W^O
$$

$$
\text{where}~\mathrm{head_i} = \mathrm{Attention}(QW^Q_i, KW^K_i, VW^V_i)
$$

其中投影参数矩阵为 $W^Q_i \in \mathbb{R}^{d_{model} \times d_k}$、$W^K_i \in \mathbb{R}^{d_{model} \times d_k}$、$W^V_i \in \mathbb{R}^{d_{model} \times d_v}$、$W^O \in \mathbb{R}^{hd_v \times d_{model}}$。

- **动机**：多头让模型能**联合关注不同表示子空间、不同位置的信息**；单头时"平均"操作会抑制这一点（With a single attention head, averaging inhibits this）。
- **本文设置**：$h=8$ 个并行注意力层（头），每个头 $d_k=d_v=d_{model}/h=64$。由于每个头维度降低，**总计算成本与全维度的单头注意力相近**。

#### 3.2.3 Applications of Attention in this Model

Transformer 在**三处**使用多头注意力：
1. **Encoder-decoder attention**：queries 来自前一个 decoder 层，memory keys/values 来自 encoder 栈输出 —— 让 decoder 每个位置都能关注输入序列的所有位置（类似经典 seq2seq 的 encoder-decoder attention）。
2. **Encoder self-attention**：keys/values/queries 都来自同一处（encoder 前一层的输出），encoder 每个位置可关注前一层的所有位置。
3. **Decoder self-attention**：decoder 每个位置可关注 decoder 中**直到并包含该位置**的所有位置。为保持自回归性质，必须阻止左向信息流 —— 实现方式是在 scaled dot-product attention 内部，把 softmax 输入中对应非法连接的值 **mask 成 $-\infty$**。

### 3.3 Position-wise Feed-Forward Networks

除注意力子层外，encoder/decoder 每层还包含一个全连接前馈网络，**对每个位置单独且相同地**施加（separately and identically）：两次线性变换，中间夹一个 ReLU：

$$
\mathrm{FFN}(x)=\max(0, xW_1 + b_1) W_2 + b_2
$$

- 线性变换在不同位置间共享，但**层与层之间参数不同**。
- 另一种理解方式：等价于**两个 kernel size 为 1 的卷积**。
- 输入输出维度 $d_{model}=512$，内层维度 $d_{ff}=2048$。

### 3.4 Embeddings and Softmax

- 与其他序列转换模型类似，使用 learned embeddings 把输入/输出 token 转为 $d_{model}$ 维向量；用常规的 learned linear transformation + softmax 把 decoder 输出转为下一 token 的预测概率。
- 本文在**两个 embedding 层与 pre-softmax 线性变换之间共享同一个权重矩阵**。
- 在 embedding 层中，把这些权重**乘以 $\sqrt{d_{model}}$**。

### 3.5 Positional Encoding

**核心思想**：由于模型不含循环与卷积，为了让模型利用序列顺序，必须注入 token 在序列中相对或绝对位置的信息。做法是：在 encoder 与 decoder 栈的底部，把"位置编码"加到输入 embedding 上；位置编码与 embedding 同维度 $d_{model}$，故可直接相加。本文使用不同频率的正弦与余弦函数：

$$
PE_{(pos,2i)} = sin(pos / 10000^{2i/d_{model}})
$$

$$
PE_{(pos,2i+1)} = cos(pos / 10000^{2i/d_{model}})
$$

其中 $pos$ 是位置、$i$ 是维度。即位置编码的每个维度对应一个正弦曲线，波长构成从 $2\pi$ 到 $10000 \cdot 2\pi$ 的**几何级数**。
- **选它的理由（假设）**：作者假设它能让模型更容易学会**按相对位置**来关注，因为对任意固定偏移 $k$，$PE_{pos+k}$ 都可以表示为 $PE_{pos}$ 的**线性函数**。
- **对照实验**：也试过 learned positional embedding，两者结果**几乎相同**（见 Table 3 行 (E)）。最终选择正弦版本，因为它**可能让模型外推到训练时未见过的更长序列**。

---

## 4. Why Self-Attention

**核心动机（三个评价指标）**：本节把自注意力层与常用于把变长序列 $(x_1,...,x_n)$ 映射为等长序列 $(z_1,...,z_n)$ 的循环层、卷积层做对比，用三个指标衡量：
1. 每层的**总计算复杂度**；
2. 可并行化的计算量，用所需的**最少顺序操作数**衡量；
3. 网络中**长程依赖之间的路径长度**（path length）—— 前向/反向信号在任意输入输出位置组合间要穿过的路径越短，越容易学到长程依赖。

| Layer Type | Complexity per Layer | Sequential Operations | Maximum Path Length |
|---|---|---|---|
| Self-Attention | $O(n^2 \cdot d)$ | $O(1)$ | $O(1)$ |
| Recurrent | $O(n \cdot d^2)$ | $O(n)$ | $O(n)$ |
| Convolutional | $O(k \cdot n \cdot d^2)$ | $O(1)$ | $O(log_k(n))$ |
| Self-Attention (restricted) | $O(r \cdot n \cdot d)$ | $O(1)$ | $O(n/r)$ |

（Table 1；$n$ 为序列长度，$d$ 为表示维度，$k$ 为卷积核大小，$r$ 为受限自注意力的邻域大小。）

**逐条解读**:
- 自注意力层用**常数次顺序执行的操作**连接所有位置，而循环层需要 $O(n)$ 次顺序操作。
- 计算复杂度上，当序列长度 $n$ 小于表示维度 $d$ 时，自注意力比循环层更快 —— 而机器翻译 SOTA 模型常用的 word-piece / byte-pair 句子表示**通常正属于这种情况**。
- 对超长序列，可把自注意力**限制在大小为 $r$ 的邻域**内，代价是最大路径长度增至 $O(n/r)$（本文把这条留作未来工作）。
- 卷积层（kernel width $k<n$）单层不能连接所有输入输出位置对；需要 $O(n/k)$ 层（连续核）或 $O(log_k(n))$ 层（扩张卷积），从而**拉长最长路径**。
- 卷积层一般比循环层贵 $k$ 倍；separable convolutions 可把复杂度降到 $O(k \cdot n \cdot d + n \cdot d^2)$。即便 $k=n$，separable convolution 的复杂度也等于"自注意力层 + 逐点前馈层"的组合（即本文做法）。

**附加收益（可解释性）**：自注意力可能带来**更可解释的模型**。作者检查了模型的注意力分布并在附录给出示例：不仅不同的注意力头 clearly learn to perform different tasks，许多头还表现出与句子的**句法和语义结构**相关的行为。

---

## 5. Training

### 5.1 Training Data and Batching

- **EN-DE**：标准 WMT 2014 英德数据集，约 **4.5M** 句对；用 **byte-pair encoding (BPE)** 编码，源-目标**共享词表约 37000 tokens**。
- **EN-FR**：更大的 WMT 2014 英法数据集，**36M** 句子；切分为 **32000 的 word-piece 词表**。
- **批构造**：句对按近似序列长度分批；每个训练 batch 包含约 **25000 个源 token 与 25000 个目标 token**。

### 5.2 Hardware and Schedule

- 硬件：**1 台机器、8 块 NVIDIA P100 GPU**。
- **base 模型**：每步约 **0.4 秒**，共训练 **100,000 步 / 12 小时**。
- **big 模型**：每步 **1.0 秒**，共训练 **300,000 步 / 3.5 天**。

### 5.3 Optimizer

使用 **Adam** 优化器，$\beta_1=0.9$、$\beta_2=0.98$、$\epsilon=10^{-9}$。学习率随训练过程按下列公式变化：

$$
lrate = d_{model}^{-0.5} \cdot \min({step\_num}^{-0.5}, {step\_num} \cdot {warmup\_steps}^{-1.5})
$$

即：前 $warmup\_steps$ 步**线性增大**学习率，之后按**步数的反平方根**衰减。本文取 $warmup\_steps=4000$。

> **代码对照**：tensor2tensor 的 `transformer_base()` 中 `learning_rate_warmup_steps = 4000`、`learning_rate = 0.1`、`num_hidden_layers = 6`、`hidden_size = 512`、`filter_size = 2048`、`num_heads = 8`、`label_smoothing = 0.1`、`shared_embedding_and_softmax_weights = True`，与论文一致；另有变体的 `learning_rate_schedule = "constant*linear_warmup*rsqrt_decay*rsqrt_hidden_size"`，正对应上式（含 $d_{model}^{-0.5}$、linear warmup、rsqrt decay 三项）。`transformer_big()` 中 `hidden_size = 1024`、`filter_size = 4096`、`num_heads = 16`、`layer_prepostprocess_dropout = 0.3`，与 Table 3 的 big 行一致。

### 5.4 Regularization

训练中使用三种正则化：
1. **Residual Dropout**：在每个子层的输出上施加 dropout，然后再加到子层输入并做归一化；此外对 encoder/decoder 栈中 **embedding 与位置编码之和**也施加 dropout。base 模型 dropout 率 $P_{drop}=0.1$。
2. **Label Smoothing**：训练时使用 $\epsilon_{ls}=0.1$ 的标签平滑。**它会损害 perplexity**（因为模型学得更"不确定"），但**提升 accuracy 与 BLEU**。

---

## 6. Results

### 6.1 Machine Translation

| Model | EN-DE BLEU | EN-FR BLEU | EN-DE Train Cost (FLOPs) | EN-FR Train Cost (FLOPs) |
|---|---|---|---|---|
| ByteNet | 23.75 | — | — | — |
| Deep-Att + PosUnk | — | 39.2 | — | $1.0\cdot10^{20}$ |
| GNMT + RL | 24.6 | 39.92 | $2.3\cdot10^{19}$ | $1.4\cdot10^{20}$ |
| ConvS2S | 25.16 | 40.46 | $9.6\cdot10^{18}$ | $1.5\cdot10^{20}$ |
| MoE | 26.03 | 40.56 | $2.0\cdot10^{19}$ | $1.2\cdot10^{20}$ |
| Deep-Att + PosUnk Ensemble | — | 40.4 | — | $8.0\cdot10^{20}$ |
| GNMT + RL Ensemble | 26.30 | 41.16 | $1.8\cdot10^{20}$ | $1.1\cdot10^{21}$ |
| ConvS2S Ensemble | 26.36 | 41.29 | $7.7\cdot10^{19}$ | $1.2\cdot10^{21}$ |
| **Transformer (base)** | 27.3 | 38.1 | $3.3\cdot10^{18}$ | $3.3\cdot10^{18}$ |
| **Transformer (big)** | **28.4** | **41.8** | $2.3\cdot10^{19}$ | $2.3\cdot10^{19}$ |

（Table 2。base/big 的训练成本栏原文为跨两列合并的单一数值，未区分语向。）

**关键结论**：
- **EN-DE**：big 模型比此前最好的已发表结果（**含集成模型**）高出 **2.0 BLEU 以上**，创下 **28.4** 的新 SOTA；训练在 **8 块 P100 上耗时 3.5 天**。**即便 base 模型**也超过了所有已发表模型与集成，而训练成本只是任何竞争模型的极小一部分。
- **EN-FR**：big 模型 **41.8 BLEU**，超过所有已发表单模型，训练成本不到此前 SOTA 的 **1/4**；该 EN-FR big 模型使用 dropout $P_{drop}=0.1$（而非 0.3）。
- **推理设置**：base 模型取**最后 5 个 checkpoint 的平均**（每 10 分钟写一次），big 模型取**最后 20 个 checkpoint 的平均**；beam search **beam size = 4**、**length penalty $\alpha=0.6$**；推理最大输出长度设为 **输入长度 + 50**，但尽可能提前终止。
- **训练成本估算口径**：训练时间 × GPU 数量 × 各 GPU 持续单精度浮点能力估计值（脚注给出 K80=2.8、K40=3.7、M40=6.0、P100=9.5 TFLOPS）。

### 6.2 Model Variations

在 EN-DE 的开发集 **newstest2013** 上做消融（使用上一节的 beam search，但**不做 checkpoint 平均**）。base 配置：$N=6$、$d_{model}=512$、$d_{ff}=2048$、$h=8$、$d_k=d_v=64$、$P_{drop}=0.1$、$\epsilon_{ls}=0.1$、100K steps，**PPL 4.92 / BLEU 25.8 / 参数 65M**（Table 3）。

| 组 | 变化 | PPL (dev) | BLEU (dev) | 参数 ×10⁶ |
|---|---|---|---|---|
| base | — | 4.92 | 25.8 | 65 |
| (A) | $h=1$, $d_k=d_v=512$ | 5.29 | 24.9 | — |
| (A) | $h=4$, $d_k=d_v=128$ | 5.00 | 25.5 | — |
| (A) | $h=16$, $d_k=d_v=32$ | 4.91 | 25.8 | — |
| (A) | $h=32$, $d_k=d_v=16$ | 5.01 | 25.4 | — |
| (B) | $d_k=16$ | 5.16 | 25.1 | 58 |
| (B) | $d_k=32$ | 5.01 | 25.4 | 60 |
| (C) | $N=2$ | 6.11 | 23.7 | 36 |
| (C) | $N=4$ | 5.19 | 25.3 | 50 |
| (C) | $N=8$ | 4.88 | 25.5 | 80 |
| (C) | $d_{model}=256$ | 5.75 | 24.5 | 28 |
| (C) | $d_{model}=1024$ | 4.66 | 26.0 | 168 |
| (C) | $d_{ff}=1024$ | 5.12 | 25.4 | 53 |
| (C) | $d_{ff}=4096$ | 4.75 | 26.2 | 90 |
| (D) | $P_{drop}=0.0$ | 5.77 | 24.6 | — |
| (D) | $P_{drop}=0.2$ | 4.95 | 25.5 | — |
| (D) | $\epsilon_{ls}=0.0$ | 4.67 | 25.3 | — |
| (D) | $\epsilon_{ls}=0.2$ | 5.47 | 25.7 | — |
| (E) | positional embedding 替代 sinusoids | 4.92 | 25.7 | — |
| big | $N=6$, $d_{model}=1024$, $d_{ff}=4096$, $h=16$, $P_{drop}=0.3$, 300K steps | **4.33** | **26.4** | 213 |

**逐条解读**（原文 Section 6.2）:
- **(A) 头数**：保持计算量不变改变头数与 $d_k, d_v$。**单头注意力比最佳设置差 0.9 BLEU**；头数**过多时质量也会下降**（$h=32$ 时降到 25.4）。
- **(B) 键维度**：减小 $d_k$ 会**损害模型质量**。这说明"判断兼容性并不容易"，比点积更复杂的兼容函数可能有帮助。
- **(C)/(D)**：如预期，**模型越大越好**；dropout 对避免过拟合非常有帮助。
- **(E) 位置编码**：把正弦位置编码换成 learned positional embedding，结果与 base **几乎一致**。

### 6.3 English Constituency Parsing

为验证泛化能力，在英语成分句法分析上实验。该任务的特殊挑战：**输出受强结构约束**，且**输出显著长于输入**；此外 RNN seq2seq 模型在小数据场景一直拿不到 SOTA。

| Parser | Training | WSJ 23 F1 |
|---|---|---|
| Vinyals & Kaiser et al. (2014) | WSJ only, discriminative | 88.3 |
| Petrov et al. (2006) | WSJ only, discriminative | 90.4 |
| Zhu et al. (2013) | WSJ only, discriminative | 90.4 |
| Dyer et al. (2016) | WSJ only, discriminative | 91.7 |
| **Transformer (4 layers)** | WSJ only, discriminative | **91.3** |
| Zhu et al. (2013) | semi-supervised | 91.3 |
| Huang & Harper (2009) | semi-supervised | 91.3 |
| McClosky et al. (2006) | semi-supervised | 92.1 |
| Vinyals & Kaiser et al. (2014) | semi-supervised | 92.1 |
| **Transformer (4 layers)** | semi-supervised | **92.7** |
| Luong et al. (2015) | multi-task | 93.0 |
| Dyer et al. (2016) | generative | 93.3 |

（Table 4，结果均在 WSJ Section 23 上）

**设置**：4 层 Transformer，$d_{model}=1024$，在 Penn Treebank 的 WSJ 部分（约 **40K** 训练句）上训练；半监督设置使用 high-confidence 与 BerkeleyParser 语料，约 **17M** 句。词表：WSJ-only 用 16K tokens，半监督用 32K tokens。只在 Section 22 开发集上少量试选 dropout（attention 与 residual）、学习率与 beam size，**其他参数保持与 EN-DE base 翻译模型相同**。推理时最大输出长度提升到 **输入长度 + 300**，**beam size = 21**、$\alpha=0.3$。

**结论**：尽管**没有做任务专属调参**，模型表现仍然出人意料地好 —— 除 Recurrent Neural Network Grammar 外，超过所有已发表模型；并且与 RNN seq2seq 不同，它**仅在 40K 句的 WSJ 训练集上训练就超过了 BerkeleyParser**。

---

## 7. Conclusion

**总结**：提出 Transformer —— **第一个完全基于注意力的序列转换模型**，用多头自注意力取代了 encoder-decoder 架构中最常用的循环层。对翻译任务，Transformer 的训练速度显著快于基于循环层或卷积层的架构；在 WMT 2014 英德、英法两个任务上都取得新 SOTA（前者 best model 甚至超过了此前所有已报告的集成模型）。

**作者展望**：对注意力模型的前景感到兴奋，计划：
- 应用到其他任务；
- 扩展到**文本之外的输入输出模态**；
- 研究 **local / restricted attention**，以高效处理图像、音频、视频等大输入大输出；
- 让生成过程**更少地顺序化**（making generation less sequential）。

**代码**：训练与评测代码开源于 https://github.com/tensorflow/tensor2tensor。

---

## Attention Visualizations（附录）

附录给出编码器自注意力（第 5 / 6 层）的可视化示例：

1. **长距离依赖**：许多注意力头都关注动词 `making` 的远距离依赖，补全短语 `making...more difficult`（仅展示 `making` 一词的注意力，不同颜色代表不同头）。
2. **指代消解（anaphora resolution）**：第 5 层的两个头（head 5、head 6）似乎参与了指代消解；对词 `its` 的注意力**非常尖锐**（very sharp）。
3. **结构相关的头**：多个头表现出与句子结构相关的行为，两个来自不同头的例子显示 —— **不同的头 clearly learned to perform different tasks**（明显学会了执行不同任务）。

---

## 个人评价

**核心价值**：这篇论文的意义远超机器翻译本身 —— 它把"顺序计算是序列建模的必要代价"这一隐含前提彻底推翻，用 $O(1)$ 的路径长度 + $O(1)$ 的顺序操作换掉了 RNN 的 $O(n)$ 顺序链，从而把序列建模变成了**可充分并行、可扩展**的问题。今天几乎所有大模型（BERT/GPT/ViT/DETR 及其在遥感、检测领域的众多衍生）都建立在本文的 encoder/decoder + 多头自注意力 + 残差 + LayerNorm + FFN 这一套最小构件之上。

**可借鉴点**（对遥感/视觉检测研究）：
- **"用多头对抗平均导致的分辨率损失"** 这一思路可迁移：任何用加权聚合做特征融合的模块（如遥感图像中的多尺度融合、跨模态注意力）都可以用"多子空间并行 + 拼接投影"的方式保留细节，而不是单头平均。
- **缩放点积**：在自己的注意力模块中若出现 attention 饱和（权重趋近 0/1、梯度消失），先检查是否缺少 $1/\sqrt{d_k}$ 缩放 —— 本文给出了方差为 $d_k$ 的定量解释。
- **位置编码的选择逻辑值得学习**：正弦编码与 learned embedding 效果几乎相同，作者选正弦的理由是"可外推到更长序列"。这提示在遥感等**输入尺度变化大**（不同分辨率、不同影像幅宽）的场景中，位置编码的**外推性**可能比拟合能力更重要。
- **实验设计上"低成本对照"的做法**：消融全部放在开发集 newstest2013、不做 checkpoint 平均、其余参数冻结，只改一个变量（见 Table 3），是低成本高信息量的消融范式。

**实验设计亮点**：
- 用 **FLOPs（训练时间 × GPU 数 × 单卡持续算力）** 把不同硬件、不同架构的训练成本放到同一把尺子上比较（Table 2），并明确列出各 GPU 的 TFLOPS 假设值（脚注），口径透明、可复核。
- **checkpoint 平均**（base 取最后 5 个、big 取最后 20 个）是一个几乎零成本却稳定提分的技巧。
- **主动报告负面结果**：标签平滑"损害 perplexity 但提升 BLEU"（Table 3 行 D 中 $\epsilon_{ls}=0.0$ 的 PPL 4.67 优于 0.1 的 4.92，BLEU 却更差），诚实地把"指标不一致"摆在台面上，而不是只报好数字。

---

## 疑问解答 (Q&A)

### Q1: 为什么一定要除以 $\sqrt{d_k}$？不缩放会怎样？

**答**：原文给出了定量解释（Section 3.2.1 及脚注）。假设 $q$ 与 $k$ 各分量独立、均值 0、方差 1，则点积 $q\cdot k=\sum_{i=1}^{d_k}q_i k_i$ 的**均值为 0、方差为 $d_k$**：$d_k$ 越大，点积的幅值波动越大，softmax 越容易进入**梯度极小的饱和区**。论文引用 Britz et al. 2017 的结果：$d_k$ 较小时加性与点积注意力表现相近，$d_k$ 较大时**不缩放的点积注意力会被加性注意力超过**。除以 $\sqrt{d_k}$ 正是把方差拉回 1。

### Q2: 多头注意力的计算量不会随头数线性增长吗？

**答**：不会明显增长。原文用 $h=8$、$d_k=d_v=d_{model}/h=64$，即**每个头的维度按头数等比例缩小**，各头的计算量之和与"全维度单头"大致相当（Due to the reduced dimension of each head, the total computational cost is similar to that of single-head attention with full dimensionality）。Table 3 行 (A) 的四组设置（$h=1/4/16/32$）就是在**保持计算量恒定**的前提下改变头数，因此比较是公平的：单头（24.9 BLEU）比最佳设置差 0.9 BLEU，而头数过多（$h=32$）也会掉到 25.4。

### Q3: 论文正文与表格里英法 BLEU 数字不一致（41.0 vs 41.8），以哪个为准？

**答**：这是一个**真实存在的原文不一致**，本笔记如实记录。LaTeX 源码（v7，`results.tex`）中，Section 6.1 正文写 "our big model achieves a BLEU score of **41.0**"，而 Table 2 与摘要写的是 **41.8**。摘要与 Table 2 的 41.8 是**被广泛引用并被后续文献采信**的口径；正文的 41.0 应属版本迭代中未同步更新的遗留表述。本笔记所有位置统一采用 **41.8（Table 2 / 摘要）**，此处标注差异以免误读。

### Q4: Transformer 是完全没有顺序计算了吗？推理时也并行？

**答**：**训练可并行，推理仍自回归**。原文 Section 3.1 明确：decoder 在生成时每一步都是 auto-regressive 的，要消费此前已生成的符号；为保证这一点，decoder 自注意力用 mask（把非法连接置为 $-\infty$）加输出 embedding 偏移一位来阻断右向信息流。Table 1 中的 "Sequential Operations = $O(1)$" 指的是**层内**的顺序操作数（对比 RNN 的 $O(n)$），而非解码步数。论文的 Conclusion 也把"让生成过程更少地顺序化"列为未来工作。

### Q5: 论文里的复杂度对比（Table 1）在什么前提成立？自注意力一定比 RNN 快吗？

**答**：不一定，**取决于 $n$ 与 $d$ 的关系**。原文给出的条件是：当序列长度 $n$ 小于表示维度 $d$ 时，自注意力层比循环层更快 —— 而机器翻译 SOTA 常用的 word-piece / byte-pair 表示通常属于这种情况（$n \lesssim d$）。自注意力每层复杂度是 $O(n^2\cdot d)$，循环层是 $O(n\cdot d^2)$：**当 $n \gg d$（超长序列）时自注意力的 $n^2$ 项会成为瓶颈**，为此原文提出可把自注意力限制在大小 $r$ 的邻域（复杂度 $O(r\cdot n\cdot d)$、路径长度 $O(n/r)$），并留作未来工作。

### Q6: 论文开源了代码吗？与论文描述一致吗？

**答**：开源，地址 https://github.com/tensorflow/tensor2tensor（Conclusion 给出）。本笔记已 clone 到 `code/transformer/` 对照验证，关键超参与论文一致：`transformer_base()` 中 `num_hidden_layers=6`、`hidden_size=512`、`filter_size=2048`、`num_heads=8`、`learning_rate_warmup_steps=4000`、`label_smoothing=0.1`、`shared_embedding_and_softmax_weights=True`；`transformer_big()` 中 `hidden_size=1024`、`filter_size=4096`、`num_heads=16`、`layer_prepostprocess_dropout=0.3`，对应 Table 3 的 big 行。学习率调度在部分变体中写作 `constant*linear_warmup*rsqrt_decay*rsqrt_hidden_size`，与论文公式 (3) 的三段结构（$d_{model}^{-0.5}$ × linear warmup × rsqrt decay）一致。注意：仓库当前 HEAD 已历经多年迭代，`transformer_base()` 的 `learning_rate_schedule` 标为 `"legacy"`，论文公式对应的是 TPU/后续变体中的那套调度实现。

### Q7: 正弦位置编码相比 learned positional embedding 有什么实际优势？论文有证据吗？

**答**：**性能上几乎没有优势，优势在假设层面**。Table 3 行 (E)：换成 learned positional embedding 后 PPL 4.92、BLEU 25.7，与 base 的 4.92 / 25.8 **几乎相同**。原文明确说明选择正弦版本的理由是"**它可能让模型外推到训练时遇到的序列长度之外**"（may allow the model to extrapolate to sequence lengths longer than the ones encountered during training），以及假设它能让模型更容易按相对位置关注（$PE_{pos+k}$ 可表示为 $PE_{pos}$ 的线性函数）。原文**未给出长序列外推的实测实验**来验证这一假设 —— 这是作者自陈的动机，不是被实验证明的结论。

### Q8: 句法分析实验为什么值得单独做？它说明了什么？

**答**：它验证的是**架构泛化性**，而非翻译性能。原文给出的挑战点：输出受强结构约束、输出显著长于输入，且 RNN seq2seq 在**小数据**场景下拿不到 SOTA。结果是 4 层 Transformer 在**几乎没有任务专属调参**（只调 dropout/学习率/beam，其余沿用 EN-DE base 配置）的情况下，WSJ-only 得 **F1 91.3**、半监督 **92.7**，仅低于 Recurrent Neural Network Grammar（93.3）。特别值得注意的是：与 RNN seq2seq 不同，它**仅用 40K 句的 WSJ 训练集就超过了 BerkeleyParser** —— 这是对小数据场景泛化能力的直接证据。

---

## 关联笔记

- [DEIMv2: Real-Time Object Detection Meets DINOv3](/readings/2025-09-25-deimv2) — `transformer` 标签，实时 DETR 检测器，Transformer 架构在检测领域的直接衍生
- [RF-DETR: Neural Architecture Search](/readings/2025-11-15-rf-detr) — `DETR` 标签，DETR 系列骨干/编码器设计
- [Grounding DINO: Marrying DINO with Grounded Pre-Training](/readings/2023-03-09-groundingdino) — Transformer-based 开放词汇检测器
- [EdgeCrafter: Compact ViT Edge Dense Prediction](/readings/2026-03-19-edgecrafter) — `ViT` 标签，ViT 在边缘端密集预测的落地
- [TinyFormer: YOLO-DETR Hybrid Tiny Object Detection](/readings/2026-05-24-tinyformer) — 卷积与 Transformer 混合的轻量检测架构
