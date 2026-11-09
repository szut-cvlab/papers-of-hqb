# 低光图像数据集专题调研

> Low-Light / Night-time Datasets for Object Detection — 面向"低光场景小目标检测"，逐数据集核实**图片规模 / 标注规模 / 能否下载 / 是否含夜间低光**

（链接核实时间 2026-11-08。网盘与镜像会失效，用前建议再点一次。）

---

## 一、核心结论

1. **能"点开就下 + 有检测框 + 真夜间 + 监控视角"的数据集，目前只有 NightSurveillance 一个。** 3.8 万图、9.3 万框、46% 是小目标，百度网盘直接给。
2. **DarkVision 是理论上最理想的配对数据集，但目前拿不到。** 论文写"发表后开放学术使用"，实际没有公开下载入口，需邮件联系作者。
3. **UA-DETRAC 官方域名已下线**（`detrac-db.rit.albany.edu` 301 跳转到实验室首页，实测确认），原直链全部失效，只能走 Kaggle 镜像或作者的个人镜像页。
4. **ExDark 官方仓库已撤掉下载链接**，只留作者邮箱；现在能下的是第三方 Google Drive 镜像（VOC 格式，含增强图）。
5. **"只拉特定文件"比"整库 clone"更现实**：绝大多数夜间数据集是 100GB 量级（NightOwls 训练集 135GB），你的场景不需要全量。

---

## 二、总表：四个必填字段 + 适配判断

### 2.1 真实采集 · 有检测标注（核心候选）

| 数据集 | 图片规模 | 标注规模 | 能否下载 / 下载方式 | 含夜间/低光 | 适配低光小目标检测 |
|---|---|---|---|---|---|
| **NightSurveillance** | **38,113** 张，1920×1080 | **93,000** 个行人框；训练/测试 1:1；小目标（<90px）占 46–47% | 能下。百度网盘 `pan.baidu.com/s/1uIhMBKmQgPdbVjR2EK7o5g`，提取码 `akzx` | 是，100% 夜间（17:00–5:00） | ★★★★★ **首选主基准**：唯一"监控俯视 + 夜间 + 小目标占比高 + 有框" |
| **NightOwls** | **279,000** 帧，1024×640 | 全部帧有 2D 框 + 遮挡/姿态/难度/截断属性 + 跟踪 ID；官方未公布总框数 | 能下。官方 HTTP 直链，train 135GB / val 50GB / test 100GB | 是，100% 夜间 | ★★☆☆☆ 车载平视，非监控；分辨率低、目标不小，只作跨视角泛化对照 |
| **ExDark** | **7,363** 张，分辨率不等 | 12 类，图像级类别 + 目标级框；官方未给总框数 | 官方仓库已撤链接，只能走第三方镜像：Google Drive `1X_zB_OSp_thhk9o26y1ZZ-F85UeS0OAC`，或百度网盘（提取码 `1234`） | 是，100% 低光（10 档：极暗到暮光） | ★★★☆☆ 多类 + 光照分层，适合验证"跨光照档"；但非监控视角、规模小 |
| **DARK FACE** | 训练 **6,000** + 测试 **4,000** + 未标注 **9,000** + 配对 **789**，1080×720 | 训练 **43,849** 张脸框；测试 37,711（另有资料写 37,047） | 能下。官方页 Google Drive / 百度网盘，提取码 `babu` | 是，100% 夜间欠曝 | ★★☆☆☆ 只有"人脸"一类，不是通用目标检测；789 组真实配对极有价值 |
| **DarkVision** | 静态 **13,455** 张（900 场景）+ 视频 **89,411** 帧（32 场景） | 静态 **7,344** 个框（15 类）；视频 **320,667** 个框（4 类）；像素级明暗配对 | **不能下。** 没有任何公开入口，只能邮件联系作者申请学术使用 | 是，100% 低光，5 档照度 × 3 档相机 | ★★★★★（若拿到）配对最完整，但目前不可得 |
| **ACDC（night 子集）** | **4,006** 张（雾/夜/雨/雪 各约 1,000），另附 4,006 张同场景正常光图 | 2,006 张有检测框 GT（4MB 包）；19 类像素级语义 | 能下。直链 `acdc.vision.ee.ethz.ch/rgb_anon_trainvaltest.zip`（15.6GB）、`gt_detection_trainval.zip`（4MB），无需登录 | 是，night 约 **1,000** 张，且同时有正常光对应图 | ★★★★☆ 车载、目标不小；能提供"同场景明暗对照"，适合验证光照鲁棒性 |
| **BDD100K（night 子集）** | **100,000** 张，1280×720；夜间 **31,900** + 黄昏 5,805 | 10 类逐框标注（官方页面未给总框数）+ 语义/车道 | 能下，但需注册登录官网 `bdd-data.berkeley.edu` | 是，夜间 31,900 张（占 32%） | ★★★☆☆ 车载多类、规模大；目标尺度偏大，做夜间切片与多类泛化 |
| **UA-DETRAC（night 子集）** | **140,000+** 帧，960×540 | **1,210,000** 个框 / 8,250 辆车 / 4 类 | 官方入口已下线（原直链 301 跳转到实验室首页）。可走 Kaggle 镜像 `kaggle.com/datasets/bratjay/ua-detrac-orig`（约 11GB）或作者镜像页 | 是，夜间是 4 种天气之一（晴/阴/雨/夜），但无独立夜间标注统计 | ★★★☆☆ 监控路口视角最接近你的场景；但无小目标、无夜间单独划分 |
| **Dark Zurich** | 151 段夜间/黄昏视频，约 **2,416** 帧标注 | 19 类**语义分割**，无检测框 | 官方页面 `trace.ethz.ch` 本次访问失败 | 是，夜间 + 黄昏 | ★☆☆☆☆ 只做分割，无框，不适合 |

### 2.2 真实采集 · 可见光-红外配对

| 数据集 | 图片规模 | 标注规模 | 能否下载 / 下载方式 | 含夜间/低光 | 适配判断 |
|---|---|---|---|---|---|
| **LLVIP** | **30,976** 张 = **15,488** 对；可见 1920×1080 + 红外 1280×720 | 1 类（行人）；官方未公布框总数 | 能下。Google Drive `1VTlT3Y7e1h-Zsne4zahjx5q0TK2ClMVv`，或百度网盘 `pan.baidu.com/s/1eQO1Is2NPyd-mgmv1Csbfg` 提取码 `14lc` | 是，傍晚 18:00–22:00，极暗 | ★★★★☆ 配对严格（时空对齐）。但红外不是低光图，是"光照无关参照"；官方自陈目标偏中尺度，不适合远距离小目标 |
| **M3FD** | **8,400** 张 = **4,200** 对，1024×768 | **34,407** 框 / 6 类（人/车/公交/摩托/车灯/卡车） | 能下。Google Drive 文件夹 `1H-oO7bgRuVFYDcMGvxstT1nmy0WF_Y_6`，或百度网盘 `pan.baidu.com/s/1GoJrrl_mn2HNQVDSUdPCrw?pwd=M3FD` | 是，昼/阴/夜/极端挑战四组 | ★★★★☆ 多类 + 夜间组 + 配准好；同样是"可见-红外"而非明暗对 |
| **KAIST** | **95,000** 对，640×480 | **103,128** 框 / 1,182 个行人实例，含遮挡标注 | 能下，无需申请。用 gdown：预览 `11nhHpmuh2FUjrLNfGs51R2Mqqy1GTjY8`（1.44GB）、全量 `1sBcAmFqNJmNMBZdMtKmO2X4BRjKPyKMc`（36.32GB） | 是，昼夜均有（官方未给昼夜比例） | ★★★☆☆ 分辨率低（640×480），小目标基本没戏；作跨模态对照 |

### 2.3 合成 / 混合（含自动生成流程）

| 数据集 | 图片规模 | 标注规模 | 能否下载 / 下载方式 | 含夜间/低光 | 适配判断 |
|---|---|---|---|---|---|
| **Dark-traffic** | **10,425** 张 | **99,014** 实例（检测 + 实例分割 + 光流） | 能下。Google Drive 文件夹 `1B8EzDn64bGBgyRCfppL_jhcOA3hIwnzi`（第三方整理，完整性需自查） | 是，全部低光（合成 + 真实 LIS-traffic 混合） | ★★★★☆ 2025 年新集，最像你的任务（交通低光）；合成方式是 sRGB 光照衰减，可作你逆 ISP 方案的对照 |
| **SpaceDG** | **160,000+** 张，近 1,000 个室内场景 | **971,000** QA 对；1,102 道 benchmark 题，**无检测框** | 能下。HuggingFace `huggingface.co/datasets/xlzhou126/SpaceDG-Bench` + GitHub | 是，9 类退化之一（低光 = 曝光衰减 + Poisson + 高斯读出噪声） | ★☆☆☆☆ VQA 任务，不能训检测器；只能借"物理退化合成引擎"的思路 |
| **CD-COCO** | **123,000** 张（train 95K / val 5K / test 23K） | 80 类，框与掩码沿用 COCO | 能下。GitHub `github.com/Aymanbegh/CD-COCO`（图像列表 + 失真生成代码 + 标注） | 部分，含"对比度变化""局部逆光"，不是严格意义的低光 | ★★☆☆☆ 生成代码可参考（深度/语义感知的局部失真），但弱光建模不足 |

### 2.4 仅增强用途（无检测框，做预训练或感知损失参考）

| 数据集 | 图片规模 | 标注 | 能否下载 / 下载方式 | 低光 | 用途 |
|---|---|---|---|---|---|
| **LOL** | 500 对（485/15） | 无框 | 能下。`daooshee.github.io/BMVC2018website/` | 是 | 增强预训练 |
| **LOL-v2** | real 789 对 + syn 1,000 对 | 无框 | 能下。随 LOL 仓库/作者页 | 是 | 同上 |
| **VE-LOL-L** | 2,500 对（400/100） | 无框 | 能下。`flyywh.github.io/IJCV2021LowLight_VELOL/` | 是 | 含 RAW 级噪声建模 |
| **SID** | 5,094 短曝 RAW + 424 长曝参考 | 无框 | 能下。`github.com/cchen156/Learning-to-See-in-the-Dark` | 是，极暗 | RAW 噪声管线参照 |
| **MIT-Adobe FiveK** | 5,000 张 × 5 版 | 无框 | 能下。`data.csail.mit.edu/graphics/fivek/` | 否，属修图风格，非物理暗光 | 色彩/色调参考 |
| **LIME/DICM/NPE/MEF/VV** | 10–84 张 | 无框 | 能下。各自 GitHub | 是 | 仅定性评测 |

---

## 三、逐数据集详情

### 3.1 NightSurveillance（主基准）

| 项目 | 内容 |
|---|---|
| **论文** | IJCAI 2020, *When Pedestrian Detection Meets Nighttime Surveillance: A New Benchmark* |
| **官方仓库** | https://github.com/xiaowang1516/NightSurveillance |
| **图片规模** | 38,113 张（1920×1080） |
| **标注规模** | 93,000 个行人框；训练/测试约 1:1 |
| **下载方式** | 百度网盘：https://pan.baidu.com/s/1uIhMBKmQgPdbVjR2EK7o5g 提取码 **akzx**（README 原文给出） |
| **含夜间** | 全部为 17:00–次日 5:00 采集 |

**获取步骤**：GitHub README → 复制网盘链接 → 输提取码 → 下载。仅限学术用途（README 明确要求）。

**为什么最适合你**：
- 唯一"监控俯视 + 夜间 + 小目标占比 46–47% + 有检测框"的组合；
- 行人密度 2.46 人/帧（NightOwls 仅 0.20），密集场景多；
- 有光照（低/中/高）、模糊、雨天属性，可做分层评测；
- 论文报告 S3D 漏检率 21.73%、小目标 38.96%，与申请书引用一致。

**注意**：单类别、单校园；相邻帧高度相似，训练前建议抽帧（有工作每 3 帧取 1 帧，得 7,002 张）。

---

### 3.2 NightOwls

| 项目 | 内容 |
|---|---|
| **官方** | https://www.nightowls-dataset.org/download/ |
| **图片规模** | 279,000 帧 / 40 序列，1024×640，5.33fps |
| **标注规模** | 全部帧标注：4 类 + 遮挡/姿态/难度/截断 + 跟踪 ID；官方未公布总框数 |
| **下载方式** | 官方 HTTP 直链，见下表，无需申请 |
| **含夜间** | 100% |

| 文件 | 链接 | 大小 |
|---|---|---|
| 训练图像 | `http://thor.robots.ox.ac.uk/~vgg/data/nightowls/python/nightowls_training.zip` | 135 GB |
| 训练标注 | `.../nightowls_training.json` | 32 MB |
| 验证图像 | `.../nightowls_validation.zip` | 50 GB |
| 验证标注 | `.../nightowls_validation.json` | 10 MB |
| 测试图像 | `.../nightowls_test.zip` | 100 GB |

**建议**：别下全量。先下 train 的 json（32MB）看标注结构，再按序列下载需要的子集。车载视角，只做"跨视角泛化"一栏。

---

### 3.3 ExDark

| 项目 | 内容 |
|---|---|
| **官方仓库** | https://github.com/cs-chan/Exclusively-Dark-Image-Dataset |
| **图片规模** | 7,363 张，10 档光照 |
| **标注规模** | 12 类；图像级类别 + 目标级框（官方未给总框数） |
| **下载方式** | 官方 README 已不再提供链接，只留邮箱（lexloh2009@hotmail.com / cs.chan@um.edu.my）。可用第三方镜像：Google Drive `https://drive.google.com/uc?export=download&id=1X_zB_OSp_thhk9o26y1ZZ-F85UeS0OAC`（EXDark.tar.gz，VOC 格式，含 Kind、Zero-DCE、MBLLEN 增强版），或百度网盘提取码 `1234` |
| **含夜间** | 100% 低光（极暗到暮光 10 档） |

> 注意：该 Drive ID 来自 IAT 官方复现仓库（MAET, ICCV 2021），不是 ExDark 作者原链。下载后请核对图像数与标注数是否为 7,363。
> 注意：许可方面，代码是 BSD-3，但底层照片不在该许可内；2026-10 起有营利机构参与的项目需书面授权。

---

### 3.4 DARK FACE（UG²+ Track 2.2）

| 项目 | 内容 |
|---|---|
| **官方页** | https://flyywh.github.io/CVPRW2019LowLight/ |
| **图片规模** | 训练 6,000 + 测试 4,000 + 未标注 9,000 + 配对 789 组，1080×720 |
| **标注规模** | 训练 43,849 个脸框；测试 37,711（部分资料写 37,047） |
| **下载方式** | 官方页给出：训练/验证图像与标注用 Google Drive 或百度网盘，提取码 **babu**；样例测试图网盘码 **429h** |
| **含夜间** | 100% 夜间欠曝 |

**对你的用法**：不要当检测训练集（只有人脸）。那 789 组"低光/正常光"真实配对是稀缺资源，用来检验你的合成管线像不像真的（见 4.4 节）。

---

### 3.5 DarkVision（想用但拿不到）

| 项目 | 内容 |
|---|---|
| **论文** | Computational Visual Media 2025, DOI 10.26599/CVM.2025.9450461；预印本 arXiv:2301.06269 |
| **图片规模** | 静态 13,455 张（900 场景）；视频 89,411 帧（32 场景） |
| **标注规模** | 静态 7,344 框 / 15 类；视频 320,667 框 / 4 类；像素级配对的明暗图 |
| **下载方式** | 无。没有任何公开仓库或网盘入口。预印本原话是"论文发表后开放学术使用"，但正式发表后仍未开放 |
| **含夜间** | 5 档照度（3.2%–0.2% 透射率）× 3 档相机 |

**结论**：目前无法直接下载。可行路径只有一条——邮件联系作者（jlsuo@tsinghua.edu.cn、b-zhang18@mails.tsinghua.edu.cn），说明课题用途申请学术使用。等回复期间不要把它写进实验计划的关键路径。它在设计上最适配（唯一像素级明暗配对 + 检测框），但在获取上不可得。

---

### 3.6 ACDC（night 子集）

| 项目 | 内容 |
|---|---|
| **官方** | https://acdc.vision.ee.ethz.ch/ ，下载页 `/download` |
| **图片规模** | 4,006 张（雾/夜/雨/雪 各约 1,000），另附 4,006 张同场景正常光图 |
| **标注规模** | 检测框 GT：2,006 张（train+val）；语义/全景：19 类 |
| **下载方式** | 直链，无需登录，见下表 |
| **含夜间** | night 约 1,000 张，且有对应正常光图 |

| 文件 | 链接 | 大小 |
|---|---|---|
| 全部图像（含正常光对应图） | `https://acdc.vision.ee.ethz.ch/rgb_anon_trainvaltest.zip` | 15.6 GB |
| 检测框标注 | `https://acdc.vision.ee.ethz.ch/gt_detection_trainval.zip` | 4 MB |
| 语义分割标注 | `https://acdc.vision.ee.ethz.ch/gt_trainval.zip` | 127 MB |
| 正常光图检测标注 | `https://acdc.vision.ee.ethz.ch/gt_detection_trainval_ref.zip` | 1 MB |

**价值**：它是"同一场景、不同光照"的对照。夜图与其正常光对应图成对存在（虽非像素级严格配准，但是同一地点）。你的对比学习可以在这里拿到近似配对。

---

### 3.7 BDD100K（night 切片）

| 项目 | 内容 |
|---|---|
| **官方** | https://bdd-data.berkeley.edu/ |
| **图片规模** | 100,000 张，1280×720；夜间 31,900、黄昏 5,805 |
| **标注规模** | 10 类逐框 + 语义 + 车道（官方页面未给框总数） |
| **下载方式** | 需注册登录官网下载。镜像 `dl.cv.ethz.ch/bdd100k/data/` 本次访问失败，可自行试 |
| **含夜间** | 32% 夜间 |

**注意**：time-of-day 标签约 1.2% 误标（有人系统核查过 951 张），严格实验前建议人工过一遍。

---

### 3.8 UA-DETRAC（官方已下线）

| 项目 | 内容 |
|---|---|
| **图片规模** | 140,000+ 帧，960×540，25fps |
| **标注规模** | 1,210,000 框 / 8,250 辆车 / 4 类（car/bus/van/other） |
| **下载方式** | 官方不可用：`detrac-db.rit.albany.edu` 已 301 跳转至实验室首页，所有原直链失效（本次实测）。替代：Kaggle https://www.kaggle.com/datasets/bratjay/ua-detrac-orig （约 11GB，140k 帧）；作者镜像页 https://sites.google.com/view/daweidu/projects/ua-detrac |
| **含夜间** | 夜间是 4 种天气之一，但官方没有给出夜间帧数，需自己按 XML 的 weather 属性筛 |

**筛夜间的方法**：标注 XML 里有 `weather` 字段（night/sunny/rainy/cloudy），按序列级统计即可切出夜间子集。

---

### 3.9 LLVIP

| 项目 | 内容 |
|---|---|
| **官方** | https://bupt-ai-cz.github.io/LLVIP/ ，下载说明见 `github.com/bupt-ai-cz/LLVIP/blob/main/download_dataset.md` |
| **图片规模** | 30,976 张 = 15,488 对（可见 1920×1080 + 红外 1280×720） |
| **标注规模** | 1 类（行人），官方未公布框总数 |
| **下载方式** | Google Drive `1VTlT3Y7e1h-Zsne4zahjx5q0TK2ClMVv`；百度网盘 `https://pan.baidu.com/s/1eQO1Is2NPyd-mgmv1Csbfg` 提取码 **14lc**；原始未配准数据另有链接 |
| **含夜间** | 傍晚 18:00–22:00，极暗 |

**注意**：官网首页是"填表留邮箱"的方式，但 GitHub 的 `download_dataset.md` 直接给了网盘链接，不用等邮件。

---

### 3.10 M3FD

| 项目 | 内容 |
|---|---|
| **官方仓库** | https://github.com/JinyuanLiu-CV/TarDAL |
| **图片规模** | 8,400 张 = 4,200 对，1024×768（另有 600 张独立场景） |
| **标注规模** | 34,407 框 / 6 类 |
| **下载方式** | Google Drive `https://drive.google.com/drive/folders/1H-oO7bgRuVFYDcMGvxstT1nmy0WF_Y_6`；百度网盘 `https://pan.baidu.com/s/1GoJrrl_mn2HNQVDSUdPCrw?pwd=M3FD` |
| **含夜间** | 昼/阴/夜/极端挑战四组 |

---

### 3.11 KAIST

| 项目 | 内容 |
|---|---|
| **官方** | https://soonminhwang.github.io/rgbt-ped-detection/ |
| **图片规模** | 95,000 对，640×480，20Hz |
| **标注规模** | 103,128 框 / 1,182 个行人实例 |
| **下载方式** | 无需申请，用 gdown 从 Google Drive 拉取：`gdown 11nhHpmuh2FUjrLNfGs51R2Mqqy1GTjY8`（预览 1.44GB）、`gdown 1sBcAmFqNJmNMBZdMtKmO2X4BRjKPyKMc`（全量 36.32GB） |
| **含夜间** | 昼夜均有（官方未给比例） |

---

### 3.12 VisDrone（小目标上限验证 + 合成底图）

| 项目 | 内容 |
|---|---|
| **官方仓库** | https://github.com/VisDrone/VisDrone-Dataset |
| **图片规模** | 10,209 张静态图 + 288 段视频（261,908 帧） |
| **标注规模** | 260 万+ 框 / 10 类 |
| **下载方式** | DET 训练集 1.44GB：Google Drive `1a2oHjcEcwXP8oUF95qiwrqzACb2YlUhn`；百度网盘 `https://pan.baidu.com/s/1K-JtLnlHw98UuBDrYJvw3A` |
| **含夜间** | 无专门夜间子集（README 只说"不同天气与光照条件"） |

**用法**：这不是低光数据集。你的用法应该是"小目标最丰富"的底图，用逆 ISP 管线合成配对低光图，标注直接复用。

---

### 3.13 Dark-traffic（2025 新集）

| 项目 | 内容 |
|---|---|
| **来源** | 重庆交通大学，2025-09 发布 |
| **图片规模** | 10,425 张 |
| **标注规模** | 99,014 个实例级标注（检测 + 实例分割 + 光流） |
| **下载方式** | Google Drive 文件夹 `https://drive.google.com/drive/folders/1B8EzDn64bGBgyRCfppL_jhcOA3hIwnzi` |
| **含夜间** | 全部低光（COCO-traffic 光照衰减合成 + 真实 LIS-traffic） |

**注意**：这个 Drive 链接来自第三方数据集索引站，不是论文仓库，下载后务必核对图像数 10,425 与标注数 99,014。

---

### 3.14 SpaceDG（不能用于检测）

| 项目 | 内容 |
|---|---|
| **官方** | https://visionary-laboratory.github.io/SpaceDG/ ，代码 github.com/Visionary-Laboratory/SpaceDG |
| **论文** | arXiv:2605.22536 |
| **图片规模** | 160,000+ 张，近 1,000 个室内场景（3DGS 渲染） |
| **标注规模** | 971,000 个 QA 对；1,102 道 benchmark 题；无检测框 |
| **下载方式** | HuggingFace `https://huggingface.co/datasets/xlzhou126/SpaceDG-Bench` |
| **含夜间** | 9 类退化之一（低光、过曝、运动模糊、雾、失焦、低分辨率、水滴、JPEG、畸变） |

**结论**：任务是空间推理 VQA，没有框，不能训练或评测检测器。可借鉴的是它的物理退化合成引擎——低光 = 曝光衰减 + Poisson 散粒噪声 + 高斯读出噪声，且嵌在渲染管线里。

---

## 四、怎么自动生成低光图像

### 4.1 方法总览

| # | 方法 | 代表工作 | 真实性 | 成本 | 推荐度 |
|---|---|---|---|---|---|
| ① | 逆 ISP + RAW 域物理噪声注入 | Brooks 2019；BMVC 2021 *Crafting Object Detection in Very Low Light*；IJCV 2023 *Instance Segmentation in the Dark* | ★★★★★ | 中 | 主线 |
| ② | 全 ISP 仿真（多相机 + 200 条 tone curve + 白平衡扰动） | *Enhancing Low-Light Images: A Synthetic Data Perspective* | ★★★★☆ | 中高 | 预训练 |
| ③ | sRGB 光照衰减 + 统计对齐 | Dark-traffic | ★★☆☆☆ | 极低 | 只做消融 |
| ④ | 3D 渲染内嵌退化 | SpaceDG | ★★★★★ | 高 | 不适用检测 |
| ⑤ | GAN 日到夜迁移 | CycleGAN 类 | ★★★☆☆ | 中 | 慎用 |

### 4.2 主线四步

```
正常光 sRGB X
  → ① Unprocessing（逆 tone mapping → 逆 gamma → 逆色彩矩阵 → 逆白平衡 → 重插 Bayer）
  → ② 曝光衰减 I_dark = I_raw × k，k ∈ [0.01, 0.1]
  → ③ 噪声注入 Poisson(散粒) + Tukey(读出) + 条带 + 量化
  → ④ 前向 ISP（可随机化白平衡/色彩矩阵/色调曲线）
  → 低光 sRGB X_dark（标注与 X 完全一致）
```

**为什么这条最适合你**：标注零成本、天然配对（可直接喂对比学习的三元组损失）、曝光系数可扫（构造轻度到极暗的梯度）、已在检测/分割任务上验证过。

### 4.3 已知问题

- 网络图 ISP 未知不可精确反演，粗糙近似即可（论文明确说不需要恢复原始 RAW）；
- 强 JPEG 压缩的底图，逆 ISP 会把压缩块当内容放大；
- 每台"虚拟相机"要单独标定噪声参数，否则跨数据集的暗图噪声分布不一致。

### 4.4 用真实配对校验合成质量（这一步别省）

拿 DARK FACE 的 789 组真实配对，对你合成出的暗图做三件事：
1. 亮度直方图对比（合成是否过亮或过暗）；
2. 噪声统计对比（暗部方差随亮度的变化曲线）；
3. 下游掉点幅度对比（同一检测器在"真实暗图"与"合成暗图"上的 mAP 差距）。

差距过大说明噪声模型不准，回到第 ③ 步调参。ACDC 的同场景正常光/夜图对也可做粗校验。

---

## 五、适配性排序（针对低光小目标检测）

| 排名 | 数据集 | 理由 | 角色 |
|---|---|---|---|
| 1 | **NightSurveillance** | 监控俯视 + 夜间 + 46% 小目标 + 有框 + 可下载 | 主评测基准 |
| 2 | **ACDC（night）** | 夜图与正常光图成对 + 有框 + 直链可下 | 光照鲁棒性对照 |
| 3 | **Dark-traffic** | 2025 交通低光 + 检测/分割/光流 + 规模适中 | 辅助训练 / 消融 |
| 4 | **LLVIP / M3FD** | 严格配对的可见-红外，夜间组明确 | 跨模态对照 |
| 5 | **UA-DETRAC（夜）** | 监控路口视角最像，但需镜像、无夜间统计 | 多类车辆补充 |
| 6 | **BDD100K（夜）** | 规模大、多类，但车载、目标大 | 夜间切片泛化 |
| 7 | **ExDark** | 多类 + 光照 10 档，但规模小、需镜像 | 跨光照档评测 |
| 8 | **DARK FACE** | 只有人脸，但 789 组真实配对极值钱 | 合成管线校验 |
| 9 | **NightOwls** | 夜间但车载、100GB 以上 | 跨视角泛化 |
| 10 | **KAIST** | 分辨率太低 | 跨模态对照 |
| — | **DarkVision** | 设计最理想，但拿不到 | 申请中，不作关键路径 |
| — | **SpaceDG / Dark Zurich** | 无检测框 | 不使用 |

---

## 六、推荐数据组合（可直接写进实验方案）

```
训练：VisDrone / UA-DETRAC 白天图
      → 逆 ISP + 物理噪声 → 合成配对低光集（标注免费）
      → 与原始正常光图组成 (x_normal, x_low) 正样本对

校验：DARK FACE 789 组真实配对 + ACDC 同场景明暗对
      → 验证合成管线的物理真实性（4.4 节三件事）

终测：NightSurveillance（主）      ← 唯一真监控夜间小目标基准
      ACDC night（光照鲁棒）
      UA-DETRAC night / BDD100K night（多类泛化）
      ExDark（跨光照档）
```

**规模估算**：VisDrone 10,209 张加 UA-DETRAC 夜间子集约 3 万帧，合成后可得到 4 万对配对样本，足够支撑对比学习训练；评测侧 NightSurveillance 3.8 万张加 ACDC 1,000 张，量级匹配。

---

## 七、参考文献

1. **When Pedestrian Detection Meets Nighttime Surveillance: A New Benchmark** — IJCAI 2020, pp.509–515. https://www.ijcai.org/proceedings/2020/0071.pdf
2. **NightOwls: A Pedestrians at Night Dataset** — ACCV 2018, LNCS 11361:691–705.
3. **Getting to Know Low-light Images with The Exclusively Dark Dataset** — CVIU 2019, 178:30–42.
4. **UG² Track 2: A Collective Benchmark Effort...** — arXiv:1904.04474.
5. **LLVIP: A Visible-infrared Paired Dataset for Low-light Vision** — ICCVW 2021, arXiv:2108.10831.
6. **Target-aware Dual Adversarial Learning and M3FD** — CVPR 2022 (Oral), arXiv:2203.16220.
7. **Multispectral Pedestrian Detection: Benchmark Dataset and Baselines** — CVPR 2015.
8. **ACDC: The Adverse Conditions Dataset with Correspondences** — ICCV 2021.
9. **DarkVision: A benchmark and study for low-light image/video analysis** — Computational Visual Media, 2025. DOI 10.26599/CVM.2025.9450461；arXiv:2301.06269.
10. **Crafting Object Detection in Very Low Light** — BMVC 2021. https://www.bmva-archive.org.uk/bmvc/2021/assets/papers/0085.pdf
11. **Instance Segmentation in the Dark** — IJCV 2023, arXiv:2304.14298.
12. **Unprocessing Images for Learned Raw Denoising** — Brooks et al., CVPR 2019.
13. **CD-COCO** — arXiv:2311.06976.
14. **SpaceDG: Benchmarking Spatial Intelligence under Visual Degradation** — arXiv:2605.22536.

---

## 八、参考链接

| 数据集 | 入口 |
|---|---|
| NightSurveillance | https://github.com/xiaowang1516/NightSurveillance （内含百度网盘） |
| NightOwls | https://www.nightowls-dataset.org/download/ |
| ExDark | https://github.com/cs-chan/Exclusively-Dark-Image-Dataset （无链接，见 3.3 镜像） |
| DARK FACE | https://flyywh.github.io/CVPRW2019LowLight/ |
| DarkVision | https://www.sciopen.com/article/10.26599/CVM.2025.9450461 （仅论文） |
| ACDC | https://acdc.vision.ee.ethz.ch/download |
| BDD100K | https://bdd-data.berkeley.edu/ |
| UA-DETRAC | https://www.kaggle.com/datasets/bratjay/ua-detrac-orig ；https://sites.google.com/view/daweidu/projects/ua-detrac |
| LLVIP | https://github.com/bupt-ai-cz/LLVIP/blob/main/download_dataset.md |
| M3FD | https://github.com/JinyuanLiu-CV/TarDAL |
| KAIST | https://soonminhwang.github.io/rgbt-ped-detection/ |
| VisDrone | https://github.com/VisDrone/VisDrone-Dataset |
| Dark-traffic | https://drive.google.com/drive/folders/1B8EzDn64bGBgyRCfppL_jhcOA3hIwnzi |
| SpaceDG | https://huggingface.co/datasets/xlzhou126/SpaceDG-Bench |
| CD-COCO | https://github.com/Aymanbegh/CD-COCO |
| LOL | https://daooshee.github.io/BMVC2018website/ |
| VE-LOL | https://flyywh.github.io/IJCV2021LowLight_VELOL/ |
| SID | https://github.com/cchen156/Learning-to-See-in-the-Dark |
| MIT-Adobe FiveK | https://data.csail.mit.edu/graphics/fivek/ |

---

## 九、一句话总结

**能用且好用**：NightSurveillance（主基准，网盘直给）、ACDC night（直链、有框、有明暗对照）、Dark-traffic（2025 交通低光）、LLVIP / M3FD（可见-红外配对）。

**能用但别当关键路径**：DarkVision（拿不到）、UA-DETRAC（官网已死，走镜像）、ExDark（只有镜像）、DARK FACE（单类，但配对值钱）。

**不能用于检测**：SpaceDG、Dark Zurich。

**不是低光集**：VisDrone（只做合成底图与小目标上限验证）。

**核心策略不变**：拿白天大数据集加逆 ISP 合成配对低光集来训练，用真实夜间基准来评测，用真实配对数据校准合成管线的物理真实性。
