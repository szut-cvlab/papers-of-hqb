# 低光场景数据集调研报告

> 面向"低光场景小目标检测"，逐数据集核实 **图片规模 / 标注规模 / 图像形式 / 目标尺度 / 单多目标 / 类别 / 分辨率 / 模糊程度**，并注明各自的处理脚本。
> 数据来源：服务器 `10.18.32.108` 的 `/data1/datasets/hqb/` 目录（已清洗后实际状态），核实时间 **2026-11-16**。
> 处理脚本仓库：**https://github.com/szut-cvlab/DataProcessScripts**

---

## 一、核心结论：已收集三类低光数据集

围绕"低光场景小目标检测"课题，目前在服务器 `/data1/datasets/hqb/` 已收集并整理好 **9 个低光数据集**，按拍摄视角分为三大类：

**① 监控图片（5 个）** —— 最贴近"夜间视频侦查"部署场景，是本课题的主战场：

- **NightSurveillance**：监控俯视 + 夜间 + 行人，唯一"监控+夜间+有框"主基准（去重后 train 17,309 / test 17,298 图，92,787 框）；
- **LLVIP**：监控平视 + 夜间（清洗后 train 5,714 / test 3,190 图，27,190 框）；
- **DETRAC**：监控路口 + 交通车辆多类（清洗后剩 test 5,660 图，47,804 框）；
- **IntersectionFlow5K**：高密度路口监控，8 类交通目标，小目标极密集（1,291 图，69,549 框）；
- **FishEye8K**：鱼眼镜头监控，5 类，仅夜间(N)+黄昏(E)（2,159 图，32,094 框）。

**② 车载夜景图片（3 个）** —— 小目标占比最高，但视角是车载前视，与监控有域差异：

- **ACDC（night）**：车载前视 + 夜间，同场景有白天对照图（train 400 / val 106 图，3,534 框）；
- **NightCity**：车载前视 + 夜间，分割标注转检测框（train 2,998 / val 1,299 图，55,145 框）；
- **BDD100K（night）**：车载前视 + 夜间，10 类、规模最大（train 27,888 / val 3,906 图，520,736 框）。

**③ 近景图片（1 个）** —— 相机手持拍摄，大目标为主：

- **NOD**：DSLR 近景拍摄，3 类，高分辨率（train 464 / val 51 / test 53，共 568 图，7,979 框）。

> **一句话总结**：监控类提供"部署场景 + 行人 + 配对"（但目标偏大），车载类提供"小目标 + 多类"（但视角不同），近景类提供"跨域泛化对照"（大目标）。三类互补，共同支撑低光小目标检测研究。

---

## 二、总表（按图像形式分三类）

### 2.1 监控图片

| 数据集 | 图片规模 | 标注规模（框数） | 图像形式 | 目标尺度 | 单/多目标 | 类别（编号） | 分辨率 | 模糊 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **NightSurveillance** | train 17,309 / test 17,298（原 38,113 去重后） | train 46,773 / test 46,014，**合计 92,787** | 监控俯视（夜间，清晰） | 框高中位 **141px**，小目标(<50px)仅 1.1% | 多目标为主（0 目标 7,255 张） | 1 类：`1=pedestrian` | 1920×1080 | 部分帧运动模糊 |
| **LLVIP** | train 5,714 / test 3,190（清洗后） | train 19,453 / test 7,737，**合计 27,190** | 监控平视（夜间+红外配对，偏近） | 框高中位 **225px**，大目标为主（<50px 仅 0.3%） | 多目标（均值 2.7） | 1 类：`1=person` | 1280×1024 | 否 |
| **DETRAC** | test 5,660（train 已删空） | test **47,804** | 监控路口（交通，较暗非纯夜间） | 框高中位 **61px**，中等偏小（<50px 占 37.3%） | 多目标（车辆密集） | 4 类：`1=bus, 2=car, 3=others, 4=van` | 960×540 | 部分帧模糊 |
| **IntersectionFlow5K** | train 992 / val 147 / test 152（**共 1,291**，官方 6,928） | 53,748 / 7,674 / 8,127，**合计 69,549** | 高密度路口监控（已删白天图） | 框高中位 **32px**，<50px 占 **61.8%**，小目标极密集 | 多目标（均值 **53.9** 个/图，密集遮挡） | 8 类：`0=vehicle, 1=bus, 2=bicycle, 3=pedestrian, 4=engine, 5=truck, 6=tricycle, 7=obstacle` | 1920×1080 | 否（夜间有眩光） |
| **FishEye8K** | train 1,727 / test 432（**共 2,159**，仅 N+E） | train 25,617 / test 6,477，**合计 32,094** | 鱼眼镜头监控（夜间+黄昏） | 框高中位 **43px**，<50px 占 **62.8%**，小目标为主 | 多目标（均值 14.9 个/图） | 5 类：`0=bus, 1=motorcycle or scooter, 2=car, 3=pedestrian, 4=truck` | 1240×1088 | 鱼眼畸变 |

### 2.2 车载夜景图片

| 数据集 | 图片规模 | 标注规模（框数） | 图像形式 | 目标尺度 | 单/多目标 | 类别（编号） | 分辨率 | 模糊 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **ACDC（night）** | train 400 / val 106 / test 500（无 GT）；另 ref：train_ref 200 / val_ref 53 / test_ref 125 | train 2,975 / val 559，**合计 3,534**；ref 另 2,726 + 539 | 车载前视（夜间，有同场景白天对照图） | 框高中位 **42px**，小目标(<50px)占 **58.4%** | 多目标（0~26 个/图，均值 7.0） | 8 类：`24=person, 25=rider, 26=car, 27=truck, 28=bus, 31=train, 32=motorcycle, 33=bicycle` | 1920×1080 | 行人轮廓模糊 |
| **NightCity** | train 2,998 / val 1,299 | train 36,606 / val 18,539，**合计 55,145** | 车载前视（夜间，灯光噪声多） | 框高中位 **14px**，小目标(<50px)占 **83.1%**（分割转框，框偏小） | 多目标（0~30+ 个/图） | 8 类：`1=person, 2=rider, 3=car, 4=truck, 5=bus, 6=train, 7=motorcycle, 8=bicycle` | 1024×512 | 灯光干扰明显 |
| **BDD100K（night）** | train 27,888 / val 3,906 | train 456,218 / val 64,518，**合计 520,736** | 车载前视（夜间，质量较 ACDC 好） | 框高中位 **27px**，小目标(<50px)占 **74.2%** | 多目标 | 10 类：`1=pedestrian, 2=rider, 3=car, 4=truck, 5=bus, 6=train, 7=motorcycle, 8=bicycle, 9=traffic light, 10=traffic sign` | 1280×720 | 否（较清晰） |

### 2.3 近景图片

| 数据集 | 图片规模 | 标注规模（框数） | 图像形式 | 目标尺度 | 单/多目标 | 类别（编号） | 分辨率 | 模糊 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **NOD** | train 464 / val 51 / test 53（**共 568**） | train 6,516 / val 713 / test 750，**合计 7,979** | 相机近景（DSLR 手持拍摄） | 框高中位 **266px**，大目标为主（<50px 仅 1.5%） | 多目标（3~30+ 个/图，均值 14.0） | 3 类：`1=person, 2=bicycle, 3=car` | 3936×2624 / 5472×3648 | 否（高分辨率清晰） |

> **全部 9 个数据集合计**：**90,286 张图、856,818 个标注框**（不含 ACDC 的 ref 对照图；若含 ref，为 90,664 图 / 860,083 框）。

---

## 三、逐数据集详情

### 3.1 NightSurveillance（监控 · 主基准）

| 项目 | 内容 |
| --- | --- |
| **来源** | IJCAI 2020，*When Pedestrian Detection Meets Nighttime Surveillance* |
| **下载** | GitHub `github.com/xiaowang1516/NightSurveillance`（内含百度网盘） |
| **图片规模** | 原始 38,113 张；已去重相邻帧后 **train 17,309 / test 17,298** |
| **标注规模** | **train 46,773 框 / test 46,014 框（合计 92,787 框）**，单类 pedestrian；train 另有 7,255 张无标注帧（负样本） |
| **图像形式** | 监控俯视，夜间 17:00–次日 5:00 |
| **目标尺度** | 框高中位 141px，**小目标(<50px)仅 1.1%**（实质是中尺度行人） |
| **单/多目标** | 多目标为主，0 目标帧 7,255 张 |
| **类别** | 1 类：`1=pedestrian` |
| **分辨率** | 1920×1080 |
| **模糊** | 部分帧运动模糊 |
| **处理脚本** | [`process_nightsurveillance.py`](https://github.com/szut-cvlab/DataProcessScripts/blob/master/NightSurveillance/process_nightsurveillance.py) —— 相邻帧去重（帧号差=1 且 MAE≤0.5 且差异像素≤0.5%）+ VOC→COCO |

> 说明：官方只给 train/test 划分（无 val），且 train/test 的视频段 ID 完全相同、帧号交错切分，存在**时序泄漏**。建议自行按视频段重划验证集，且小目标占比极低，若做"小目标"实验需注意口径。

### 3.2 LLVIP（监控）

| 项目 | 内容 |
| --- | --- |
| **来源** | ICCVW 2021，*LLVIP: A Visible-infrared Paired Dataset for Low-light Vision* |
| **下载** | GitHub `github.com/bupt-ai-cz/LLVIP/blob/main/download_dataset.md` |
| **图片规模** | 清洗后 **train 5,714 / test 3,190**（原始 15,488 对） |
| **标注规模** | **train 19,453 框 / test 7,737 框（合计 27,190 框）**，单类 person；空标注 0 张 |
| **图像形式** | 监控平视，夜间（原含部分白天图，已删亮度>80 的 273 张） |
| **目标尺度** | 框高中位 225px，**大目标为主**（<50px 仅 0.3%） |
| **单/多目标** | 多目标，均值 2.7 个/图 |
| **类别** | 1 类：`1=person` |
| **分辨率** | 1280×1024 |
| **模糊** | 否 |
| **处理脚本** | [`process_llvip.py`](https://github.com/szut-cvlab/DataProcessScripts/blob/master/LLVIP/process_llvip.py) —— 保留前缀 01,03~09（删 02,10~18,25）+ 删白天图 + VOC→COCO |

> 注意：服务器上**红外目录缺失**，只有可见光 + COCO 标注。红外是它的核心价值（可见-红外严格配对、一套标注两模态共用），需从本地补齐 infrared 才能做红外蒸馏。

### 3.3 DETRAC（监控 · 交通）

| 项目 | 内容 |
| --- | --- |
| **来源** | UA-DETRAC（官网已下线），服务器上是 night 子集 |
| **下载** | Kaggle `kaggle.com/datasets/bratjay/ua-detrac-orig` |
| **图片规模** | **test 5,660**（train 已删空） |
| **标注规模** | **test 47,804 框**（4 个序列 MVI_40763/40773/40775/40793） |
| **图像形式** | 监控路口，较暗（非纯夜间） |
| **目标尺度** | 框高中位 61px，中等偏小（<50px 占 37.3%） |
| **单/多目标** | 多目标（车辆密集） |
| **类别** | 4 类：`1=bus, 2=car, 3=others, 4=van`（粒度粗，others 是兜底类） |
| **分辨率** | 960×540 |
| **模糊** | 部分序列模糊（原序列 1 尤甚，已删） |
| **处理脚本** | [`process_detrac_simple.py`](https://github.com/szut-cvlab/DataProcessScripts/blob/master/DETRAC/process_detrac_simple.py) —— 序列级亮度筛选（抽 5 帧灰度中位≤85）+ 人工删 4 个序列 + XML→COCO |

> ⚠️ **重要**：被删的 4 个序列恰好等于原 train 集，删完后 DETRAC 只剩 test（5,660 图），**无训练集**。若要用于训练，需从这 5,660 张按序列重新划分 train/val/test。

### 3.4 IntersectionFlow5K（监控 · 高密度路口）

| 项目 | 内容 |
| --- | --- |
| **来源** | PRCV 2025，*FlowDet: Overcoming Perspective and Scale Challenges in Real-Time End-to-End Traffic Detection* |
| **下载** | arXiv:2508.19565 / 官方 Intersection-Flow-5K 仓库 |
| **图片规模** | 官方 6,928 张；服务器子集 **train 992 / val 147 / test 152（共 1,291）** |
| **标注规模** | **train 53,748 / val 7,674 / test 8,127 框（合计 69,549 框）** |
| **图像形式** | 固定摄像头高密度路口监控，已删白天图（按拍摄时间戳判定） |
| **目标尺度** | 框高中位 **32px**，<50px 占 **61.8%**，框高/图高<5% 占 63.5%，极端尺度变化 |
| **单/多目标** | 多目标，均值 **53.9 个/图**，严重遮挡（最高 85% 遮挡也被标注） |
| **类别** | 8 类：`0=vehicle, 1=bus, 2=bicycle, 3=pedestrian, 4=engine, 5=truck, 6=tricycle, 7=obstacle` |
| **分辨率** | 1920×1080 |
| **模糊** | 否（夜间有眩光干扰） |
| **处理脚本** | [`process_if5k.py`](https://github.com/szut-cvlab/DataProcessScripts/blob/master/IntersectionFlow5K/process_if5k.py) —— sky_mean<100 粗筛 + 按时间戳删 33 张白天图（1,324→1,291）；YOLO→YOLO，不转 COCO |

> 说明：这是"高密度交通监控"基准，小目标 + 密集遮挡是核心挑战，比 NightSurveillance 更适合做"监控场景小目标检测"。但服务器上只有 1,291 张子集（官方 6,928 张），且 owner 是另一位同学（pyc）。

### 3.5 FishEye8K（监控 · 鱼眼镜头）

| 项目 | 内容 |
| --- | --- |
| **来源** | CVPR 2023 Workshop，*FishEye8K: A Benchmark and Dataset for Fisheye Camera Object Detection*（台湾新竹市 18 路鱼眼监控） |
| **下载** | GitHub `github.com/MoyoG/FishEye8K` |
| **图片规模** | **train 1,727 / test 432（共 2,159）**，仅保留夜间 `_N_` + 黄昏 `_E_` |
| **标注规模** | **train 25,617 / test 6,477 框（合计 32,094 框）**，COCO + YOLO 双格式 |
| **图像形式** | 鱼眼镜头监控，只保留低光时段（camera3_N + camera4_N + camera4_E） |
| **目标尺度** | 框高中位 **43px**，<50px 占 **62.8%**，框高/图高<5% 占 67.4% |
| **单/多目标** | 多目标，均值 14.9 个/图 |
| **类别** | 5 类：`0=bus, 1=motorcycle or scooter, 2=car, 3=pedestrian, 4=truck` |
| **分辨率** | 1240×1088（鱼眼畸变，非标准 16:9） |
| **模糊** | 鱼眼畸变（边缘拉伸） |
| **处理脚本** | [`process_fisheye8k.py`](https://github.com/szut-cvlab/DataProcessScripts/blob/master/Fisheye8K/process_fisheye8k.py) —— 只留 `_N_`/`_E_`，混合随机 8:2 重划 train/test，输出 COCO + YOLO |

> 说明：**已替换掉原来 pyc 的旧版本**（旧版 3,090 张，混入 M 早晨白天图且未删干净）。现版本只含真正的低光时段，并按 8:2 混合重划（打破官方"train/test 不共享相机"约束，两个相机都进 train 和 test）。鱼眼视角独特（边缘畸变），是现有数据里没有的视角类型。

### 3.6 ACDC（night 子集）（车载 · 明暗对照）

| 项目 | 内容 |
| --- | --- |
| **来源** | ICCV 2021，*ACDC: The Adverse Conditions Dataset with Correspondences* |
| **下载** | `acdc.vision.ee.ethz.ch/download` |
| **图片规模** | night **train 400 / val 106 / test 500（无 GT）**；对应正常光 ref：train_ref 200 / val_ref 53 / test_ref 125 |
| **标注规模** | **train 2,975 / val 559 框（合计 3,534 框）**；ref 另 train_ref 2,726 / val_ref 539 框 |
| **图像形式** | 车载前视，夜间；**同场景有白天对照图**（`*_ref`） |
| **目标尺度** | 框高中位 42px，**小目标(<50px)占 58.4%**，框高/图高<5% 占 61.6% |
| **单/多目标** | 多目标，均值 7.0 个/图（506 张中 15 张 0 目标） |
| **类别** | 8 类：`24=person, 25=rider, 26=car, 27=truck, 28=bus, 31=train, 32=motorcycle, 33=bicycle` |
| **分辨率** | 1920×1080 |
| **模糊** | 行人轮廓模糊 |
| **处理脚本** | [`process_acdc.py`](https://github.com/szut-cvlab/DataProcessScripts/blob/master/ACDC/process_acdc.py) —— 只保留 night、扁平化目录、`file_name` 去掉条件层、输出 COCO |

> 注意：服务器版本已重构——只保留 night（fog/rain/snow 全删），目录扁平化为 `train/ val/ test/ *_ref/`，标注统一在 `gt_detection/night/` 下且路径与图片严格对应（之前的旧版残留在根目录的全条件 json 已清理）。
> `segm_file` 字段指向的语义分割标注（labelIds）在包里不存在，只有检测标注。类别 ID 沿用 Cityscapes（跳过 29/30，不连续，训练时需自行映射）。

> **个人提示**：个人认为该数据集为车载图片，图像中小目标/目标较少，多为开阔路段，且车灯会影响检测，之后会考虑是否要删去。

### 3.7 NightCity（车载 · 分割转框）

| 项目 | 内容 |
| --- | --- |
| **来源** | *Night-time Scene Parsing with a Large Real Dataset* |
| **下载** | 见 NightCity 官方页 |
| **图片规模** | **train 2,998 / val 1,299** |
| **标注规模** | **train 36,606 / val 18,539 框（合计 55,145 框）**（val 已换用官方重新标注版） |
| **图像形式** | 车载前视，夜间，灯光多、噪声多（比监控清晰、比照片模糊） |
| **目标尺度** | 框高中位 **14px**，**小目标(<50px)占 83.1%**（分割外接框，框偏小、碎片多） |
| **单/多目标** | 多目标 |
| **类别** | 8 类：`1=person, 2=rider, 3=car, 4=truck, 5=bus, 6=train, 7=motorcycle, 8=bicycle` |
| **分辨率** | 1024×512 |
| **模糊** | 灯光干扰明显，车灯易被误检为小目标 |
| **处理脚本** | [`process_nightcity_simple.py`](https://github.com/szut-cvlab/DataProcessScripts/blob/master/NightCity/process_nightcity_simple.py) —— 语义分割 mask → 8 连通域 → 外接矩形 → COCO |

> 注意：由分割 mask 转框，框质量偏弱（遮挡目标框偏小、碎片多），建议只作弱监督训练数据，不进验证/测试集。

### 3.8 BDD100K（night 切片）（车载 · 质量较好）

| 项目 | 内容 |
| --- | --- |
| **来源** | BDD100K（Berkeley） |
| **下载** | `bdd-data.berkeley.edu`（需注册）或镜像 `dl.cv.ethz.ch/bdd100k/data/` |
| **图片规模** | night 切片 **train 27,888 / val 3,906** |
| **标注规模** | **train 456,218 / val 64,518 框（合计 520,736 框，占全部数据集标注量的 61%）** |
| **图像形式** | 车载前视，夜间，质量比 ACDC 好一点 |
| **目标尺度** | 框高中位 27px，**小目标(<50px)占 74.2%** |
| **单/多目标** | 多目标 |
| **类别** | 10 类：`1=pedestrian, 2=rider, 3=car, 4=truck, 5=bus, 6=train, 7=motorcycle, 8=bicycle, 9=traffic light, 10=traffic sign` |
| **分辨率** | 1280×720 |
| **模糊** | 否 |
| **处理脚本** | [`process_bdd100k_night_simple.py`](https://github.com/szut-cvlab/DataProcessScripts/blob/master/BDD100K/process_bdd100k_night_simple.py) —— timeofday==night 筛选 + 视觉过亮筛 + 类别映射（已修正历史 bug：补回 pedestrian/bicycle/motorcycle 三类）+ COCO/YOLO |

### 3.9 NOD（近景 · 相机拍摄）

| 项目 | 内容 |
| --- | --- |
| **来源** | BMVC 2021，*NOD: Taking a Closer Look at Detection under Extreme Low-Light* |
| **下载** | GitHub `github.com/igor-morawski/NOD` |
| **图片规模** | 服务器 **train 464 / val 51 / test 53（共 568 张子集）**，本地完整版 7,219 张 |
| **标注规模** | **train 6,516 / val 713 / test 750 框（合计 7,979 框）** |
| **图像形式** | 相机近景（DSLR 手持：Nikon D750 + Sony RX100 VII） |
| **目标尺度** | 框高中位 **266px**，大目标为主（<50px 仅 1.5%，框高/图高<5% 占 20.2%） |
| **单/多目标** | 多目标，均值 14.0 个/图（行人密集街景） |
| **类别** | 3 类：`1=person, 2=bicycle, 3=car` |
| **分辨率** | 3936×2624 / 5472×3648（高分辨率） |
| **模糊** | 否 |
| **处理脚本** | [`process_nod_simple.py`](https://github.com/szut-cvlab/DataProcessScripts/blob/master/NOD/process_nod_simple.py) —— 两阶段：① 大目标/过亮初筛 ② 低光区间 + 相对小目标 + 欠曝 + 人工复核名单（257 张内置） |

> 注意：NOD 是"近景大目标"，与监控/车载视角差异最大，宜作跨域泛化测试，不宜混入主训练集。
> 服务器版本已与脚本最新产出一致（568 张 / 7,979 框）；此前的 564 张版本缺 4 张图（源文件补齐后符合筛选条件被自动补回：train +3、test +1），旧标注备份在 `NOD/_backup_564/`。

---

## 四、处理脚本仓库

所有数据集的清洗/筛选脚本已整理为可复现版本，统一放在：

**https://github.com/szut-cvlab/DataProcessScripts**

每个数据集一个子目录，内含「一个处理脚本 + 一份 README」（写清官方下载路径、处理规则、运行方式）。原则是 **「原始数据 + 脚本」即可复现出与服务器一致的子集**，脚本不修改原始数据，只做筛选与标注格式转换。

| 数据集 | 脚本 | 核心处理 |
| --- | --- | --- |
| NightSurveillance | `NightSurveillance/process_nightsurveillance.py` | 相邻帧去重 + VOC→COCO |
| LLVIP | `LLVIP/process_llvip.py` | 前缀筛选 + 删白天图 + VOC→COCO |
| IntersectionFlow5K | `IntersectionFlow5K/process_if5k.py` | sky_mean 粗筛 + 按时间戳删白天图 |
| DETRAC | `DETRAC/process_detrac_simple.py` | 序列亮度筛选 + 人工删序列 + XML→COCO |
| BDD100K | `BDD100K/process_bdd100k_night_simple.py` | timeofday 筛选 + 过亮筛 + 类别映射修正 |
| NOD | `NOD/process_nod_simple.py` | 大目标/过亮初筛 + 低光小目标筛选 + 人工复核 |
| NightCity | `NightCity/process_nightcity_simple.py` | 分割 mask → 连通域 → 外接框 |
| FishEye8K | `Fisheye8K/process_fisheye8k.py` | 只留 `_N_`/`_E_` + 混合重分 + COCO/YOLO |
| ACDC | `ACDC/process_acdc.py` | 只留 night + 扁平化 + COCO |

> 运行环境：多数脚本仅需 Python 标准库；涉及亮度/边缘计算的（NightSurveillance、IntersectionFlow5K、NOD、LLVIP）需 `Pillow + numpy + opencv-python`。各 README 内有具体依赖与命令。

---

## 五、适配性排序（针对低光小目标检测）

| 排名 | 数据集 | 视角 | 小目标占比 | 标注量 | 角色 |
| --- | --- | --- | --- | --- | --- |
| 1 | **NightSurveillance** | 监控 | 低（1.1%） | 92,787 | 主评测基准（监控夜间行人） |
| 2 | **BDD100K night** | 车载 | 高（74%） | 520,736 | 多类小目标训练/泛化 |
| 3 | **NightCity** | 车载 | 极高（83%） | 55,145 | 小目标弱监督（框质量弱） |
| 4 | **ACDC night** | 车载 | 高（58%） | 3,534（+ref 3,265） | 光照鲁棒 + 明暗对照 |
| 5 | **IntersectionFlow5K** | 监控 | 极高（62%） | 69,549 | 高密度监控小目标（密集遮挡） |
| 6 | **FishEye8K** | 监控 | 高（63%） | 32,094 | 鱼眼视角监控小目标 |
| 7 | **LLVIP** | 监控 | 低 | 27,190 | 可见-红外配对（蒸馏） |
| 8 | **DETRAC** | 监控 | 中（37%） | 47,804 | 车辆多类（需重划 train） |
| 9 | **NOD** | 近景 | 极低 | 7,979 | 跨域泛化测试 |

---

## 六、关键结论与建议

1. **"小目标"和"监控"在原有数据里是错位的**：监控数据集（NightSurveillance/LLVIP/DETRAC）目标都偏大，小目标反而集中在车载数据集（NightCity/BDD100K/ACDC）。**IntersectionFlow5K 和 FishEye8K 补上了"监控 + 小目标"这一缺口**——两者都是监控视角且小目标占比 60%+，是监控小目标检测的更好候选。
2. **标注量分布极不均衡**：BDD100K 一个数据集就占了全部标注量的 61%（52 万框），而 NOD 只有 7,979 框、ACDC 只有 3,534 框。做类别平衡采样时需要考虑这一点。
3. **遗留问题**：NightSurveillance 需自行重划验证集（时序泄漏）；LLVIP 缺红外目录；DETRAC 删空 train 后需重划；IntersectionFlow5K 仍是 YOLO 格式且是 pyc 同学的。
4. **车载三件套（BDD100K/NightCity/ACDC）小目标占比极高**，是做"小目标检测"的主要数据来源，但视角是车载前视，与监控俯视有域差异。
5. **NOD 是近景大目标**，仅作跨域泛化测试集。
6. **类别统一**：若要合并训练，需把各数据集映射到统一类别表（person/car 等），非目标类别的框删除。注意各数据集类别 id 起点不同：IntersectionFlow5K/FishEye8K 从 0 开始，其余从 1 开始，ACDC 沿用 Cityscapes 的不连续 id（24~33）。

---

## 七、参考链接

| 数据集 | 入口 |
| --- | --- |
| 处理脚本仓库 | <https://github.com/szut-cvlab/DataProcessScripts> |
| NightSurveillance | <https://github.com/xiaowang1516/NightSurveillance> |
| LLVIP | <https://github.com/bupt-ai-cz/LLVIP/blob/main/download_dataset.md> |
| UA-DETRAC | <https://www.kaggle.com/datasets/bratjay/ua-detrac-orig> |
| ACDC | <https://acdc.vision.ee.ethz.ch/download> |
| NightCity | <https://github.com/yuyang18/NightCity（见原论文页）> |
| BDD100K | <https://bdd-data.berkeley.edu/> |
| NOD | <https://github.com/igor-morawski/NOD> |
| IntersectionFlow5K | <https://arxiv.org/abs/2508.19565> |
| FishEye8K | <https://github.com/MoyoG/FishEye8K> |

---

## 八、怎么生成低光图像

### 8.1 方法总览

| # | 方法 | 代表工作 | 真实性 | 成本 | 推荐度 |
| - | --- | --- | --- | --- | --- |
| ① | 逆 ISP + RAW 域物理噪声注入 | Brooks 2019；BMVC 2021 *Crafting Object Detection in Very Low Light*；IJCV 2023 *Instance Segmentation in the Dark* | ★★★★★ | 中 | 主线 |
| ② | 全 ISP 仿真（多相机 + 200 条 tone curve + 白平衡扰动） | *Enhancing Low-Light Images: A Synthetic Data Perspective* | ★★★★☆ | 中高 | 预训练 |
| ③ | sRGB 光照衰减 + 统计对齐 | Dark-traffic | ★★☆☆☆ | 极低 | 只做消融 |
| ④ | 3D 渲染内嵌退化 | SpaceDG | ★★★★★ | 高 | 不适用检测 |
| ⑤ | GAN 日到夜迁移 | CycleGAN 类 | ★★★☆☆ | 中 | 慎用 |

### 8.2 主线四步

```
正常光 sRGB X
  → ① Unprocessing（逆 tone mapping → 逆 gamma → 逆色彩矩阵 → 逆白平衡 → 重插 Bayer）
  → ② 曝光衰减 I_dark = I_raw × k，k ∈ [0.01, 0.1]
  → ③ 噪声注入 Poisson(散粒) + Tukey(读出) + 条带 + 量化
  → ④ 前向 ISP（可随机化白平衡/色彩矩阵/色调曲线）
  → 低光 sRGB X_dark（标注与 X 完全一致）
```

**为什么这条最适合**：标注零成本、天然配对（可直接喂对比学习的三元组损失）、曝光系数可扫（构造轻度到极暗的梯度）、已在检测/分割任务上验证过。

### 8.3 已知问题

- 网络图 ISP 未知不可精确反演，粗糙近似即可（论文明确说不需要恢复原始 RAW）；
- 强 JPEG 压缩的底图，逆 ISP 会把压缩块当内容放大；
- 每台"虚拟相机"要单独标定噪声参数，否则跨数据集的暗图噪声分布不一致。

### 8.4 用真实配对校验合成质量（这一步别省）

拿 DARK FACE 的 789 组真实配对，对合成出的暗图做三件事：

1. 亮度直方图对比（合成是否过亮或过暗）；
2. 噪声统计对比（暗部方差随亮度的变化曲线）；
3. 下游掉点幅度对比（同一检测器在"真实暗图"与"合成暗图"上的 mAP 差距）。

差距过大说明噪声模型不准，回到第 ③ 步调参。ACDC 的同场景正常光/夜图对也可做粗校验。
