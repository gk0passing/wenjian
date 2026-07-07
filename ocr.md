1. 团队与论文基本信息

PP-OCRv6 来自 百度 PaddlePaddle / PaddleOCR 团队，论文题目是 “PP-OCRv6: From 1.5M to 34.5M Parameters, Surpassing Billion-Scale VLMs on OCR Tasks”。作者团队包括 Yubo Zhang、Xueqing Wang、Manhui Lin、Yue Zhang 等，项目负责人是 Cheng Cui。论文给出的单位是 PaddlePaddle Team, Baidu Inc.，并提供了 PaddleOCR 官网、GitHub 源码和 HuggingFace 模型地址。

从定位上看，PP-OCRv6 不是一个 VLM OCR 模型，而是 PaddleOCR 系列中最新一代的轻量级专用 OCR 系统。它延续了 PP-OCR 系列的两阶段 OCR pipeline，即：

文本检测模型 → 文本框裁剪 / 矫正 → 文本识别模型

论文明确强调，PP-OCRv6 面向的是工业级 OCR 部署场景，关注高精度、低延迟、低参数量、低幻觉、强定位能力，而不是追求一个通用视觉语言模型。

2. 背景与研究动机
2.1 大模型时代下，为什么还需要专用 OCR 小模型？

近几年 OCR 领域受到 VLM 的冲击很大。GPT、Qwen-VL、Gemini、Kimi、MiniMax 这类大视觉语言模型具备较强的图像理解和文本生成能力，看起来可以直接做 OCR。但是 PP-OCRv6 论文认为，VLM 在真实 OCR 场景中存在三个核心问题：

第一，定位不精确。OCR 不只是读出文字，还需要给出准确文本框、版面结构和文字区域。VLM 往往可以描述图中有什么文字，但难以稳定输出紧致、准确的 polygon / bbox。

第二，容易幻觉。VLM 有很强的语言先验，可能会把图像中拼错的单词、重复字符、特殊符号“纠正”为语言上更合理的内容。这对于金融、医疗、合同、票据、工业铭牌等数据敏感场景风险很大。

第三，计算成本高。VLM 参数量大、推理慢、部署成本高，不适合高吞吐、低延迟、端侧或 CPU 场景。

因此，PP-OCRv6 的核心动机是证明：

在 OCR 这种高精度视觉感知任务上，专用小模型 + 高质量数据 + 针对性架构设计，仍然可以比大规模 VLM 更实用、更稳定、更适合工业部署。

2.2 PP-OCR 系列的发展背景

PP-OCR 系列一直是 PaddleOCR 的核心轻量化 OCR 方案。前几代大致可以理解为：

PP-OCR / PP-OCRv2：
面向轻量 OCR 系统的基础工程化探索。

PP-OCRv3：
引入 SVTR-LCNet、LK-PAN 等结构，提升移动端 OCR 效果。

PP-OCRv4：
server 侧采用 PPHGNetV2，mobile 侧采用 LCNetV3。

PP-OCRv5：
基本继承 PP-OCRv4 的结构，更强调 data-centric，即通过数据难度、准确性、多样性提升性能。

PP-OCRv6：
在继承 PP-OCRv5 数据方法的基础上，重点重构模型架构。

论文指出，PP-OCRv5 已经证明了一个 5M 参数左右的小模型，只要数据足够高质量，性能上限可以很高。但 PP-OCRv5 仍存在结构瓶颈：server 和 mobile 使用不同 backbone，工程维护复杂；LCNetV3 仍是 MobileNet-style 的 DW → SE → PW 堆叠；检测 neck 中的卷积结构也有进一步轻量化和扩大感受野的空间。

3. PP-OCRv6 的核心贡献

PP-OCRv6 的贡献可以概括为五点。

3.1 贡献一：统一且可扩展的三档模型家族

PP-OCRv6 提供了 medium、small、tiny 三个版本：

版本	检测模型参数	识别模型参数	端到端参数	主要定位
PP-OCRv6_tiny	0.43M	1.1M	1.5M	极致轻量，边缘端 / CPU
PP-OCRv6_small	2.48M	5.2M	7.7M	移动端、普通工业部署
PP-OCRv6_medium	15.5M	19M	34.5M	服务端高精度部署

这三个版本共享相同的基本 block 设计，只是深度和宽度不同，因此在工程上更容易统一训练、部署和维护。

3.2 贡献二：提出统一 backbone —— LCNetV4

PP-OCRv6 使用 LCNetV4 替代 PP-OCRv5 中 server / mobile 分离的 backbone 设计。LCNetV4 的核心是 MetaFormer-style block，把每个 block 显式拆成：

Token Mixer：负责空间信息聚合
Channel Mixer：负责通道特征交互

同时在 token mixer 中加入结构重参数化 RepDWConv，训练时多分支增强表达，推理时融合成单个 depthwise convolution，不增加推理成本。

3.3 贡献三：检测端提出 RepLKFPN + 深监督 + Dice/Focal Loss

PP-OCRv6 检测端的主要改进包括：

LCNetV4 Backbone
+
RepLKFPN Neck
+
DB Head
+
Auxiliary Deep Supervision
+
Dice Loss + Focal Loss

其中 RepLKFPN 用可重参数化的大核 depthwise 卷积替代原来的普通卷积，在减少参数的同时扩大局部感受野；辅助深监督增强中间层训练信号；Focal Loss 补充 Dice Loss 对困难像素的监督。

3.4 贡献四：识别端提出 EncoderWithLightSVTR

识别端采用 LCNetV4 + EncoderWithLightSVTR + CTC/NRTR 多头训练。

相比 PP-OCRv5 的 EncoderWithSVTR，PP-OCRv6 的 LightSVTR 做了两点关键优化：

1. 加入 1×7 depthwise convolution，建模文本横向局部上下文。
2. 用 additive skip 替代 concat skip，减少参数量。

训练时使用 CTC 和 NRTR 两个 head，推理时只保留 CTC head，因此可以在增强上下文学习的同时不增加推理开销。

3.5 贡献五：扩展多语言和工业场景能力

PP-OCRv6 medium / small 支持 50 种语言，tiny 支持 49 种语言。论文还专门增强了工业场景数据，包括：

数码管
点阵字符
轮胎文字
电源适配器文字
工业铭牌
屏幕显示
特殊字符
艺术字
旋转文本
古文 / 日文 / 多语言文本

这说明 PP-OCRv6 的提升不是单纯靠结构，而是结构改进 + 数据体系扩展共同作用。

4. 方法细节
4.1 整体 pipeline

PP-OCRv6 仍然是检测识别分离的经典 OCR 系统：

输入图像
 ↓
文本检测模型
 ↓
文本区域 crop & resize
 ↓
文本识别模型
 ↓
输出文字

论文第 3 页 Figure 2 给出了整体架构图。检测和识别都使用 LCNetV4 backbone，但检测模式输出多尺度 2D 特征，识别模式输出 1D 序列特征。

4.2 LCNetV4 Backbone
4.2.1 LCNetV4Block 的结构

LCNetV4Block 的核心结构是：

输入 x
 ↓
Token Mixer:
    RepDWConv 3×3
    SE Attention
    residual connection
 ↓
Channel Mixer:
    1×1 Conv Expand，通道扩展 2 倍
    GELU
    1×1 Conv Compress
    residual connection
 ↓
输出 y

可以理解为：

空间建模和通道建模被拆开了。

这和 LCNetV3 的 MobileNet-style block 不同。LCNetV3 是 DWConv、SE、PWConv 顺序堆叠，空间建模和通道建模耦合在一起；LCNetV4 则显式分离两者，从而让空间感受野和通道交互可以分别设计。

4.2.2 RepDWConv 的作用

RepDWConv 是 LCNetV4 的关键细节。

训练时：

3×3 DWConv + BN
+
1×1 DWConv + BN
+
Identity + BN

推理时：

融合成一个 3×3 DWConv

这样做的好处是：

训练阶段：多分支结构提供更强表达能力和更好的优化路径。
推理阶段：结构融合后没有额外分支，不增加延迟。

这类设计非常适合工业 OCR，因为 OCR 小模型往往受限于参数量和延迟，不能在推理阶段引入复杂模块。

4.2.3 检测模式与识别模式的差异

检测任务需要保留 2D 空间结构，因此 LCNetV4 检测模式输出：

P1: 1/4
P2: 1/8
P3: 1/16
P4: 1/32

这些多尺度特征送入 FPN 做融合。

识别任务需要保留横向文本序列，因此 LCNetV4 识别模式在 Stage 3 和 Stage 4 使用非对称 stride：

stride = (2, 1)

也就是高度方向下采样，宽度方向尽量不下采样。最后通过 height average pooling 得到：

[B, C, 1, W]

再转成序列送入识别 neck 和 CTC / NRTR head。

这个设计非常重要，因为它使同一个 backbone 可以同时服务：

检测：2D dense prediction
识别：1D sequence recognition
4.3 文本检测方法

PP-OCRv6 的检测模型结构是：

Input
 ↓
LCNetV4 Backbone
 ↓
RepLKFPN Neck
 ↓
DB Head
 ↓
PostProcess

训练阶段额外有 auxiliary DB heads，推理阶段删除。

4.3.1 RepLKFPN

PP-OCRv5_mobile_det 使用 RSEFPN，里面的 refinement block 主要是普通 3×3 conv。PP-OCRv6 改为 RepLKFPN，其 refinement block 使用 DilatedReparamBlock。

训练时这个 block 包含：

7×7 DWConv 主分支
5×5 DWConv dilation=1
3×3 DWConv dilation=2
3×3 DWConv dilation=3

推理时所有分支融合成单个 7×7 depthwise convolution。

论文中给出的参数和感受野对比是：

Neck	参数量	refinement block	局部感受野
RSEFPN	172K	3×3 Conv	3×3
RepLKFPN train	140K	DilatedReparamBlock	7×7
RepLKFPN infer	118K	7×7 DWConv	7×7

所以 RepLKFPN 的价值是：

参数更少
感受野更大
推理更简单
对大文本、密集文本、小文本上下文更友好

4.3.2 Auxiliary Deep Supervision

PP-OCRv6 在 FPN 的 P2、P3、P4 三个尺度上加辅助 DB Head。每个辅助 head 都预测：

shrink map
threshold map
binary map

训练时这些辅助输出也参与 loss：

P2 权重 0.4
P3 权重 0.3
P4 权重 0.2

推理时辅助 head 全部移除。

它的核心作用是让中间层特征也获得直接监督，改善深层网络训练时梯度传递不足的问题。对文本检测来说，这有助于：

提升小文本检测
减少漏检
增强多尺度特征学习
提升 FPN 中间层表达能力

4.3.3 Dice Loss + Focal Loss

PP-OCRv5 检测主要依赖 DiceLoss。DiceLoss 强调整体区域重叠，但对每个像素的困难程度区分不够。

PP-OCRv6 在 shrink map 和 binary map 上加入 Focal Loss：

L_shrink = L_Dice + L_Focal

Dice Loss 负责整体区域 overlap，Focal Loss 负责困难像素，尤其是：

边界像素
小文本像素
低对比度文本像素
容易与背景混淆的像素

论文的消融结果显示，在 PP-OCRv6_small 检测模型上，Focal Loss 带来了 +1.15% Hmean，是检测端收益较大的单项改进。

4.4 文本识别方法

PP-OCRv6 的识别模型结构是：

Text crop: 3 × 48 × W
 ↓
LCNetV4 Backbone
 ↓
EncoderWithLightSVTR Neck
 ↓
CTC Head
 ↓
CTC Decode

训练时还有 NRTR Head，推理时丢弃。

4.4.1 EncoderWithLightSVTR

PP-OCRv5 的 EncoderWithSVTR 使用 concat skip：

attention output 与 input concat
 ↓
1×1 projection

这个设计会导致通道数翻倍，参数较重。

PP-OCRv6 改成 EncoderWithLightSVTR：

输入特征 x
 ↓
1×1 Conv 降维
 ↓
1×7 DWConv 建模横向局部上下文
 ↓
Transformer Blocks 建模全局上下文
 ↓
与 1×1 skip projection 相加
 ↓
输出

这个设计背后的逻辑是：

OCR 识别是序列任务。
字符之间的局部关系很重要。
比如字符边界、字符间距、局部笔画结构。

因此，在全局 self-attention 之前先用 1×7 depthwise convolution 注入横向局部信息，会比纯全局注意力更适合文本识别。

4.4.2 CTC + NRTR 多头训练

PP-OCRv6 识别端训练时使用两个 head：

CTC Head：训练 + 推理
NRTR Head：只训练，不推理

NRTR Head 的作用类似辅助语言建模分支，它通过 cross-entropy 和 label smoothing 帮助共享 encoder 学到更好的上下文表示。

推理时只保留 CTC Head，因此：

推理速度快
可并行解码
不引入自回归生成开销
减少 VLM 式幻觉风险

这也是 PP-OCRv6 相比 VLM 更稳定的一个结构原因。

4.4.3 Tiny 识别模型的蒸馏

PP-OCRv6_tiny 为了极致轻量，识别模型去掉了 EncoderWithLightSVTR neck：

Backbone output
 ↓
Reshape
 ↓
FC projection
 ↓
CTC / NRTR heads

为了弥补容量下降，PP-OCRv6_tiny 使用 medium teacher 做 CTC KL 蒸馏：

L_tiny = L_CTC_GT + L_NRTR_GT + λ · KL(teacher CTC || student CTC)

一个关键细节是：tiny 使用较小字典，因此论文先训练了一个和 tiny 使用相同字典的 medium teacher，使 teacher 和 student 的 CTC logits 可以直接对齐，不需要额外 vocabulary mapping。

这个设计对工业落地很有参考价值：当端侧模型字典裁剪、类别减少时，teacher 也要保持同字典，否则蒸馏对齐会很麻烦。

5. 数据策略

PP-OCRv6 继承了 PP-OCRv5 的 data-centric 思路。论文提到 PP-OCRv5 的数据构建包括：

difficulty filtering
accuracy verification
diversity sampling

其中 difficulty filtering 使用 confidence 0.95–0.97 的“难度甜点区间”，diversity sampling 使用 1000 个 K-means clusters。PP-OCRv6 在这个基础上进一步扩展了工业场景数据，例如数码管、点阵字符、轮胎参数、电源适配器文字等。

从方法论上看，PP-OCRv6 的数据策略不是简单堆数据，而是强调：

难度适中
标注可靠
场景多样
工业长尾覆盖

这和你在荣耀做 OCR 数据校验非常相关。你现在用 GLM-OCR、dots.mOCR 等大模型做数据校验，本质上也是在做类似的数据中心化优化：用强 Teacher 提升训练数据质量，再反哺轻量 OCR 小模型。

6. 实验结果概要

需要注意一点：论文中的主要检测和识别结果来自 in-house benchmark，也就是百度内部构建的 OCR 评测集。因此这些结果具有参考价值，但如果用于公司内部技术选型，最好再结合自有业务数据复测。

6.1 文本检测结果

PP-OCRv6 在检测任务上明显优于 PP-OCRv5 和 VLM。

模型	平均检测 Hmean
PP-OCRv5_server	81.6%
PP-OCRv5_mobile	75.2%
PP-OCRv6_medium	86.2%
PP-OCRv6_small	84.1%
PP-OCRv6_tiny	80.6%
Gemini-3.1-Pro	46.8%
GPT-5.5	45.6%
Qwen3-VL-235B	38.3%

PP-OCRv6_medium 相比 PP-OCRv5_server 提升 +4.6%，small 版本也超过了 PP-OCRv5_server，tiny 版本接近 server baseline 且明显超过 PP-OCRv5_mobile。论文中特别指出，PP-OCRv6 在日文、古文、旋转文本、工业字符等场景上提升明显。

从这个结果可以看出：当前 VLM 在文本定位上仍然不适合替代专用 OCR 检测模型。VLM 可以读图、理解图，但要输出精确文本 polygon，仍然有明显短板。

6.2 文本识别结果
模型	识别加权平均准确率
PP-OCRv5_server	78.1%
PP-OCRv5_mobile	73.7%
PP-OCRv6_medium	83.2%
PP-OCRv6_small	81.3%
PP-OCRv6_tiny	73.5%
Qwen3-VL-235B	74.9%
Gemini-3.1-Pro	71.4%
GPT-5.5	64.2%

PP-OCRv6_medium 相比 PP-OCRv5_server 提升 +5.1%。论文指出，提升尤其明显的类别包括日文、古文、屏幕显示等。small 版本只有 5.2M 识别参数，却超过了 21M 参数左右的 PP-OCRv5_server_rec，说明 LCNetV4 和 LightSVTR 的结构效率较高。

6.3 幻觉评估

论文专门构建了 hallucination benchmark，评估模型是否会生成图像中不存在的文本。

模型	Hallucination Benchmark Accuracy
PP-OCRv6_medium	93.2%
PP-OCRv6_small	88.2%
PP-OCRv6_tiny	86.8%
Kimi-K2.6	85.0%
Qwen3-VL-235B	80.56%
GPT-5.5	78.0%
MiniMax-M3	72.6%

论文第 23 页的可视化例子显示，VLM 往往会把图像中的错误拼写、重复字符、非标准表达自动“纠正”，而 PP-OCRv6 更倾向于忠实输出图像内容。

这点对工业 OCR 非常关键。比如设备序列号、票据编号、银行卡号、药品批号、合同金额，这些内容本身不一定符合自然语言规律，如果模型用语言先验去“纠错”，反而会造成严重错误。

6.4 鲁棒性结果

PP-OCRv6 还做了两个鲁棒性实验。

第一是检测输入分辨率变化鲁棒性。PP-OCRv6_medium 在不同缩放比例下平均 Hmean 达到 86.67%，CV 为 5.19%，优于 PP-OCRv5_server 的 79.98% 和 8.02%。这说明 PP-OCRv6 对输入尺度变化更稳定。

第二是识别 crop margin 变化鲁棒性。PP-OCRv6_medium 在不同 crop 边界下识别一致性为 75.32%，相比 PP-OCRv5_server 的 54.82% 提升明显。这个结果说明 LCNetV4 和 LightSVTR 对检测框松紧变化更不敏感。

这在真实 OCR pipeline 中很重要，因为检测框不可能永远精准，识别模型必须能容忍一定的 crop 偏移、背景 padding 或边界裁切。

6.5 推理速度

端到端速度方面，PP-OCRv6_tiny 是最快的模型。在 Intel Xeon OpenVINO 后端上，PP-OCRv6_tiny 达到 0.20s/image，而 PP-OCRv5_mobile 是 0.78s/image，大约 3.9 倍加速。PP-OCRv6_medium 在 CPU OpenVINO 上也比 PP-OCRv5_server 快很多，论文报告为 1.40s/image 对 7.30s/image。

论文附录还单独比较了检测模型速度。在 2048 输入分辨率下，PP-OCRv6_medium GPU 推理为 106.89ms，而 PP-OCRv5_server 为 253.52ms；CPU ONNX 下 PP-OCRv6_medium 为 2327.23ms，PP-OCRv5_server 为 3034.93ms。

这说明 PP-OCRv6 的结构设计确实更利于 CPU / OpenVINO / ONNX 这类工业部署后端。

7. 消融实验分析

PP-OCRv6 的消融实验比较完整，能帮助我们理解每个模块到底有没有用。

7.1 Backbone 消融

在识别任务上，从 LCNetV3 baseline 逐步升级到 LCNetV4：

配置	准确率	增益
LCNetV3 baseline	76.01%	-
+ MetaFormer-style block	78.24%	+2.23%
+ RepDWConv	78.30%	+0.06%
+ Multi-branch StemBlock	78.96%	+0.66%
+ Per-tier depth/width config	79.21%	+0.25%
+ BN zero-init	79.35%	+0.14%

最大收益来自 MetaFormer-style token/channel mixer 解耦，说明 LCNetV4 的核心贡献不是简单加参数，而是 block 设计范式改变。RepDWConv 的精度收益只有 +0.06%，但它训练时增强表达、推理时零成本，更多是工程友好的训练增强。

7.2 检测消融

以 PP-OCRv5_mobile_det 为 baseline：

配置	Hmean	增益
Baseline	77.84%	-
+ LCNetV4 Backbone	79.37%	+1.53%
+ RepLKFPN	79.75%	+0.38%
+ Auxiliary Deep Supervision	80.28%	+0.53%
+ Focal Loss	81.43%	+1.15%
+ Full Data Training	84.12%	+2.69%

这里可以看到，检测提升主要来自三个方面：

LCNetV4 提升 backbone 表达能力
Focal Loss 增强困难像素监督
Full Data Training 体现数据规模和质量的重要性

RepLKFPN 增益不算最大，但它同时减少了参数、扩大了感受野，是一个性价比很高的结构替换。

7.3 识别消融

识别端消融显示：

配置	准确率	增益
EncoderWithSVTR	79.35%	-
LightSVTR additive skip	79.77%	+0.42%
LightSVTR + DWConv 1×7	80.24%	+0.89%
No neck	74.52%	-4.83%
CTC only	79.08%	-
CTC + NRTR-384	80.24%	+1.16%
CTC + NRTR-512	80.61%	+1.53%
Tiny + CTC KL distillation	74.52%	+2.69%

这个结果说明：

1. 识别 neck 很重要，直接去掉会明显掉点。
2. 1×7 DWConv 的局部横向建模有效。
3. CTC + NRTR 多头训练明显优于 CTC-only。
4. tiny 模型蒸馏收益很大。

8. 深度分析
8.1 PP-OCRv6 的本质：不是单点 trick，而是系统工程

PP-OCRv6 不是靠某一个模块大幅提升，而是典型的工业 OCR 系统工程：

Backbone 统一
Neck 轻量化
Loss 优化
多尺度监督
训练数据扩展
小模型蒸馏
多后端部署

它真正体现的是：

OCR 小模型的上限不是只由参数量决定，而是由架构设计、训练策略、数据质量、部署约束共同决定。

这一点对企业内部 OCR 模型非常重要。很多时候模型不需要盲目换成 VLM，而是应该围绕业务场景做数据闭环、难例挖掘和轻量模型结构优化。

8.2 为什么 PP-OCRv6 能比 VLM 更适合 OCR？

VLM 的优势是通用理解，但 OCR 需要的是非常细粒度的视觉对齐。

OCR 任务本质上包括：

像素级文本区域定位
字符级视觉区分
序列级文本解码
版面级结构保持

VLM 的语言先验很强，但 OCR 的很多内容并不遵循自然语言规律，比如：

SN: X7A0O1
195/65B15
0.0
eeeeeeeeeeee
点阵字符
工业批号
银行卡号
错别字
非标准拼写

这些内容如果用语言先验去“理解”，反而容易出错。PP-OCRv6 采用 CTC 解码和专用检测结构，更接近“看见什么输出什么”，因此更适合严肃 OCR 任务。

8.3 为什么 LCNetV4 是合理的？

LCNetV4 的合理性在于它抓住了轻量 CNN 的两个核心矛盾：

要有足够空间建模能力
又不能增加太多计算量

MetaFormer-style 的 token mixer / channel mixer 解耦，使空间建模和通道建模分别优化。RepDWConv 训练多分支、推理单分支，让模型训练时更强，部署时仍然简单。

这比单纯堆 MobileNet block 更适合 OCR，因为 OCR 对局部纹理、字符边界、细粒度结构非常敏感。

8.4 为什么 LightSVTR 对识别有效？

文本识别不是普通图像分类。它有很强的横向序列结构：

字符之间有顺序
相邻字符边界重要
字符宽度变化明显
局部笔画和上下文共同决定结果

LightSVTR 先用 1×7 DWConv 建模局部横向邻域，再用 Transformer 建模全局关系，相当于：

先看局部字符结构
再看整行上下文

这比直接做 global attention 更符合文本识别的任务特性。

8.5 PP-OCRv6 的局限性

虽然 PP-OCRv6 表现很好，但从技术调研角度也要看到它的局限。

第一，主要结果来自百度内部 benchmark。内部数据是否覆盖荣耀业务场景，例如手机相册、系统截图、拍照翻译、文档扫描、票据、商品包装、屏幕反光、低光拍摄等，需要重新验证。

第二，PP-OCRv6 仍然是两阶段 pipeline。检测错误会传递到识别，识别对 crop 质量仍然有依赖。虽然论文做了 margin robustness，但真实业务中的框偏移、漏检、重叠文本、弯曲文本仍然需要专项验证。

第三，VLM 对比需要谨慎。VLM 是否经过专门 OCR prompt、是否支持坐标输出、是否针对检测任务调优，都会影响结果。因此可以说 PP-OCRv6 在论文设定下显著优于 VLM，但不能简单泛化为所有 VLM OCR 场景都更差。

第四，PP-OCRv6 的很多提升依赖高质量数据。对于企业内部落地，真正难点往往不是复现结构，而是构建同等质量和覆盖度的数据集。

9. 对你在荣耀 OCR 实习工作的启发

你现在做的是 OCR 数据校验，用 GLM-OCR、dots.mOCR 等大模型对训练数据和回流数据进行校验。PP-OCRv6 对你的工作有很强启发。

9.1 大模型不一定直接部署，但很适合做 Teacher

PP-OCRv6 的路线说明：

VLM / 大 OCR 模型：
适合做数据校验、伪标签生成、难例审核、质量评估。

轻量 OCR 小模型：
适合最终部署、实时推理、端侧运行。

所以你可以把自己的实习工作抽象成：

利用大模型 Teacher 提升训练数据质量，从而提升轻量 OCR Student 在工业场景中的效果。

这和 PP-OCRv6 的“高质量数据 + 小模型部署”思想高度一致。

9.2 可以设计一个类似 PP-OCRv6 的数据质量闭环

你可以把数据校验体系拆成四个维度：

1. 准确性：
   文本是否错标、漏标、多标，检测框是否偏移。

2. 难度：
   模糊、反光、低光、小字、旋转、弯曲、遮挡、复杂背景。

3. 多样性：
   语言、字体、颜色、拍摄角度、设备类型、业务场景。

4. 业务价值：
   手机截图、相册图片、票据、文档、包装、铭牌、屏幕显示。

然后用多 Teacher 做一致性判断：

原始标注
GLM-OCR 输出
dots.mOCR 输出
内部 OCR baseline 输出
规则校验结果

对每个样本计算质量分：

字符级编辑距离
检测框 IoU / Polygon IoU
文本合法性规则
语言类型一致性
置信度分布
版面结构一致性
多模型投票一致性

最后将数据分为：

高质量样本：直接入训
疑似错标样本：人工复核
高价值难例：进入 hard case pool
低质量噪声：剔除或降权
未标注高置信样本：生成伪标签入训

这套思路可以很好地和 PP-OCRv6 的 data-centric 方法对应起来。