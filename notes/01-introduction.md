# 01. Introduction：给 AI 助手一套端侧记忆

> 课程：Building AI Assistants with On-Device Memory  
> 讲师：Dylan Couzon（Qdrant Developer Experience Engineer）  
> 本课性质：课程导论与学习路线说明，不包含独立编程实验  
> 主要材料：[01 transcript](../transcripts/01-introduction.md) · [课程代码说明](../code/lessons/README.md) · [DeepLearning.AI 官方课程页](https://www.deeplearning.ai/courses/building-ai-assistants-with-on-device-memory)

## 1. 这门课要解决什么问题？

一般的 AI 应用往往依赖云端保存和处理用户信息：设备需要联网，个人数据也必须发送到远程服务器。本课程要构建的是另一种方案——让 AI 助手在设备本地形成、保存和检索自己的“经历”。

完成课程后，助手能够：

- 记住文字笔记、照片和语音记录；
- 不依赖关键词，而是按照**语义（meaning）**找回相关记忆；
- 使用类别、价格、位置、时间等 **metadata** 缩小检索范围；
- 按要求明确删除记忆，即“遗忘”；
- 从少量照片中记住新物体，并在新视角下尝试识别它；
- 在模型与数据准备完成后离线运行，使私人记忆留在本地。

官方课程页将课程概括为约 1 小时 4 分钟的短课，适合希望应用能离线工作、并将用户数据保留在设备上的 AI 开发者；动手实验建议具备 Python 基础。

## 2. 为什么需要 on-device memory？

### 个性化需要跨会话记忆

如果助手只能处理当前输入，它每次交互都像第一次见到用户。持续记忆可以让它利用过去的事件、偏好、物体和位置，提供真正与用户相关的结果。

### 端侧存储的主要价值

| 价值 | 含义 |
| --- | --- |
| 隐私 | 笔记、照片、录音等私人经历不必上传到远程服务器 |
| 离线可用 | 召回不依赖持续网络连接，适合机器人、智能眼镜和野外设备 |
| 低网络依赖 | 避免网络往返带来的延迟和连接失败 |
| 本地控制 | 应用可以直接添加、更新、筛选和删除记忆 |

端侧方案并不意味着云端永远无用。课程的核心实验只在本地运行，但附录还展示了端侧记忆与云端同步的可选混合方案。选择本地、云端或二者结合，应由隐私、算力、存储和共享需求共同决定。

## 3. 核心心智模型：记忆循环

课程围绕同一个循环展开：

```text
Capture → Encode → Store → Recall → Forget
 捕获       编码      存储      召回       遗忘
```

| 阶段 | 系统做什么 | 关键产物 |
| --- | --- | --- |
| Capture | 从文本、摄像头或麦克风获得一次经历 | 原始笔记、图片或音频 |
| Encode | 用相应模型把内容转换成 embedding | 表示语义或视觉特征的向量 |
| Store | 将向量与可读内容、时间、位置、名称等信息一起保存 | vector + metadata/payload |
| Recall | 将问题用兼容的模型编码，并搜索最相近的记忆 | 按相似度排序的候选结果 |
| Forget | 删除不再需要的向量及其 metadata | 记忆不再参与后续检索 |

这里的“记忆”不是一段塞进 prompt 的聊天记录，而是可以持久化、查询和删除的数据对象。后续实验使用 Qdrant Edge 将每条记忆保存为一个 **point**：它包含 ID、一个或多个向量，以及相应的 metadata（在 Qdrant 中也称 payload）。

## 4. 咖啡杯示例：一次经历如何变成记忆

导论用“这是我的咖啡杯”说明完整流程：

1. 摄像头捕获咖啡杯的图像。
2. 图像编码模型把画面转换为 embedding。
3. 系统保存该向量，并附上“Dylan 的咖啡杯”、位置和时间等 metadata。
4. 设备以后再次看到咖啡杯时，把新画面编码为同一向量空间中的向量。
5. 本地向量搜索返回最相似的已知视觉记忆。
6. 只有当相似度达到经过校准的阈值时，系统才把候选结果当作一次真正的“识别”。
7. 新的目击记录又可以带着新的位置和时间存入记忆。

这个例子展示了课程最重要的设计思想：系统学习新对象时，改变的是本地记忆中的向量和 metadata，而不是重新训练模型参数。

## 5. 多模态内容如何进入同一套记忆系统？

不同模态需要不同的编码入口，但最终都变成可以检索的向量及上下文信息。

| 输入 | 处理方式 | 后续如何检索 |
| --- | --- | --- |
| 文本笔记 | 文本 embedding 模型直接编码 | 用问题的文本向量进行语义搜索 |
| 语音记录 | 本地 Whisper 先转成文字，再生成文本向量 | 与普通文字笔记共用文本检索流程 |
| 照片 | CLIP image encoder 生成图像向量 | 可用图像查图像，也可用自然语言描述查图片 |

CLIP 包含兼容的文本与图像编码器。它能让“一张猫的图片”和“a cat”这样的文字描述落入可比较的表示空间，因此用户不需要准备查询图片，也可以直接用自然语言搜索照片。

课程自带的架构图强调：Qdrant Edge、EdgeStorage 和 Local API 可以随应用一起运行在不同设备上，本地设备共享的是同一种记忆架构，而不是必须使用同一种硬件。

![Qdrant Edge 端侧记忆架构：不同设备各自运行本地存储和 API，并可选择与上层服务连接](../code/lessons/assets/qdrant-edge-scheme.png)

## 6. Retrieval 不等于 Recognition

这是后续物体学习实验的核心区别。

### Retrieval：返回“最接近的候选”

最近邻检索总能从已有记忆中选出一个最相似的结果。即使系统从未见过橡皮鸭，它仍可能返回自行车，因为自行车只是当前记忆库里“相对最接近”的对象。

### Recognition：判断“候选是否足够像”

要把检索结果变成“认识 / 不认识”的判断，还要设置 **similarity threshold**：

```text
最高相似度 ≥ threshold  →  接受为识别结果
最高相似度 < threshold  →  标记为未知对象
```

阈值不能只凭感觉设置。课程后续会比较已知匹配与非匹配样本的得分分布，再根据业务中 false positive 和 false negative 的代价选择阈值。换言之，向量检索负责生成候选，阈值负责作出是否接受候选的决策。

## 7. “不需要 LLM”与“不需要重新训练”到底指什么？

- **不需要 LLM：** 本课程中的核心问答是 retrieval。系统把问题编码后，返回已存储的原始笔记、转写或照片；它不依赖大语言模型生成答案。
- **不需要重新训练：** 教会助手一个新物体时，只需保存几张样例照片的 embedding 和标签，不更新视觉模型的权重。
- **不代表没有模型：** 系统仍然使用文本 embedding、CLIP 和本地语音转写等预训练模型。
- **不代表任何候选都可信：** 最近邻始终会返回结果，因此真实识别还需要阈值、样本覆盖和误差权衡。

这套方法的优势是新增知识速度快、成本低，而且可以完全在本地完成；它的能力边界则取决于 embedding 的表示质量、示例是否覆盖足够多的视角，以及阈值是否适合当前场景。

## 8. 导论中的承诺会如何在后续课程落地？

> 以下是根据后续 transcript 和实际 notebook 整理的学习导航，不是 01 中已经展开的操作细节。

| 课程材料 | 实际要做的事 | 对应导论概念 |
| --- | --- | --- |
| [02 Overview](../transcripts/02-overview-of-on-device-ai-memory.md) | 用文本、图片和机器人示例理解语义检索 | embedding、CLIP、on-device 的价值 |
| [03 Inside the Device](../transcripts/03-inside-the-ai-assistant-device.md) | 理解摄像头、计算设备、存储和手机界面的关系 | 捕获到本地记忆的完整路径 |
| [L3 notebook](../code/lessons/L3/Lesson3.ipynb) | 创建 shard；存储与检索文本；按类别和价格过滤；用文字搜图；删除记忆 | Store、Recall、Forget |
| [L4 notebook](../code/lessons/L4/Lesson4.ipynb) | 本地转写语音，把文本、语音和照片合并进一天的记忆 | 多模态离线助手 |
| [L5 notebook](../code/lessons/L5/Lesson5.ipynb) | 用少量图片教会新对象；校准阈值；加入时间新鲜度排序 | 学习、识别与 soft forgetting |
| [07 Running Locally](../transcripts/07-running-on-a-local-computer.md) | 在 macOS、Windows 或 Linux 电脑上运行完整应用 | 同一记忆架构适配不同设备 |
| [Appendix notebook](../code/lessons/Appendix/Appendix.ipynb) | 将端侧记忆推送到云端，再同步到另一台设备 | 可选的 local + cloud 混合架构 |

课程代码使用的主要本地组件包括：Qdrant Edge（向量记忆层）、FastEmbed（文本与 CLIP embedding）、ONNX Whisper（语音转写）和 ONNX Runtime（本地模型运行时）。首次准备模型可能需要下载；模型就绪后，核心实验不需要 API key，也可离线执行。

## 9. 关键词速查

| 术语 | 本课中的含义 |
| --- | --- |
| Embedding | 用一组数字表示文本语义或图像特征的向量 |
| Vector search | 在大量向量中查找与查询向量最接近的结果 |
| Semantic retrieval | 按含义而不是仅按字面关键词召回内容 |
| Metadata / Payload | 与向量一起保存的类别、时间、位置、价格、名称等结构化信息 |
| Multimodal memory | 能统一管理文本、语音和图像经历的记忆系统 |
| Similarity score | 查询向量与候选记忆的接近程度 |
| Similarity threshold | 将最近邻候选转成“匹配 / 不匹配”决策的最低分数 |
| Hard forgetting | 明确删除某条记忆，使它不再出现在任何查询中 |
| Soft forgetting | 不删除记忆，而是在排序时降低旧记忆的权重 |

## 10. 易混淆点

1. **向量不是原始内容。** 它适合相似度计算；实际应用通常还要保留可展示的文本、图片路径及 metadata。
2. **语义相近不代表满足硬约束。** “适合吃饭的地方”可以靠相似度排序，但“价格低于 15 美元”应使用 metadata filter。
3. **召回第一名不代表识别正确。** 记忆库中没有正确对象时，最近邻仍然存在。
4. **端侧不等于特定机器人。** 机器人只是示例，同一循环也可用于个人电脑、Raspberry Pi、智能眼镜或其他现场设备。
5. **记住新对象不等于训练新模型。** 本课程通过添加样本向量扩展记忆，而不是微调或重新训练 CLIP。

## 11. 本课小结

- AI 助手的持久记忆可以抽象为 `Capture → Encode → Store → Recall → Forget`。
- embedding 让文本、图像等非结构化经历变成可搜索的向量；metadata 补充时间、位置和类别等上下文。
- 本地向量记忆兼顾隐私、离线能力和较低的网络依赖。
- 多模态助手可以把文本、语音转写和照片组织进同一套检索逻辑。
- 最近邻检索只负责返回候选；可靠识别还需要经过验证的 similarity threshold。
- 新知识可以通过增加记忆样本获得，不一定需要 LLM，也不一定需要重新训练模型。

## 12. 复习问题

1. 为什么只有原始照片或录音文件，还不能称为“可检索的记忆系统”？
2. embedding 和 metadata 分别解决什么问题？为什么两者不能互相替代？
3. 用自然语言搜索照片时，CLIP 的两个 encoder 分别处理什么输入？
4. 为什么最近邻结果不能直接当作对象识别结果？
5. “从两个样例学会一个新物体”为什么不等同于重新训练视觉模型？
6. 哪些场景特别适合 on-device memory？哪些场景可能仍然需要云端？
7. 如果旧的门锁密码与新的门锁密码语义同样接近问题，应如何让新记忆优先？

## 资料来源

- [DeepLearning.AI：Building AI Assistants with On-Device Memory](https://www.deeplearning.ai/courses/building-ai-assistants-with-on-device-memory)
- [课程导论 transcript](../transcripts/01-introduction.md)
- [课程清单与来源链接](../manifest.json)
- [本地代码说明与运行要求](../code/lessons/README.md)
- 课程 notebooks：[L3](../code/lessons/L3/Lesson3.ipynb) · [L4](../code/lessons/L4/Lesson4.ipynb) · [L5](../code/lessons/L5/Lesson5.ipynb) · [Appendix](../code/lessons/Appendix/Appendix.ipynb)

