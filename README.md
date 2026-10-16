# Building AI Assistants with On-Device Memory 学习笔记

这是 DeepLearning.AI 短课程 [Building AI Assistants with On-Device Memory](https://www.deeplearning.ai/courses/building-ai-assistants-with-on-device-memory) 的个人学习归档，包含中文笔记、课程 transcript、实验 notebook 和配套学习素材。

课程由 DeepLearning.AI 与 [Qdrant](https://qdrant.tech) 合作推出，讲师为 Dylan Couzon。课程围绕 Qdrant Edge 构建端侧 AI 记忆：在本地保存和检索文本、语音转写与图片，使用少量样例教学新对象，并将这些能力接入实时摄像头应用。

## 中文笔记

| 课程 | 笔记 |
| --- | --- |
| 01 Introduction | [课程导论与完整学习路线](notes/01-introduction.md) |
| 02 Overview of On-Device AI Memory | [端侧 AI 记忆概览](notes/02-overview-of-on-device-ai-memory.md) |
| 03 Inside the AI Assistant Device | [AI 助手设备内部架构](notes/03-inside-the-ai-assistant-device.md) |
| 04 Building the Memory | [构建本地向量记忆](notes/04-building-the-memory.md) |
| 05 Building the Voice Assistant | [构建本地多模态语音助手](notes/05-building-the-voice-assistant.md) |
| 06 Learning Through Memory | [通过记忆教学新对象](notes/06-learning-through-memory.md) |
| 07 Running on a Local Computer | [在个人电脑上运行实时助手](notes/07-running-on-a-local-computer.md) |
| 08 Conclusion | [全课程知识地图与实践路线](notes/08-conclusion.md) |

## 仓库结构

```text
.
├── notes/          # 基于课程材料整理的中文学习笔记
├── transcripts/    # 课程视频文字稿及原始课程链接
├── code/lessons/   # L3–L5、Appendix notebooks 与配套素材
└── manifest.json   # 课程项目、来源链接与本地文件索引
```

课程代码的环境要求、目录说明和运行方式见 [`code/lessons/README.md`](code/lessons/README.md)。建议使用 Python 3.12 或更新版本，并在独立虚拟环境中安装固定依赖。首次执行会下载本地 embedding、语音转写等模型。

## 学习主线

```text
Capture → Encode → Store → Retrieve → Recognize/Recall → Update/Forget
```

- Nomic 将文字和语音 transcript 编码为文本向量；
- CLIP 将图片和图片描述映射到跨模态向量空间；
- Qdrant Edge 在设备本地保存 named vectors 与 payload；
- Metadata filter、freshness reranking 与 similarity threshold 将近邻候选转化为可用结果；
- 多视角样例让助手立即认识用户教学的对象，不需要重新训练模型。

## 资料来源与版权说明

本仓库是非官方个人学习归档，与 DeepLearning.AI、Qdrant、课程讲师及其他内容权利人不存在隶属、赞助或背书关系。

- 课程名称、视频内容、transcripts、notebooks、示例代码和原始教学材料的相关权利归各自权利人所有。
- 本仓库不对第三方课程材料授予或暗示任何新的许可；使用者应遵守原课程平台条款及各材料自身的许可条件。
- 中文笔记是基于课程内容整理的个人学习记录，其中引用的课程概念、示例和名称仍归原权利人所有。
- 示例图片分别采用 CC0、Public Domain、CC BY 或 CC BY-SA 等许可；具体作者、来源和许可见各素材目录中的 `CREDITS.json`。
- 请优先通过[官方课程页面](https://www.deeplearning.ai/courses/building-ai-assistants-with-on-device-memory)学习并获取最新材料。

本仓库仅用于学习、研究与技术交流，不提供任何担保，也不替代官方课程材料。如果你是相关权利人，并认为仓库中的内容需要修改或移除，请通过本仓库的 GitHub Issue 联系维护者。
