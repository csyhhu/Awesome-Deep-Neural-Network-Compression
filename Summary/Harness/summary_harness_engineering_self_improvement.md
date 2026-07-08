# Harness Engineering for Self-Improvement

- **原文链接**: [https://lilianweng.github.io/posts/2026-07-04-harness/](https://lilianweng.github.io/posts/2026-07-04-harness/)
- **作者**: Lilian Weng
- **发布时间**: 2026年7月4日
- **关键词**: Harness 工程、递归自我改进 (RSI)、智能体设计模式、上下文工程、进化搜索、自我改进 Harness

---

## 1. 概述

这篇博客系统性地阐述了 **Harness（外壳/框架）工程** 如何成为通往 AI **递归自我改进 (Recursive Self-Improvement, RSI)** 的一条近路。

核心论点：在基础模型能力不断增长的当下，Harness — 连接模型与真实世界之间的中间层 — 决定了模型能多好地思考、规划、调用工具、管理上下文和评估结果。与其等待模型本身进化到能自我改写权重的地步，不如先从 **Harness 层入手**，让系统能够优化自身的运行机制。

博客将 Harness 定义为超越早期「智能体 = LLM + 记忆 + 工具 + 规划 + 行动」框架的更丰富系统层，额外包含：工作流设计、评估、权限控制、持久状态管理等。

---

## 2. Harness 设计模式

### 2.1 工作流自动化

构建模型能在其中**操作、测试和迭代**的工作流环境。核心循环为：

> **计划 → 执行 → 观察/测试 → 改进 → 再次执行**

典型案例是 Karpathy 的 `autoresearch` 仓库。关键在于模型要能分析自身的**轨迹和失败案例**，通过"智能体运行时"迭代改进，而非简单的静态提示模板。

### 2.2 文件系统作为持久记忆

长周期智能体系统中，不应将所有内容保留在上下文窗口中，而应使用**文件系统存储持久状态**。产出物（实验日志、代码差异、论文摘要、错误跟踪等）通常远超上下文窗口长度。

LLM 通过 `bash` 等命令读写文件系统是一项基础能力，便于利用模型能力的持续提升。

### 2.3 子智能体与后台任务

Harness 可生成多个**子智能体**并行执行并监控后台任务。主智能体需要一个小型进程管理器：启动任务、检查日志、取消失败运行、将结果合并回主线程。

**关键设计原则**：并行性必须**显式可检查**。子智能体输出应保存为文件、日志、状态记录，使模型能恢复中断并从执行历史中推理。

### 2.4 案例研究：编码智能体 Harness

主流编码智能体（Claude Code、Codex、OpenCode、Cursor）的核心接口已趋稳定，使用类似循环。典型工具集包括：

| 类别 | 工具 |
|------|------|
| 文件系统 | `glob`, `grep`, `ls`, `read`, `write`, `edit`, `apply_patch` |
| Shell 执行 | `bash`, `PowerShell` |
| IO | LSP, `git_status`, `git_diff`, `git_commit` |
| 外部上下文 | MCP 工具, Skills |
| 网络搜索 | `web_search`, `web_fetch`, 浏览器工具 |
| 制品 | 阅读文档/图片, 生成 HTML/图片 |
| 后台进程 | `CronCreate`, `CronDelete` |
| 智能体委派 | `spawn_agent`, `resume_agent`, `wait_agent`, `list_agents` |

### 2.5 Harness 层 vs 核心智能

预测的 RSI 演进路径：
1. 先由 Harness 工程向**元方法论**演进（改进"获得更好答案的机制"本身）
2. 成熟后 Harness 支持模型自我改进循环
3. 更智能的模型反过来**防止 Harness 过度工程化**

未来许多 Harness 的改进可能被**内部化**到模型行为中，但与外部上下文和工具的接口将保留（类比提示工程技巧被微调取代，但目标、约束、评估的指定永不过时）。

---

## 3. Harness 优化方法

**优化对象的演进层次**：

> 指令提示 → 结构化上下文 → 工作流 → Harness 代码 → 优化器代码

### 3.1 上下文工程 (Context Engineering)

#### ACE (Agentic Context Engineering) - Zhang et al. 2025
- 将上下文视为不断演进的"**操作手册**"
- 三个维护组件：**生成器、反思器、策展器**
- 上下文被结构化为 (标识符, 描述) 条目清单，策展器不重写整个提示，而是输出结构化条目并进行去重和周期性精炼

#### MCE (Meta Context Engineering) - Ye et al. 2026
- 分离"如何管理上下文"（机制）和"上下文内容"
- **两层架构**：元级别技能进化 + 基级别上下文优化
- 技能定义上下文函数，包含静态组件和动态运算符
- 元级别智能体通过**交叉 (crossover)** 创造新技能

#### Meta-Harness - Lee et al. 2026
- 优化对象是决定**存储、检索和呈现信息**的代码本身
- 外部循环优化 Harness，内部智能体生成候选
- 使用编码智能体在文件系统上操作，仅在合格时保留到帕累托前沿

### 3.2 工作流设计

| 工作 | 核心特点 |
|------|----------|
| **AI Scientist** (Lu et al. 2026) | 自动化研究全流程：提出想法、编写代码、实验、手稿、同行评审 |
| **ScientistOne** (Meng et al. 2026) | 以可验证性为核心，每条声明必须追溯到证据来源并接受证据链审计 |
| **Autodata** (Kulikov et al. 2026) | 管理挑战者/弱解算器/强解算器/验证者，合成"恰好难度"的数据 |
| **ADAS** (Hu et al. 2025) | 将智能体设计形式化为优化问题，元智能体搜索并编程新的智能体工作流 |
| **AFlow** (Zhang et al. 2025) | 智能体工作流表示为图（节点=LLM 调用，边=逻辑操作），MCTS 搜索 |

### 3.3 自我改进 Harness (Self-Improving Harness)

#### STOP (Self-Taught Optimizer) - Zelikman et al. 2023
- **递归改善"改进器"**，而非直接改进解决方案
- 定义元效用为下游任务平均效用：$I_t = I_{t-1}(û, I_{t-1}; M)$
- 自我发现的策略包括：遗传算法、分解改进、多臂老虎机、模拟退火
- **关键发现**：递归结构仅在基础模型足够强时有效（GPT-4 有效，GPT-3.5/Mixtral 失败）

#### Self-Harness - Zhang et al. 2026
- **三阶段循环**：
  1. **弱点挖掘**：聚类失败为可验证模式
  2. **Harness 提案**：基于有限可编辑表面提出针对性编辑
  3. **提案验证**：在留入/留出数据上回归测试，仅接受无回归的编辑
- 在不同模型上学习到特定于模型的 Harness 指令
- **警示**：可编辑表面必须仔细设计，权限和安全层应置于循环之外

### 3.4 进化搜索 (Evolutionary Search)

适用于搜索空间庞大/非凸、评估简单但梯度困难的场景：

| 方法 | 核心创新 |
|------|----------|
| **Promptbreeder** (Fernando et al. 2023) | 进化任务特定提示，连*变异提示*自身也进化 |
| **GEPA** (Agrawal et al. 2025) | 结合反思提示与进化搜索 |
| **AlphaEvolve** (Novikov et al. 2025) | 编码智能体进化系统，维护候选代码池，diff 改进，元提示协同进化 |
| **ThetaEvolve** (Wang et al. 2025) | 结合进化搜索、RL 和上下文学习 |
| **ShinkaEvolve** (Lange et al. 2025) | 改进采样效率（父代采样平衡、代码新颖性拒绝采样、元草稿板模式识别） |
| **Darwin Gödel Machine** (Zhang et al. 2025) | 让编码智能体修改**自身 Harness** 的代码仓库 |
| **Hyperagents** (Zhang et al. 2026) | 元智能体控制如何修改已有任务智能体创造新智能体 |

### 3.5 与模型权重的联合优化

**SIA** (Hebbar et al. 2026)：同时允许更新 Harness 和模型权重，包含元智能体、任务特定智能体、回馈智能体（决定下一轮更新 Harness 还是权重）。方向有趣但实验存在混杂因素，证据尚不充分。

---

## 4. 未来挑战

### 4.1 弱且模糊的评估器
研究品味、新颖性、长期科学价值难以量化，很多真实任务缺少快速精确的验证器。

### 4.2 上下文和记忆生命周期
随着智能体自主性提高，记忆持续增长，上下文工程必须成为智能的核心部分，而不仅仅是软件系统层。

### 4.3 负面结果
人类文献偏向成功，模型可能不擅长放弃假设或报告失败。Harness 应让失败尝试易于保存，以便从失败中剪枝搜索空间。

### 4.4 多样性崩溃
进化和 RL 循环趋向利用已知高奖励模式。需要探索机制防止种群坍缩为单一解，对开放式研究至关重要。

### 4.5 奖励黑客 (Reward Hacking)
自我改进循环会优化任何给予的信号。如果奖励来自单元测试、评委模型或基准分数，可能学到投机取巧行为。评估器和权限控制应置于循环之外，通过留出测试、轨迹审计和人工审查来监督。

### 4.6 长期成功
当前优化目标偏短期（如代码任务完成），难捕获可维护性、向后兼容性、迁移成本等长期仓库健康度量。

### 4.7 人类的角色
人类应**向上提升**而非被移除出循环，在正确的抽象层次和时间点提供监督。

---

## 5. 相关基准测试

| 基准 | 内容 | 关键结果 |
|------|------|----------|
| **PaperBench** | 复现 20 篇 ICML 2024 Spotlight/Oral | Claude 3.5 Sonnet 仅 21% |
| **CORE-Bench** | 270 个计算可复现性任务 | 最佳准确率 21% |
| **ScienceAgentBench** | 102 个数据驱动科学发现任务 | 涵盖数学/化学/生物/地理 |
| **RE-Bench** | 7 个真实 ML 研究工程环境 | AI 短时(2h)是人的 4 倍，长时间人反超 |
| **MLE-bench** | 75 个 Kaggle 竞赛离线任务 | o1-preview + AIDE 达铜牌 16.9% |
| **KernelBench** | 250 个 PyTorch GPU 内核生成 | 衡量正确性和速度 |

---

## 6. 总结

这篇博客系统性地梳理了 Harness 工程在 AI 自我改进中的核心地位。核心观点可以概括为：

1. **Harness 是通往 RSI 的近路**：与其等待模型进化到能自我改写权重，不如先构建能优化自身运行机制的系统框架

2. **设计原则**：简单、通用、可检查、可持久化 — 以文件系统为持久记忆，子智能体显式并行，输出可审计

3. **优化层次递进**：从提示词到上下文到工作流到 Harness 代码到优化器代码，每一步都在抽象层次上升级

4. **自我改进的关键瓶颈**：评估器的质量、负面信息的利用、多样性维持、奖励黑客防御 — 这些是 Harness 工程当前最核心的未解决问题

5. **人类不可替代**：人类应向上移动到更高层次的监督角色，而非被完全移除出循环

整篇博客将 2023-2026 年间的大量相关工作编织成一个连贯的叙事，展示了从手工设计 Harness 到 Harness 自我进化的演进图景。

---

## 7. 关键参考文献

- Good (1965) - 超智能机器的原始构想
- Yudkowsky (2008) - 递归自我改进
- Zhang et al. (2025) - ACE: Agentic Context Engineering
- Ye et al. (2026) - MCE: Meta Context Engineering
- Lee et al. (2026) - Meta-Harness
- Lu et al. (2026) - AI Scientist
- Zelikman et al. (2023) - STOP: Self-Taught Optimizer
- Zhang et al. (2026) - Self-Harness
- Novikov et al. (2025) - AlphaEvolve
- Zhang et al. (2025) - Darwin Gödel Machine (DGM)
- Hebbar et al. (2026) - SIA: Simultaneous Improvement of Agent

---

## 8. Q&A

### Q1: 本文讨论的方法，会在单次样本 eval 期间修改 harness 吗？还是只在训练/优化阶段修改，优化结束后 harness 就固定了？

**简短回答**：绝大多数方法都是**离线优化 harness，优化完成后 harness 在推理时固定不变**。不存在单次 eval 期间动态修改 harness 自身代码/结构的机制。但有几个方法存在"在线"成分，需要区分清楚。

---

#### 分类分析

| 方法 | 优化阶段 | 推理时 Harness 是否固定 | 说明 |
|------|:---:|:---:|------|
| **STOP** | 离线元训练 | ✅ 固定 | 递归改进"改进器"，但改进器本身在优化完成后就不再变化，用于推理时作为固定的 scaffold |
| **Self-Harness** | 离线三阶段循环 | ✅ 固定 | 弱点挖掘→提案→验证是离线流程，通过验证的编辑才被接受并固化为稳定的 Harness 版本 |
| **Darwin Gödel Machine** | 离线进化搜索 | ✅ 固定 | 编码智能体在进化阶段修改自身 Harness 代码仓库，但搜索结束后选出的最优 Harness 在推理时固定 |
| **AlphaEvolve** | 离线进化搜索 | ✅ 固定 | 候选代码池进化，最终选出最优代码作为固定 Harness |
| **Promptbreeder** | 离线进化搜索 | ✅ 固定 | 进化出最优提示后，推理时使用该固定提示 |
| **ADAS** | 离线搜索+编程 | ✅ 固定 | 元智能体搜索并生成智能体工作流代码，之后归档为固定程序 |
| **Meta-Harness** | 离线优化 | ✅ 固定 | 外部循环优化 Harness 代码，内部智能体生成候选，帕累托前沿选出最优后固定 |
| **ACE** | **在线维护** | ❌ 部分动态 | 上下文被视为不断演进的"操作手册"，生成器/反思器/策展器**持续运行**来更新上下文条目。但 Harness 的**结构代码**不变，变的是**上下文内容** |
| **MCE** | 离线元优化 + 在线基优化 | ❌ 部分动态 | 元级别技能进化是离线的，但基级别上下文可根据反馈在线学习更新 |

---

#### 关键区分：Harness 代码 vs Harness 管理的内容

理解这个问题的关键在于区分两个层面：

```
Harness 的"骨架"（代码/结构）：      → 离线优化，推理时固定
Harness 的"血肉"（上下文/提示/记忆）： → 可能在推理时被在线更新
```

| 层面 | 何时修改 | 代表方法 |
|------|----------|----------|
| **Harness 代码/结构**（工作流定义、工具接口、评估逻辑、控制流） | 离线优化，推理时**固定** | STOP, Self-Harness, DGM, AlphaEvolve, ADAS, Meta-Harness |
| **上下文/提示内容**（系统提示、few-shot 示例、记忆条目） | 可能**在线更新** | ACE（通过策展器持续精炼）、MCE（基级别在线学习） |

---

#### 为什么推理时不修改 Harness 代码？

1. **安全与可控性**：Self-Harness 论文明确警示——"可编辑表面必须仔细设计，权限和安全层应置于循环之外"。如果允许推理时修改 Harness，奖励黑客风险将不可控。

2. **稳定性与可复现性**：如果每次推理都修改 Harness，行为将无法调试和审计。离线优化 + 固定部署是当前工程实践的基本范式。

3. **评估信号稀疏**：单次 eval 不足以提供修改 Harness 所需的足够反馈信号。Harness 优化需要**聚合大量轨迹**才能识别模式（如 Self-Harness 的"弱点聚类"步骤）。

4. **进化搜索的成本**：Darwin Gödel Machine 这类方法需要维护候选种群、交叉变异、多轮评估，这些操作在单样本推理的时间尺度上不可行。

---

#### 一个灰色地带：SIA

SIA (Hebbar et al. 2026) 同时允许更新 Harness 和模型权重，由"回馈智能体"**在线决定**下一轮是更新 Harness 还是更新权重。这看起来像在线修改，但博客指出其实验存在混杂因素、证据尚不充分，且这种在线决策仍然发生在**多轮交互的"训练"阶段**，而非单次样本 eval。

---

#### 总结

| 问题 | 答案 |
|------|------|
| 单次 eval 期间会修改 Harness 代码吗？ | ❌ **不会**。所有方法的 Harness 结构代码都在离线优化后固定 |
| 推理时上下文/记忆会变化吗？ | ✅ **会**。ACE、MCE 等方法管理的内容可以在推理时在线更新 |
| 离线优化完后 Harness 就永久固定吗？ | 基本是，但 Self-Harness、DGM 等方法支持**周期性离线重新优化**，即 Harness 版本会迭代，但每次迭代后仍然是固定的 |
| 这是否意味着 Harness 缺乏适应性？ | 不——Harness 代码是固定的，但 Harness **管理的上下文、记忆和运行时状态**可以动态变化，这提供了足够的适应性 |

---

### Q2: 详细介绍本文提到的 AI Coding 的例子

**简短回答**：博客将"编码智能体 Harness"作为 Harness 设计模式**最成熟的案例研究**，核心论点是——Claude Code、Codex (OpenAI)、OpenCode、Cursor 等主流编码智能体虽然来自不同团队，但其 Harness 接口已经**高度收敛**到一套几乎相同的工具集和执行循环。这个收敛本身就是 Harness 设计模式具有普适性的强证据。

---

#### 一、编码智能体的通用执行循环

博客指出，所有主流编码智能体共享同一个核心循环范式：

```
用户任务 → 智能体规划 → 调用工具执行 → 观察结果 → 迭代改进 → 完成交付
```

这恰好是 Harness 设计模式中**"工作流自动化"模式**的最纯粹体现（章节 2.1）：

> Plan → Execute → Observe/Test → Improve → Execute again

在这个循环中：
- 智能体不仅执行代码，还会**读取自己的执行轨迹和失败日志**
- 通过"智能体运行时"（agent runtime）迭代，而非依赖静态提示模板
- 每一步的工具调用结果都被记录，形成可审计的执行历史

---

#### 二、收敛的工具集（完整分类）

博客详细列举了编码智能体 Harness 的标准工具集，这是各家产品**不约而同收敛到的接口**：

##### 1. 文件系统操作
| 工具 | 功能 | 为什么必要 |
|------|------|-----------|
| `glob` | 按模式匹配文件名 | 快速定位项目中的相关文件 |
| `grep` | 正则搜索文件内容 | 查找函数定义、引用、TODO 等 |
| `ls` | 列出目录内容 | 理解项目结构 |
| `read` | 读取文件内容 | 获取上下文、理解代码 |
| `write` | 创建/覆写文件 | 生成新代码或重构 |
| `edit` | 精确编辑文件片段 | 最小化改动，避免重写整个文件 |
| `apply_patch` | 应用 diff 补丁 | 批量应用多处修改 |

> 这套文件系统工具直接体现了 Harness 设计模式 2.2"**文件系统作为持久记忆**"——编码智能体的产出物（代码文件、测试结果、日志）远比上下文窗口大，必须持久化到文件系统。

##### 2. Shell 执行
| 工具 | 功能 |
|------|------|
| `bash` | Unix/Linux shell 命令执行 |
| `PowerShell` | Windows shell 命令执行 |

执行内容包括：运行测试、构建项目、安装依赖、执行脚本、检查运行结果。Shell 是智能体与真实执行环境交互的**唯一通道**。

##### 3. IO 与版本控制
| 工具 | 功能 |
|------|------|
| **LSP (Language Server Protocol)** | 获取类型信息、自动补全、跳转定义、诊断错误 |
| `git_status` | 查看仓库状态（哪些文件被修改） |
| `git_diff` | 查看具体的代码变更 |
| `git_commit` | 提交变更 |

LSP 集成尤其关键——它让智能体能够利用与人类开发者完全相同的 IDE 基础设施来理解代码。

##### 4. 外部上下文获取
| 工具 | 功能 |
|------|------|
| **MCP (Model Context Protocol)** 工具 | 连接到外部数据源、API、数据库 |
| **Skills** | 加载专业领域知识或工作流 |

MCP 是 Anthropic 提出的开放协议，本质上让智能体能够动态扩展其 Harness 的能力边界。

##### 5. 网络搜索
| 工具 | 功能 |
|------|------|
| `web_search` | 搜索最新文档、错误信息、技术方案 |
| `web_fetch` | 抓取网页内容（如文档页面） |
| 浏览器工具 | 与需要渲染的网页交互 |

##### 6. 制品（Artifacts）
| 能力 | 说明 |
|------|------|
| 阅读文档/图片 | 理解输入的文档和图像 |
| 生成 HTML | 创建网页预览 |
| 生成图片 | 产出可视化结果 |

##### 7. 后台进程管理
| 工具 | 功能 |
|------|------|
| `CronCreate` | 创建定时任务（如定期运行测试） |
| `CronDelete` | 删除定时任务 |

这直接对应 Harness 设计模式 2.3 **"子智能体与后台任务"**——主智能体需要进程管理能力来协调长时间运行的后台任务。

##### 8. 智能体委派（最关键的工具类别）
| 工具 | 功能 |
|------|------|
| `spawn_agent` | 孵化新的子智能体处理子任务 |
| `resume_agent` | 恢复之前暂停的智能体继续工作 |
| `wait_agent` | 等待子智能体完成并收集结果 |
| `list_agents` | 列出所有活跃的子智能体 |

这是 Harness 设计模式 2.3 的直接实现：**子智能体并行执行，主智能体作为进程管理器**。每个子智能体的输出保存为文件/日志/状态记录，保证**显式可检查性**（博客强调的关键原则）。

---

#### 三、Karpathy 的 `autoresearch` 示例

博客在"工作流自动化"一节特别提到 Andrej Karpathy 的 `autoresearch` 仓库作为典型案例：

- 它构建了一个模型能在其中**操作、测试和迭代**的完整工作流
- 智能体在循环中不断分析自己的**轨迹和失败案例**
- 并非简单的"给个 prompt 就执行"，而是通过运行时迭代来逼近目标
- 这体现了 Harness 工程从"静态提示"到"动态运行时"的范式转变

---

#### 四、编码智能体作为 Harness 收敛的"活证据"

博客选择编码智能体作为案例研究的深层意图：

| 观察 | 含义 |
|------|------|
| 多家公司（Anthropic、OpenAI、开源社区）**独立**设计 | 不是一家公司的偏好，而是行业共识 |
| 最终收敛到**几乎相同的工具集** | Harness 设计空间存在天然的"吸引子" |
| 收敛到**几乎相同的执行循环** | Plan→Execute→Observe→Improve 是编码任务的本质结构 |
| 工具类别覆盖了三大设计模式 | 文件系统 = 持久记忆、Shell = 执行环境、子智能体 = 并行委派 |

博客的潜台词是：**如果编码领域的 Harness 已经收敛，那么其他领域（科研、数据分析、通用智能体）的 Harness 也会经历类似的收敛过程**——这正是研究 Harness 工程的动机。

---

#### 五、相关的编码基准测试

博客附录中列出了几个直接衡量编码智能体能力的基准：

| 基准 | 内容 | 关键发现 |
|------|------|----------|
| **RE-Bench** | 7 个真实 ML 研究工程环境，61 位人类专家 8 小时对比 | AI 短时（2h）是人类的 4 倍，但长时人类反超 —— Harness 的局限性暴露 |
| **MLE-bench** | 75 个 Kaggle 竞赛离线任务 | `o1-preview` + AIDE 框架在 16.9% 竞赛中达铜牌水平 |
| **KernelBench** | 250 个 PyTorch GPU 内核生成任务 | 衡量正确性和速度 |

这些基准反过来又成为评估 Harness 优化效果的标尺——比如 Self-Harness 方法会用这些基准上的表现提升来证明 Harness 优化的价值。

---

#### 六、编码智能体 Harness 与三大设计模式的映射

```
编码智能体 Harness
│
├── 模式1: 工作流自动化
│   └── Plan → Execute (bash, edit, write) → Observe (test results, lints) → Improve
│
├── 模式2: 文件系统作为持久记忆
│   └── 代码文件、测试输出、git history 全部持久化在文件系统中
│      上下文窗口只需保留"当前正在做什么"，不需要塞入整个项目
│
└── 模式3: 子智能体与后台任务
    └── spawn_agent 将子任务委派给独立智能体并行处理
        CronCreate 管理定时任务
        输出全部可检查（文件/日志）
```

这正是博客的核心论点：**Harness 不是 ad-hoc 的工具堆砌，而是由可复用的设计模式构成的系统层**——编码智能体是目前最好的例证。

---

### Q3: 用一个具体例子说明 AI Coding 的完整过程

以 **"为一个 React 电商应用添加暗色模式（Dark Mode）功能"** 为例，展示一个编码智能体 Harness 如何通过与文件系统、Shell、子智能体的交互，多轮迭代完成任务。

---

#### 第一阶段：理解现状（Plan 的前半段）

用户输入：
> "给这个项目加个暗色模式，用户在导航栏点一下按钮就能切换"

**Turn 1 — 智能体开始探索项目结构：**

```
智能体调用: glob(pattern="**/*.css")
→ 返回: ["src/styles/global.css", "src/styles/variables.css",
          "src/components/Navbar/Navbar.css", ...]

智能体调用: grep(pattern="theme|color|background", glob="*.css")
→ 返回: 24 处匹配，发现项目使用 CSS 变量定义颜色，没有暗色主题

智能体调用: read("src/styles/variables.css")
→ 内容:
  :root {
    --color-bg: #ffffff;
    --color-text: #1a1a1a;
    --color-primary: #0066cc;
    --color-card: #f5f5f5;
    /* ... 20+ 个颜色变量 */
  }

智能体调用: read("src/components/Navbar/Navbar.tsx")
→ 了解了导航栏结构，找到了可以添加切换按钮的位置
```

> 这四步体现了 Harness 模式 2（文件系统作为持久记忆）：智能体不依赖一次性把所有代码塞进上下文，而是按需探索文件系统。每一步的 `read` 结果被记录在执行轨迹中。

**Turn 2 — 智能体制定计划（Plan 的后半段）：**

智能体在内部推理（Think）：

> 1. `variables.css` 已经用 CSS 变量管理颜色 → 最佳方案：添加 `[data-theme="dark"]` 选择器覆盖变量
> 2. 需要在 Navbar 添加切换按钮
> 3. 需要创建 `useTheme` hook 管理状态 + 持久化到 localStorage
> 4. 需要更新全局 CSS 以支持过渡动画
> 5. 最后运行测试确保不改坏现有功能

---

#### 第二阶段：并行执行（Execute）

**Turn 3 — 智能体同时孵化 3 个子智能体（Harness 模式 3）：**

```
主智能体调用: spawn_agent(
  name="css-agent",
  task="修改 src/styles/variables.css: 在 :root 后添加 [data-theme='dark']
        选择器，将所有颜色变量翻转为暗色方案"
)

主智能体调用: spawn_agent(
  name="hook-agent",
  task="创建 src/hooks/useTheme.ts: 管理 theme 状态，
        从 localStorage 读取初始值，提供 toggle 函数，
        在 <html> 上设置 data-theme 属性"
)

主智能体调用: spawn_agent(
  name="navbar-agent",
  task="修改 src/components/Navbar/Navbar.tsx: 添加暗色模式切换按钮，
        引入 useTheme hook，按钮显示当前主题图标"
)
```

> 三个子智能体并行工作，各自独立读写文件。主智能体通过 `list_agents` 监控进度。

**Turn 4 — 收集结果：**

```
主智能体调用: wait_agent("css-agent")   → 完成，修改了 variables.css
主智能体调用: wait_agent("hook-agent")  → 完成，创建了 useTheme.ts
主智能体调用: wait_agent("navbar-agent") → 完成，修改了 Navbar.tsx
```

每个子智能体的输出被保存为文件，主智能体可以随时 `read` 检查产出物——符合博客强调的"**显式可检查**"原则。

---

#### 第三阶段：观察与测试（Observe/Test）

**Turn 5 — 主智能体运行测试：**

```
主智能体调用: bash("npm run test")
→ 输出:
  ✓ Navbar renders correctly (passed)
  ✓ Theme toggle switches theme (passed)
  ✗ ProductCard matches snapshot (FAILED — 快照中背景色变了)
  ✓ Cart functionality (passed)
  3 passed, 1 failed
```

**Turn 6 — 智能体分析失败原因：**

```
智能体调用: bash("npm run test -- --verbose ProductCard")
→ 查看完整错误输出

智能体调用: read("src/components/ProductCard/__snapshots__/ProductCard.test.tsx.snap")
→ 发现快照中包含硬编码的颜色值 #f5f5f5，
   而暗色模式下变为了 #1a1a1a ——这是预期行为
```

> 这是 Harness 循环中关键的 **Observe** 步骤：智能体不是盲目地"再试一次"，而是精确诊断失败原因。

---

#### 第四阶段：修复问题（Improve）

**Turn 7 — 智能体修复：**

```
智能体调用: bash("npm run test -- -u")
→ 更新快照以反映新颜色

智能体调用: bash("npm run test")
→ 输出:
  ✓ Navbar renders correctly (passed)
  ✓ Theme toggle switches theme (passed)
  ✓ ProductCard matches snapshot (passed)
  ✓ Cart functionality (passed)
  4 passed, 0 failed
```

**Turn 8 — 智能体添加更多测试覆盖：**

```
智能体调用: edit(
  file="src/hooks/useTheme.test.ts",
  old_str="// TODO: add tests",
  new_str="describe('useTheme', () => {
    it('默认读取 localStorage 的主题', ...);
    it('toggle 切换主题并更新 localStorage', ...);
    it('无 localStorage 时回退到系统偏好', ...);
  })"
)

智能体调用: bash("npm run test")
→ 全部通过
```

---

#### 第五阶段：交付（Deliver）

**Turn 9 — 提交并与用户确认：**

```
智能体调用: git_status()
→ modified: src/styles/variables.css
   modified: src/components/Navbar/Navbar.tsx
   new file: src/hooks/useTheme.ts
   modified: src/components/ProductCard/__snapshots__/...
   new file: src/hooks/useTheme.test.ts

智能体调用: git_diff()
→ 展示所有变更摘要

智能体向用户报告:
  "暗色模式已添加完成：
   - CSS 变量支持 [data-theme='dark'] 覆盖
   - 导航栏添加了切换按钮（☀️/🌙 图标）
   - 主题选择持久化到 localStorage
   - 所有现有测试通过，新增 3 个 hook 测试
   
   你可以运行 npm start 查看效果。需要我调整配色方案吗？"
```

---

#### 整个过程与 Harness 设计模式的映射

```
任务: "给 React 应用添加暗色模式"

Turn 1-2: PLAN（探索+规划）
├── glob, grep, read  →  探索项目结构       [模式2: 文件系统 = 持久记忆]
└── 内部推理          →  制定修改计划

Turn 3-4: EXECUTE（并行执行）
├── spawn_agent × 3   →  并行修改 CSS/Hook/Navbar  [模式3: 子智能体并行]
└── wait_agent × 3    →  收集结果（文件可检查）

Turn 5-6: OBSERVE（观察结果）
├── bash(npm test)    →  运行测试发现失败
└── read(快照) + 分析  →  诊断原因

Turn 7-8: IMPROVE（修正+增强）
├── bash(npm test -u) →  更新快照
├── edit              →  添加额外测试
└── bash(npm test)    →  验证全部通过

Turn 9: DELIVER（交付）
├── git_status/git_diff →  审查变更
└── 报告              →  与用户确认
                   [模式1: Plan→Execute→Observe→Improve 完整闭环]
```

---

#### 关键观察

1. **上下文窗口从未溢出**：智能体每次只 `read` 当前需要的文件，不一次性加载整个项目。文件系统充当了"外部记忆"。

2. **失败是正常流程的一部分**：第 5 轮的测试失败没有被视为错误，而是触发了 O**bserve → Improve** 循环。这直接对应博客强调的"负面结果"价值。

3. **子智能体使并行成为可能**：CSS、Hook、Navbar 三个修改互不依赖，并行执行将 3 个串行步骤压缩为 1 步。但博客的警告也在起作用——每个子智能体的输出是**文件**，主智能体可以事后检查，确保不会出现隐藏的错误。

4. **git 是 Harness 的安全网**：`git_diff` 在交付前让智能体（和用户）审查所有变更，`git_status` 让智能体清楚知道"自己改了哪些文件"——防止遗漏或重复修改。

5. **Harness 工具是"正交"的**：`glob`/`grep` 负责探索，`read`/`edit` 负责修改，`bash` 负责验证，`git` 负责追踪——每个工具只做一件事，组合起来覆盖完整工作流。

---

### Q4: 详细介绍 CORE-Bench 并补充一个例子

**简短回答**：CORE-Bench（Computational Reproducibility Agent Benchmark）是由普林斯顿大学提出的基准测试，评估 AI 智能体**复现已发表科学论文计算结果**的能力。它包含 **270 个任务**（来自 90+ 篇论文，覆盖计算机科学、社会科学、医学），分为三个难度等级。2024 年发布时，最强智能体 CORE-Agent + GPT-4o 的总体准确率仅 **45.93%**，在最难的 CORE-Hard 级别仅 **22.22%**——说明论文复现对 AI 来说极为困难。

---

#### 一、CORE-Bench 的设计理念

CORE-Bench 的存在本身就是在回答一个问题：**如果 AI 智能体连"跑通已有代码并读取结果"都做不好，那它凭什么声称能"做科研"？**

与 MLE-bench（Kaggle 竞赛）或 RE-Bench（ML 工程）不同，CORE-Bench 不要求智能体**创造**新东西，只要求它**复现**已有工作。但即便是这个"简单"任务，AI 的表现远未达到可用水平。

---

#### 二、三个难度等级

| 级别 | 智能体需要做什么 | 真实度 |
|------|-----------------|:------:|
| **CORE-Retrieve** | 代码**已经跑完**，结果文件/图像/日志都已就绪。智能体只需在已有输出中**查找答案**。 | 最低 |
| **CORE-Easy** | 提供完整 **Dockerfile**，智能体只需 `docker build && docker run`，然后从输出中提取答案。 | 中等 |
| **CORE-Hard** | **只有裸代码仓库**，没有 Dockerfile，没有环境说明。智能体必须自己安装依赖、处理兼容性问题、跑通代码、提取答案。 | 最接近真实论文复现 |

关键设计：**任务只有全部子问题都回答正确才算通过**。不存在"部分得分"。

---

#### 三、完整例子：复现一篇交通预测论文

以下是一个**CORE-Hard** 级别任务的完整 walkthrough，展示 CORE-Bench 如何测试智能体的计算复现能力。

---

##### 任务初始状态

智能体收到的唯一输入是一段任务描述和论文的代码仓库：

```
environment/
├── task.txt                     ← 任务说明和需要回答的问题
├── capsule-0a1b2c3/             ← 论文代码仓库
│   ├── README.md
│   ├── requirements.txt
│   ├── main.py
│   ├── run_experiments.sh
│   ├── data/                    ← 数据集
│   └── utils/
└── report.json                  ← 智能体需要填写的答案模板
```

**`task.txt` 的内容**（摘要）：

> 这篇论文提出了 HyperETA，一个基于超图的到达时间预测模型。
> 请在代码仓库中复现以下实验结果，并将答案写入 report.json：
>
> 1. "在不使用 DTW 的情况下，HyperETA 的 MAPE 是多少？"
> 2. "在不使用 DTW 的情况下，HyperETA 的 RMSE 是多少？"
> 3. "在不使用 DTW 的情况下，HyperETA 的 MAE 是多少？"
>
> 注意：你需要自己安装依赖、运行代码、并从输出中提取准确的数值。

---

##### Turn 1 — 探索代码仓库

```
智能体调用: ls("environment/capsule-0a1b2c3/")
→ README.md, requirements.txt, main.py, run_experiments.sh, data/, utils/

智能体调用: read("environment/capsule-0a1b2c3/README.md")
→ 了解了项目结构：
  - Python 3.8+, PyTorch 1.10+
  - 运行 run_experiments.sh 执行实验
  - 结果保存到 results/metrics.json
  - 参数 --no_dtw 控制是否使用 DTW

智能体调用: read("environment/capsule-0a1b2c3/requirements.txt")
→ numpy==1.21.0, torch==1.10.0, pandas==1.3.0, scikit-learn==0.24.2, ...
```

> 第一步完全是 Harness 模式 2（文件系统作为持久记忆）——智能体通过按需探索文件系统建立对项目的理解。

---

##### Turn 2 — 安装依赖（第一次尝试）

```
智能体调用: bash("cd environment/capsule-0a1b2c3 && pip install -r requirements.txt")
→ 输出:
  ERROR: Could not find a version that satisfies the requirement torch==1.10.0
  ERROR: No matching distribution found for torch==1.10.0
```

> **这就是 CORE-Hard 的关键挑战**：真实论文的代码往往有依赖兼容性问题。智能体不能只是机械执行，必须**诊断和修复**。

---

##### Turn 3 — 诊断并修复依赖

```
智能体调用: bash("python --version")
→ Python 3.11.5

智能体分析:
  "torch==1.10.0 在 Python 3.11 上不可用，需要升级 torch 版本，
   但需要确认新版 torch 的 API 是否兼容"

智能体调用: bash("pip install torch==2.0.0 --index-url https://download.pytorch.org/whl/cpu")
→ 安装成功

智能体调用: bash("pip install -r requirements.txt --no-deps")
→ 安装其余依赖

智能体调用: bash("pip install numpy pandas scikit-learn")
→ 手动安装兼容版本的其余包
```

> 这一步体现了 AI Coding 示例中没有出现的能力：**环境故障排除**。智能体需要理解 Python 版本兼容性、torch 的安装源、以及 `--no-deps` 标志的用法。

---

##### Turn 4 — 运行实验

```
智能体调用: bash("cd environment/capsule-0a1b2c3 && bash run_experiments.sh --no_dtw")
→ 输出:
  Loading data...
  Training model... Epoch 1/50...
  Epoch 2/50... [loss: 0.342]
  ...
  Epoch 50/50... [loss: 0.018]
  Evaluating...
  Results saved to results/metrics.json
```

---

##### Turn 5 — 提取答案

```
智能体调用: read("environment/capsule-0a1b2c3/results/metrics.json")
→ {
    "MAPE": 17.374344500709498,
    "RMSE": 459.7782074000463,
    "MAE": 323.0
  }

智能体向 report.json 写入答案:
→ {
    "Report the HyperETA MAPE with no DTW.": 17.374344500709498,
    "Report the HyperETA RMSE with no DTW.": 459.7782074000463,
    "Report the HyperETA MAE with no DTW.": 323.0
  }
```

> 注意：答案必须与标准答案**精确匹配**到小数点后若干位。如果智能体的运行环境不同导致微小的数值差异（如浮点舍入），任务也会被判失败。

---

##### 可能的失败路径

在实际 CORE-Bench 评估中，智能体经常在以下环节失败：

| 失败点 | 典型原因 | 难度等级 |
|--------|----------|:--------:|
| **依赖安装** | 版本冲突、已废弃的包、系统库依赖 | Hard 最严重 |
| **代码运行** | 硬编码路径、GPU/CPU 假设、shell 脚本语法差异 | Hard |
| **结果定位** | 输出文件名与预期不同、格式变化 | 全部 |
| **数值精度** | 不同硬件/库版本产生的浮点舍入差异 | 全部 |
| **多模态输出** | 需要从图中读取数值（曲线图的极值点等） | 全部 |
| **时间/资源** | 代码运行时间超预算 | Hard |

---

#### 四、CORE-Bench 与 Harness 设计模式的关系

```
CORE-Bench 任务执行流
│
├── 模式1: 工作流自动化
│   └── Read task → Explore repo → Install deps → Run code → Extract results
│       （每一步都可能失败，触发子循环：Diagnose → Fix → Retry）
│
├── 模式2: 文件系统作为持久记忆
│   └── README 提供项目理解
│       requirements.txt 提供依赖信息
│       metrics.json 存储运行结果
│       report.json 作为最终交付物
│
└── 模式3: 子智能体与后台任务
    └── （CORE-Bench 本身不要求子智能体，但可以扩展：
         一个智能体负责安装，另一个并行阅读论文补充理解）
```

CORE-Bench 的核心洞察：**计算复现本质上就是一个 Harness 工程问题**——智能体需要在受限的环境中自主导航文件系统、执行 shell 命令、解析输出、并做出**精确的数值回答**。每一步都是对 Harness 工具能力和鲁棒性的测试。

---

#### 五、与博客中其他基准的对比

| 基准 | 测试什么 | AI 能力要求 |
|------|----------|------------|
| **CORE-Bench** | 复现已发表论文的计算结果 | 环境配置、代码调试、精确结果提取 |
| **PaperBench** | 从零复现 ICML 论文的全部实验 | 完全自主科研（Claude 3.5 仅 21%） |
| **RE-Bench** | 7 个真实 ML 工程任务 | 工程迭代能力 |
| **MLE-bench** | Kaggle 竞赛 | 从数据到模型的完整 pipeline |

CORE-Bench 的特殊之处在于：它测试的不是"创造力"，而是**精确性和鲁棒性**——能否一丝不苟地复现指定结果。这和博客强调的"可验证性"（ScientistOne 的核心理念）一脉相承。

---

#### 六、当前表现与含义

| 智能体 + 模型 | Retrieve | Easy | Hard | 总体 |
|:---|:---:|:---:|:---:|:---:|
| CORE-Agent + GPT-4o | 57.78% | 57.78% | 22.22% | 45.93% |
| CORE-Agent + GPT-4o-mini | 44.44% | 26.67% | 15.56% | 28.89% |
| AutoGPT + GPT-4o | 35.6% | 37.8% | 6.7% | 26.7% |

**Hard 级别仅 22.22%** 意味着：即使是最强智能体，在 5 次尝试中也有近 4 次无法成功复现一篇论文的计算结果。这直接击中了博客第 4.1 节"弱且模糊的评估器"挑战——如果连"代码能不能跑通"这种客观验证都做不到，何谈评估"研究是否有价值"这种模糊目标？
