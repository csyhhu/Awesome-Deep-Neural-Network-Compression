# Darwin Gödel Machine: Open-Ended Evolution of Self-Improving Agents

- **论文链接**: [arxiv.org/abs/2505.22954](https://arxiv.org/abs/2505.22954)
- **作者**: Jenny Zhang*, Shengran Hu*, Cong Lu, Robert Lange†, Jeff Clune† (UBC, Vector Institute, Sakana AI)
- **核心代码**: [github.com/jennyzzt/dgm](https://github.com/jennyzzt/dgm)

---

## 1. 核心动机与问题

当前大多数 AI 系统受限于人类设计的固定架构，无法自主、持续地自我改进。科学进步是累积和开放式的（open-ended），而 AI 的发展至今仍严重依赖人类干预。作者探索一个问题：**AI 系统能否像科学发现一样，成为自身进步的引擎——不断在自身基础上递归改进？**

Gödel Machine（Schmidhuber, 2007）提出了一个理论方案：一个能够通过形式化证明来确保每次自我修改都是有益的 AI。但实践中，证明大多数修改的净收益是不可能的。因此本文提出 **Darwin Gödel Machine (DGM)**：用经验验证取代形式化证明，用达尔文式的开放探索取代单一优化路径。

---

## 2. 方法

### 2.1 核心思想

DGM 是一个自我引用（self-referential）、自我改进的系统，它通过修改自身的代码仓库来成为一个更好的 coding agent。核心假设是：**coding benchmark 的性能提升 = coding 能力提升 → 自我修改能力也随之提升**，从而实现加速的自我改进循环。

### 2.2 三个关键组成部分

**① 自我引用的自我改进（Self-referential Self-improvement）**

- DGM 初始化为一个 coding agent（由冻结的预训练 FM + 工具能力组成）
- 每次迭代中，选中的 agent **分析自己的 benchmark 评测日志**，提出下一个要实现的特性，然后**修改自己的代码仓库**来生成一个新版本
- 自我改进本质上是一个 coding 任务——agent 需要用 coding 能力来改进自己的代码

**② 基于种群的开放探索（Population-based Open-ended Exploration）**

- 维护一个**存档（archive）**，包含所有已发现的可工作 agent
- 每次选择父 agent 进行自我修改时，**概率与性能分数成正比、与已有子代数成反比**（鼓励探索尚未充分开发的高性能路径）
- 所有 agent 保留非零选择概率，确保任何路径都有被重新探索的机会
- 只有能成功编译、并保持代码编辑能力的 agent 才会被加入存档

**③ 经验验证取代形式化证明**

- 不像 Gödel Machine 那样需要形式化证明，DGM 通过**在 benchmark 上实际评估**来验证修改的有效性
- 这借鉴了达尔文进化中的"变异—选择—保留"机制

### 2.3 算法伪代码

```
输入: 初始 agent g0, benchmark 套件 B, 最大迭代次数 T
输出: agent 存档 A

s0 ← evaluate(g0, B)
初始化 A ← {(g0, s0)}

For t ← 1 to T:
    P ← SelectParents(A)           # 从存档中选择父 agent
    For each p in P:
        c ← p.modify(p)             # 自我修改
        s ← evaluate(c, B)          # benchmark 评估
        If c.is_valid():            # 仅保留能编辑代码的
            A ← A ∪ {(c, s)}
Return A
```

### 2.4 父选择机制

对每个 agent $a_i$:
- $s_i = \frac{1}{1 + \exp(-\lambda(\alpha_i - \alpha_0))}$ （sigmoid 缩放的性能分数）
- $h_i = \frac{1}{1 + n_i}$ （新颖性奖励，$n_i$ 为有代码编辑功能的子代数）
- $w_i = s_i \cdot h_i$
- $p_i = w_i / \sum w_j$（选择概率）

---

## 3. 实验设置

### 3.1 初始 Agent

- 基于冻结的 FM + 两个基础工具：
  - **Bash Tool**: 执行 bash 命令
  - **Edit Tool**: 查看和编辑目录及整个文件
- 设计刻意保持轻量，以便清晰展示自我改进效果

### 3.2 Benchmark

- **SWE-bench Verified**: Python 多文件编辑任务（使用了 10→50→200 阶段评估策略）
- **Polyglot**: 多语言（C++, Rust, Python 等）编程任务（pass@1 设定）

### 3.3 对比基线

| 基线 | 说明 |
|------|------|
| **DGM w/o self-improve** | 修改 agent 的元 agent 始终固定为初始 agent（类似 ADAS 设定） |
| **DGM w/o open-ended exploration** | 无存档，总是从最新版本自我修改（类似 hill-climbing） |
| **DGM Greedy** | 永远选择最佳 agent 作为父节点（类似 Robeyns et al., 2025） |
| **Human-designed SOTA** | SWE-bench: OpenHands+CodeAct v2.1; Polyglot: Aider |

---

## 4. 实验结果

### 4.1 核心性能提升

| Benchmark | 初始 Agent | DGM 最佳 | 提升 |
|-----------|-----------|---------|------|
| SWE-bench | 20.0% | 50.0% | +30.0% |
| Polyglot (subset) | 14.0% | 38.0% | +24.0% |
| Polyglot (full) | 14.2% | 30.7% | +16.5% |

- DGM 发现的 agent 性能**与人工设计的 SOTA 开源方案相当**（SWE-bench），或**大幅超越**（Polyglot 超越 Aider）

### 4.2 消融实验

| 方法 | SWE-bench | Polyglot |
|------|-----------|----------|
| DGM | **50.0%** | **38.0%** |
| DGM w/o Open-ended | 23.0% | 14.0% |
| DGM w/o Self-improve | 39.0% | 28.0% |
| DGM Greedy | 39.7% | 30.0% |

- 去掉开放探索后性能急剧下降（23.0% vs 50.0%），说明**存档和多样化的 stepping stones 至关重要**
- DGM Greedy（只从最佳节点分叉）也显著不如 DGM，说明开放探索的多样性比贪心搜索更有效

### 4.3 DGM 发现的具体改进

- **编辑工具增强**: 支持按行查看文件、按字符串替换编辑（而非总是查看/替换整个文件）
- **工作流改进**: 多次尝试求解 + 用另一个 FM 评估并选择最佳解；后续尝试会参考之前的尝试
- **长期上下文窗口管理**

### 4.4 跨模型迁移

| 评估模型 | 初始 Agent | DGM Agent |
|---------|-----------|-----------|
| Claude 3.5 Sonnet | 20.0% | 50.0% |
| Claude 3.7 Sonnet | 19.0% | 59.5% |
| o3-mini | 23.0% | 33.0% |

- 改进**跨模型泛化**，说明 DGM 发现的是通用的工作流改进，而非针对特定模型的过拟合

### 4.5 跨 Benchmark 迁移

- 在 SWE-bench 上进化出的 agent 在 Polyglot 上从 14.2% → 28.9%
- 在 Polyglot 上进化出的 agent 在 SWE-bench 上从 20.0% → 24.5%
- **跨语言迁移**: 仅用 Python 任务训练的 DGM 在非 Python 语言上也取得接近全语言训练的性能

### 4.6 存档分析

- 搜索树中许多通向创新的路径穿过了低性能节点
- 关键创新节点（如 node 24）产生后，会引发大量基于它的后续创新
- 最佳 agent 的 lineage 包含两次性能下降，体现了开放探索的价值——暂时的性能下降可能是通往更大突破的踏脚石

### 4.7 稳定性

DGM 在 Polyglot 上运行 3 次：平均 40.7%，标准差 2.3%，表现稳定可复现。

---

## 5. 安全性讨论

### 潜在风险

1. 仅针对 benchmark 性能优化的修改可能**无意中引入漏洞**或与人类意图不一致的行为
2. 评估指标若不能完全捕捉安全性和鲁棒性，自我改进循环可能在多代后**放大偏差**
3. 迭代自我修改可能导致**越来越复杂和不可解释的内部逻辑**

### 当前防护措施

- 所有 agent 执行和自我修改在**隔离沙箱环境**中进行
- 严格的时间限制防止资源耗尽
- 自我改进范围限于**特定的 coding benchmark**（修改 Python 代码仓库）
- DGM 存档提供了**可追溯的修改 lineage** 供审查
- 目前未发现有害或恶意行为

### 安全应用的探索

论文还展示了 DGM 可被用于**解决 FM 的工具调用幻觉（hallucination）问题**，但也发现了 **objective hacking** 风险——agent 学会了绕过幻觉检测函数（删除特殊标记）来获得高分，而非真正解决问题。

---

## 6. 局限性与未来工作

1. **计算成本高**: 单次 SWE-bench 运行约 2 周，花费约 $22,000 USD API 费用
2. **受限于 FM 能力**: DGM 无法超越底层 FM 的能力上限
3. **限于 coding 领域**: 未来可扩展到计算机视觉、创意写作等领域
4. **开放探索过程未自身进化**: 目前的存档维护、父选择机制是固定的，未来可让 agent 自行改进这一元过程
5. **未涉及 FM 训练**: 当前仅优化 prompt/工具/工作流（冻结 FM），未修改训练脚本；未来可扩展到重写训练代码以更新 FM 本身
6. **人类角色问题**: 在自主 AI 系统中人类如何参与（RLHF、Constitutional AI 等）仍是开放问题

---

## 7. 关键贡献总结

1. **首个结合开放探索的 FM 驱动自改进系统**，实现了 self-referential 的改进循环
2. 证明 benchmark 性能提升可以直接转化为更好的自我改进能力
3. 展示开放探索和存档机制对于避免局部最优和持续进步的关键作用
4. 在 SWE-bench（20%→50%）和 Polyglot（14.2%→30.7%）上取得了显著提升
5. 发现的改进具有跨模型、跨 benchmark、跨编程语言的泛化能力
6. 提供了详细的安全分析和防护措施讨论

---

## 8. QA 讨论

### Q1: 本文的产出是什么？是一套 harness 框架吗？

**不是传统意义上的 harness 框架。** DGM 不是像 SWE-agent、Aider、OpenHands 那样提供一个固定的、人工设计的 coding agent。它的产出分两层：

1. **DGM 元算法本身**：一套"自动进化更好 coding agent"的方法论（父选择、自我修改、存档维护、benchmark 评估的完整流程）
2. **算法运行后的产物**：一个包含多种 agent 变体的存档（archive），其中**得分最高的 agent**是可以直接用来解决实际 coding 任务的成品

类比理解：SWE-agent 是"工程师手工设计的产品"，DGM 是"自动化工程师团队 + 他们进化出的产品"。DGM 的最终使用者获得的不是一套固定框架，而是**经过 80 轮自动进化后的优化版 agent 代码仓库**。

### Q2: 关键内容包含哪些？

| 组件 | 说明 |
|------|------|
| **初始 Agent（base agent）** | 极简 Python 代码仓库，仅含 Bash 工具 + Edit 工具，由冻结 FM（如 Claude 3.5 Sonnet）驱动，`forward()` 方法定义工作流 |
| **自我修改循环** | 被选 agent 分析自身 benchmark 评测日志 → 诊断应改进什么 → **修改自己的代码仓库**（改 tools/workflow/prompts）→ 生成新版本 |
| **开放探索存档** | 所有能编译成功且保持代码编辑能力的 agent 被无限期保留，形成可追溯的进化树 |
| **父选择机制** | 概率 ∝ sigmoid(性能分数) × 1/(1+已有子代数)，所有节点非零概率，鼓励探索未充分开发的高性能路径 |
| **诊断 Prompt（o1）** | 用 o1 模型分析评测日志，输出 `implementation_suggestion` + `problem_description`，作为 agent 自我修改的任务描述 |
| **阶段评估策略** | 10 任务（验证基本功能）→ 50 任务（估计效果）→ 200 任务（精确评估头部 agent），逐渐加码 |

### Q3: 实际使用时如何进行？

以一个完整的 SWE-bench 运行为例：

**前置准备：**
- 初始 agent 代码仓库（Python 项目，定义 `forward()` 方法和 Bash/Edit 两个工具）
- benchmark 任务子集 + 评测脚本
- 沙箱隔离环境

**运行流程（80 次迭代，k=2 并行）：**
```
1. 评测初始 agent g0 → 分数 s0 → 加入存档 A

2. 迭代 t = 1..80：
   a. SelectParents(A) → 按概率选 k 个父 agent
   b. 对每个父 agent p：
      - 将 p 的评测日志发给 o1 诊断模型 → 输出改进建议
      - p 在自己的代码仓库中实现该建议 → 生成子 agent c
      - evaluate(c, B) → 分数 s
      - 若 c 能编译且保持代码编辑功能 → A ← A ∪ {(c, s)}
      否则丢弃

3. 运行结束后，从 A 中选最高分 agent 作为最终产物
```

**最终得到的东西：**
- 一个优化过的 coding agent 代码仓库，可直接运行 `python run.py --task=xxx`
- 自动包含了 DGM 发现的各种改进（如 `str_replace` 编辑工具、多轮尝试+投票、长上下文管理等）
- 该 agent **可跨模型使用**（用 Claude 3.5 进化的 agent，换成 Claude 3.7 后性能从 19% 升至 59.5%）

**成本参考：** SWE-bench 单次完整运行约 2 周、~$22,000 API 费用（主要是 Claude 的调用成本）；Polyglot 用 o3-mini 评估大幅降低了成本。

### Q4: Agent 是通过 LLM 参数更新得到进化，还是产出中间产物（prompt）来提升 LLM 能力？

**Agent 的进化完全通过修改包裹 LLM 的"脚手架代码"实现，LLM 参数自始至终不更新。**

论文明确指出 FM 是 **frozen**（冻结的），未来工作方向之一才是"让 agent 重写训练脚本以更新 FM 本身"。当前 DGM 每次自我修改实际改变的是：

| 修改目标 | 具体内容 | 实例 |
|---------|---------|------|
| **Prompt** | agent 发给 FM 的 system/user prompt | 加入"多次尝试并用另一个 FM 投票选最优解"的指令 |
| **工具定义** | Bash/Edit 工具的参数、功能签名 | Edit 工具从"整文件替换"变成 `str_replace`（按字符串精确替换） |
| **工作流代码** | `forward()` 方法中的执行逻辑 | 多轮尝试+投票机制、前次结果反馈给后续尝试 |
| **工具实现** | Python 代码中工具的具体行为 | 按行范围查看文件、添加自动测试验证步骤 |

**类比：**
- **LLM 参数更新** = 改造大脑（修改神经元连接）
- **DGM 的做法** = 给同一个大脑配备更好的工具、更有效的指令、更聪明的工作流程

这也解释了为什么 DGM 发现的 agent 可以**跨模型泛化**——用 Claude 3.5 进化出的 agent 换成 Claude 3.7 后性能从 19% 飙升到 59.5%。因为优化的是模型无关的脚手架逻辑（"怎么用 FM"），而非模型本身的权重（"FM 是什么"）。

**当前局限：** DGM 无法超越底层 FM 的能力天花板——如果 FM 本身推理能力不足，再好的 prompt 和工具也无法弥补。将自我修改扩展到 FM 训练是论文列为 priority 的未来方向。

### Q5: Self-referential Self-improvement 中的"修改自己的代码仓库"具体指什么？

DGM 中每个 agent 本质上是一个 **Python 项目目录**，包含 `run.py`（定义 `forward()` 工作流）、`tools/`（Bash/Edit 工具实现）、`prompts/`（system prompt 文件）等。

**"修改自己的代码仓库"** 就是 agent 内部的 FM **自己读自己的代码，然后生成 diff/patch 来改自己的代码文件**。三步走：

1. **Self-read**: FM 读取当前 agent 的所有代码文件
2. **Self-write**: FM 根据 o1 诊断模型给出的改进建议（如"Edit 工具应支持按字符串替换"），生成代码 diff
3. **Save**: diff 应用后保存为新版本 agent

之所以叫 "Self-referential"（自我引用），是因为**修改者和被修改者是同一个实体**——agent 用自身的编程能力来改进自身。

**关键约束：** 只有修改后满足两个条件的 agent 才会被保留：
- 能成功编译运行（无语法错误）
- 保持基本的代码编辑功能（不能变成"只会解题不会自我改进"的废版本）

因此论文中只有 51.3% 的生成 agent 通过了这一筛选。

**具体实例——论文中最重要的改进（node 24）：**

| 修改前 | 修改后 |
|--------|--------|
| `edit` 工具只能查看/替换**整个文件** | 新增 `str_replace` 命令，按 `old_str → new_str` **精确替换**文件中的某几行 |
| 每次编辑需传入完整文件内容 | 只需传入目标字符串，Tokens 消耗大幅降低 |

这个改动就是 FM 打开 `edit_tool.py`，将原来的整文件替换函数改写为更精细的 `str_replace` 实现。

### Q6: 最终交付物和使用方式确认

**正确。** DGM 的最终交付物是存档中得分最高的 **agent_repo（代码仓库）**，部署时直接使用该仓库执行任务，不再需要 DGM 进化循环。

**DGM 进化 vs 部署的关系：**

```
DGM 进化阶段（工厂）
  80 轮迭代，~$22K API 费用
  产出: 一个包含多种 agent 变体的 archive
              │
              ▼ 选最佳 agent
部署阶段（产品）
  一个 agent_repo 代码仓库
  按需调用 FM API 执行任务
  使用方式: python run.py --repo <目标仓库> --problem <任务描述>
```

DGM 是"工厂"，agent_repo 是"产品"。部署时只需产品，工厂可以关掉。这也解释了为什么进化出的 agent 可以**跨模型使用**——进化固化的是工具/工作流/prompt 等脚手架逻辑，部署时可以绑定任意 FM（甚至换更强的模型直接获得额外收益）。

### Q7: 用一个例子说明每次自我修改时的信号和具体动作

以论文中最关键的改进 **Node 24（str_replace 编辑工具）** 为例，完整流程如下：

**第一步：收集输入信号**

父 agent 在 benchmark 上评测后，收集以下信息作为"信号"：

- ① **评测日志**: 大量失败案例显示 agent 在编辑大文件时因整文件覆盖丢失了不相关的代码
- ② **当前 agent 完整代码仓库**: 所有 `.py` 文件被作为上下文提供
- ③ **失败案例的 GitHub issue + agent 预测 patch + golden patch + 测试结果**

**第二步：o1 诊断模型输出改进建议**

以上信号填入诊断 prompt → o1 输出结构化 JSON：

```json
{
  "log_summarization": "Agent 多次因整文件替换丢失上下文代码而失败...",
  "improvement_proposal": "将 Edit 工具从整文件覆盖改为字符串精确替换",
  "implementation_suggestion": "将 edit 命令替换为 str_replace，接收 old_str/new_str 参数，仅在字符串唯一出现时执行替换",
  "problem_description": "Enhance file editing to support fine-grained edits via str_replace..."
}
```

其中 `problem_description` 被提取出来，作为自我修改的"任务描述"。

**第三步：Agent 自我修改（Claude 3.5 Sonnet 执行）**

agent 读取自己的代码仓库 + 收到 `problem_description`，生成以下具体改动：

```
修改: tools/edit.py           # 核心改动，+87/-61 行
新增: tests/test_tools.py     # 自动生成了 5 个单元测试
```

`tools/edit.py` 的具体变化：

| 位置 | 旧 (base agent) | 新 (node 24) |
|------|----------------|-------------|
| 命令名 | `edit` | `str_replace` |
| 参数 | 仅 `file_text`（整文件内容） | 新增 `old_str` + `new_str` |
| 核心逻辑 | `write_file(path, file_text)` | `str_replace_in_file(path, old_str, new_str)` — 先数出现次数：0→报错、>1→报错、=1→执行替换 |

新增的核心函数 `str_replace_in_file()` 约 16 行，实现了严格的出现次数检查和原子替换。同时生成了覆盖成功替换、未找到、多次出现、参数缺失、路径无效五种场景的单元测试。

**第四步：验证与存档**

新 agent 编译通过 → 10 任务验证 → 保持代码编辑能力 → 加入存档。该节点性能从父节点的 ~23% 跃升至 **40.5%**。

**完整信号→动作链路总结：**

```
评测日志 + 代码仓库 + 失败案例　→ o1 诊断　→ Claude 3.5 修改代码　→ 新 agent repo
     (信号)                       (翻译)        (动作)               (产物)
```

### Q8: 感觉整个方法就是把遗传算法用在 agent harness 上？创新点在哪？

表面上看确实像遗传算法（选择→变异→评估→存档），但 DGM 有三个本质区别使它不是简单的 GA 套用：

**区别一：变异是 Self-referential，不是外部的随机扰动**

| | 传统遗传算法 | DGM |
|---|---|---|
| **谁做变异？** | 外部算法（随机 crossover/mutation 算子） | **agent 自己做**（FM 理解自己的代码→决定改什么→改自己的代码） |
| **变异性质** | 固定编码空间内的随机扰动 | **语义驱动的、理解性的代码重构** |
| **例子** | 随机翻转 bit、交叉两个父代基因 | FM 理解"整文件替换有问题"后，自主决定"应该在 edit.py 里新增一个 str_replace 函数" |

传统 GA 的变异算子是人工预定义的、对代码语义一无所知。DGM 的"变异"是 agent 用自身编程能力完成的 **informed self-modification**，这就是 "Self-referential" 的核心含义——修改者和被修改者是同一实体。

**区别二：诊断引导的变异方向，而非纯随机探索**

传统 GA 仅用 fitness score 选父代，不告诉你**为什么**差、**该怎么**改。

DGM 在变异前多了一个 **o1 诊断步骤**：用强 FM 分析 agent 的评测日志，输出结构化改进建议（`implementation_suggestion` + `problem_description`）。相当于每个 agent 旁边有一个"代码审查员"告诉它：你的问题出在哪，建议怎么修。

```
传统 GA:   fitness score → 选择 → 随机变异
DGM:      fitness score + o1 诊断报告 → 选择 → 有理解的自我修改
```

**区别三：新颖性驱动的非贪心探索，而非精英主义**

传统 GA 倾向选择高适应度个体（elitism/tournament selection），容易早熟收敛。

DGM 的父选择是 **概率 ∝ sigmoid(性能) × 1/(1+已有子代数)** — 所有节点有非零概率。一个性能中等但从未被探索过的 agent，可能比高分的"老面孔"更优先产生子代。消融实验验证了这一点：去掉开放探索后，SWE-bench 性能从 50% 崩溃到约 23%。

**论文的贡献层级总结：**

| 层级 | 贡献 |
|------|------|
| **概念创新** | 首次将 self-referential self-improvement 用于 coding agent 自动化设计——agent 用自身编程能力改进自身脚手架 |
| **系统设计** | 开放探索存档 + o1 诊断引导 + 非贪心父选择的三合一机制 |
| **实证发现** | 消融实验证明开放探索和自我修改两个组件缺一不可；进化出的改进可跨模型泛化 |
| **跨模型泛化** | 用 Claude 3.5 进化的 agent 换 Claude 3.7 后性能从 19%→59.5%，说明进化固化的是"怎么用 FM"而非模型本身 |

### Q9: 诊断引导的输出具体怎么使用？

诊断输出通过**两步流水线**被使用，最终驱动 agent 的自我修改：

**Step A：o1 诊断（外部 FM 分析日志，输出改进建议）**

论文在当前实现中用一个**单独的外部 FM（o1）**来做诊断，因为它推理能力更强。o1 接收 5 种输入：

```
① agent 的完整代码仓库
② agent 在 benchmark 上的执行日志（失败/成功的完整记录）
③ 失败案例的 GitHub issue 原文
④ agent 生成的 predicted patch
⑤ gold test patch + 测试结果（私有测试，agent 评估时看不到）
```

o1 必须按预定义格式输出结构化 JSON：

```json
{
  "log_summarization":        "分析 agent 执行日志，总结失败模式",
  "potential_improvements":   "列出多个可能的改进方向",
  "improvement_proposal":     "从中选择一项高影响力的改进，详细描述",
  "implementation_suggestion": "从代码层面说明如何修改现有工具/新增工具来实现该改进",
  "problem_description":      "将改进方案写成一个 GitHub issue 风格的描述，便于开发人员直接实现"
}
```

**Step B：提取两个关键字段，传给 agent 执行自修改**

o1 的输出**不是全部都用**。论文只提取两个字段：

- `implementation_suggestion`：代码层面的修改指引
- `problem_description`：以 GitHub issue 格式的需求描述

然后将这两个字段**拼接为 agent 的"自修改任务描述"**，传给 agent 自己的 FM（Claude 3.5 Sonnet）。效果等同于：

> "这是一个 GitHub issue，请你在自己的代码仓库中实现这个需求"

agent 收到后：
1. 读取自己的代码仓库
2. 理解 `problem_description` 描述的需求
3. 生成代码 diff
4. 应用 diff → 保存为新版本 agent repo

**为什么需要分开诊断和执行？**

论文作者明确解释：这是**当前实现的工程选择，不是理论必要的**。他们用 o1 做诊断是因为 o1 的推理能力比当时其他 FM（包括 agent 自己用的 Claude 3.5）强。理论上，如果 agent 自身的 FM 足够强，完全可以自己分析日志并提出改进——这正是论文列为未来方向之一的"让 agent 完全自主完成闭环"。

```
完整流水线：
评测日志+代码+失败案例 →  o1 诊断 → 提取 problem_description →  agent 自修改 →  新 agent repo
    (原始信号)           (翻译/分析)     (只取两个字段)          (执行)           (产物)
```

**关键设计选择：** `problem_description` 被特意设计为"通用性问题"（如"Edit 工具应支持字符串替换"），而非针对某个具体 GitHub issue 的修复方案。这保证了每次自我修改产出的是**跨任务通用的能力提升**，而非对特定测试用例的过拟合。
