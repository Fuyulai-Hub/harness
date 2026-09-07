# FylHarness

> 一个轻量级、开箱即用、Windows 友好的大语言模型评估 Harness，兼容任意 OpenAI 风格 API（OpenAI 官方、vLLM、Ollama、LM Studio、TGI、llama.cpp server 等）。

FylHarness 借鉴 [DeepSeek-LLM](https://github.com/deepseek-ai/DeepSeek-LLM) 与 `lm-evaluation-harness` 的模块化设计思想，但**完全独立实现**：核心依赖仅 `pyyaml` + `requests`，所有指标（BLEU/ROUGE/F1/Accuracy）均为纯 Python 实现，无原生编译依赖，在 Windows 10/11 上可直接安装运行。

- **Python ≥ 3.9**（已在 Windows 11 / Python 3.12、3.14 验证）
- **Windows 安全并发**：默认 `ThreadPoolExecutor`，可选 `asyncio`，不使用任何 `os.fork`
- **断点续跑**：每个样本结果落盘 JSONL，重启自动跳过已完成项
- **三种内置任务**：多项选择（MMLU 风格）、开放式问答、文本生成
- **四种内置指标** + 自定义指标注册装饰器
- **JSON / CSV / Markdown** 三份报告自动生成
- **Web 仪表盘**：运行后自动打开浏览器，像 DeepSeek harness 一样可视化查看结果
- **交互式 Harness**：`fylharness serve` 启动带对话界面的 Web 应用——聊天 Playground + Workspace 模型管理 + Agent 工作区 + 在线评估 + 结果查看，五合一
- **Agent 工作区**：指定本地项目目录，用自然语言描述任务，模型像 agent 一样自主读文件、写代码、跑命令完成复杂任务

---

## 目录

1. [快速开始](#1-快速开始)
2. [项目结构](#2-项目结构)
3. [核心模块实现详解](#3-核心模块实现详解)
4. [配置说明](#4-配置说明)
5. [自定义扩展指南](#5-自定义扩展指南)
6. [Windows 兼容性处理](#6-windows-兼容性处理)
7. [性能优化与扩展性](#7-性能优化与扩展性)
8. [交互式 Harness（对话界面）](#8-交互式-harness对话界面)
9. [Web 仪表盘](#9-web-仪表盘)
10. [常见问题](#10-常见问题-faq)

---

## 1. 快速开始

### 安装

```bash
# 克隆或进入项目目录
cd fylharness

# 安装核心依赖（仅需两个包）
python -m pip install -r requirements.txt

# （可选）作为可安装包安装，获得 fylharness 命令行入口
python -m pip install -e .
```

可选 extras：

```bash
python -m pip install -e ".[async]"   # 启用 asyncio 并发执行器（需要 aiohttp）
python -m pip install -e ".[hf]"     # 启用 HuggingFace datasets 适配器
python -m pip install -e ".[dev]"     # 安装 pytest 用于跑测试
```

### 3 分钟跑通示例（完全离线，无需任何 LLM 服务器）

仓库内置一个 `mock` 模型后端，不发起任何网络请求，可在干净机器上立即验证整条流水线：

```bash
# 方式一：Python 脚本
python examples/run_demo.py

# 方式二：CLI（安装包后可用 fylharness 命令；未安装可用 python -m fylharness.cli）
fylharness run examples/config_mock.yaml
```

输出示例：

```
[mock-model] mmlu_demo: samples=8 errors=0 accuracy=0.1250
[mock-model] openqa_demo: samples=8 errors=0 accuracy=0.0000, f1=0.0000
[mock-model] gsm8k_demo: samples=8 errors=0 accuracy=0.0000, f1=0.0000
[mock-lucky] mmlu_demo: samples=8 errors=0 accuracy=0.6250
...

Reports written to: outputs/demo_mock
```

报告目录下会生成三份文件：`results.json`（完整数据）、`summary.csv`（汇总表）、`report.md`（人类可读摘要）。

### 启动交互式 Harness（对话界面，像 DeepSeek 一样）

```bash
# 启动带对话界面的 Web 应用，自动打开浏览器
fylharness serve examples/config_mock.yaml

# 指定端口 / 不自动开浏览器
fylharness serve examples/config_mock.yaml --port 8080 --no-browser
```

浏览器打开后有四个 Tab：
- **Playground** — 选模型后直接对话，响应逐字流式输出（SSE），与 DeepSeek 网页体验一致
- **Workspace** — 手动添加/删除模型，配置推理强度（Reasoning Effort）、Temperature、Max Tokens
- **Agent** — 指定本地项目目录 + 自然语言任务，模型像 agent 自主读文件、写代码、跑命令
- **Evaluate** — 选模型 + 任务 + 样本上限，点击 Run，进度条与逐样本预测实时刷新
- **Results** — 浏览任意历史评估结果，含指标矩阵 + 逐样本下钻

### 运行后自动打开 Web 仪表盘

像 DeepSeek harness 一样，FylHarness 可以在评估完成后自动启动一个本地 Web 服务器并打开浏览器，可视化查看结果：

```bash
# 评估完成后自动打开浏览器仪表盘（服务器会一直运行，按 Ctrl+C 停止）
fylharness run examples/config_mock.yaml --open

# 也可以单独查看任意已有结果目录
fylharness web outputs/demo_mock
fylharness web outputs/demo_mock --port 8080   # 指定端口
```

仪表盘功能：模型 × 任务指标矩阵（颜色分级）、多模型对比柱状图、逐样本预测 vs 参考答案下钻（含错误高亮、全文过滤、仅看错误项）。详见 [第 8 节](#8-web-仪表盘)。

### 对接真实 LLM 服务器

```bash
# 设置 API Key（敏感信息绝不写入配置文件）
set OPENAI_API_KEY=sk-xxxx          # Windows cmd
$env:OPENAI_API_KEY="sk-xxxx"       # PowerShell

# 编辑 examples/config_openai.yaml 指向你的端点，然后运行
fylharness run examples/config_openai.yaml --limit 4
```

`config_openai.yaml` 内同时给出了 OpenAI 官方 API 与本地服务器（vLLM/Ollama）两种写法模板。

---

## 2. 项目结构

```
fylharness/
├── README.md                      # 本文档
├── requirements.txt              # 核心依赖
├── pyproject.toml                # 打包配置 + console_scripts 入口
├── fylharness/                   # 主包
│   ├── __init__.py               # 公共 API 导出，导入即注册内置组件
│   ├── config.py                 # 配置加载、环境变量插值、数据类
│   ├── runner.py                 # 评估编排：并发、断点续跑、聚合
│   ├── reporting.py              # JSON/CSV/Markdown 报告生成
│   ├── results.py                # SampleResult / TaskResult 数据类（解耦循环导入）
│   ├── web.py                    # 结果仪表盘：自包含 HTML + 标准库 HTTP 服务器
│   ├── server.py                 # 交互式 Harness：Flask 应用（聊天 Playground + Workspace + Agent + 在线评估 + 结果）
│   ├── agent.py                  # Agent 循环：ReAct 式自主任务执行（思考→工具调用→观察→迭代）
│   ├── tools.py                  # Agent 工具：list_dir/read_file/write_file/edit_file/run_command（沙箱化）
│   ├── cli.py                    # argparse 命令行入口（run/list/show/web/serve）
│   ├── tasks/                    # 任务抽象与内置任务
│   │   ├── __init__.py           # 导入即注册内置任务
│   │   ├── base.py               # Task 抽象基类 + 注册表 + 装饰器
│   │   ├── multiple_choice.py    # MMLU 风格多项选择任务
│   │   ├── text_generation.py    # 指令式文本生成任务
│   │   └── open_qa.py            # 短答案开放式问答
│   ├── data/                     # 数据集加载
│   │   ├── __init__.py
│   │   └── adapters.py           # LocalAdapter / HuggingFaceAdapter / InlineAdapter + 注册表
│   ├── models/                   # 模型接口
│   │   ├── __init__.py
│   │   ├── base.py               # Model 抽象基类 + ModelResponse
│   │   ├── openai_api.py         # OpenAI 兼容 HTTP 客户端（重试+退避+限流）
│   │   ├── mock.py               # 离线 mock 模型（演示/测试用）
│   │   └── registry.py          # 模型类型工厂 build_model()
│   └── metrics/                  # 评估指标
│       ├── __init__.py           # 导入即注册内置指标
│       ├── base.py               # Metric 数据类 + 注册表 + 装饰器
│       ├── accuracy.py           # 准确率
│       ├── f1.py                 # F1（token 重叠 + 按标签宏平均）
│       ├── bleu.py               # BLEU-4（纯 Python，含 brevity penalty）
│       └── rouge.py              # ROUGE-1/2/L（纯 Python LCS 实现）
├── examples/                     # 示例
│   ├── config_mock.yaml          # 离线 demo 配置（mock 模型）
│   ├── config_openai.yaml       # 真实 OpenAI 兼容端点配置
│   ├── run_demo.py               # 一键运行离线 demo
│   └── data/
│       ├── mmlu_sample.json      # 8 条多项选择题样例
│       ├── open_qa_sample.json    # 8 条开放式问答样例
│       └── gsm8k_sample.json      # 8 条数学题样例
└── tests/
    └── test_basic.py             # 离线 smoke 测试（pytest）
```

**模块职责一句话总结**：

| 模块 | 职责 |
|---|---|
| `config.py` | 把 YAML/JSON 配置 + 环境变量解析为强类型 `HarnessConfig` |
| `tasks/` | 定义"样本→prompt→预测→参考答案"的转换逻辑 |
| `data/adapters.py` | 把数据集规格（文件/HF/内联）变成统一的 `list[dict]` |
| `models/` | 把 prompt 变成模型补全文本，封装重试/并发/限流 |
| `metrics/` | 把 `(预测, 参考)` 列表变成数值分数 |
| `runner.py` | 串联上述模块，跑完所有 (model, task) 组合并收集结果 |
| `reporting.py` | 把结果序列化为三种格式 |
| `web.py` | 把 `results.json` 渲染成自包含 HTML 仪表盘，标准库 HTTP 服务器托管，自动开浏览器 |
| `server.py` | 交互式 Harness：Flask 应用，含聊天 Playground（SSE 流式）、Workspace（模型管理）、Agent（自主任务执行）、在线评估、结果查看五个界面 |
| `agent.py` | ReAct 循环：构建系统提示词 → 让模型输出 JSON 工具调用 → 执行工具 → 把结果喂回模型 → 迭代直到完成 |
| `tools.py` | 沙箱化工具集：list_dir/read_file/write_file/edit_file/run_command/finish，所有路径限制在工作区内 |
| `cli.py` | 提供 `fylharness run/list/show/web/serve` 命令 |

**数据流**：

```
config.yaml ──load_config──▶ HarnessConfig
                                    │
                         Runner 遍历 (model, task) 对
                                    │
   ┌────────────────────────────────┼────────────────────────────────┐
   ▼                                ▼                                ▼
Task.load_dataset()         Model.generate(payload)          metrics[...](records)
(Adapter: 文件/HF)           (HTTP/async，重试退避)            (accuracy/f1/...)
   │ sample dict                ▲ prompt                         │
   └─▶ Task.build_prompt()──────┘                                ▼
                            Task.postprocess(resp) ─▶ SampleResult ─▶ checkpoint JSONL
                                                                          ▼
                                                            ReportWriter ─▶ results.json / summary.csv / report.md
```

---

## 3. 核心模块实现详解

### 3.1 任务抽象与注册（`fylharness/tasks/base.py`）

`Task` 是评估的基本单元。它只关心三件事：怎么把样本变成 prompt、怎么把模型输出变成预测、参考答案是什么。数据加载、模型调用、指标计算都委托给其他模块，所以新增一个任务通常只需 ~40 行。

核心接口签名：

```python
class Task(ABC):
    name: str = ""  # 注册表键，子类必须设置

    def __init__(self, config: TaskConfig, harness_config=None) -> None: ...
    def load_dataset(self) -> list[dict]: ...        # 委托给 DataAdapter
    @abstractmethod
    def build_prompt(self, sample: dict) -> dict:   # 返回 {"prompt": str, "messages": [...]}
        ...
    @abstractmethod
    def postprocess(self, raw_output: str, sample: dict) -> Any: ...
    @abstractmethod
    def reference(self, sample: dict) -> Any: ...
    def make_record(self, sample, prediction, raw_output) -> SampleRecord: ...
    def evaluate(self, records: list[SampleRecord]) -> dict[str, MetricValue]: ...
```

注册机制是"装饰器 + 全局字典"，与指标、适配器保持一致的模式：

```python
@register_task
class MyTask(Task):
    name = "my_task"
    ...
```

之后在配置里 `type: my_task` 即可引用。内置任务在 `fylharness/tasks/__init__.py` 被 import 时自动注册，所以用户只要 `import fylharness.tasks` 即可获得全部内置任务。

**多项选择任务的关键逻辑**（`multiple_choice.py`）：prompt 末尾以 `Answer:` 结束，`postprocess` 用正则 `\b([A-Z])\b` 抓取第一个独立大写字母作为答案，与 gold letter 比对。支持 few-shot：从数据集头部取 N 条作为范例拼进 prompt。

### 3.2 数据集管理（`fylharness/data/adapters.py`）

适配器把"数据集规格"变成 `list[dict]`。三个内置适配器：

| 适配器 | 说明 | 依赖 |
|---|---|---|
| `local` | JSON / JSONL / CSV 本地文件，支持 `field_map` 字段重命名 | 无 |
| `huggingface` | HuggingFace Hub 数据集，`name/subset/split/limit` | 可选 `datasets` |
| `inline` | 配置文件内联样本，适合 demo/测试 | 无 |

相对路径解析基于配置文件所在目录（通过 `resolve_path(path, reference=config.source)`），而非 CWD，这样无论从哪里运行命令都能找到数据文件。

自定义适配器：

```python
from fylharness.data import DataAdapter, register_adapter

@register_adapter
class MyAdapter(DataAdapter):
    name = "my_source"
    def load(self, spec, reference=None):
        ...  # 返回 list[dict]
```

### 3.3 模型接口适配（`fylharness/models/`）

统一接口只有两个方法：`generate(payload)`（同步）和 `agenerate(payload)`（异步），返回 `ModelResponse(text, prompt_tokens, completion_tokens, finish_reason, latency_ms, raw, error)`。

`OpenAICompatibleModel` 是默认实现，要点：

- **双协议**：`chat: true` 走 `/chat/completions`（messages 数组），`chat: false` 走 `/completions`（prompt 字符串）。任务 `build_prompt` 同时产出两种格式，单份任务可服务于任一协议。
- **指数退避重试**：对 408/409/429/500/502/503/504 重试，退避 `min(uniform(0, base·2^attempt), 60s)` 带全抖动，尊重 `Retry-After` 响应头。
- **并发限流**：异步路径用 `asyncio.Semaphore(concurrency)` 按模型限流；同步路径由 Runner 的 `ThreadPoolExecutor(max_workers)` 控制。
- **API Key 解析**：`resolved_api_key()` 优先环境变量 `api_key_env`，其次配置字面量 `api_key`。本地服务器无 key 时用占位符 `"EMPTY"` 并告警。

模型工厂 `build_model(config)` 根据 `config.type` 分发（`openai` / `mock`），新增后端只需在 `models/registry.py` 注册。

### 3.4 评估指标（`fylharness/metrics/`）

每个指标是一个被 `Metric` 数据类包装的纯函数，接收 `list[SampleRecord]`，返回 `float` 或 `dict[str, float]`（如 ROUGE）。

```python
@register_metric("my_metric", needs=("prediction", "reference"))
def my_metric(samples):
    return sum(...) / len(samples)
```

`needs` 字段声明该指标依赖样本记录里的哪些键，`Task.evaluate` 会在调用前校验，缺字段时立即报清晰错误而非静默算错。

内置指标均为纯 Python（无 nltk/sacrebleu 依赖）：
- `accuracy`：归一化后精确匹配占比
- `f1`：token 集合重叠的 macro-F1；另有 `f1_macro_by_label` 按类别宏平均
- `bleu`：corpus 级 BLEU-4，含 brevity penalty 与全抖动退避
- `rouge`：ROUGE-1/2/L，ROUGE-L 用动态规划求 LCS

### 3.5 运行与调度（`fylharness/runner.py`）

`Runner` 是唯一需要直接调用的执行对象。核心循环（伪代码）：

```python
for model_cfg in config.models:
    model = build_model(model_cfg)
    for task_cfg in config.tasks:
        task = get_task(task_cfg.type)(task_cfg, config)
        samples = task.load_dataset()   # 应用 limit / sample_range
        checkpoint = load_checkpoint(...) if config.run.resume else {}
        todo = [(i, s) for i, s in samples if i not in checkpoint]
        results = execute(model, task, todo)   # ThreadPoolExecutor 或 asyncio
        append_to_checkpoint(results)          # 每个 SampleResult 落盘 JSONL
        metrics = task.evaluate(records)
```

**断点续跑**：每个 `(model, task)` 对应一个 `checkpoints/{model}__{task}.jsonl` 文件，逐条 append 写入。重启时先读取已完成样本索引，已完成的直接从缓存复用，未完成的才调用模型。即使中途 Ctrl+C，已写条目也不会丢失。

**并发选择**：`run.executor: thread`（默认）用 `ThreadPoolExecutor`，对 HTTP 密集型负载足够且最稳；`async` 用 `asyncio` + `aiohttp`，单进程可撑更高并发。两者在 Windows 上均安全。

**Python API**：

```python
from fylharness import run
summary = run("examples/config_mock.yaml")          # 默认 thread 执行器
summary = run("config.yaml", executor="async")      # 强制 async
```

### 3.6 结果记录与报告（`fylharness/reporting.py`）

`ReportWriter.write(results, metadata, save_predictions)` 产出三份文件：

- **`results.json`**：完整结构，含 `metadata`（开始时间、Python 版本、平台、git commit、配置源、tags、seed）和每个 `(model, task)` 的指标 + 每样本预测详情。是后续分析的真相源。
- **`summary.csv`**：一行一个 `(model, task)`，列含 `num_samples/errors/latency/tokens` + 扁平化后的所有指标键（如 `rouge.rouge_1`），适合 Excel 查看。
- **`report.md`**：按模型分组的 Markdown 表格，可直接贴进 PR/README。

---

## 4. 配置说明

配置为单个 YAML 或 JSON 文件，顶层三个块：`run`、`models`、`tasks`。

### `run` 块（全局执行选项）

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `output_dir` | str | `outputs` | 结果输出目录（相对路径基于 CWD） |
| `log_level` | str | `INFO` | 日志级别 |
| `executor` | str | `thread` | `thread` 或 `async` |
| `max_workers` | int | 8 | 总并发上限 |
| `save_predictions` | bool | true | JSON 是否含每样本预测 |
| `resume` | bool | true | 是否启用断点续跑 |
| `seed` | int | 42 | 随机种子 |
| `tags` | list | `[]` | 自由标签，写入报告元数据 |

### `models` 块（列表）

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `name` | str | 必填 | 模型名（报告与 checkpoint 文件名用） |
| `type` | str | `openai` | `openai`（HTTP）或 `mock`（离线） |
| `base_url` | str | `http://127.0.0.1:8000/v1` | API 根地址 |
| `api_key` | str | `""` | 字面量 key（不推荐） |
| `api_key_env` | str | `OPENAI_API_KEY` | 读 key 的环境变量名（推荐） |
| `model` | str | `=name` | 实际模型 ID |
| `chat` | bool | true | true→Chat Completions，false→Completions |
| `max_tokens` | int | 256 | 最大生成长度 |
| `temperature` | float | 0.0 | 采样温度（评估建议 0） |
| `timeout` | float | 120 | 单请求超时秒数 |
| `max_retries` | int | 5 | 重试次数 |
| `retry_backoff` | float | 2.0 | 指数退避基数 |
| `concurrency` | int | 4 | 该模型并发上限 |
| `extra` | dict | `{}` | 透传到请求 body 的额外字段 |

**环境变量插值**：配置里任何字符串都可用 `${VAR}` 或 `${VAR:-default}` 语法引用环境变量，加载时自动替换。例如 `api_key: ${OPENAI_API_KEY}` 会在运行时从环境读取，未设则空字符串。这保证密钥不进配置文件。

### `tasks` 块（列表）

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `name` | str | 必填 | 任务实例名（报告用） |
| `type` | str | 必填 | 任务注册键，如 `multiple_choice` |
| `dataset` | dict | `{}` | 数据集规格，见下 |
| `params` | dict | `{}` | 透传给任务构造函数的参数 |
| `metrics` | list | `[]` | 指标名列表，如 `[accuracy, f1]` |
| `limit` | int? | - | 每任务样本上限（调试用） |
| `sample_range` | str? | - | 切片，如 `0:100` 或 `::2` |

`dataset` 子字段：

```yaml
dataset:
  adapter: local              # local / huggingface / inline
  path: data/mmlu.json        # 相对配置文件目录
  format: json                # json / jsonl / csv（省略则按扩展名）
  field_map: {answer: reference}  # 重命名列
```

完整示例见 `examples/config_mock.yaml` 与 `examples/config_openai.yaml`。

---

## 5. 自定义扩展指南

### 5.1 添加新任务

```python
# my_tasks.py
from fylharness import Task, register_task

@register_task
class CodeCompletionTask(Task):
    name = "code_completion"

    def build_prompt(self, sample):
        prompt = f"Complete this function:\n{sample['prompt']}\n"
        return {"prompt": prompt, "messages": [{"role": "user", "content": prompt}]}

    def postprocess(self, raw_output, sample):
        return raw_output.split("```")[0].strip()   # 取代码块

    def reference(self, sample):
        return sample["canonical_solution"]
```

配置引用：

```yaml
tasks:
  - name: humaneval
    type: code_completion          # 对应 cls.name
    dataset: {adapter: local, path: data/humaneval.jsonl}
    metrics: [bleu, rouge]
    params: {max_lines: 20}
```

运行前确保你的任务模块被 import（在入口脚本顶部 `import my_tasks`，或在 `fylharness/tasks/__init__.py` 里加一行）。

### 5.2 添加新指标

```python
from fylharness import register_metric

@register_metric("pass_at_1", needs=("prediction", "reference"))
def pass_at_1(samples):
    passed = sum(1 for s in samples if s["prediction"].strip() == s["reference"].strip())
    return passed / max(1, len(samples))
```

指标可返回 dict（如 ROUGE），会自动扁平化为 `pass_at_1.sub_key`。

### 5.3 添加新数据格式

实现 `DataAdapter` 并 `@register_adapter` 即可（见 3.2 节）。也可不改代码，直接用 `local` 适配器 + `field_map` 适配非标准 JSON。

### 5.4 添加新模型后端

```python
from fylharness.models import Model, ModelResponse, register_model_type
from fylharness.config import ModelConfig

@register_model_type("transformers")
class TransformersModel(Model):
    def __init__(self, config: ModelConfig):
        # 加载本地 HF 模型...
        ...
    def generate(self, payload):
        ...  # 返回 ModelResponse(text=...)
```

配置 `type: transformers` 即用。这让 harness 未来可支持本地 GPU 推理而无需改 Runner。

---

## 6. Windows 兼容性处理

FylHarness 从设计伊始就保证 Windows 10/11 原生可运行，所有方案均经过实测：

### 6.1 并发

- **默认 `ThreadPoolExecutor`**：HTTP 评估是 I/O 密集型，线程池完全够用，且避开了 `multiprocessing` 在 Windows 上的 `spawn` 启动开销与 pickle 序列化限制。GIL 对纯网络 I/O 无影响。
- **可选 `asyncio`**：Python 3.9+ 在 Windows 上默认 `ProactorEventLoop`，原生支持，无需 `WindowsSelectorEventLoopPolicy` hack。
- **绝不使用** `os.fork`、`signal.SIGKILL` 等 Unix-only 特性。

验证方法：

```bash
fylharness run examples/config_mock.yaml                    # 默认 thread
fylharness run examples/config_mock.yaml --executor async   # async 路径
```

两者均应无报错完成（需 `pip install aiohttp`）。

### 6.2 路径

- 全部使用 `pathlib.Path`，避免手拼字符串与 `/` `\` 混淆。
- 配置中的相对数据路径基于**配置文件目录**解析（`resolve_path`），与 CWD 解耦。
- 文件读写统一 `encoding="utf-8"`（CSV 用 `utf-8-sig` 以兼容 Excel）。

### 6.3 编码

- 所有文件 I/O 显式指定 `encoding="utf-8"`，规避 Windows 默认 GBK 控制台导致的中文乱码。
- JSON 写入用 `ensure_ascii=False`，中文预测结果可读。

### 6.4 依赖

核心依赖仅 `pyyaml`、`requests`，二者在 PyPI 提供 Windows 轮子（`cp39`-`cp314` 全覆盖）。`aiohttp`、`datasets` 为可选 extras，按需安装。所有指标纯 Python，无 C 扩展、无需编译器。

### 验证命令

```bash
python -m pytest tests/ -q          # 8 项离线测试全过
python examples/run_demo.py        # 端到端离线跑通
```

---

## 7. 性能优化与扩展性

### 当前已具备

- **并发**：线程池/异步双引擎，每模型独立并发预算。
- **断点续跑**：JSONL 增量落盘，长任务可安全中断恢复。
- **内存友好**：本地大 JSONL 流式逐行读；HF 适配器按 split 加载后转 list。
- **指标纯 Python**：零原生依赖，冷启动快。

### 未来扩展方向

1. **更多模型后端**：通过 `register_model_type` 接入本地 `transformers` / `vLLM` 直连，绕过 HTTP。Runner 与指标层无需改动。
2. **分布式运行**：当前单进程。可引入任务分片：把样本索引区间分给多个 worker 进程（用 `spawn`），各自写独立 checkpoint，最后合并 `results.json`。Runner 的 `sample_range` 字段已为此预留。
3. **批量推理**：`Model` 接口可扩展 `batch_generate`，对支持 server-side batching 的 vLLM 一次发多条，吞吐显著提升。
4. **指标丰富化**：可按需接入 `sacrebleu` / `evaluate` 作为可选依赖，在自定义指标函数里调用，与内置指标共存。
5. **结果对比**：基于 `results.json` 写一个 `fylharness compare` 子命令，生成多模型横向对比表。

### 设计取舍说明

- **未集成 `lm-evaluation-harness` 全量任务**：保持轻量，避免数百 MB 依赖与复杂 YAML 继承。用户按 5.1 节几十行即可加自定义任务。
- **BLEU/ROUGE 自实现而非调 sacrebleu**：满足"核心依赖尽量少 + Windows 零编译"要求；研究级复现可替换为 sacrebleu（5.2 节）。
- **默认同步而非全异步**：`requests` 比 `aiohttp` 更易调试且无额外依赖；HTTP 评估线程池足够，把异步作为 opt-in。

---

## 8. 交互式 Harness（对话界面）

`fylharness serve` 启动一个 Flask Web 应用，提供**真实的对话界面**——像 DeepSeek 网页一样在浏览器里与模型对话、管理模型配置、Agent 自主任务、交互式跑评估、查看结果，五合一。

### 启动

```bash
fylharness serve examples/config_mock.yaml                    # 默认 127.0.0.1:5000，自动开浏览器
fylharness serve examples/config_openai.yaml --port 8080       # 对接真实模型
fylharness serve config.yaml --no-browser                      # 不自动开浏览器（远程场景）
```

启动后浏览器自动打开 `http://127.0.0.1:5000/`，页面有五个 Tab：Playground（对话）、Workspace（模型管理）、Agent（自主任务）、Evaluate（在线评估）、Results（结果查看）。

### 8.1 Playground（对话界面）

像 DeepSeek / ChatGPT 一样的聊天界面：

- 左侧边栏选择配置里定义的任意模型，可设置 Reasoning Effort / Temperature / Max Tokens
- 打开页面即可直接输入，Enter 发送（Shift+Enter 换行）；首条消息自动创建会话，无需先点 "+ New Conversation"
- 左侧会话列表支持新建 / 切换 / 删除（✕）会话，会话标题自动取首条消息；会话保存在浏览器内存中，刷新页面后清空
- 模型响应**逐字流式输出**（Server-Sent Events），不是等满才显示
- 支持多轮对话（完整 message history 发给模型）

技术实现：`/api/chat` POST 端点用 Flask 的 `stream_with_context` 逐 chunk 返回 SSE 数据流。OpenAI 兼容模型走真正的 `stream: true` 协议（每 token 一个 delta）；mock 模型逐词模拟流式。

### 8.2 Evaluate（在线评估）

不写命令行，直接在浏览器里跑评估：

- 选模型 + 任务 + 样本上限
- 点 Run，后台线程开始执行
- **进度条实时更新**（`done/total`），每完成一个样本就刷新
- 实时显示最近 5 个样本的预测 vs 参考答案
- 完成后显示指标分数（如 `accuracy=0.5000`）

技术实现：`/api/evaluate` POST 启动后台线程跑 `Runner._infer_one`，`/api/runs/<id>` GET 轮询进度（500ms 间隔），前端 `setInterval` 刷新。

### 8.3 Results（结果查看）

- 下拉列出 `output_dir` 下所有历史评估（自动发现 `results.json`）
- 选中后展示：元信息（时间/平台/git commit）、模型×任务指标矩阵（颜色分级）、逐样本预测 vs 参考下钻（前 50 条，错误行红底）

### 8.4 Python API

```python
from fylharness import load_config, HarnessServer

cfg = load_config("examples/config_mock.yaml")
server = HarnessServer(cfg)
server.run(port=5000)          # 阻塞，Ctrl+C 退出
```

### 8.5 对接真实模型

把 `config_mock.yaml` 换成指向真实端点的配置即可，无需改任何代码：

```bash
set OPENAI_API_KEY=sk-xxxx
fylharness serve examples/config_openai.yaml
```

浏览器里的 Playground 立刻能和 GPT-4o / Qwen / DeepSeek 等真实模型对话，Evaluate Tab 能跑真实评估。

### 8.6 Agent 工作区（自主任务执行）

Agent Tab 让 harness 像 AI 编程助手一样工作：指定一个本地项目目录 + 用自然语言描述任务，模型会自主探索文件、写代码、跑命令、迭代直到完成。

**使用方式**：
1. 在 Agent Tab 填写：模型、工作区目录（如 `C:\projects\my-app`）、最大步数、任务描述
2. 点击 Run Agent，右侧以**步骤卡片**实时展示执行轨迹：每一步的 🧠 思考过程、⚡ 执行步骤（工具名 + 格式化 JSON 参数）、👁 观察结果（超过 600 字符自动折叠，点击标题展开/收起）
3. 顶部状态栏实时显示运行状态、当前步数 / 总步数与已用时；任务完成后显示绿色总结卡片；可随时点 Stop 中止

**任务示例**：
```
任务：Read main.py, find the function that parses CLI args, add a --verbose flag, then run `python main.py --help` to verify.
```

```
任务：这个项目有个 bug 导致 pytest 失败。运行测试，读报错的文件，修复，再跑一次测试确认通过。
```

**工具集**（沙箱化，路径限制在工作区内）：
- `list_dir(path)` — 列目录
- `read_file(path)` — 读文件（带行号，自动截断长文件）
- `write_file(path, content)` — 创建/覆盖文件
- `edit_file(path, old, new)` — 精确替换文件中的文本块
- `run_command(command)` — 在工作区目录执行 shell 命令（60s 超时，Windows 安全）
- `finish(summary)` — 标记任务完成

系统提示词中的工具目录会带上述参数签名，模型无需猜测参数名即可正确调用。

**安全机制**：
- 所有路径操作拒绝绝对路径和 `..` 越界，强制限制在工作区根目录内
- `run_command` 屏蔽危险命令（`rm -rf /` 等），60 秒硬超时
- 最大步数可配（默认 25），防止失控循环
- 用户可随时点 Stop 中止运行中的 agent

**对接真实模型**：Agent 需要能输出 JSON 工具调用的模型（如 DeepSeek-V3/R1、GPT-4o、Qwen2.5-Coder、Claude）。Mock 模型只用于验证 UI 流程：

```bash
set OPENAI_API_KEY=sk-xxxx
fylharness serve examples/config_openai.yaml
# 然后在浏览器 Agent Tab 选真实模型
```

**技术实现**：[agent.py](fylharness/agent.py) 实现 ReAct 循环——构建含工具目录的系统提示词 → 模型输出 `{"thought":..., "tool":..., "args":...}` JSON → [tools.py](fylharness/tools.py) 执行工具 → 结果喂回模型 → 重复直到 `finish` 或步数耗尽。每一步通过 SSE 流式推送到浏览器，用户实时看到 agent 的思考链路。

---

## 9. Web 仪表盘

FylHarness 内置一个零依赖的 Web 结果查看器（`fylharness/web.py`），使用 Python 标准库 `http.server` 提供服务，生成自包含 HTML（内联 CSS+JS，无 CDN，离线可用），运行后自动打开浏览器，体验与 DeepSeek 评估 harness 一致。

### 启动方式

```bash
# 方式一：评估完成后自动打开（推荐）
fylharness run config.yaml --open

# 方式二：查看任意已有结果目录（不重新评估）
fylharness web outputs/demo_mock
fylharness web outputs/demo_mock --port 8080 --no-browser
```

`--open` 会在评估写完 `results.json` 后启动服务器并打开浏览器；服务器持续运行直到按 `Ctrl+C`，期间页面会实时读取最新的 `results.json`（适合边改配置边查看）。

### Python API

```python
from fylharness import run, serve_dashboard

# 跑完评估后直接启动仪表盘
summary = run("config.yaml")
serve_dashboard(summary["output_dir"])          # 阻塞，按 Ctrl+C 退出

# 或单独查看已有结果
serve_dashboard("outputs/demo_mock", port=8080, open_browser=True)
```

### 仪表盘功能

页面是一个深色主题的单页应用，所有渲染在浏览器端完成（数据嵌入 HTML，无需后端 API）：

| 区域 | 功能 |
|---|---|
| **顶部元信息** | 开始时间、平台、Python 版本、git commit、配置源、tags、seed |
| **Summary 矩阵** | 模型 × 任务表格，每个指标单元格按分数颜色分级（绿 ≥0.75 / 黄 ≥0.4 / 红 <0.4），含样本数、错误数、延迟、token 用量 |
| **Model Comparison** | 多模型时按主指标绘纯 CSS 柱状图横向对比，一眼看出哪个模型更强 |
| **Per-Sample 下钻** | 按任务分 Tab，列出每个样本的预测 vs 参考，错误行红底高亮；支持模型下拉过滤、全文搜索、仅看错误项；点击可展开原始模型输出 |

整页可 `Ctrl+S` 存成单个 `.html` 文件分享给同事，无需服务器即可打开查看。

### 技术取舍

- **不用 Flask / FastAPI**：标准库 `http.server.ThreadingHTTPServer` 已够用，零额外依赖，Windows 原生支持。
- **不用前端框架 / CDN**：纯 DOM 操作的内联 JS，离线可跑，页面 < 50KB。
- **数据嵌入而非 fetch**：`results.json` 直接内联到 `<script>` 里，避免 CORS / 路径问题，单文件即可分享。
- **多线程服务器**：`ThreadingHTTPServer` 处理并发请求，刷新页面时不会阻塞。

---

## 10. 常见问题 (FAQ)

**Q: 报 `ModuleNotFoundError: No module named 'yaml'`**  
A: 未装依赖。`pip install -r requirements.txt`。

**Q: 评估中途 Ctrl+C，重跑会从头开始吗？**  
A: 不会。默认 `resume: true`，已完成的样本从 `outputs/.../checkpoints/*.jsonl` 恢复。强制重跑加 `--no-resume`。

**Q: 本地 Ollama/vLLM 报 401 未授权？**  
A: 本地服务器一般接受任意非空 key。配置写 `api_key: EMPTY`，或设环境变量 `OPENAI_API_KEY=any`。

**Q: 指标值都是 0？**  
A: 多半是 mock 模型在跑（开放式任务 mock 只回显文本，对不上参考答案）。换真实模型，或检查 `postprocess` 输出与 `reference` 是否可比。可用 `results.json` 里的 `samples` 逐条核对。

**Q: 如何只跑某个模型/任务？**  
A: `fylharness run config.yaml --model gpt-4o-mini --task mmlu_demo --limit 10`。

**Q: 支持非 OpenAI 协议（如 Anthropic）吗？**  
A: 当前内置仅 OpenAI 兼容。用 `register_model_type`（5.4 节）实现自定义后端即可，Runner/指标层不变。

**Q: Web 仪表盘打不开 / 浏览器没自动弹出来？**  
A: 服务器默认绑 `127.0.0.1` + 随机端口。手动看终端打印的 URL；或 `fylharness web <目录> --port 8080` 指定端口。`--no-browser` 不自动开浏览器，可手动复制 URL。Linux/无头服务器场景把 `--host 0.0.0.0` 加上即可远程访问（注意安全）。

**Q: 仪表盘能脱机查看吗？**  
A: 能。页面是单文件 HTML，所有数据内联。浏览器 `Ctrl+S` 存下来后双击即可离线打开，无需服务器。

**Q: `fylharness serve` 和 `fylharness web` 有什么区别？**  
A: `serve` 启动完整的交互式应用（聊天 Playground + 在线评估 + 结果查看，基于 Flask）；`web` 只查看已有 `results.json`（基于标准库 http.server，零依赖）。日常用 `serve`，轻量查看用 `web`。

**Q: 对话界面里能对接真实 DeepSeek / Qwen / GPT 模型吗？**  
A: 能。把 `config_mock.yaml` 换成指向真实端点的配置（`examples/config_openai.yaml` 有模板），`fylharness serve examples/config_openai.yaml` 即可在浏览器里和真实模型对话。

**Q: Agent Tab 的模型为什么一直报 "Could not parse tool call"？**  
A: Mock 模型只能输出对话文本，不会输出 JSON 工具调用，所以 agent 循环解析失败。Agent 需要能遵循 JSON 指令的真实模型（DeepSeek-V3/R1、GPT-4o、Qwen2.5-Coder、Claude 等）。配置真实端点后在 Agent Tab 选对应模型即可。

**Q: Agent 会不会越界删除工作区外的文件？**  
A: 不会。所有路径工具强制限制在指定的工作区目录内，绝对路径和 `..` 越界会被拒绝。`run_command` 也在工作区目录内执行，并屏蔽危险命令、60 秒超时。

---

## 许可证

MIT。借鉴了 DeepSeek-LLM 与 lm-evaluation-harness 的设计思想，但全部代码独立实现。
