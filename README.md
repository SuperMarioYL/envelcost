[English](./README.en.md) · [Website](https://envelcost.lei6393.com) · [GitHub](https://github.com/SuperMarioYL/envelcost)

<picture>
  <source media="(max-width: 600px) and (prefers-color-scheme: dark)" srcset="./assets/presentation/hero-mobile-dark.svg">
  <source media="(max-width: 600px)" srcset="./assets/presentation/hero-mobile-light.svg">
  <source media="(prefers-color-scheme: dark)" srcset="./assets/presentation/hero-dark.svg">
  <img src="./assets/presentation/hero-light.svg" width="960" alt="Hero diagram">
</picture>

# envelcost

**检查工具封装带来的文本开销。**

envelcost 序列化配置的工具封装模板、计算文本 token，并记录可供容量情景使用的 profile。

## 为什么需要它

冗长工具 schema 会在任务本身之前增加输入文本。比较实际序列化表示，可以看清统计了什么，以及各表示保留了哪些 schema 信息。

- **展示被统计文本** — 可以检查确切序列化内容。
- **区分格式与模型** — 比较不同表示时可使用同一 tokenizer。
- **明确投影假设** — 容量输出仍是基于给定比值的情景。

## 架构

<picture>
  <source media="(max-width: 600px) and (prefers-color-scheme: dark)" srcset="./assets/presentation/architecture-mobile-dark.svg">
  <source media="(max-width: 600px)" srcset="./assets/presentation/architecture-mobile-light.svg">
  <source media="(prefers-color-scheme: dark)" srcset="./assets/presentation/architecture-dark.svg">
  <img src="./assets/presentation/architecture-light.svg" width="960" alt="Architecture diagram">
</picture>

ToolDef 提供紧凑与 JSON-schema 序列化。EnvelopeConfig 与任务、轮次包装组合，Tokenizer 计算 token，Runner 保存 EnvelopeProfile，Projector 按配置容量假设使用这些比值。

| 组件 | 职责 |
| --- | --- |
| `Tool definitions` | envelcost/envelope.py |
| `Envelope templates` | Configured serialization |
| `Tokenizer / profile` | tokenizer.py; runner.py |
| `Scenario report` | projector.py; report.py |

## 安装与快速上手

使用仓库清单指定的运行时版本构建，并在仓库根目录运行示例。

```bash
git clone https://github.com/SuperMarioYL/envelcost.git
cd envelcost
uv venv .venv
uv pip install --python .venv/bin/python -e .
source .venv/bin/activate
```

将一个完整 read 工具序列化为紧凑和 JSON-schema 形式，并使用同一个可用 cl100k tokenizer 统计。

```bash
.venv/bin/python examples/presentation-demo.py
```

## 实际运行示例

<picture>
  <source media="(max-width: 600px) and (prefers-color-scheme: dark)" srcset="./assets/presentation/process-mobile-dark.svg">
  <source media="(max-width: 600px)" srcset="./assets/presentation/process-mobile-light.svg">
  <source media="(prefers-color-scheme: dark)" srcset="./assets/presentation/process-dark.svg">
  <img src="./assets/presentation/process-light.svg" width="960" alt="Process diagram">
</picture>

The output shows both exact representations and their tokenizer counts; it makes no live harness performance claim.

```text
{"representation": "compact", "text": "[tool:read] Read a file\nargs: path", "tokens_same_cl100k": 12}
{"representation": "json-schema", "text": "{\"type\":\"function\",\"function\":{\"name\":\"read\",\"description\":\"Read a file\",\"parameters\":{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\"}},\"required\":[\"path\"]}}}", "tokens_same_cl100k": 37}
cl100k available: True
```

完整命令与输出保存在 [docs/demo-results.json](./docs/demo-results.json). 输入和复现代码均随仓提供。

![已有终端录制](./assets/demo.gif)

保留已有录制供参考；上方文字示例给出当前可复现的操作。

## 用法

CLI 提供以下操作。示例之外的命令需要替换成你的文件路径或标识。

```bash
envelcost run
envelcost report
# Scenario projection, not a load test:
envelcost project --gpus 8xH100 --seats 50
```

## 配置

run 接收配置封装列表和任务选择，默认是离线模板分词；可选在线模式会实际请求。Tokenizer 回退可能使用字符估算，因此比较 profile 时应保留实际可用 tokenizer 信息。

## 集成与职责分工

<picture>
  <source media="(max-width: 600px) and (prefers-color-scheme: dark)" srcset="./assets/presentation/integrations-mobile-dark.svg">
  <source media="(max-width: 600px)" srcset="./assets/presentation/integrations-mobile-light.svg">
  <source media="(prefers-color-scheme: dark)" srcset="./assets/presentation/integrations-dark.svg">
  <img src="./assets/presentation/integrations-light.svg" width="960" alt="Integrations diagram">
</picture>

以下路径已有源码实现。按任务选择输入，并把生成的结果与项目一起保存。

| 路径 | 已实现职责 |
| --- | --- |
| Task YAML | Configured task inputs |
| ToolDef | Compact and full-schema representations |
| Tokenizers | cl100k and optional DeepSeek path |
| Profile JSONL | Measured serialized-text counts |
| Capacity model | Explicit scenario assumptions |

## 限制与后续方向

- 封装模板是项目建模，不是证明当前同名调用框架行为的抓包记录。
- 紧凑形式省略 JSON 保留的递归 schema 细节；文本更小不证明工具调用行为等价。
- token 比值不能单独测量吞吐或 GPU 座位容量；容量和费用输出依赖假设系数与登记数据。

可靠容量决策需要真实封装记录、等价任务质量，以及目标环境的服务性能实测。

## 许可与贡献

许可见 [LICENSE](./LICENSE). 反馈问题时请提供最小输入、执行命令和实际输出。
