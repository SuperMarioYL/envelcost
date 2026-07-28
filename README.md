[English](./README.en.md) | **简体中文**

<div align="right"><sub><a href="./README.en.md"><b>EN</b></a>&nbsp;&nbsp;⇄&nbsp;&nbsp;中文</sub></div>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./assets/hero-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="./assets/hero-light.svg">
    <img src="./assets/hero-light.svg" width="880" alt="envelcost — DeepSeek 工具调用 envelope token 成本投影器">
  </picture>
</p>

<p align="center"><sub>把同一 DeepSeek V4 跨编码 agent harness 的 4x token 差异，量化成信创 on-prem GPU 集群容量规划数字的 CLI。</sub></p>

<p align="center">
  <a href="./LICENSE"><img src="https://img.shields.io/github/license/SuperMarioYL/envelcost?label=license&color=0071E3" alt="License: MIT"></a>
  <a href="https://github.com/SuperMarioYL/envelcost/releases"><img src="https://img.shields.io/github/v/release/SuperMarioYL/envelcost?label=release&color=0071E3" alt="Latest release"></a>
  <a href="https://github.com/SuperMarioYL/envelcost/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/SuperMarioYL/envelcost/ci.yml?branch=main&label=CI" alt="CI status"></a>
  <img src="https://img.shields.io/badge/python-3.12%2B-3776AB?logo=python&logoColor=white" alt="Python 3.12+">
</p>

<p align="center"><strong>同一 DeepSeek V4，在 Claude Code / OpenCode / Pi 三种 harness 下烧的 token 差 4 倍——envelcost 把这个差异算成你 H100 集群撑得住几个座位。</strong></p>

---

## 目录

- [架构](#架构)
- [为什么需要它](#为什么需要它)
- [安装](#安装)
- [快速开始](#快速开始)
- [用法](#用法)
- [演示](#演示)
- [配置](#配置)
- [付费](#付费)
- [路线图](#路线图)
- [License](#license)

<h2><img src="https://api.iconify.design/tabler:topology-star-3.svg?color=%230071E3&width=24" height="22" align="absmiddle" alt=""> 架构</h2>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./assets/atlas-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="./assets/atlas-light.svg">
    <img src="./assets/atlas-light.svg" width="880" alt="架构：CLI→Runner→DeepSeek API→EnvelopeProfile store→Reporter/Projector">
  </picture>
</p>

一个 binary，一个进程，无 DB。状态是 `.envelcost/` 下的 JSON 文件。`Runner` 把一组固定编码任务通过 N 个 envelope 配置回放到 DeepSeek，把每请求 usage 捕获成 `EnvelopeProfile` 记录存盘；`Reporter` 渲染 per-envelope token 表，`Projector` 把已测 multiplier 投影成 GPU 座位容量。无 microservices，无 Kubernetes——超过 3 个进程就 scope down。

<h2><img src="https://api.iconify.design/tabler:bulb.svg?color=%230071E3&width=24" height="22" align="absmiddle" alt=""> 为什么需要它</h2>

信创 enterprise ML-platform 工程师今天把内部编码 agent 标准化在 DeepSeek V4 Flash 上，却无法从模型本身做容量规划：同一模型在 Claude Code（经 CLIProxyAPI）、OpenCode、Pi 三种 OpenAI-shaped harness 下，跑出 quality 持平的 diff 却烧掉 2–5x 不同的 token——社区 benchmark 实测 Claude-Code-wrapped-DeepSeek 比最快 harness 慢 ~4x。在「数据不出境」约束下集群不能弹性扩容吃掉最坏 harness 的开销，**harness envelope 而非模型本身**成了硬吞吐约束。ops 今天没有工具能说：给定内部 dev 舰队规模，每个 harness 选择意味着多少张 H200。envelcost 把这 ~4x 差异变成 per-harness token 成本投影，让 GPU 预算照着它定。

新命名的 primitive 是 `EnvelopeProfile`——一条类型化记录，把「归因于工具调用 envelope」的那部分 token 成本（harness 包在每个 DeepSeek 请求外的 schema 框架）从模型 intrinsic token 成本里隔离出来。`envelope_overhead_tokens` 字段正是社区 benchmark 看见却无法隔离的成本驱动；这是 5 个已 shipped DeepSeek/cost 工具（tokensched / dscache / cachepin / tokenctl / agentfuse）都不碰的层——它们都在单个 harness 的 budget/cache 内操作，`EnvelopeProfile` 比较的是跨 harness envelope。

<h2><img src="https://api.iconify.design/tabler:rocket.svg?color=%230071E3&width=24" height="22" align="absmiddle" alt=""> 安装</h2>

```bash
# 纯 Python，单命令
uv tool install envelcost        # 或：pipx install envelcost
# 从源码：uv pip install -e .
```

需要 Python ≥ 3.12。核心 `run` / `project` 路径**不需要联网、不需要 DeepSeek API key**——envelope 成本由实际序列化文本离线 tokenize 计算，可复现。可选 `pip install -e ".[deepseek]"` 装上 transformers 拿到 DeepSeek 原生 BPE 的绝对 token 数（离线 fallback 已足够复现 >2x 差异）。

<h2><img src="https://api.iconify.design/tabler:rocket.svg?color=%230071E3&width=24" height="22" align="absmiddle" alt=""> 快速开始</h2>

```bash
envelcost run                                    # 回放 5 任务 × 2 envelope，断言 >2x 差异门
envelcost project --gpus 8xH100 --seats 50       # 读 multiplier，投影座位容量 + 成本/座
envelcost report                                 # 打印 per-envelope token 表，写 .envelcost/envelcost-report.{json,md}
```

<details><summary>样例输出</summary>

```
task_id              harness                input overhead   mult
-------------------------------------------------------------------
swe-bench-mini-001   deepseek-native         3238        0  1.00x
swe-bench-mini-001   openai-shape            9102     5864  2.81x
...
m1 variance gate: 5/5 tasks above 2.0x (done bar PASSED); kill floor (1.5x) HELD

cluster: 8×NVIDIA H100 80GB  target seats: 50  total capex: ¥2,080,000
harness              mult   raw   eff   fits  deficit      ¥/seat   ¥/seat/yr
deepseek-native     1.00x    64    64    yes      -14 ¥    32,500 ¥   10,833
openai-shape        2.89x    64    22     NO      +28 ¥    94,545 ¥   31,515
on 8×H100 / 50 seats: deepseek-native fits, openai-shape does NOT fit
```

</details>

到首价值总用户 effort < 5 分钟，每步 < 60 秒。

<h2><img src="https://api.iconify.design/tabler:terminal-2.svg?color=%230071E3&width=24" height="22" align="absmiddle" alt=""> 用法</h2>

三个子命令对应 m1/m2/m3 里程碑。完整 CLI 参考 `envelcost --help`。

```bash
# 1) 三 envelope 全扫（含 Claude Code via CLIProxyAPI，严格比 openai-shape 更重）
envelcost run --harnesses deepseek-native,openai-shape,claude-code-cliproxy

# 2) 容量规划：H200 集群、80 座位——看哪个 harness 撑不住
envelcost project --gpus 4xH200 --seats 80

# 3) 联网模式（可选，schema_unverified）：用真实 DeepSeek usage 填 output_tokens
export DEEPSEEK_API_KEY=...
envelcost run --online
```

`envelcost project` 读 `.envelcost/profiles.jsonl` 里已测的 multiplier；若还没 `run`，projection 退化为 parity（1.0x）仍可跑。更多示例见 [`examples/quickstart.sh`](./examples/quickstart.sh)。

<h2><img src="https://api.iconify.design/tabler:photo.svg?color=%230071E3&width=24" height="22" align="absmiddle" alt=""> 演示</h2>

![演示](assets/demo.gif)

tape 脚本在 [`docs/demo.tape`](./docs/demo.tape)，由 [`.github/workflows/demo.yml`](./.github/workflows/demo.yml) 手动触发重渲染。本地跑：`vhs docs/demo.tape`（需 vhs + ttyd + ffmpeg）。

<h2><img src="https://api.iconify.design/tabler:adjustments.svg?color=%230071E3&width=24" height="22" align="absmiddle" alt=""> 配置</h2>

CLI 顶层键：

| 参数 | 类型 | 默认 | 含义 |
|---|---|---|---|
| `--task` | str | `swe-bench-mini` | benchmark 任务集 id |
| `--harnesses` | csv | `deepseek-native,openai-shape` | 回放的 envelope 配置（逗号分隔） |
| `--gpus` | spec | `8xH100` | GPU 集群规格，如 `4xH200` / `2xH3` |
| `--seats` | int | `50` | 目标并发编码 agent 座位数 |
| `--store` | path | `.envelcost/` | store 目录覆盖 |
| `--online` | flag | off | 经 DeepSeek API 回放（需 `DEEPSEEK_API_KEY`） |

envelope 配置（`envelcost/envelope.py`）：`deepseek-native`（baseline 1.0x，紧凑内联协议）、`openai-shape`（OpenAI function-calling envelope，verbose JSON tools 数组每轮重发）、`claude-code-cliproxy`（Claude Code 经 CLIProxyAPI，严格比 openai-shape 更重）。

GPU 目录（`envelcost/config.py`，信创 on-prem 在架）：

| GPU | 有效 tokens/s/座 | 单卡 capex（¥，未税） | 单卡功耗 |
|---|---:|---:|---:|
| H100 80GB | 3200 | 260,000 | 700W |
| H200 141GB | 3600 | 300,000 | 700W |
| H3 288GB | 5200 | 420,000 | 1000W |

<h2><img src="https://api.iconify.design/tabler:credit-card.svg?color=%230071E3&width=24" height="22" align="absmiddle" alt=""> 付费</h2>

OSS 核心（benchmark + `EnvelopeProfile` + `envelcost project` 投影）**自托管免费**（MIT）。不是「永远免费」的许诺——是自托管可跑、`数据不出境` 友好的可信度锚。

付费 = **企业版 on-prem license**：在 m3 投影器上扩展多集群 `EnvelopeProfile` 聚合 + 信创 compliance 留痕导出（给国产化 officer）+ 配额告警 + 1 年支持。形态契合「自托管免费 + 付费企业版」——因为数据不出境排斥纯 SaaS 托管，license + on-prem 是信创标配。

| 层 | 形态 | 价格（educated guess） | 适合谁 |
|---|---|---|---|
| OSS 核心 | 自托管，MIT | 免费 | 任何想复现 4x 差异、单集群容量规划的信创 ML 平台工程师 |
| 企业版 license | on-prem，多集群聚合 + 合规导出 + 告警 + 支持 | **¥48,000/年起/集群**（按座增量约 ¥1,200/座/年） | 已在 DeepSeek V4 上标准化编码 agent + 固定 on-prem 集群 + 数据不出境 的信创 ML 平台团队 |
| 自服务 SaaS tier（非信创、cloud-API devs 外圈） | hosted | ¥1,990/月（非 v0.1 主路径） | 云 API 成本优化方 |

**最小「这是我卡/对公打款」path：** design-partner 免费跑一次 `envelcost run` + `envelcost project` 出 `EnvelopeProfile` 报告 + 容量建议（OSS 免费）→ 客户要「3 个集群汇总 + 合规留痕导出」恰是付费层独有 → 升级 enterprise license → 首次对公打款 ~¥48,000/年，开增值税专票（信创企业走对公转账 + 增值税专票，不依赖 Stripe）。

<h2><img src="https://api.iconify.design/tabler:map-2.svg?color=%230071E3&width=24" height="22" align="absmiddle" alt=""> 路线图</h2>

- [x] **m1 — 复现差异**：5 任务 × 2 envelope 可复现 benchmark，>2x 差异在 ≥3/5 任务上确认（kill 门：<1.5x 即 halt abandon）。**已达成**：当前 5/5 任务 2.83–3.27x。
- [ ] **m2 — ship envelope report**：3 envelope 配置 + 完整 `EnvelopeProfile` JSON+MD 输出 + 双语 README（README.md zh + README.en.md）的双向打磨。
- [ ] **m3 — project GPU budget**：`envelcost project` 已可跑；待扩展多集群聚合预览（企业版 license 的 hook）。
- [ ] 未来：非 DeepSeek 模型（Qwen / Kimi / GLM）、通用 function-calling envelope、IDE 插件（VS Code / JetBrains）——均 deferred，见 v0.1 out-of-scope。

**Kill criteria（可证伪，非永远 ship）：** (1) m1 benchmark 跨 envelope variance <1.5x → 核心论点证伪，halt；(2) 30 天 Gitee+GitHub <20 stars 且零 design-partner 询盘 → kill；(3) Claude Code / OpenCode / Pi 上线 native DeepSeek tool-call 支持使 envelope 向 1x 收敛 → 2 季度内 kill。

<h2><img src="https://api.iconify.design/tabler:license.svg?color=%230071E3&width=24" height="22" align="absmiddle" alt=""> License</h2>

MIT，见 [`LICENSE`](./LICENSE)。发现 bug 或想加 harness 配置？开 [issue](https://github.com/SuperMarioYL/envelcost/issues) 或提 PR——尤其欢迎补一个真实 DeepSeek V4 endpoint 的可复现 benchmark gist。

<p align="center"><sub><a href="./LICENSE">MIT</a> © 2026 SuperMarioYL</sub></p>
