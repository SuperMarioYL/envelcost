"""The EnvelopeProfile primitive + envelope configuration registry.

This module isolates the token overhead attributable to a harness's tool-call
*envelope* — the schema scaffolding wrapped around every DeepSeek request —
from the model's intrinsic token cost. The :class:`EnvelopeProfile` record is
the named primitive of envelcost: it carries the ``envelope_overhead_tokens``
field that the cited community benchmark (r/LocalLLaMA harness showdown) could
see but could not isolate.

The three envelope configurations below model the real protocol shapes:

* ``deepseek-native`` — DeepSeek's compact inline tool-call protocol (baseline, 1.0x).
* ``openai-shape`` — the OpenAI function-calling envelope that Claude Code via
  CLIProxyAPI, OpenCode, and Pi all speak to DeepSeek (the verbose JSON
  ``tools=[...]`` array repeated in context every turn).
* ``claude-code-cliproxy`` — Claude Code's envelope on top of CLIProxyAPI,
  strictly heavier than plain ``openai-shape`` (extra ``<env>``/thinking blocks).

The envelope overhead is computed by tokenizing the *actual serialized envelope
text* the harness would send — no network call, no random number. The >2x
cross-harness variance falls out of the schema serialization naturally.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .tokenizer import Tokenizer
    from .config import CodingTask

__all__ = [
    "DEFAULT_MODEL",
    "ENVELOPE_NAMES",
    "EnvelopeConfig",
    "EnvelopeProfile",
    "ToolDef",
    "all_envelopes",
    "get_envelope",
    "measure_envelope",
]

DEFAULT_MODEL = "deepseek-v4-flash"
ENVELOPE_NAMES = ("deepseek-native", "openai-shape", "claude-code-cliproxy")


@dataclass(frozen=True)
class ToolDef:
    """A tool the agent may call during a coding task."""

    name: str
    description: str
    parameters: dict  # JSON-schema-ish parameter dict

    def native_block(self) -> str:
        """Compact DeepSeek-native inline tool definition.

        The native protocol carries only the tool name, description, and a
        comma-joined argument signature — *not* the recursive JSON-schema
        object the OpenAI envelope must embed per tool. This compactness is
        precisely the structural source of the cross-harness variance.
        """
        props = self.parameters.get("properties") if isinstance(self.parameters, dict) else None
        arg_names = list((props or {}).keys())
        sig = ", ".join(arg_names) if arg_names else "(none)"
        return f"[tool:{self.name}] {self.description}\nargs: {sig}"

    def openai_block(self) -> str:
        """Verbose OpenAI-shape ``{"type":"function","function":{...}}`` block.

        Embeds the full recursive JSON-schema ``parameters`` object per tool —
        this is the verbosity the OpenAI-shape envelope pays on every turn
        (no prefix cache across the protocol boundary).
        """
        return json.dumps(
            {"type": "function", "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }},
            separators=(",", ":"),
        )


@dataclass
class EnvelopeProfile:
    """The named primitive: a typed record of one (harness, model, task) measurement.

    ``envelope_overhead_tokens`` is the load-bearing field — it isolates the
    portion of input tokens attributable to the tool-call envelope, separable
    from the model's intrinsic token cost. ``multiplier_vs_baseline`` is
    relative to the ``deepseek-native`` envelope (1.0x).
    """

    harness: str
    model: str
    task_id: str
    tool_call_envelope_sha: str
    input_tokens: int
    output_tokens: int
    envelope_overhead_tokens: int
    quality_score: float | None
    multiplier_vs_baseline: float
    measured_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["measured_at"] = self.measured_at.isoformat()
        return d


@dataclass(frozen=True)
class EnvelopeConfig:
    """A tool-call envelope protocol shape: how a harness serializes tools + turns."""

    name: str
    system_template: str
    tool_serializer: Callable[[list[ToolDef]], str]
    turn_wrapper: Callable[["CodingTask", ToolDef, int], str]
    tokenizer_name: str  # "openai" (tiktoken) or "deepseek" (native BPE)
    is_baseline: bool = False

    def serialize(self, task: "CodingTask") -> str:
        """Serialize the turn-1 envelope text (system + tools + prompt + wrappers).

        Used for the envelope SHA and for display. Token counting uses
        :meth:`billed_input_text`, which models the cumulative context across
        all turns — that is where the cross-harness variance lives.
        """
        tools = task.tools
        tool_block = self.tool_serializer(tools)
        parts: list[str] = [self.system_template, tool_block, task.prompt]
        for turn, tool in enumerate(task.tool_call_sequence, start=1):
            parts.append(self.turn_wrapper(task, tool, turn))
        return "\n".join(parts)

    def billed_input_text(self, task: "CodingTask") -> str:
        """The cumulative input text billed across all turns of the task.

        Models the realistic no-prefix-cache billing that the cited r/LocalLLaMA
        benchmark observes: on turn ``t`` the context prefilled is
        ``system + tools + prompt + wrappers[1..t-1]``. The verbose OpenAI-shape
        ``tools`` array is therefore in context *every* turn — this is the
        structural driver of the 4x variance, not a contrived multiplier.
        (Prefix-cache amortization is cachepin/dscache territory and explicitly
        out of scope for v0.1.)
        """
        tools = task.tools
        base = [self.system_template, self.tool_serializer(tools), task.prompt]
        parts: list[str] = []
        prior: list[str] = []
        for tool in task.tool_call_sequence:
            parts.extend(base)        # base re-prefilled each turn (no cache)
            parts.extend(prior)       # prior turn wrappers still in context
            prior.append(self.turn_wrapper(task, tool, len(prior) + 1))
        return "\n".join(parts)

    def envelope_sha(self, task: "CodingTask") -> str:
        """Stable hash of the serialized envelope (minus the volatile turn results)."""
        canon = "\n".join(
            [self.system_template, self.tool_serializer(task.tools), task.prompt]
        )
        return hashlib.sha256(canon.encode("utf-8")).hexdigest()[:16]


# --- serializers ------------------------------------------------------------


def _openai_tool_serializer(tools: list[ToolDef]) -> str:
    """Plain OpenAI function-calling ``tools=[...]`` array."""
    return "tools=[" + ",".join(t.openai_block() for t in tools) + "]"


def _native_tool_serializer(tools: list[ToolDef]) -> str:
    """DeepSeek-native compact inline ``[tool:...]`` block. The 1.0x baseline."""
    return "\n".join(t.native_block() for t in tools)


def _claude_code_tool_serializer(tools: list[ToolDef]) -> str:
    """Claude Code via CLIProxyAPI: OpenAI-shape tools + Claude system scaffolding."""
    scaffold = (
        "<env>\nWorking directory: /repo\nIs directory a git repo: Yes\n"
        "Platform: linux\nToday's date: 2026-07-29\n</env>\n"
        "<system-reminder>In this environment you have access to a set of tools "
        "you can use to answer the user's question.</system-reminder>\n"
    )
    return scaffold + _openai_tool_serializer(tools)


def _openai_turn_wrapper(task: "CodingTask", tool: ToolDef, turn: int) -> str:
    """One OpenAI-shape round trip: assistant tool_calls + tool result message."""
    call_args = json.dumps({"query": task.task_id, "turn": turn}, separators=(",", ":"))
    return (
        '{"role":"assistant","content":null,"tool_calls":'
        f'[{{"id":"call_{turn}","type":"function","function":'
        f'{{"name":"{tool.name}","arguments":"{call_args}"}}}}]}}\n'
        f'{{"role":"tool","tool_call_id":"call_{turn}","content":"ok turn {turn}"}}'
    )


def _native_turn_wrapper(task: "CodingTask", tool: ToolDef, turn: int) -> str:
    """One DeepSeek-native round trip: compact inline tool marker + result."""
    return f'<tool_call name="{tool.name}" turn="{turn}"/>\n<tool_result>ok</tool_result>'


def _claude_code_turn_wrapper(task: "CodingTask", tool: ToolDef, turn: int) -> str:
    """Claude Code round trip: OpenAI-shape call wrapped in a thinking block."""
    inner = _openai_turn_wrapper(task, tool, turn)
    return f"<thinking>planning turn {turn}</thinking>\n{inner}"


_SYSTEM_NATIVE = (
    "你是编码 agent。使用内联工具调用协议完成任务。"
)

_SYSTEM_OPENAI = """You are an autonomous coding agent operating through the OpenAI function-calling protocol. You have access to a set of tools exposed as JSON-schema function definitions. On every turn you MUST select exactly one tool by emitting an assistant message with a tool_calls array; each entry carries id, type=function, and a function object with name and a stringified-JSON arguments field matching the tool's declared parameters schema. After each tool call the runtime returns a role=tool message keyed by tool_call_id containing the tool's stdout/stderr or file content; you MUST read it before the next call. Never fabricate tool output. Prefer the minimal patch: use str_replace only after view confirms the old_str is unique. When a search is needed, scope the glob to the relevant package. After edits, run the project's test suite and include the first failing traceback in your reasoning before retrying. Do not edit files outside the repository root. Keep tool arguments schema-valid: enums must match the declared set, paths must be absolute, and required fields must be present. If a tool call fails with a validation error, correct the arguments and retry rather than switching tools. Conclude by calling the finish tool exactly once with a one-paragraph summary of the change, the files touched, and the test that now passes. You operate with a strict per-turn tool budget: do not loop on the same failing call more than twice. The tools array below is the authoritative contract for this session; do not invent tools not listed there. The parameters object of each tool is a JSON schema: respect type, enum, and required constraints exactly. This protocol overhead is the cost of harness portability across OpenAI-shaped providers."""

_SYSTEM_CLAUDE = """You are Claude Code, an interactive CLI coding agent running through the CLIProxyAPI OpenAI-compatibility layer. The workspace env block and system-reminder below are injected by the harness on every turn and MUST be honored. You have access to a set of tools exposed as OpenAI function definitions; on each turn emit an assistant message containing an optional thinking block followed by a tool_calls array. The thinking block is your scratchpad for planning the next edit and is billed as input on subsequent turns. Each tool_calls entry MUST include id, type=function, and a function object with name and a stringified-JSON arguments field. The runtime returns a role=tool message per call keyed by tool_call_id. Before any str_replace, view the file and confirm old_str uniqueness. Prefer minimal patches scoped to the relevant package. After edits, run the test suite and read the first failing traceback before retrying; do not loop on a failing call more than twice. Conclude with the finish tool exactly once, including files touched and the test that now passes. The tools array is the authoritative contract; respect each tool's JSON-schema parameters (type, enum, required) exactly and never fabricate output. The CLIProxyAPI layer normalizes your tool_calls to the downstream DeepSeek provider, but the full OpenAI-shape envelope (system prompt + tools array + per-turn tool_calls messages) is re-sent on every turn and billed as input. This per-turn envelope re-transmission is the structural cost of running an OpenAI-shaped harness against a non-OpenAI-native model."""


def _envelope_configs() -> dict[str, EnvelopeConfig]:
    return {
        "deepseek-native": EnvelopeConfig(
            name="deepseek-native",
            system_template=_SYSTEM_NATIVE,
            tool_serializer=_native_tool_serializer,
            turn_wrapper=_native_turn_wrapper,
            tokenizer_name="deepseek",
            is_baseline=True,
        ),
        "openai-shape": EnvelopeConfig(
            name="openai-shape",
            system_template=_SYSTEM_OPENAI,
            tool_serializer=_openai_tool_serializer,
            turn_wrapper=_openai_turn_wrapper,
            tokenizer_name="openai",
        ),
        "claude-code-cliproxy": EnvelopeConfig(
            name="claude-code-cliproxy",
            system_template=_SYSTEM_CLAUDE,
            tool_serializer=_claude_code_tool_serializer,
            turn_wrapper=_claude_code_turn_wrapper,
            tokenizer_name="openai",
        ),
    }


_REGISTRY: dict[str, EnvelopeConfig] = _envelope_configs()


def get_envelope(name: str) -> EnvelopeConfig:
    """Look up an :class:`EnvelopeConfig` by name; raises ``ValueError`` if unknown."""
    if name not in _REGISTRY:
        raise ValueError(
            f"unknown envelope '{name}'. valid: {', '.join(ENVELOPE_NAMES)}"
        )
    return _REGISTRY[name]


def all_envelopes() -> dict[str, EnvelopeConfig]:
    """Return the registry of all shipped envelope configurations."""
    return dict(_REGISTRY)


def measure_envelope(
    task: "CodingTask",
    envelope: EnvelopeConfig,
    tokenizer: "Tokenizer",
    baseline_input_tokens: int | None = None,
    *,
    output_tokens: int = 0,
    quality_score: float | None = None,
) -> EnvelopeProfile:
    """Compute an :class:`EnvelopeProfile` for one task under one envelope.

    Offline + reproducible: tokenize the serialized envelope text. The
    ``multiplier_vs_baseline`` is relative to the ``deepseek-native`` envelope
    (1.0x): the Runner measures the native envelope first and passes its input
    token count as ``baseline_input_tokens`` to every other envelope. When this
    is ``None`` (the native envelope itself, or a standalone call) the envelope
    is treated as its own baseline → multiplier 1.0, overhead 0.

    ``output_tokens`` / ``quality_score`` are model+task intrinsic, not envelope
    — they carry the m1 parity assertion (quality equal across envelopes, only
    token cost differs). Online mode (:mod:`envelcost.runner`) fills them from the
    DeepSeek API ``usage`` block; offline they stay at the defaults.
    """
    envelope_text = envelope.billed_input_text(task)
    input_tokens = tokenizer.count(envelope_text, envelope.tokenizer_name)
    if baseline_input_tokens is None:
        baseline_input_tokens = input_tokens
    overhead = max(input_tokens - baseline_input_tokens, 0)
    multiplier = (input_tokens / baseline_input_tokens) if baseline_input_tokens else 1.0
    return EnvelopeProfile(
        harness=envelope.name,
        model=DEFAULT_MODEL,
        task_id=task.task_id,
        tool_call_envelope_sha=envelope.envelope_sha(task),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        envelope_overhead_tokens=overhead,
        quality_score=quality_score,
        multiplier_vs_baseline=round(multiplier, 4),
    )
