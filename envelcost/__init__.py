"""envelcost — per-harness DeepSeek tool-call-envelope token-cost profiler.

A CLI that replays a fixed set of coding-agent tasks through several OpenAI-shaped
tool-call *envelope* configurations against DeepSeek V4 Flash, isolates the token
overhead attributable to each envelope, and projects that variance onto a fixed
on-prem GPU cluster so a 信创 ML-platform engineer can read off per-harness
seat-capacity in one line.

The named primitive is :class:`~envelcost.envelope.EnvelopeProfile` — a typed
record that separates the *envelope* token cost (the schema scaffolding a harness
wraps around every DeepSeek request) from the model's intrinsic token cost.
"""

from __future__ import annotations

__version__ = "0.6.0"

__all__ = ["__version__"]
