"""Only LLM is exported at the top level (mirrors mini-sglang's __init__.py).

Usage:
    from python import LLM
    from python.core import SamplingParams

Model internals (Qwen3ForCausalLM etc.) are not exported here — they live
under python.model.*. The top level keeps the single responsibility of
being the offline inference entry point, so `import python` does not leak
implementation details.
"""

from python.llm import LLM

__all__ = ["LLM"]
