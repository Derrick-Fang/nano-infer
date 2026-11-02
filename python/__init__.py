"""顶层只导出 LLM（对齐 mini-sglang 的 __init__.py）。

用法：
    from python import LLM
    from python.core import SamplingParams

模型内部组件（Qwen3ForCausalLM 等）不从顶层导出，走 python.model.*——
顶层保持"离线推理入口"这一个职责，避免 import python 时暴露实现细节。
"""

from python.llm import LLM

__all__ = ["LLM"]
