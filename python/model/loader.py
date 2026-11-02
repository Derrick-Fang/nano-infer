"""Writes HF safetensors weights into the packed model.

The adaptation layer between "checkpoint layout" and "model layout":
- unpacked weights: matched by name, written through each parameter's
  weight_loader (defaulting to a plain copy_);
- packed weights (q/k/v_proj -> qkv_proj, gate/up_proj -> gate_up_proj):
  the parameter name is rewritten per packed_modules_mapping, and the
  shard's row range is sliced in.

Note: the model currently uses ReplicatedLinear everywhere (which does no
sharding of its own), so packed slicing lives here for now; once we switch
back to QKVParallelLinear / MergedColumnParallelLinear, the slicing logic
moves back down into each Linear layer's weight_loader, and this file can
return to being a pure dispatcher like nano-vllm's.

Returns the set of parameter names successfully written, so tests can
verify completeness (loaded == model.state_dict().keys() means nothing
was missed).

Supports mini-sglang-style meta construction: when the model is created
under `with torch.device("meta")`, parameters are memory-free
placeholders; this function materializes them before the first write
(_materialize), keeping the dtype fixed at construction time (see
python/utils.torch_dtype).
"""

from pathlib import Path

import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def _materialize(model: nn.Module, device: torch.device) -> None:
    """Materializes meta-device parameters into real storage (keeping the
    shape/dtype fixed at construction time).

    Note that `param.data = cpu_tensor` does not work — Variable.set_data
    rejects crossing tensor types (meta vs dense); the whole Parameter
    object must be replaced, and the weight_loader hook attached to the
    old object carried over to the new one.
    Side effect: tied parameters (lm_head/embed_tokens) are split into
    independent storage, so model.tie_weights() must be called after
    loading to re-share them. On a non-meta model this is a no-op.
    """
    for module in model.modules():
        for name, param in module._parameters.items():
            if param is not None and param.device.type == "meta":
                new_param = nn.Parameter(
                    torch.empty_like(param, device=device), requires_grad=False
                )
                if hasattr(param, "weight_loader"):
                    new_param.weight_loader = param.weight_loader
                module._parameters[name] = new_param


def load_weights(
    model: nn.Module,
    model_path: str | Path,
    device: str | torch.device = "cpu",
) -> set[str]:
    device = torch.device(device)
    _materialize(model, device)
    config = model.config
    q_size = config.num_attention_heads * config.head_dim
    kv_size = config.num_key_value_heads * config.head_dim
    inter = config.intermediate_size
    # Row ranges inside packed parameters: packed name -> shard id -> (start, end)
    shard_spans = {
        "qkv_proj": {
            "q": (0, q_size),
            "k": (q_size, q_size + kv_size),
            "v": (q_size + kv_size, q_size + 2 * kv_size),
        },
        "gate_up_proj": {0: (0, inter), 1: (inter, 2 * inter)},
    }
    packed = model.packed_modules_mapping
    loaded = set()
    # glob handles both a single file and model-0000N-of-XXXXX shards
    for file in sorted(Path(model_path).glob("*.safetensors")):
        with safe_open(file, framework="pt", device="cpu") as f:
            for weight_name in f.keys():
                for src, (dst, shard_id) in packed.items():
                    if src in weight_name:
                        param_name = weight_name.replace(src, dst)
                        param = model.get_parameter(param_name)
                        start, end = shard_spans[dst][shard_id]
                        with torch.no_grad():
                            param[start:end].copy_(f.get_tensor(weight_name))
                        loaded.add(param_name)
                        break
                else:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
                    loaded.add(weight_name)
    return loaded
