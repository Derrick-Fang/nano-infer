"""把 HF safetensors 权重写入 packed 模型。

"checkpoint 布局 -> 模型布局" 的适配层：
- 未打包的权重：按名字对号入座，走参数的 weight_loader（缺省直接 copy_）
- 打包的权重（q/k/v_proj -> qkv_proj，gate/up_proj -> gate_up_proj）：
  按 packed_modules_mapping 改写参数名，按 shard 的行区间切片写入

注意：当前模型全部使用 ReplicatedLinear（不会自己切分片），所以 packed
切片暂时在这里做；换回 QKVParallelLinear / MergedColumnParallelLinear
后，切片逻辑会随 weight_loader 下沉回各 Linear 层，本文件可恢复成
nano-vllm 那样的纯调度。

返回成功写入的参数名集合，供测试做完整性校验
（loaded == model.state_dict().keys() 才算一个不漏）。

支持 mini-sglang 风格的 meta 构造：模型在 `with torch.device("meta")`
下创建时参数只是不占内存的占位符，本函数在首次写入前将其物化
（_materialize），dtype 沿用构造期定下的值（见 python/utils.torch_dtype）。
"""

from pathlib import Path

import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def _materialize(model: nn.Module, device: torch.device) -> None:
    """把 meta 设备上的参数整体落成真实存储（保留构造期定下的 shape/dtype）。

    注意不能直接 `param.data = cpu_tensor`——Variable.set_data 拒绝跨
    tensor 类型（meta vs dense），只能整个替换 Parameter 对象，并把挂在
    旧对象上的 weight_loader 钩子搬到新对象上。
    副作用：tied 参数（lm_head/embed_tokens）会被拆成独立存储，加载后
    需调用 model.tie_weights() 重新共享。非 meta 模型本函数是 no-op。
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
    # packed 参数内部的行区间：packed 名 -> shard id -> (start, end)
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
    # glob 兼容单文件与 model-0000N-of-XXXXX 分片两种 checkpoint
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
