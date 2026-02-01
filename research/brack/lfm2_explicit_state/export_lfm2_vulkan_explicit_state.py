from __future__ import annotations

import argparse
import json
import os

import torch
from torch.export import export

from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.exir import to_edge_transform_and_lower
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.extension.export_util.utils import save_pte_program

from lfm2_explicit_state_model import Lfm2ExplicitStateModel
from executorch.examples.models.llama.model_args import ModelArgs


def _resolve_opkey(name: str):
    """
    Convert human-friendly names like:
      - aten.linear.default
      - edge.aten.linear.default
      - et_vk.prepack.default
    into the canonical EdgeOpOverload keys (exir_ops.edge.*) used by the Vulkan registry.
    Falls back to returning the original string if resolution fails.
    """
    s = name.strip()
    if not s:
        return s

    # Normalize prefix
    if s.startswith("edge."):
        s = s[len("edge.") :]

    root = exir_ops.edge

    parts = s.split(".")
    try:
        cur = root
        for p in parts:
            cur = getattr(cur, p)
        return cur
    except Exception:
        # Allow passing through raw string keys for custom ops registered by name.
        return name.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_json", required=True, help="lfm2_*_config.json path")
    parser.add_argument("--checkpoint_pt", required=True, help="model.pt (torch.save) path")
    parser.add_argument("--tokenizer_json", required=True, help="tokenizer.json path (used for vocab size sanity only)")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--output_name", default="lfm2_explicit_vulkan", help="base filename without extension")
    parser.add_argument("--max_context_len", type=int, default=256)
    parser.add_argument("--seq_len", type=int, default=1)
    parser.add_argument("--force_fp16", action="store_true", default=True)
    parser.add_argument(
        "--buffer_limit",
        type=int,
        default=500_000,
        help="Vulkan partitioner buffer_limit (numel). Lower skips huge weights to avoid GPU OOM.",
    )
    parser.add_argument(
        "--small_texture_limits",
        action="store_true",
        default=False,
        help="Use smaller texture limits (2048^3) for broader device compatibility.",
    )
    parser.add_argument(
        "--operator_blocklist",
        default="",
        help="Comma-separated Vulkan op keys to blocklist (e.g. 'aten.linear.default,aten.embedding.default').",
    )
    args = parser.parse_args()

    with open(args.config_json, "r") as f:
        cfg = json.load(f)

    # Build ModelArgs compatible with executorch examples.
    model_args = ModelArgs(
        dim=int(cfg["dim"]),
        n_layers=int(cfg["n_layers"]),
        n_heads=int(cfg["n_heads"]),
        n_kv_heads=int(cfg.get("n_kv_heads") or cfg["n_heads"]),
        vocab_size=int(cfg["vocab_size"]),
        hidden_dim=int(cfg["hidden_dim"]),
        norm_eps=float(cfg.get("norm_eps", 1e-5)),
        rope_theta=float(cfg.get("rope_theta", 10000.0)),
        use_hf_rope=bool(cfg.get("use_hf_rope", False)),
        use_qk_norm=bool(cfg.get("use_qk_norm", False)),
        qk_norm_before_rope=bool(cfg.get("qk_norm_before_rope", False)),
        max_batch_size=1,
        max_seq_len=args.max_context_len,
        max_context_len=args.max_context_len,
        use_kv_cache=True,
        enable_dynamic_shape=False,
        layer_types=cfg.get("layer_types"),
    )

    model = Lfm2ExplicitStateModel(model_args).eval()

    # Load checkpoint weights. We expect missing keys for removed internal state buffers.
    sd = torch.load(args.checkpoint_pt, map_location="cpu")
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"Loaded checkpoint. missing={len(missing)} unexpected={len(unexpected)}")

    # Prepare explicit state inputs
    conv_ids = model.layout.conv_layer_ids
    attn_ids = model.layout.attn_layer_ids
    num_conv = len(conv_ids)
    num_attn = len(attn_ids)

    D = model_args.dim
    H_kv = model_args.n_kv_heads
    Hd = model_args.head_dim
    C = model_args.max_context_len
    Lm1 = model.conv_L_cache_minus_1  # 2

    tokens = torch.ones((1, args.seq_len), dtype=torch.long)
    input_pos = torch.arange(args.seq_len, dtype=torch.long)

    conv_state = torch.zeros((num_conv, D, Lm1), dtype=torch.float32)
    k_cache = torch.zeros((num_attn, H_kv, C, Hd), dtype=torch.float32)
    v_cache = torch.zeros((num_attn, H_kv, C, Hd), dtype=torch.float32)

    # Export
    print("Exporting with explicit state (no internal mutation).")
    ep = export(model, (tokens, input_pos, conv_state, k_cache, v_cache), strict=True)

    compile_options = {}
    if args.force_fp16:
        compile_options["force_fp16"] = True
    compile_options["buffer_limit"] = args.buffer_limit
    if args.small_texture_limits:
        compile_options["small_texture_limits"] = True

    operator_blocklist = None
    if args.operator_blocklist.strip():
        operator_blocklist = [
            _resolve_opkey(s) for s in args.operator_blocklist.split(",") if s.strip()
        ]

    edge_program = to_edge_transform_and_lower(
        ep,
        partitioner=[VulkanPartitioner(compile_options, operator_blocklist=operator_blocklist)],
    )
    exec_prog = edge_program.to_executorch()

    os.makedirs(args.output_dir, exist_ok=True)
    save_pte_program(exec_prog, args.output_name, args.output_dir)
    print(f"Wrote {os.path.join(args.output_dir, args.output_name)}.pte")


if __name__ == "__main__":
    with torch.no_grad():
        main()

