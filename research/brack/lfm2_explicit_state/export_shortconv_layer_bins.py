from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Tuple

import torch

from lfm2_explicit_state_model import Lfm2ExplicitStateModel
from executorch.examples.models.llama.model_args import ModelArgs


def _save_f32(path: Path, t: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tt = t.detach().to(dtype=torch.float32, device="cpu").contiguous()
    tt.numpy().tofile(str(path))


@torch.no_grad()
def _shortconv_step(
    conv: torch.nn.Module,
    x_norm: torch.Tensor,         # [1, 1, D]
    conv_state: torch.Tensor,     # [1, D, 2]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
      y_pre: [D] before out_proj
      y_out: [D] after out_proj
      next_state: [D,2]
    """
    # conv is ShortConvExplicit
    B = conv.B_proj(x_norm).transpose(-1, -2)      # [1, D, 1]
    C = conv.C_proj(x_norm).transpose(-1, -2)      # [1, D, 1]
    x_proj = conv.x_proj(x_norm).transpose(-1, -2) # [1, D, 1]
    Bx = B * x_proj                                # [1, D, 1]

    Bx_padded = torch.cat([conv_state, Bx], dim=-1)  # [1, D, 3]

    # conv.conv expects [N,C,L] where C==D
    conv_out = conv.conv(Bx_padded)[..., : x_proj.size(-1)]  # [1, D, 1]
    y_pre = (C * conv_out).squeeze(0).squeeze(-1)            # [D]

    y = y_pre.view(1, 1, -1)
    y_out = conv.out_proj(y).squeeze(0).squeeze(0)           # [D]

    # next_state should be [D,2] with last dim fastest (matches C++ expectation)
    next_state = Bx_padded.squeeze(0)[..., -(conv.L_cache - 1) :]  # [D,2]
    return y_pre, y_out, next_state


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_json", required=True)
    ap.add_argument("--checkpoint_pt", required=True)
    ap.add_argument("--output_dir", required=True, help="Output directory (single layer) or root (suite mode).")
    ap.add_argument("--layer_id", type=int, default=None, help="Conv layer_id in params.layer_types space; defaults to first conv.")
    ap.add_argument(
        "--all_conv_layers",
        action="store_true",
        default=False,
        help="Export bins for all conv layers into subdirectories under output_dir.",
    )
    ap.add_argument("--waves", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with open(args.config_json, "r") as f:
        cfg = json.load(f)

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
        max_seq_len=int(cfg.get("max_seq_len", cfg.get("max_context_len", 256))),
        max_context_len=int(cfg.get("max_context_len", 256)),
        use_kv_cache=True,
        enable_dynamic_shape=False,
        layer_types=cfg.get("layer_types"),
    )

    model = Lfm2ExplicitStateModel(model_args).eval()
    sd = torch.load(args.checkpoint_pt, map_location="cpu")
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    model.load_state_dict(sd, strict=False)

    # Pick a conv layer
    conv_layer_ids = model.layout.conv_layer_ids
    if not conv_layer_ids:
        raise RuntimeError("No conv layers in this config (layer_types).")
    if args.all_conv_layers:
        out_root = Path(args.output_dir)
        for layer_id in conv_layer_ids:
            sub = out_root / f"layer_{layer_id:02d}"
            _export_one(model, model_args, layer_id, sub, args.waves, args.seed)
        print(f"Wrote conv layer suite to: {out_root}")
        print(f"conv_layer_ids={conv_layer_ids}")
        return

    if args.layer_id is None:
        layer_id = conv_layer_ids[0]
    else:
        layer_id = int(args.layer_id)
        if layer_id not in conv_layer_ids:
            raise ValueError(f"layer_id={layer_id} is not a conv layer. conv_layer_ids={conv_layer_ids}")

    block = model.layers[layer_id]
    # ShortConvBlockExplicit has: attention_norm + conv (ShortConvExplicit)
    conv = block.conv
    D = model_args.dim
    L = conv.L_cache
    assert L == 3, "This exporter currently assumes L_cache=3 for LFM2."

    torch.manual_seed(args.seed)
    x = torch.randn((1, 1, D), dtype=torch.float32)
    x_norm = block.attention_norm(x)
    conv_state = torch.randn((1, D, L - 1), dtype=torch.float32) * 0.1

    _export_one(model, model_args, layer_id, Path(args.output_dir), args.waves, args.seed)


@torch.no_grad()
def _export_one(
    model: Lfm2ExplicitStateModel,
    model_args: ModelArgs,
    layer_id: int,
    out_dir: Path,
    waves: int,
    seed: int,
) -> None:
    block = model.layers[layer_id]
    conv = block.conv
    D = model_args.dim
    L = conv.L_cache
    assert L == 3, "This exporter currently assumes L_cache=3 for LFM2."

    torch.manual_seed(seed + layer_id)
    x = torch.randn((1, 1, D), dtype=torch.float32)
    x_norm = block.attention_norm(x)
    conv_state = torch.randn((1, D, L - 1), dtype=torch.float32) * 0.1

    # Export weights
    _save_f32(out_dir / "Wb.bin", conv.B_proj.weight)
    _save_f32(out_dir / "Wc.bin", conv.C_proj.weight)
    _save_f32(out_dir / "Wx.bin", conv.x_proj.weight)
    _save_f32(out_dir / "Wout.bin", conv.out_proj.weight)

    # Depthwise conv weights [D,1,3] -> [D,3]
    w_dw = conv.conv.weight.squeeze(1)
    _save_f32(out_dir / "Wdw.bin", w_dw)

    # Inputs
    _save_f32(out_dir / "x_norm.bin", x_norm.squeeze(0).squeeze(0))  # [D]
    _save_f32(out_dir / "state0.bin", conv_state.squeeze(0))         # [D,2]

    # Expected wave-by-wave outputs/state evolution
    y_pre_all = torch.empty((waves, D), dtype=torch.float32)
    y_out_all = torch.empty((waves, D), dtype=torch.float32)
    state_all = torch.empty((waves, D, L - 1), dtype=torch.float32)

    cur_state = conv_state
    for w in range(waves):
        y_pre, y_out, next_state = _shortconv_step(conv, x_norm, cur_state)
        y_pre_all[w] = y_pre
        y_out_all[w] = y_out
        state_all[w] = next_state
        cur_state = next_state.unsqueeze(0)  # [1,D,2]

    _save_f32(out_dir / "expected_y_pre.bin", y_pre_all)
    _save_f32(out_dir / "expected_out.bin", y_out_all)
    _save_f32(out_dir / "expected_state.bin", state_all)

    meta: Dict[str, object] = {
        "layer_id": int(layer_id),
        "dim": int(D),
        "L_cache": int(L),
        "waves": int(waves),
        "seed": int(seed),
        "files": {
            "Wb": "Wb.bin",
            "Wc": "Wc.bin",
            "Wx": "Wx.bin",
            "Wout": "Wout.bin",
            "Wdw": "Wdw.bin",
            "x_norm": "x_norm.bin",
            "state0": "state0.bin",
            "expected_y_pre": "expected_y_pre.bin",
            "expected_out": "expected_out.bin",
            "expected_state": "expected_state.bin",
        },
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote ShortConv layer bins to: {out_dir}")


if __name__ == "__main__":
    main()

