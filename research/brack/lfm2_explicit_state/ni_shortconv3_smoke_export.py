#!/usr/bin/env python3
"""
Exports a tiny ExecuTorch program that calls our custom op:

  ni::shortconv3_step.out

This is the “make it real” bridge:
- Python side registers the custom op schema via torch.library (DEF + Meta)
- Export produces a .pte that requires the runtime kernel registration
- Android runner links `ni_softchip_ops` to provide the Vulkan-backed kernel

Usage (host):
  python research/brack/lfm2_explicit_state/ni_shortconv3_smoke_export.py --out smoke_shortconv3.pte --D 1024

Device runtime requirements:
- Push shader to device (or set env var NI_SHORTCONV3_SPV):
    adb push research/brack/neural_interposer_demo/shaders/shortconv_chip.spv /data/local/tmp/shortconv_chip.spv
"""

from __future__ import annotations

import argparse

import torch
# Workaround: importing ExecuTorch pulls in torchao, which can import
# `torch._inductor.decomposition`. In this environment, importing that module
# via torchao can fail because of dict-unpacking expecting a Mapping.
# Importing it here first ensures it is initialized (or fails fast) before
# ExecuTorch import.
import torch._inductor.decomposition  # noqa: F401


def register_ni_shortconv3_op() -> None:
    # Define functional + out variants.
    lib = torch.library.Library("ni", "DEF")
    lib.define("shortconv3_step(Tensor bx, Tensor c, Tensor state, Tensor w) -> Tensor")
    lib.define(
        # NOTE: `Tensor(a!)` is only used for the out tensor here. Marking a
        # non-out input as aliasing (e.g. `state(a!)`) prevents schema conversion
        # in this ExecuTorch version.
        "shortconv3_step.out(Tensor bx, Tensor c, Tensor state, Tensor w, *, Tensor(a!) output) -> Tensor(a!)"
    )

    impl = torch.library.Library("ni", "IMPL")

    @torch.library.impl(impl, "shortconv3_step", "Meta")
    def _meta(bx, c, state, w):
        # output is [D] like bx
        return torch.empty_like(bx)


class Smoke(torch.nn.Module):
    def __init__(self, D: int):
        super().__init__()
        self.D = D

    def forward(self, bx, c, state, w):
        # Calls the custom op (functional); exporter will lower to out variant.
        return torch.ops.ni.shortconv3_step(bx, c, state, w)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output .pte path")
    ap.add_argument("--D", type=int, default=1024)
    args = ap.parse_args()

    register_ni_shortconv3_op()

    m = Smoke(args.D).eval()

    # Smoke inputs. The runner (by default) can also fill with ones, but
    # exporting with explicit shapes avoids surprises.
    bx = torch.ones((args.D,), dtype=torch.float32)
    c = torch.ones((args.D,), dtype=torch.float32)
    state = torch.zeros((args.D, 2), dtype=torch.float32)
    w = torch.ones((args.D, 3), dtype=torch.float32)

    # ExecuTorch export (best-effort; API surface can vary by version).
    try:
        # Work around torch.export default_decompositions() returning a
        # CustomDecompTable (not a Mapping) in newer PyTorch versions.
        import executorch.exir.program._program as _exir_program

        _orig_default_table = _exir_program._default_decomposition_table
        _exir_program._default_decomposition_table = lambda *a, **k: dict(
            _orig_default_table(*a, **k)
        )

        from executorch.extension.export_util.utils import export_to_exec_prog

        with torch.no_grad():
            prog = export_to_exec_prog(m, (bx, c, state, w), strict=True)

        with open(args.out, "wb") as f:
            prog.write_to_file(f)
        print(f"Wrote {args.out}")
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise SystemExit(
            "Failed to export via ExecuTorch EXIR APIs. "
            "This repo’s ExecuTorch version may require a different export call path.\n"
            f"Error: {e}"
        )


if __name__ == "__main__":
    main()

