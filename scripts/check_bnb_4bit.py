"""Pod-side preflight: does the 4-bit NF4 path actually RUN on this GPU?

The bootstrap's sanity step only does `import bitsandbytes`, which passes even when
the 4-bit CUDA kernels are broken for the card's arch. On Blackwell (sm_120, cu128)
the nf4 quant/dequant/matmul path was historically fragile, so this script does a
MINIMAL but REAL exercise of it before committing GPU hours to training:

  1. report torch build + device compute capability + CUDA version (cu128 expected
     for Blackwell; capability (12, 0) for sm_120),
  2. quantize a small Linear to NF4 with the SAME knobs as
     ``train_sft.build_bnb_config`` (nf4, double-quant, bf16 compute), move it to the
     GPU, and run a forward pass — exercising the exact kernels training relies on.

Run ON THE POD (needs CUDA):  python scripts/check_bnb_4bit.py
Exit code 0 = the 4-bit path works; non-zero (and a printed FAIL) = do not train.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    import torch

    print("== environment ==")
    print(f"torch              : {torch.__version__}")
    print(f"torch.version.cuda : {torch.version.cuda}   (expect 12.8 / cu128 on Blackwell)")
    if not torch.cuda.is_available():
        print("FAIL: torch.cuda.is_available() is False — no usable GPU. Do not train.")
        return 2
    cap = torch.cuda.get_device_capability(0)
    print(f"device             : {torch.cuda.get_device_name(0)}")
    print(f"compute capability : {cap}   (expect (12, 0) for sm_120 Blackwell)")

    # mirror the EXACT quantization training uses, so this tests the trained path.
    from scripts.train_sft import build_bnb_config
    bnb_cfg = build_bnb_config()
    print("\n== build_bnb_config (the config training loads the base model with) ==")
    print(f"  load_in_4bit={bnb_cfg.load_in_4bit} quant_type={bnb_cfg.bnb_4bit_quant_type} "
          f"double_quant={bnb_cfg.bnb_4bit_use_double_quant} "
          f"compute_dtype={bnb_cfg.bnb_4bit_compute_dtype}")

    print("\n== minimal NF4 quantized load + forward ==")
    try:
        import bitsandbytes as bnb
        from bitsandbytes.nn import Linear4bit
        print(f"bitsandbytes       : {bnb.__version__}")

        # Same knobs as build_bnb_config: nf4 + double-quant (compress_statistics)
        # + bf16 compute. Constructing on GPU quantizes the weight to NF4; the
        # forward then dequantizes + matmuls — the kernels that were fragile here.
        torch.manual_seed(0)
        lin = Linear4bit(
            256, 256, bias=False,
            compute_dtype=torch.bfloat16,
            compress_statistics=bnb_cfg.bnb_4bit_use_double_quant,
            quant_type=bnb_cfg.bnb_4bit_quant_type,
        ).cuda()
        x = torch.randn(4, 256, device="cuda", dtype=torch.bfloat16)
        y = lin(x)
        torch.cuda.synchronize()
        finite = bool(torch.isfinite(y).all())
        qstate = getattr(lin.weight, "quant_state", None)
        print(f"weight quant_state : {'present' if qstate is not None else 'MISSING'} "
              f"(blockwise={getattr(qstate, 'blocksize', '?')})")
        print(f"forward out        : shape={tuple(y.shape)} dtype={y.dtype} "
              f"all_finite={finite} abs_sum={float(y.float().abs().sum()):.3f}")
        if qstate is None or not finite:
            print("FAIL: NF4 forward did not produce a valid quantized result.")
            return 3
    except Exception as e:  # noqa: BLE001 — this is the thing we're probing; report it
        print(f"FAIL: NF4 path raised {type(e).__name__}: {e}")
        return 3

    print("\nPASS: 4-bit NF4 path runs on this GPU — safe to launch QLoRA training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
