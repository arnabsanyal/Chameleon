"""
Ablation 3, row "LCM no-BAQ": disable BOS-Aware Quantization for Chameleon-LCM.

Chameleon-LCM is now Chameleon-on-MixDQ — BAQ is MixDQ's BOS-token bypass, owned
by the MixDQ pipeline, not by a Chameleon predicate. So disabling BAQ is simply
running the driver with --no-bos. This patch injects --no-bos into the forwarded
CLI args (idempotent) and then runs the target.

(The old standalone Chameleon-LCM disabled BAQ by patching is_cross_attn_kv in
chameleon_lcm_quant; that predicate no longer exists in the MixDQ-based path.)
"""

from _shim import parse_target, run_cli


def main():
    cfg = parse_target()
    assert cfg["module"] == "chameleon_lcm_generation", \
        "lcm_disable_baq is LCM-only; pass --target lcm"

    if "--no-bos" not in cfg["forwarded"]:
        cfg["forwarded"].append("--no-bos")
    print("[ablation patch] BAQ disabled (forwarding --no-bos to MixDQ pipeline)")
    run_cli(cfg)


if __name__ == "__main__":
    main()
