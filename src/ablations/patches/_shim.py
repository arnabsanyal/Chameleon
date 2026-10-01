"""
Common shim helpers used by every patch in this directory.

Each patch is a tiny wrapper that:
    1. Imports the relevant Chameleon module(s).
    2. Replaces a symbol (e.g. route_format_by_kurtosis_snr, bos_bypass setter).
    3. Calls the target CLI's main() with the rest of sys.argv.

Invocation pattern (from a runner script):

    python -m patches.routing_kurt_only \
        --target sdxl \
        -- \
        --weight-bits 4 --num-samples 1000 ...

The `--` is consumed by the shim and everything after it is forwarded to the
target CLI verbatim.
"""

from __future__ import annotations

import argparse
import importlib
import runpy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

TARGETS = {
    "sdxl": {
        "code_dir": REPO_ROOT / "src" / "sdxl-chameleon" / "generation",
        "module":   "chameleon_generation",
        "selector": "snr_format_selector",
        "quant":    "chameleon_quant",
    },
    # Chameleon-LCM is now Chameleon-on-MixDQ: Chameleon's adaptive WEIGHT palette
    # (chameleon_lcm_quant.WEIGHT_FORMATS_*) on top of MixDQ's calibrated static
    # activations. There is NO kurtosis/SNR activation routing here, so the
    # activation-routing ablations (routing_*, palette_force/drop --drop) do not
    # apply to lcm — they are sdxl/dit-only. Valid lcm ablations: BAQ on/off
    # (MixDQ --no-bos, see lcm_disable_baq) and weight-palette drops
    # (palette_drop_format --weight-drop, which edits WEIGHT_FORMATS_* here).
    "lcm": {
        "code_dir": REPO_ROOT / "src" / "chameleon-lcm" / "generation",
        "module":   "chameleon_lcm_generation",
        "selector": "chameleon_lcm_quant",   # weight palette (no act routing)
        "quant":    "chameleon_lcm_quant",
        "act_routing": False,                # MixDQ owns activations
    },
    "dit": {
        "code_dir": REPO_ROOT / "src" / "chameleon-dit" / "generation",
        "module":   "chameleon_dit_generation",
        "selector": "chameleon_dit_quant",
        "quant":    "chameleon_dit_quant",
    },
}


def split_argv():
    """Split sys.argv on the first '--' into (shim_args, forwarded)."""
    argv = sys.argv[1:]
    if "--" not in argv:
        return argv, []
    i = argv.index("--")
    return argv[:i], argv[i + 1:]


def parse_target() -> dict:
    """Pop the --target flag from argv and return the matching config."""
    shim_args, forwarded = split_argv()
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--target", choices=list(TARGETS), required=True)
    args, leftover = p.parse_known_args(shim_args)
    cfg = dict(TARGETS[args.target])
    cfg["forwarded"] = forwarded
    cfg["leftover"]  = leftover
    return cfg


def import_target(cfg: dict):
    """Make the target's code dir importable, then import its modules."""
    sys.path.insert(0, str(cfg["code_dir"]))
    selector = importlib.import_module(cfg["selector"])
    quant    = importlib.import_module(cfg["quant"])
    return selector, quant


def patch_symbol(name, value, *modules):
    """Rebind `name` to `value` in EVERY given module that already has it.

    Fixes the from-import binding trap: the quant module does
    `from snr_format_selector import route_format_by_kurtosis_snr`, which creates
    an independent binding.  Patching only `selector.route_format_by_kurtosis_snr`
    leaves the quant module's copy untouched, so the patched routing never runs
    and every ablation arm produces identical images.  We therefore set the
    attribute on the DEFINING module *and* every consumer that imported it.

    Raises if `name` is bound in none of `modules` — fail loud instead of
    silently producing a no-op ablation.
    """
    changed = []
    for m in modules:
        if hasattr(m, name):
            setattr(m, name, value)
            changed.append(getattr(m, "__name__", str(m)))
    if not changed:
        raise RuntimeError(
            f"[shim] patch target '{name}' not found in any of "
            f"{[getattr(m, '__name__', str(m)) for m in modules]} — "
            f"the ablation would be a silent no-op; check the import path.")
    print(f"[shim] patched '{name}' in: {', '.join(changed)}")
    return changed


def run_cli(cfg: dict):
    """Replace argv with the forwarded args and invoke the target CLI.

    Ensures the target's code dir is importable first: patches that DON'T call
    import_target (e.g. lcm_disable_baq, which only forwards --no-bos and patches
    nothing) would otherwise leave the code dir off sys.path, so runpy.run_module
    fails with ImportError. Adding it here makes run_cli self-sufficient.
    """
    code_dir = str(cfg["code_dir"])
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)
    sys.argv = [cfg["module"]] + cfg["forwarded"]
    runpy.run_module(cfg["module"], run_name="__main__")
