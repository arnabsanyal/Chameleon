"""
Capture REAL SDXL UNet activations at the two ends of the denoising schedule
(t -> T, noise-dominated; t -> 0, structured signal) for a single linear layer,
then score every candidate 8-bit format with the repo's own quantizers.

Produces the motivation/toy-example table in chameleon-3.pdf §3.

Run:
    CUDA_VISIBLE_DEVICES=0 \
    python capture_toy_activations.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sdxl-chameleon", "generation"))

import torch
from diffusers import StableDiffusionXLPipeline
from snr_format_selector import (
    compute_snr_for_all_formats, compute_kurtosis, compute_diffusion_snr, QuantFormat,
)

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
DEVICE   = "cuda"
STEPS    = 50
PROMPT   = "a photograph of an astronaut riding a horse"

FORMATS = [QuantFormat.INT8_ASYM, QuantFormat.FLOAT8_E5M2, QuantFormat.FLOAT8_E4M3]


def best_fmt(snrs):
    return max(snrs, key=snrs.get)


def main():
    torch.manual_seed(0)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, variant="fp16", use_safetensors=True,
    ).to(DEVICE)
    pipe.unet.eval()

    # Instrument every Linear layer on the main path (skip QKV projections, which
    # Chameleon leaves alone). Capture input activations at the first and last step.
    targets = [(n, m) for n, m in pipe.unet.named_modules()
               if isinstance(m, torch.nn.Linear)
               and not any(s in n for s in ("to_q", "to_k", "to_v"))]
    print(f"Instrumented {len(targets)} linear layers")

    captured = {}        # (name, step_index) -> input activation (float cpu)
    cur_step = {"i": -1}

    def make_hook(name):
        def hook(module, inputs, output):
            i = cur_step["i"]
            if i in (0, STEPS - 1):
                captured[(name, i)] = inputs[0].detach().float().reshape(-1).cpu()
        return hook

    for name, mod in targets:
        mod.register_forward_hook(make_hook(name))

    # Manual denoising loop so we can tag the step index for each UNet call.
    prompt_embeds, neg, pooled, neg_pooled = pipe.encode_prompt(
        PROMPT, device=DEVICE, num_images_per_prompt=1, do_classifier_free_guidance=True)
    pe = torch.cat([neg, prompt_embeds])
    add_text = torch.cat([neg_pooled, pooled])

    add_time_ids = pipe._get_add_time_ids(
        (1024, 1024), (0, 0), (1024, 1024), dtype=prompt_embeds.dtype,
        text_encoder_projection_dim=pipe.text_encoder_2.config.projection_dim)
    add_time_ids = torch.cat([add_time_ids, add_time_ids]).to(DEVICE)

    pipe.scheduler.set_timesteps(STEPS, device=DEVICE)
    timesteps = pipe.scheduler.timesteps
    latents = torch.randn(1, pipe.unet.config.in_channels, 128, 128,
                          device=DEVICE, dtype=torch.float16) * pipe.scheduler.init_noise_sigma

    alphas = pipe.scheduler.alphas_cumprod.to(DEVICE)

    with torch.no_grad():
        for i, t in enumerate(timesteps):
            cur_step["i"] = i
            li = torch.cat([latents] * 2)
            li = pipe.scheduler.scale_model_input(li, t)
            added = {"text_embeds": add_text, "time_ids": add_time_ids}
            np_ = pipe.unet(li, t, encoder_hidden_states=pe, added_cond_kwargs=added).sample
            nu, nc = np_.chunk(2)
            np_ = nu + 7.5 * (nc - nu)
            latents = pipe.scheduler.step(np_, t, latents).prev_sample

    first_i, last_i = 0, STEPS - 1
    t_first, t_last = int(timesteps[first_i].item()), int(timesteps[last_i].item())
    alphas_cpu = alphas.cpu()

    # Score every layer at both ends; record kurtosis and per-format SNR.
    per_layer = {}   # name -> dict(end -> (t, kappa, snrs))
    for name, _ in targets:
        if (name, first_i) not in captured or (name, last_i) not in captured:
            continue
        rec = {}
        for end, i, t_val in [("T", first_i, t_first), ("0", last_i, t_last)]:
            x = captured[(name, i)]
            rec[end] = (t_val, compute_kurtosis(x), compute_snr_for_all_formats(x, FORMATS))
        per_layer[name] = rec

    # Aggregate: how often is each format the MSQE winner at each end?
    from collections import Counter
    winT = Counter(best_fmt(r["T"][2]) for r in per_layer.values())
    win0 = Counter(best_fmt(r["0"][2]) for r in per_layer.values())
    flips = [n for n, r in per_layer.items() if best_fmt(r["T"][2]) != best_fmt(r["0"][2])]

    print(f"\n=== {len(per_layer)} layers scored ===")
    print("Best format at t->T :", {f.name: winT[f] for f in FORMATS})
    print("Best format at t->0 :", {f.name: win0[f] for f in FORMATS})
    print(f"Layers whose MSQE-optimal format FLIPS between ends: {len(flips)}")

    # Pick the most dramatic flip (largest swing in winning-format margin).
    def swing(n):
        r = per_layer[n]
        return abs((sorted(r["T"][2].values())[-1] - sorted(r["T"][2].values())[-2]) +
                   (sorted(r["0"][2].values())[-1] - sorted(r["0"][2].values())[-2]))
    if flips:
        ex = max(flips, key=swing)
        print(f"\nExample flip layer: {ex}")
        for end in ("T", "0"):
            t_val, kappa, snrs = per_layer[ex][end]
            print(f"  t={t_val:4d} kappa={kappa:6.1f}  " +
                  "  ".join(f"{f.name}={snrs[f]:.2f}" for f in FORMATS) +
                  f"   -> {best_fmt(snrs).name}")

    # Also report the single layer with the widest INT8-vs-FP8 gap reversal,
    # and emit a paste-ready LaTeX table for the chosen example.
    fmt_names = {QuantFormat.INT8_ASYM: r"\texttt{INT8} (asym)",
                 QuantFormat.FLOAT8_E5M2: r"\texttt{FP8 E5M2}",
                 QuantFormat.FLOAT8_E4M3: r"\texttt{FP8 E4M3}"}
    chosen = flips and max(flips, key=swing) or list(per_layer)[0]
    def cell(snrs, f):
        best = max(snrs.values())
        s = f"{snrs[f]:.1f}"
        return (r"\mathbf{" + s + "}") if snrs[f] == best else s
    print("\n% --- paste-ready (example layer: " + chosen + ") ---")
    print(r"\begin{tabular}{lccc}\toprule")
    print("regime & " + " & ".join(fmt_names[f] for f in FORMATS) + r" \\ \midrule")
    for end in ("T", "0"):
        t_val, kappa, snrs = per_layer[chosen][end]
        tag = r"$t\!\to\!T$" if end == "T" else r"$t\!\to\!0$"
        print(f"{tag} ($\\kappa\\!\\approx\\!{kappa:.0f}$) & " +
              " & ".join("$%s$" % cell(snrs, f) for f in FORMATS) + r" \\")
    print(r"\bottomrule\end{tabular}")


if __name__ == "__main__":
    main()
