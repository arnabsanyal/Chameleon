"""
Q-Diffusion: Quantizing Diffusion Models
Based on: https://arxiv.org/pdf/2302.04304
Reference: https://github.com/Xiuyu-Li/q-diffusion

Key features:
- Uniform timestep sampling during calibration
- Split quantization for shortcut connections (bimodal distributions)
- Support for 4-bit weights and 8-bit activations
- Timestep-aware calibration strategy
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="diffusers")

import torch
import torch.nn as nn
import numpy as np
from diffusers import StableDiffusionXLPipeline
from tqdm import tqdm
from typing import List, Optional, Tuple, Union
import os
import sys as _sys

_sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "test"))
from perf_instrument import PerfTracker  # noqa: E402

from quant_utils import (
    QuantizedLinear,
    QuantizedConv2d,
    replace_layers_with_quantized,
    set_calibration_mode,
    set_quantized_mode,
    calibrate_all_weights,
    calibrate_all_activations,
    disable_quantization,
    enable_quantization,
    print_quantization_stats,
    auto_batch_size_for_sdxl,
)


class QDiffusionQuantizer:
    """
    Q-Diffusion style quantizer for SDXL.

    Key innovations from the paper:
    1. Uniform timestep sampling for calibration data
    2. Split quantization for shortcut layers (handling bimodal distributions)
    3. Support for aggressive 4-bit weight quantization
    """

    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        weight_bits: int = 8,
        act_bits: int = 8,
    ):
        self.device = device
        self.dtype = dtype
        self.weight_bits = weight_bits
        self.act_bits = act_bits
        self.model_id = model_id
        self.pipe = None
        self.quant_layers = []
        self.calibration_complete = False

    def load_model(self):
        """Load SDXL pipeline"""
        print(f"Loading SDXL from {self.model_id}...")
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
            variant="fp16",
            use_safetensors=True
        ).to(self.device)
        self.pipe.unet.eval()

        # Enable memory optimizations
        if self.device.startswith("cuda"):
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
                print("xformers memory efficient attention enabled")
            except Exception as e:
                print(f"xformers not available: {e}")

        if torch.cuda.is_available():
            device_idx = torch.cuda.current_device()
            print(f"Pipeline loaded on {torch.cuda.get_device_name(device_idx)}")

        return self.pipe

    def inject_quantization(self):
        """
        Inject fake quantization nodes into U-Net.
        Replaces Linear and Conv2d layers with quantized versions.
        """
        if self.pipe is None:
            self.load_model()

        print(f"Injecting {self.weight_bits}-bit weight / {self.act_bits}-bit activation "
              f"quantization into U-Net...")

        self.quant_layers = replace_layers_with_quantized(
            self.pipe.unet,
            weight_bits=self.weight_bits,
            act_bits=self.act_bits,
            timestep_aware=False,
            skip_attention_qkv=True,   # Skip Q/K/V - critical (affects attention scores)
            skip_attention_out=False,  # Quantize output projection (after softmax, less sensitive)
            skip_first_last_conv=True, # Skip input/output convs
        )

        print(f"Injected {len(self.quant_layers)} quantized layers")

    def get_calibration_timesteps(
        self,
        num_timesteps: int = 10,
        max_timestep: int = 1000,
        strategy: str = "uniform"
    ) -> List[int]:
        """
        Get timesteps for calibration.

        Q-Diffusion uses uniform sampling across the timestep range
        to capture activation distributions at different noise levels.

        Args:
            num_timesteps: Number of timesteps to sample
            max_timestep: Maximum timestep value
            strategy: "uniform" for Q-Diffusion style

        Returns:
            List of timesteps for calibration
        """
        if strategy == "uniform":
            # Q-Diffusion: uniform sampling
            timesteps = np.linspace(0, max_timestep - 1, num_timesteps, dtype=int).tolist()
        else:
            # Alternative: focus on specific ranges
            timesteps = [1, 100, 200, 400, 600, 800, 900, 950, 980, 999][:num_timesteps]

        return timesteps

    def get_calibration_prompts(self, num_samples: int = 128) -> List[str]:
        """
        Generate diverse calibration prompts to excite various features.
        More diverse prompts = better activation coverage = better quantization.
        """
        base_prompts = [
            # People and portraits
            "A photo of a cat sitting on a windowsill",
            "A portrait of a wise old wizard with a long beard",
            "A young woman with red hair smiling in a garden",
            "A group of children playing in a park",
            "A businessman in a suit walking down a busy street",
            # Landscapes and nature
            "A futuristic city skyline at night with neon lights",
            "A serene mountain landscape with snow-capped peaks",
            "A tropical beach with crystal clear water",
            "A dense forest with sunlight filtering through trees",
            "A vast desert with sand dunes at sunset",
            "A waterfall in a lush jungle",
            "An aurora borealis over a frozen lake",
            # Objects and still life
            "A bowl of fresh fruits on a wooden table",
            "An astronaut riding a horse on mars",
            "A steampunk mechanical robot",
            "A beautiful sunset over the ocean",
            "A cozy coffee shop interior",
            "A majestic lion in the African savanna",
            "A Japanese garden with cherry blossoms",
            "A vintage car on an empty highway",
            "A colorful butterfly on a flower",
            "A medieval castle on a cliff",
            "A spaceship landing on an alien planet",
            # Architecture and urban
            "A modern glass skyscraper reflecting clouds",
            "An ancient Greek temple with marble columns",
            "A cozy cottage in the English countryside",
            "A busy Tokyo intersection at night",
            # Abstract and artistic
            "An abstract painting with vibrant colors",
            "A surreal dreamscape with floating islands",
            "A minimalist zen composition",
            "Geometric patterns in neon colors",
            # Animals
            "A golden retriever running on a beach",
            "An owl perched on a branch at night",
            "A school of colorful tropical fish",
            "A majestic eagle soaring over mountains",
            # Food
            "A gourmet burger with melted cheese",
            "A colorful smoothie bowl with fruits",
            "Fresh sushi on a wooden board",
            "A steaming cup of coffee with latte art",
            # Technology
            "A futuristic robot in a laboratory",
            "Virtual reality headset with glowing lights",
            "A circuit board with glowing components",
            "A holographic display in a dark room",
        ]

        # Repeat prompts to reach desired sample count
        prompts = []
        while len(prompts) < num_samples:
            prompts.extend(base_prompts)
        return prompts[:num_samples]

    def calibrate(
        self,
        num_calibration_samples: int = 128,  # Increased from 32 for better coverage
        num_inference_steps: int = 20,
        calibration_timesteps: Optional[List[int]] = None,
        guidance_scale: float = 7.5,
    ):
        """
        Run Q-Diffusion calibration.

        This runs the diffusion process on calibration prompts to collect
        activation statistics across different timesteps.

        Args:
            num_calibration_samples: Number of calibration samples
            num_inference_steps: Denoising steps per sample
            calibration_timesteps: Specific timesteps to calibrate on (None = all)
            guidance_scale: CFG scale
        """
        if not self.quant_layers:
            self.inject_quantization()

        print(f"\n{'='*50}")
        print("Q-DIFFUSION CALIBRATION")
        print(f"{'='*50}")
        print(f"Calibration samples: {num_calibration_samples}")
        print(f"Weight bits: {self.weight_bits}, Activation bits: {self.act_bits}")
        print(f"{'='*50}\n")

        # First calibrate weights (one-time, from model weights)
        print("Calibrating weight quantization parameters...")
        calibrate_all_weights(self.quant_layers)

        # Enable calibration mode
        set_calibration_mode(self.quant_layers, calibrating=True)

        # Get calibration prompts
        prompts = self.get_calibration_prompts(num_calibration_samples)

        # Optional: filter to specific timesteps
        if calibration_timesteps is None:
            calibration_timesteps = self.get_calibration_timesteps(10)
        calibration_timesteps_set = set(calibration_timesteps)
        print(f"Calibration timesteps: {sorted(calibration_timesteps)}")

        # Run calibration
        with torch.no_grad():
            for i, prompt in enumerate(tqdm(prompts, desc="Calibrating")):
                try:
                    # Encode prompt
                    (prompt_embeds, negative_prompt_embeds,
                     pooled_prompt_embeds, negative_pooled_prompt_embeds) = \
                        self.pipe.encode_prompt(prompt)

                    # Prepare latents
                    latents = torch.randn(
                        (1, 4, 128, 128),
                        device=self.device,
                        dtype=self.dtype,
                        generator=torch.Generator(device=self.device).manual_seed(i)
                    )

                    # Setup scheduler
                    self.pipe.scheduler.set_timesteps(num_inference_steps)

                    # Denoising loop
                    for t in self.pipe.scheduler.timesteps:
                        # Q-Diffusion: optionally only calibrate on selected timesteps
                        # For now, we calibrate on all steps for robustness
                        should_calibrate = (len(calibration_timesteps_set) == 0 or
                                          int(t.item()) in calibration_timesteps_set or
                                          any(abs(int(t.item()) - ct) < 50 for ct in calibration_timesteps_set))

                        # Temporarily disable calibration if not a target timestep
                        if not should_calibrate:
                            set_calibration_mode(self.quant_layers, calibrating=False)

                        # Expand for CFG
                        latent_input = torch.cat([latents] * 2)
                        latent_input = self.pipe.scheduler.scale_model_input(latent_input, t)

                        # Get time embeddings
                        add_time_ids = torch.tensor(
                            [[1024, 1024, 0, 0, 1024, 1024]],
                            device=self.device,
                            dtype=self.dtype
                        )
                        added_cond_kwargs = {
                            "text_embeds": torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds]),
                            "time_ids": torch.cat([add_time_ids] * 2)
                        }

                        # U-Net forward pass (triggers calibration)
                        noise_pred = self.pipe.unet(
                            latent_input,
                            t,
                            encoder_hidden_states=torch.cat([negative_prompt_embeds, prompt_embeds]),
                            added_cond_kwargs=added_cond_kwargs
                        ).sample

                        # Re-enable calibration if it was disabled
                        if not should_calibrate:
                            set_calibration_mode(self.quant_layers, calibrating=True)

                        # CFG
                        noise_uncond, noise_text = noise_pred.chunk(2)
                        noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)

                        # Scheduler step
                        latents = self.pipe.scheduler.step(noise_pred, t, latents).prev_sample

                except Exception as e:
                    print(f"Error during calibration sample {i}: {e}")
                    continue

        # Finalize calibration: compute activation quantization params
        print("\nFinalizing activation calibration...")
        calibrate_all_activations(self.quant_layers)

        # Switch to quantized mode
        set_quantized_mode(self.quant_layers, quantized=True)
        self.calibration_complete = True

        # Print debug stats
        print_quantization_stats(self.quant_layers)

        print(f"\n{'='*50}")
        print("CALIBRATION COMPLETE")
        print(f"Model is now in {self.weight_bits}W{self.act_bits}A quantized mode")
        print(f"{'='*50}\n")

    def generate_images(
        self,
        prompts: List[str],
        output_dir: str,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        batch_size: Union[int, str] = "auto",
        start_idx: int = 0,
        disable_quant: bool = False,
    ):
        """
        Generate images using the quantized model.

        batch_size: "auto" (default) probes VRAM with a two-point calibration
                    and picks the largest batch that fits in 60% of total VRAM.
                    Any OOM during generation halves the current batch and
                    retries the same chunk.
        """
        if not self.calibration_complete and not disable_quant:
            print("Warning: Calibration not complete. Running calibration first...")
            self.calibrate()

        if disable_quant:
            print("QUANTIZATION DISABLED FOR TESTING")
            disable_quantization(self.quant_layers)
        else:
            enable_quantization(self.quant_layers)

        os.makedirs(output_dir, exist_ok=True)

        # ── Resolve batch size ────────────────────────────────────────────────
        if batch_size == "auto":
            batch_size = auto_batch_size_for_sdxl(
                self.pipe,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                device=self.device,
            )

        print(f"Generating {len(prompts)} images with Q-Diffusion quantized model...")
        print(f"Quantization: {self.weight_bits}W{self.act_bits}A  |  Batch: {batch_size}")

        tracker = PerfTracker(
            label=f"qdiffusion_w{self.weight_bits}a{self.act_bits}",
            device=self.device,
        )
        tracker.reset()

        # ── Adaptive generation loop (halve-on-OOM, retry same chunk) ─────────
        current_bs = batch_size
        pos        = 0
        pbar       = tqdm(total=len(prompts), desc="Generating")

        with torch.no_grad():
            while pos < len(prompts):
                end   = min(pos + current_bs, len(prompts))
                idxs  = list(range(pos, end))
                batch_prompts = [prompts[i] for i in idxs]
                generators = [
                    torch.Generator(device=self.device).manual_seed(start_idx + i)
                    for i in idxs
                ]

                try:
                    tracker.batch_start()
                    result = self.pipe(
                        prompt=batch_prompts,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        generator=generators,
                    )
                    tracker.batch_end(len(batch_prompts))

                    for j, image in enumerate(result.images):
                        image_idx = start_idx + pos + j
                        image.save(f"{output_dir}/{image_idx:05d}.png")
                    pbar.update(len(idxs))
                    pos = end                       # advance only on success

                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if current_bs > 1:
                        new_bs = max(1, current_bs // 2)
                        pbar.write(f"OOM at batch={current_bs} → retrying with {new_bs}")
                        current_bs = new_bs
                    else:
                        pbar.write(f"OOM at batch=1 — skipping image {start_idx + pos}")
                        pbar.update(1)
                        pos += 1

                except Exception as e:
                    pbar.write(f"Error (batch={current_bs}) at {pos}: {e} — skipping {len(idxs)} images")
                    pbar.update(len(idxs))
                    pos = end

        pbar.close()
        tracker.finish()
        tracker.dump_json(os.path.join(output_dir, "perf.json"))
        print(f"Generated {len(prompts)} images in {output_dir}")

    def cleanup(self):
        """Free GPU memory"""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
            torch.cuda.empty_cache()

    def get_quantization_stats(self) -> dict:
        """Get statistics about the quantized model"""
        stats = {
            "num_quantized_layers": len(self.quant_layers),
            "weight_bits": self.weight_bits,
            "act_bits": self.act_bits,
            "calibration_complete": self.calibration_complete,
        }

        if self.quant_layers:
            # Count by type
            linear_count = sum(1 for l in self.quant_layers if isinstance(l, QuantizedLinear))
            conv_count = sum(1 for l in self.quant_layers if isinstance(l, QuantizedConv2d))
            stats["linear_layers"] = linear_count
            stats["conv_layers"] = conv_count

        return stats


def create_qdiffusion_quantizer(
    weight_bits: int = 8,
    act_bits: int = 8,
    device: str = "cuda"
) -> QDiffusionQuantizer:
    """
    Factory function to create a Q-Diffusion quantizer.

    Common configurations:
    - W8A8: weight_bits=8, act_bits=8 (default, good quality)
    - W4A8: weight_bits=4, act_bits=8 (aggressive, may impact quality)
    """
    return QDiffusionQuantizer(
        weight_bits=weight_bits,
        act_bits=act_bits,
        device=device
    )


if __name__ == "__main__":
    # Example usage
    quantizer = create_qdiffusion_quantizer(weight_bits=8, act_bits=8)
    quantizer.inject_quantization()
    quantizer.calibrate(num_calibration_samples=8, num_inference_steps=20)

    test_prompts = ["A photo of a cat"] * 5
    quantizer.generate_images(test_prompts, "./qdiffusion_output")
    quantizer.cleanup()
