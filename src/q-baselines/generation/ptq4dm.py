"""
PTQ4DM: Post-Training Quantization for Diffusion Models
Based on: https://arxiv.org/pdf/2211.15736
Reference: https://github.com/42Shawn/PTQ4DM

Key features:
- Timestep-aware calibration (different quantization params per timestep bucket)
- Addresses timestep-dependent activation distribution changes
- Discrete-time Normalized Time-aware Calibration (DNTC)
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="diffusers")

import torch
import torch.nn as nn
import numpy as np
from diffusers import StableDiffusionXLPipeline
from tqdm import tqdm
from typing import List, Optional, Union
import os
import sys as _sys

_sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "test"))
from perf_instrument import PerfTracker  # noqa: E402

from quant_utils import (
    TimestepAwareQuantizedLayer,
    replace_layers_with_quantized,
    set_calibration_mode,
    set_quantized_mode,
    calibrate_all_weights,
    calibrate_all_activations,
    set_timestep_for_all,
    disable_quantization,
    enable_quantization,
    print_quantization_stats,
    auto_batch_size_for_sdxl,
)


class PTQ4DMQuantizer:
    """
    PTQ4DM style quantizer for SDXL.

    Key innovation: Timestep-aware quantization that maintains separate
    calibration statistics for different timestep buckets to handle
    the dramatically changing activation distributions across timesteps.
    """

    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        weight_bits: int = 8,
        act_bits: int = 8,
        num_timestep_buckets: int = 10,
    ):
        self.device = device
        self.dtype = dtype
        self.weight_bits = weight_bits
        self.act_bits = act_bits
        self.num_timestep_buckets = num_timestep_buckets
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
        Inject timestep-aware fake quantization nodes into U-Net.
        Uses TimestepAwareQuantizedLayer which maintains per-bucket stats.
        """
        if self.pipe is None:
            self.load_model()

        print(f"Injecting PTQ4DM timestep-aware quantization into U-Net...")
        print(f"  Weight bits: {self.weight_bits}")
        print(f"  Activation bits: {self.act_bits}")
        print(f"  Timestep buckets: {self.num_timestep_buckets}")

        self.quant_layers = replace_layers_with_quantized(
            self.pipe.unet,
            weight_bits=self.weight_bits,
            act_bits=self.act_bits,
            timestep_aware=True,
            num_timestep_buckets=self.num_timestep_buckets,
            skip_attention_qkv=True,   # Skip Q/K/V - critical (affects attention scores)
            skip_attention_out=False,  # Quantize output projection (after softmax, less sensitive)
            skip_first_last_conv=True, # Skip input/output convs
        )

        print(f"Injected {len(self.quant_layers)} timestep-aware quantized layers")

    def get_calibration_timesteps(
        self,
        num_timesteps_per_bucket: int = 2,
        max_timestep: int = 1000,
    ) -> List[int]:
        """
        Get timesteps for PTQ4DM calibration.

        PTQ4DM needs to sample from each timestep bucket to capture
        the timestep-dependent activation distributions.

        Args:
            num_timesteps_per_bucket: Samples per bucket
            max_timestep: Maximum timestep

        Returns:
            List of timesteps covering all buckets
        """
        timesteps = []
        bucket_size = max_timestep // self.num_timestep_buckets

        for bucket_idx in range(self.num_timestep_buckets):
            bucket_start = bucket_idx * bucket_size
            bucket_end = (bucket_idx + 1) * bucket_size

            # Sample uniformly within each bucket
            bucket_samples = np.linspace(
                bucket_start, bucket_end - 1,
                num_timesteps_per_bucket, dtype=int
            ).tolist()
            timesteps.extend(bucket_samples)

        return sorted(set(timesteps))

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

        prompts = []
        while len(prompts) < num_samples:
            prompts.extend(base_prompts)
        return prompts[:num_samples]

    def calibrate(
        self,
        num_calibration_samples: int = 128,  # Increased from 32 for better coverage
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
    ):
        """
        Run PTQ4DM calibration with timestep-aware statistics collection.

        This method carefully tracks which timestep bucket each activation
        belongs to, building separate quantization parameters for each bucket.

        Args:
            num_calibration_samples: Number of calibration samples
            num_inference_steps: Denoising steps per sample
            guidance_scale: CFG scale
        """
        if not self.quant_layers:
            self.inject_quantization()

        print(f"\n{'='*50}")
        print("PTQ4DM CALIBRATION (Timestep-Aware)")
        print(f"{'='*50}")
        print(f"Calibration samples: {num_calibration_samples}")
        print(f"Weight bits: {self.weight_bits}, Activation bits: {self.act_bits}")
        print(f"Timestep buckets: {self.num_timestep_buckets}")
        print(f"{'='*50}\n")

        # First calibrate weights (timestep-independent)
        print("Calibrating weight quantization parameters...")
        calibrate_all_weights(self.quant_layers)

        # Enable calibration mode
        set_calibration_mode(self.quant_layers, calibrating=True)

        # Get calibration prompts
        prompts = self.get_calibration_prompts(num_calibration_samples)

        # Track which buckets have been calibrated
        bucket_sample_counts = [0] * self.num_timestep_buckets

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

                    # Denoising loop with timestep tracking
                    for t in self.pipe.scheduler.timesteps:
                        timestep_val = int(t.item())

                        # PTQ4DM key insight: set current timestep for all layers
                        # so they know which bucket to store statistics in
                        set_timestep_for_all(self.quant_layers, timestep_val, max_timestep=1000)

                        # Track bucket coverage
                        bucket_idx = min(timestep_val * self.num_timestep_buckets // 1000,
                                        self.num_timestep_buckets - 1)
                        bucket_sample_counts[bucket_idx] += 1

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

                        # U-Net forward pass (triggers timestep-aware calibration)
                        noise_pred = self.pipe.unet(
                            latent_input,
                            t,
                            encoder_hidden_states=torch.cat([negative_prompt_embeds, prompt_embeds]),
                            added_cond_kwargs=added_cond_kwargs
                        ).sample

                        # CFG
                        noise_uncond, noise_text = noise_pred.chunk(2)
                        noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)

                        # Scheduler step
                        latents = self.pipe.scheduler.step(noise_pred, t, latents).prev_sample

                except Exception as e:
                    print(f"Error during calibration sample {i}: {e}")
                    continue

        # Report bucket coverage
        print("\nTimestep bucket coverage:")
        for idx, count in enumerate(bucket_sample_counts):
            bucket_start = idx * 1000 // self.num_timestep_buckets
            bucket_end = (idx + 1) * 1000 // self.num_timestep_buckets
            print(f"  Bucket {idx} (t={bucket_start}-{bucket_end}): {count} samples")

        # Finalize calibration: compute per-bucket activation quantization params
        print("\nFinalizing per-bucket activation calibration...")
        calibrate_all_activations(self.quant_layers)

        # Switch to quantized mode
        set_quantized_mode(self.quant_layers, quantized=True)
        self.calibration_complete = True

        # Print debug stats
        print_quantization_stats(self.quant_layers)

        print(f"\n{'='*50}")
        print("PTQ4DM CALIBRATION COMPLETE")
        print(f"Model is now in {self.weight_bits}W{self.act_bits}A timestep-aware quantized mode")
        print(f"{'='*50}\n")

    def _make_step_callback(self):
        """Create a callback that sets timestep for all quantized layers at each step."""
        def callback(pipe, step_index, timestep, callback_kwargs):
            # Set timestep for all quantized layers
            timestep_val = int(timestep.item()) if hasattr(timestep, 'item') else int(timestep)
            set_timestep_for_all(self.quant_layers, timestep_val, max_timestep=1000)
            return callback_kwargs
        return callback

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
        Generate images using the PTQ4DM quantized model.

        Uses the pipeline with a step callback to set timesteps for
        timestep-aware quantization while enabling batched generation.

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

        # Create step callback for timestep-aware quantization
        step_callback = self._make_step_callback()

        # ── Resolve batch size ────────────────────────────────────────────────
        # Calibrate with the callback active so the probe exercises the same
        # code path as real generation (different timestep buckets → different
        # fake-quant scales loaded per step).
        if batch_size == "auto":
            batch_size = auto_batch_size_for_sdxl(
                self.pipe,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                device=self.device,
                extra_pipe_kwargs={
                    "callback_on_step_end": step_callback,
                    "callback_on_step_end_tensor_inputs": ["latents"],
                },
            )

        print(f"Generating {len(prompts)} images with PTQ4DM quantized model...")
        print(f"Quantization: {self.weight_bits}W{self.act_bits}A (timestep-aware)  |  Batch: {batch_size}")

        tracker = PerfTracker(
            label=f"ptq4dm_w{self.weight_bits}a{self.act_bits}",
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
                        callback_on_step_end=step_callback,
                        callback_on_step_end_tensor_inputs=["latents"],
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
            "method": "PTQ4DM",
            "num_quantized_layers": len(self.quant_layers),
            "weight_bits": self.weight_bits,
            "act_bits": self.act_bits,
            "num_timestep_buckets": self.num_timestep_buckets,
            "calibration_complete": self.calibration_complete,
        }

        if self.quant_layers and self.calibration_complete:
            # Check bucket coverage
            layer = self.quant_layers[0]
            if hasattr(layer, 'bucket_mins'):
                calibrated_buckets = sum(1 for m in layer.bucket_mins if m < float('inf'))
                stats["calibrated_buckets"] = calibrated_buckets

        return stats


def create_ptq4dm_quantizer(
    weight_bits: int = 8,
    act_bits: int = 8,
    num_timestep_buckets: int = 10,
    device: str = "cuda"
) -> PTQ4DMQuantizer:
    """
    Factory function to create a PTQ4DM quantizer.

    Args:
        weight_bits: Bit-width for weights
        act_bits: Bit-width for activations
        num_timestep_buckets: Number of timestep buckets for calibration
        device: Device to run on
    """
    return PTQ4DMQuantizer(
        weight_bits=weight_bits,
        act_bits=act_bits,
        num_timestep_buckets=num_timestep_buckets,
        device=device
    )


if __name__ == "__main__":
    # Example usage
    quantizer = create_ptq4dm_quantizer(weight_bits=8, act_bits=8, num_timestep_buckets=10)
    quantizer.inject_quantization()
    quantizer.calibrate(num_calibration_samples=8, num_inference_steps=20)

    test_prompts = ["A photo of a cat"] * 5
    quantizer.generate_images(test_prompts, "./ptq4dm_output")
    quantizer.cleanup()
