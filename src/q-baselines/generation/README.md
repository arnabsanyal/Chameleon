# Quantized Baseline Generation

This module implements two post-training quantization (PTQ) techniques for diffusion models:
- **Q-Diffusion**: Uniform timestep calibration with support for 4-bit/8-bit quantization
- **PTQ4DM**: Timestep-aware quantization with per-bucket calibration

## Quick Start

```bash
# Install requirements
pip install -r requirements.txt

# Run interactive script
./run_quantized.sh

# Or run directly
python quantized_generation.py --method qdiffusion --num-samples 100
```

## Methods

### Q-Diffusion
Based on [Q-Diffusion: Quantizing Diffusion Models](https://arxiv.org/pdf/2302.04304)

Key features:
- Uniform timestep sampling during calibration
- Split quantization for shortcut connections
- Support for aggressive 4-bit weight quantization

```python
from qdiffusion import create_qdiffusion_quantizer

quantizer = create_qdiffusion_quantizer(weight_bits=8, act_bits=8)
quantizer.inject_quantization()
quantizer.calibrate(num_calibration_samples=32)
quantizer.generate_images(prompts, output_dir)
```

### PTQ4DM
Based on [PTQ4DM: Post-Training Quantization for Diffusion Models](https://arxiv.org/pdf/2211.15736)

Key features:
- Timestep-aware calibration (separate stats per timestep bucket)
- Handles changing activation distributions across diffusion steps
- Better quality at same bit-width vs naive PTQ

```python
from ptq4dm import create_ptq4dm_quantizer

quantizer = create_ptq4dm_quantizer(weight_bits=8, act_bits=8, num_timestep_buckets=10)
quantizer.inject_quantization()
quantizer.calibrate(num_calibration_samples=32)
quantizer.generate_images(prompts, output_dir)
```

## Command Line Usage

```bash
# Q-Diffusion W8A8
python quantized_generation.py \
    --method qdiffusion \
    --weight-bits 8 \
    --act-bits 8 \
    --num-samples 1000 \
    --output-dir ./output

# PTQ4DM W8A8 with 10 timestep buckets
python quantized_generation.py \
    --method ptq4dm \
    --weight-bits 8 \
    --act-bits 8 \
    --num-timestep-buckets 10 \
    --num-samples 1000 \
    --output-dir ./output

# Both methods for comparison
python quantized_generation.py \
    --method both \
    --num-samples 1000 \
    --output-dir ./output

# With COCO captions for FID evaluation
python quantized_generation.py \
    --method qdiffusion \
    --coco-captions /path/to/captions_val2014.json \
    --num-samples 5000 \
    --calculate-fid \
    --reference-dir /path/to/coco/val2014
```

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--method` | qdiffusion | Quantization method (qdiffusion/ptq4dm/both) |
| `--weight-bits` | 8 | Bit-width for weights (4 or 8) |
| `--act-bits` | 8 | Bit-width for activations |
| `--num-timestep-buckets` | 10 | Timestep buckets for PTQ4DM |
| `--num-calibration-samples` | 32 | Samples for calibration |
| `--calibration-steps` | 20 | Inference steps during calibration |
| `--num-samples` | 100 | Images to generate |
| `--num-inference-steps` | 50 | Inference steps for generation |
| `--guidance-scale` | 7.5 | CFG guidance scale |
| `--batch-size` | 1 | Batch size for generation |
| `--output-dir` | ... | Output directory |

## Quantization Configurations

| Config | Description | Expected Quality |
|--------|-------------|------------------|
| W8A8 | 8-bit weights, 8-bit activations | Near FP16 quality |
| W4A8 | 4-bit weights, 8-bit activations | Some quality loss |

## Files

- `quant_utils.py` - Core quantization utilities and layer wrappers
- `qdiffusion.py` - Q-Diffusion implementation
- `ptq4dm.py` - PTQ4DM implementation
- `quantized_generation.py` - Main generation script
- `run_quantized.sh` - Interactive shell script
- `example_usage.py` - Usage examples

## References

- Q-Diffusion: https://arxiv.org/pdf/2302.04304
- PTQ4DM: https://arxiv.org/pdf/2211.15736
- Q-Diffusion GitHub: https://github.com/Xiuyu-Li/q-diffusion
- PTQ4DM GitHub: https://github.com/42Shawn/PTQ4DM
