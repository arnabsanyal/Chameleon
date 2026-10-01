# Baseline Generation for Image and Video Diffusion Models

This code provides a production-ready implementation for generating baseline images and videos using state-of-the-art diffusion models:
- **SDXL (Stable Diffusion XL)** for image generation
- **SVD (Stable Video Diffusion)** for video generation

Use this for calculating FID (Frechet Inception Distance) and FVD (Frechet Video Distance) scores for research papers.

## Features

- Production-ready code with proper error handling
- GPU optimization (A100/CUDA support)
- Memory-efficient generation with chunking
- Progress tracking with tqdm
- Reproducible results with seed control
- FID score calculation integration
- Command-line interface and Python API

## Requirements

### Hardware
- NVIDIA GPU with CUDA support (A100 recommended)
- Minimum 24GB VRAM for SDXL
- Minimum 40GB VRAM for SVD
- Sufficient disk space (models: ~20GB, outputs: variable)

### Software
```bash
pip install -r requirements.txt
```

Key dependencies:
- PyTorch 2.0+
- Hugging Face Diffusers
- clean-fid (for FID calculation)
- xformers (optional, for memory efficiency)

## Quick Start

### 1. Install Dependencies
```bash
cd src
pip install -r requirements.txt
```

### 2. Run Examples
```bash
python example_usage.py
```

### 3. Command-Line Usage

Generate 100 baseline images (with batch size 4 for faster generation):
```bash
python baseline_generation.py \
    --mode image \
    --num-samples 100 \
    --batch-size 4 \
    --output-dir $CHAMELEON_OUTPUT_ROOT
```

Generate videos from images:
```bash
python baseline_generation.py \
    --mode video \
    --input-images-dir $CHAMELEON_OUTPUT_ROOT/images \
    --num-samples 50 \
    --output-dir $CHAMELEON_OUTPUT_ROOT
```

Generate both images and videos:
```bash
python baseline_generation.py \
    --mode both \
    --num-samples 100 \
    --output-dir $CHAMELEON_OUTPUT_ROOT
```

### 4. Calculate FID Score
```bash
python baseline_generation.py \
    --mode image \
    --num-samples 2048 \
    --calculate-fid \
    --reference-dir /path/to/coco/images \
    --output-dir $CHAMELEON_OUTPUT_ROOT
```

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--mode` | str | `image` | Generation mode: `image`, `video`, or `both` |
| `--num-samples` | int | `100` | Number of samples to generate |
| `--output-dir` | str | `$CHAMELEON_OUTPUT_ROOT` | Output directory |
| `--prompt-file` | str | `None` | Path to file with prompts (one per line) |
| `--coco-captions` | str | `None` | Path to COCO captions JSON for proper FID evaluation |
| `--input-images-dir` | str | `None` | Input images for video generation |
| `--num-inference-steps` | int | `50` | Number of denoising steps |
| `--guidance-scale` | float | `7.5` | Classifier-free guidance scale |
| `--batch-size` | int | `16` | Batch size for image generation |
| `--start-ind` | int | `0` | Starting index (use to resume generation) |
| `--fps` | int | `7` | Frames per second for videos |
| `--calculate-fid` | flag | `False` | Calculate FID after generation |
| `--reference-dir` | str | `None` | Reference directory for FID |

## Python API Usage

### Image Generation
```python
from baseline_generation import ImageBaselineGenerator

# Create generator
generator = ImageBaselineGenerator()

# Define prompts
prompts = [
    "A photo of a cat",
    "A beautiful landscape",
    "A futuristic city"
]

# Generate images
generator.generate_images(
    prompts=prompts,
    output_dir="$CHAMELEON_OUTPUT_ROOT/images",
    num_inference_steps=50,
    guidance_scale=7.5,
    batch_size=4  # Generate 4 images in parallel
)

# Cleanup
generator.cleanup()
```

### Video Generation
```python
from baseline_generation import VideoBaselineGenerator

# Create generator
generator = VideoBaselineGenerator()

# Use generated images or load from directory
input_images = ["$CHAMELEON_OUTPUT_ROOT/images/00001.png", "$CHAMELEON_OUTPUT_ROOT/images/00002.png"]

# Generate videos
generator.generate_videos(
    input_images=input_images,
    output_dir="$CHAMELEON_OUTPUT_ROOT/videos",
    fps=7,
    num_frames=25
)

# Cleanup
generator.cleanup()
```

### FID Calculation
```python
from baseline_generation import calculate_fid

score = calculate_fid(
    real_images_dir="/path/to/coco/images",
    generated_images_dir="$CHAMELEON_OUTPUT_ROOT/images"
)
print(f"FID Score: {score:.2f}")
```

## For Research Papers

### Recommended Settings

**Image Generation (SDXL):**
- Samples: 2,048 - 10,000 (30,000 for top-tier venues)
- Inference steps: 50
- Guidance scale: 7.5
- Resolution: 1024x1024 (native SDXL)

**Video Generation (SVD):**
- Samples: 1,000 - 5,000
- Frames: 25 (default for SVD)
- FPS: 7
- Resolution: 1024x576 (native SVD)

**FID Calculation:**
- Use COCO-2014 validation set as reference
- Generate at least 2,048 samples for statistical significance
- Report mean and std over 3+ runs

### Proper FID Evaluation with COCO Captions

For accurate FID scores comparable to published results, use **COCO captions as prompts** instead of generic prompts.

**1. Download COCO dataset:**
```bash
mkdir -p $CHAMELEON_DATA_ROOT/coco
cd $CHAMELEON_DATA_ROOT/coco

# Download validation images
wget http://images.cocodataset.org/zips/val2014.zip
unzip val2014.zip

# Download annotations (captions)
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
unzip annotations_trainval2014.zip
```

**2. Generate images using COCO captions:**
```bash
python baseline_generation.py \
    --mode image \
    --num-samples 5000 \
    --batch-size 16 \
    --coco-captions $CHAMELEON_DATA_ROOT/coco/annotations/captions_val2014.json \
    --output-dir $CHAMELEON_OUTPUT_ROOT/coco_baseline
```

**3. Compute FID:**
```bash
python -c "from cleanfid import fid; print(f'FID: {fid.compute_fid(\"$CHAMELEON_OUTPUT_ROOT/coco_baseline/images\", \"$CHAMELEON_DATA_ROOT/coco/val2014\")}')"
```

### Resuming Generation

If generation is interrupted, use `--start-ind` to resume from where you left off:

```bash
# Start fresh (generates indices 0 to 4999)
python baseline_generation.py \
    --mode image \
    --num-samples 5000 \
    --coco-captions /path/to/captions.json \
    --output-dir $CHAMELEON_OUTPUT_ROOT/coco_baseline

# Resume from index 2500 (generates indices 2500 to 4999)
python baseline_generation.py \
    --mode image \
    --num-samples 5000 \
    --start-ind 2500 \
    --coco-captions /path/to/captions.json \
    --output-dir $CHAMELEON_OUTPUT_ROOT/coco_baseline
```

**Note:** Seeds are based on global indices, so resumed generation produces identical results to a full run.

### Example Pipeline for Paper

```bash
# 1. Generate 10k baseline images using COCO captions
python baseline_generation.py \
    --mode image \
    --num-samples 10000 \
    --batch-size 16 \
    --num-inference-steps 50 \
    --coco-captions $CHAMELEON_DATA_ROOT/coco/annotations/captions_val2014.json \
    --output-dir $CHAMELEON_OUTPUT_ROOT/paper_baseline

# 2. Calculate FID against COCO val2014
python -c "from cleanfid import fid; print(f'FID: {fid.compute_fid(\"$CHAMELEON_OUTPUT_ROOT/paper_baseline/images\", \"$CHAMELEON_DATA_ROOT/coco/val2014\")}')"

# 3. Generate videos from subset of images
python baseline_generation.py \
    --mode video \
    --input-images-dir $CHAMELEON_OUTPUT_ROOT/paper_baseline/images \
    --num-samples 2000 \
    --output-dir $CHAMELEON_OUTPUT_ROOT/paper_baseline
```

## Memory Optimization and Batch Processing

### Batch Size Configuration

The code now supports **batch processing** for image generation, which significantly improves throughput on high-memory GPUs.

**Default batch size: 4** (safe default for most GPUs)

```bash
# Use default batch size (4)
python baseline_generation.py --mode image --num-samples 2048

# Adjust batch size based on your VRAM
python baseline_generation.py --mode image --num-samples 100 --batch-size 8
```

**Recommended batch sizes by GPU:**
- A100 80GB: 4-8
- A100 40GB: 2-4
- A6000 (48GB): 2-4
- RTX 4090 (24GB): 1-2
- RTX 3090 (24GB): 1-2

### For Limited VRAM

If you encounter OOM errors:

1. **Reduce batch size**: Use `--batch-size 2` or `--batch-size 1`
2. **Use FP16** (already default)
3. **Enable xformers** (install with `pip install xformers`)

### Python API Batch Control

```python
# High-memory GPU (A100 80GB)
generator.generate_images(prompts, output_dir, batch_size=8)

# Medium-memory GPU (A6000/A100 40GB)
generator.generate_images(prompts, output_dir, batch_size=4)

# Lower-memory GPU (RTX 3090/4090)
generator.generate_images(prompts, output_dir, batch_size=2)

# Low-memory or OOM issues
generator.generate_images(prompts, output_dir, batch_size=1)
```

## Troubleshooting

### Out of Memory (OOM)
- Reduce `num_frames` for video generation
- Increase `decode_chunk_size` (try 4 or 2)
- Ensure `enable_model_cpu_offload()` is working
- Close other GPU processes

### Slow Generation
- Install xformers: `pip install xformers`
- Use FP16 (already default)
- Check if CUDA is properly installed: `torch.cuda.is_available()`

### Model Download Issues
- Set HuggingFace token: `huggingface-cli login`
- Check internet connection
- Verify model IDs are correct

## Citation

If you use this code in your research, please cite the original models:

**SDXL:**
```bibtex
@article{podell2023sdxl,
  title={SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis},
  author={Podell, Dustin and English, Zion and Lacey, Kyle and Blattmann, Andreas and Dockhorn, Tim and M{\"u}ller, Jonas and Penna, Joe and Rombach, Robin},
  journal={arXiv preprint arXiv:2307.01952},
  year={2023}
}
```

**SVD:**
```bibtex
@article{blattmann2023stable,
  title={Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets},
  author={Blattmann, Andreas and Dockhorn, Tim and Kulal, Sumith and Mendelevitch, Daniel and Kilian, Maciej and Lorenz, Dominik and Levi, Yam and English, Zion and Voleti, Vikram and Letts, Adam and others},
  journal={arXiv preprint arXiv:2311.15127},
  year={2023}
}
```

## License

This code is provided for research purposes. Please refer to the licenses of the underlying models:
- SDXL: https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
- SVD: https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt
