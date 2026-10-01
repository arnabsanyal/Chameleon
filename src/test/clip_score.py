"""
Calculate CLIP scores between generated images and their corresponding prompts.

Usage:
    python clip_score.py --images /path/to/images --coco-captions /path/to/captions.json
    python clip_score.py --images /path/to/images --coco-captions /path/to/captions.json --start-ind 0 --end-ind 5000
"""

import argparse
import json
import os
import random
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
import numpy as np


def load_coco_captions(captions_path, num_samples=None, shuffle=True):
    """Load captions EXACTLY as every generator does, so that image i is scored
    against the caption it was generated from.

    Every per-backbone generator (chameleon_generation, mixdq_lcm_generation,
    q_dit_generation, ...) builds its prompt list as the flat list of raw
    annotations shuffled with seed 42, and writes image i as {i:05d}.png. The
    scorer indexes prompts[idx] for {idx:05d}.png, so it must reproduce that
    list verbatim.

    A previous version instead took one caption per image_id with no shuffle,
    which produced a different list: image i was then scored against an
    unrelated caption. That made the reported CLIP scores measure
    image-vs-random-caption similarity.
    """
    with open(captions_path, 'r') as f:
        data = json.load(f)

    captions = [ann['caption'].strip() for ann in data['annotations']]

    if shuffle:
        random.seed(42)
        random.shuffle(captions)

    if num_samples:
        captions = captions[:num_samples]

    return captions


class CLIPScoreCalculator:
    def __init__(self, model_name="openai/clip-vit-large-patch14", device=None):
        """
        Initialize CLIP model for score calculation.

        Args:
            model_name: CLIP model to use (default: ViT-L/14, same as used in most papers)
            device: Device to run on (auto-detected if None)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading CLIP model: {model_name}")
        print(f"Using device: {self.device}")

        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

    @torch.no_grad()
    def calculate_scores(self, images_dir, prompts, start_ind=0, end_ind=None, batch_size=32):
        """
        Calculate CLIP scores between images and their corresponding prompts.

        Args:
            images_dir: Directory containing generated images (named as 00000.png, 00001.png, etc.)
            prompts: List of prompts (indexed to match image filenames)
            start_ind: Starting index (default: 0)
            end_ind: Ending index exclusive (default: len(prompts))
            batch_size: Batch size for processing

        Returns:
            Dictionary with scores and statistics
        """
        if end_ind is None:
            end_ind = len(prompts)

        # Validate indices
        if start_ind >= len(prompts):
            raise ValueError(f"start_ind ({start_ind}) >= num_prompts ({len(prompts)})")
        if end_ind > len(prompts):
            print(f"Warning: end_ind ({end_ind}) > num_prompts ({len(prompts)}), clamping to {len(prompts)}")
            end_ind = len(prompts)

        num_images = end_ind - start_ind
        print(f"Calculating CLIP scores for {num_images} images (indices {start_ind} to {end_ind - 1})")

        all_scores = []
        missing_images = []

        # Process in batches
        indices = list(range(start_ind, end_ind))

        for batch_start in tqdm(range(0, len(indices), batch_size), desc="Computing CLIP scores"):
            batch_end = min(batch_start + batch_size, len(indices))
            batch_indices = indices[batch_start:batch_end]

            batch_images = []
            batch_prompts = []
            batch_valid_indices = []

            for idx in batch_indices:
                image_path = os.path.join(images_dir, f"{idx:05d}.png")

                if not os.path.exists(image_path):
                    missing_images.append(idx)
                    continue

                try:
                    image = Image.open(image_path).convert("RGB")
                    batch_images.append(image)
                    batch_prompts.append(prompts[idx])
                    batch_valid_indices.append(idx)
                except Exception as e:
                    print(f"Error loading image {image_path}: {e}")
                    missing_images.append(idx)
                    continue

            if not batch_images:
                continue

            # Process batch
            inputs = self.processor(
                text=batch_prompts,
                images=batch_images,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)

            outputs = self.model(**inputs)

            # Get similarity scores (diagonal of the similarity matrix)
            # Normalize features and compute cosine similarity
            image_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
            text_embeds = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)

            # Compute pairwise similarities (we want diagonal - each image with its corresponding prompt)
            similarities = (image_embeds * text_embeds).sum(dim=-1)

            # Convert to CLIP score (multiply by 100 as per convention)
            scores = similarities.cpu().numpy() * 100

            for idx, score in zip(batch_valid_indices, scores):
                all_scores.append({
                    'index': idx,
                    'score': float(score),
                    'prompt': prompts[idx][:100] + '...' if len(prompts[idx]) > 100 else prompts[idx]
                })

        if missing_images:
            print(f"Warning: {len(missing_images)} images not found")
            if len(missing_images) <= 10:
                print(f"  Missing indices: {missing_images}")

        # Calculate statistics
        scores_array = np.array([s['score'] for s in all_scores])

        stats = {
            'num_images': len(all_scores),
            'num_missing': len(missing_images),
            'mean': float(np.mean(scores_array)),
            'std': float(np.std(scores_array)),
            'median': float(np.median(scores_array)),
            'min': float(np.min(scores_array)),
            'max': float(np.max(scores_array)),
            'percentile_5': float(np.percentile(scores_array, 5)),
            'percentile_25': float(np.percentile(scores_array, 25)),
            'percentile_75': float(np.percentile(scores_array, 75)),
            'percentile_95': float(np.percentile(scores_array, 95)),
        }

        return {
            'scores': all_scores,
            'statistics': stats,
            'missing_indices': missing_images
        }


def main():
    parser = argparse.ArgumentParser(description="Calculate CLIP scores between images and prompts")
    parser.add_argument("--images", type=str, required=True,
                       help="Path to directory containing generated images")
    parser.add_argument("--coco-captions", type=str, required=True,
                       help="Path to COCO captions JSON file")
    parser.add_argument("--num-samples", type=int, default=None,
                       help="Total number of samples (prompts) to consider")
    parser.add_argument("--start-ind", type=int, default=0,
                       help="Starting index for evaluation")
    parser.add_argument("--end-ind", type=int, default=None,
                       help="Ending index (exclusive) for evaluation")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size for CLIP inference")
    parser.add_argument("--output", type=str, default=None,
                       help="Output JSON file for detailed results")
    parser.add_argument("--model", type=str, default="openai/clip-vit-large-patch14",
                       help="CLIP model to use")

    args = parser.parse_args()

    # Load prompts
    print(f"Loading captions from: {args.coco_captions}")
    prompts = load_coco_captions(args.coco_captions, num_samples=args.num_samples)
    print(f"Loaded {len(prompts)} prompts")

    # Determine end index
    end_ind = args.end_ind if args.end_ind is not None else len(prompts)

    # Calculate CLIP scores
    calculator = CLIPScoreCalculator(model_name=args.model)
    results = calculator.calculate_scores(
        images_dir=args.images,
        prompts=prompts,
        start_ind=args.start_ind,
        end_ind=end_ind,
        batch_size=args.batch_size
    )

    # Print statistics
    stats = results['statistics']
    print("\n" + "=" * 50)
    print("CLIP SCORE RESULTS")
    print("=" * 50)
    print(f"Images evaluated: {stats['num_images']}")
    print(f"Missing images:   {stats['num_missing']}")
    print("-" * 50)
    print(f"Mean CLIP Score:  {stats['mean']:.2f} ± {stats['std']:.2f}")
    print(f"Median:           {stats['median']:.2f}")
    print(f"Min / Max:        {stats['min']:.2f} / {stats['max']:.2f}")
    print("-" * 50)
    print(f"5th percentile:   {stats['percentile_5']:.2f}")
    print(f"25th percentile:  {stats['percentile_25']:.2f}")
    print(f"75th percentile:  {stats['percentile_75']:.2f}")
    print(f"95th percentile:  {stats['percentile_95']:.2f}")
    print("=" * 50)

    # Save detailed results if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed results saved to: {args.output}")

    return results


if __name__ == "__main__":
    main()
