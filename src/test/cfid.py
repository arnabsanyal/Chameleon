import os
from cleanfid import fid
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default=os.path.join(os.environ.get("CHAMELEON_OUTPUT_ROOT", os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")), "outputs")), "research_baseline/images/"), help="Path to generated images")
parser.add_argument("--ref", type=str, default=os.path.join(os.environ.get("CHAMELEON_DATA_ROOT", os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")), "data")), "coco/val2014"), help="Path to reference images (COCO val2014)")
args = parser.parse_args()

print(f"Computing FID between:\n  Generated: {args.path}\n  Reference: {args.ref}")
score = fid.compute_fid(args.path, args.ref)
print(f"FID Score: {score:.2f}")