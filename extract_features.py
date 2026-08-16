"""
Extract CLIP ViT-L/14 visual features from a video at 1 fps and save as .pth.tar.

VidOSC expects features with shape (n_seconds, 768) — one 768-dim vector per second.
This script uses open_clip's ViT-L/14 (OpenAI weights) as a stand-in for the
InternVideo-MM-L14 backbone used in the original paper. The checkpoint was trained
with InternVideo features, so results are approximate; swap the model for
OpenGVLab/InternVideo1.0 for exact reproduction.

Usage:
    python extract_features.py --video ../Dicing_Onion.mp4 --video_id dicing_onion
"""
import argparse
import sys
from pathlib import Path

import ffmpeg
import numpy as np
import torch
import open_clip
from PIL import Image
from tqdm import tqdm


def extract_frames_1fps(video_path: str) -> np.ndarray:
    """Return uint8 array of shape (n_frames, H, W, 3) sampled at 1 fps."""
    probe = ffmpeg.probe(video_path)
    vs = next(s for s in probe["streams"] if s["codec_type"] == "video")
    w, h = int(vs["width"]), int(vs["height"])

    out, _ = (
        ffmpeg.input(video_path)
        .filter("fps", fps=1)
        .filter("scale", 224, 224)
        .output("pipe:", format="rawvideo", pix_fmt="rgb24")
        .run(capture_stdout=True, quiet=True)
    )
    frames = np.frombuffer(out, np.uint8).reshape([-1, 224, 224, 3])
    print(f"  Source resolution: {w}x{h}  →  {len(frames)} frames @ 1 fps")
    return frames


def extract_clip_features(
    video_path: str,
    output_path: str,
    model_name: str = "ViT-L-14",
    pretrained: str = "openai",
) -> torch.Tensor:
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Device: {device}")

    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model = model.to(device).eval()

    print(f"Extracting frames from {video_path} ...")
    frames = extract_frames_1fps(video_path)

    features = []
    with torch.no_grad():
        for frame in tqdm(frames, desc="Encoding frames"):
            img_t = preprocess(Image.fromarray(frame)).unsqueeze(0).to(device)
            feat = model.encode_image(img_t)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            features.append(feat.cpu().float())

    feat_tensor = torch.cat(features, dim=0)  # (n_frames, 768)
    print(f"Feature tensor: {feat_tensor.shape}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(feat_tensor, output_path)
    print(f"Saved → {output_path}")
    return feat_tensor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to input .mp4")
    parser.add_argument("--video_id", required=True, help="Unique ID used for the .pth.tar filename")
    parser.add_argument("--feat_dir", default="./data/feats", help="Output directory")
    args = parser.parse_args()

    output_path = str(Path(args.feat_dir) / f"{args.video_id}.pth.tar")
    extract_clip_features(args.video, output_path)


if __name__ == "__main__":
    main()
