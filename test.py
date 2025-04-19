from argparse import ArgumentParser
from pathlib import Path
import os
import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from config import sampling_rate
from dataset import AVSEDataModule
from model import AVSEModule
from utils.generic import str2bool


def main(args):
    # Initialize paths with proper handling
    enhanced_root = Path(args.save_root) / args.model_uid
    os.makedirs(enhanced_root, exist_ok=True)

    # Initialize datamodule
    datamodule = AVSEDataModule(batch_size=1, lips=args.lips)
    datamodule.setup(stage='test')

    # Get the correct dataset
    if args.dev_set and args.eval_set:
        raise RuntimeError("Select either dev set or test set")
    elif args.dev_set:
        dataset = datamodule.val_dataloader().dataset
    elif args.eval_set:
        dataset = datamodule.test_dataloader().dataset
    else:
        raise RuntimeError("Select one of dev set or test set")

    try:
        model = AVSEModule.load_from_checkpoint(args.ckpt_path)
        print(f"Model loaded from {args.ckpt_path}")
    except Exception as e:
        raise FileNotFoundError(f"Cannot load model weights: {args.ckpt_path}\nError: {str(e)}")

    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    model.to(device)
    model.eval()

    processed_files = 0
    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc="Processing audio files"):
            try:
                data = dataset[i]

                # Sanitize filename
                scene_name = "".join(c for c in data['scene'] if c.isalnum() or c in (' ', '-', '_')).strip()
                if not scene_name:
                    scene_name = f"audio_{i:04d}"

                filename = f"{scene_name}.wav"
                enhanced_path = enhanced_root / filename

                if enhanced_path.exists():
                    continue

                # Process audio - REMOVED .cpu() since enhance() returns numpy array
                estimated_audio = model.enhance(data).reshape(-1)

                # Validate and normalize audio
                if np.isnan(estimated_audio).any():
                    print(f"Warning: NaN values detected in {scene_name}, skipping")
                    continue

                estimated_audio = estimated_audio / (np.abs(estimated_audio).max() + 1e-7)
                estimated_audio = np.clip(estimated_audio, -0.99, 0.99)

                # Save file
                enhanced_path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(
                    str(enhanced_path),
                    estimated_audio,
                    samplerate=sampling_rate,
                    subtype='PCM_16'
                )
                processed_files += 1

            except Exception as e:
                print(f"Error processing {data.get('scene', str(i))}: {str(e)}")
                continue

    print(f"\nProcessing complete. Saved {processed_files} files to {enhanced_root}")


if __name__ == '__main__':
    parser = ArgumentParser(description="Audio-Visual Speech Enhancement Testing Script")
    parser.add_argument("--ckpt_path", type=str, required=True,
                        help="Path to model checkpoint file (.ckpt)")
    parser.add_argument("--save_root", type=str, required=True,
                        help="Root directory to save enhanced audio")
    parser.add_argument("--model_uid", type=str, required=True,
                        help="Unique identifier for this test run")
    parser.add_argument("--dev_set", type=str2bool, default=False,
                        help="Use development/validation set")
    parser.add_argument("--eval_set", type=str2bool, default=False,
                        help="Use evaluation/test set")
    parser.add_argument("--cpu", type=str2bool, default=False,
                        help="Force CPU usage (default: False)")
    parser.add_argument("--lips", type=str2bool, default=False,
                        help="Use lip data mode (default: False)")

    args = parser.parse_args()

    # Verify paths
    if not os.path.isfile(args.ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found: {args.ckpt_path}")

    main(args)