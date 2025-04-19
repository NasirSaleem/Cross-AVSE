from argparse import ArgumentParser
from pathlib import Path
import os
import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm
import time
from thop import profile, clever_format

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

    # Prepare for metrics collection
    total_latency = 0.0
    total_audio_length = 0.0
    processed_files = 0

    # Calculate MACs/FLOPs once (assuming fixed input size)
    try:
        sample_data = dataset[0]
        # Convert numpy arrays to tensors and ensure proper shape
        dummy_input = {}
        for k, v in sample_data.items():
            if isinstance(v, np.ndarray):
                dummy_input[k] = torch.from_numpy(v).unsqueeze(0).to(device)
            elif torch.is_tensor(v):
                dummy_input[k] = v.unsqueeze(0).to(device)
            else:
                dummy_input[k] = v

        macs, flops = profile(model, inputs=(dummy_input,), verbose=False)
        macs, flops = clever_format([macs, flops], "%.3f")
        print(f"Model MACs: {macs}, FLOPs: {flops}")
    except Exception as e:
        print(f"Could not calculate MACs/FLOPs: {str(e)}")
        macs, flops = "N/A", "N/A"

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

                # Prepare input for timing
                input_data = {}
                for k, v in data.items():
                    if isinstance(v, np.ndarray):
                        input_data[k] = torch.from_numpy(v).unsqueeze(0).to(device)
                    elif torch.is_tensor(v):
                        input_data[k] = v.unsqueeze(0).to(device)
                    else:
                        input_data[k] = v

                # Measure latency
                start_time = time.time()
                estimated_audio = model.enhance(input_data).reshape(-1)
                torch.cuda.synchronize() if device == 'cuda' else None
                latency = time.time() - start_time

                # Calculate audio length in seconds
                audio_length = len(estimated_audio) / sampling_rate

                # Update metrics
                total_latency += latency
                total_audio_length += audio_length
                processed_files += 1

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

            except Exception as e:
                print(f"Error processing {data.get('scene', str(i))}: {str(e)}")
                continue

    # Calculate and print final metrics
    if processed_files > 0:
        avg_latency = total_latency / processed_files
        avg_rtf = avg_latency / (total_audio_length / processed_files)

        print(f"\nProcessing complete. Saved {processed_files} files to {enhanced_root}")
        print("\nPerformance Metrics:")
        print(f"Model MACs: {macs}")
        print(f"Model FLOPs: {flops}")
        print(f"Average Latency per file: {avg_latency:.4f} seconds")
        print(f"Average Real-Time Factor (RTF): {avg_rtf:.4f}")
        print(f"Total Processing Time: {total_latency:.2f} seconds")
        print(f"Total Audio Processed: {total_audio_length:.2f} seconds")


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