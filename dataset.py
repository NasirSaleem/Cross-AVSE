import logging
import os
import random
from os.path import join, isfile
import cv2
import numpy as np
import torch
from decord import VideoReader, cpu
from pytorch_lightning import LightningDataModule
from scipy.io import wavfile
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from config import *


class AVSEDataset(Dataset):
    def __init__(self, scenes_root, shuffle=True, seed=SEED, subsample=1,
                 clipped_batch=True, sample_items=True, test_set=False, lips=False, rgb=False):
        super().__init__()
        self.img_width, self.img_height = (96, 96) if lips else (224, 224)
        self.lips = lips
        self.test_set = test_set
        self.clipped_batch = clipped_batch
        self.scenes_root = scenes_root
        self.files_list = self._build_files_list()
        if shuffle:
            random.seed(SEED)
            random.shuffle(self.files_list)
        if subsample != 1:
            self.files_list = self._subsample_list(self.files_list, sample_rate=subsample)
        logging.info(f"Found {len(self.files_list)} utterances")
        self.rgb = rgb
        self.sample_items = sample_items

    def _build_files_list(self):
        files_list = []
        for file in os.listdir(self.scenes_root):
            if file.endswith("mixed.wav"):
                if self.lips:
                    files = (
                        join(self.scenes_root, file.replace("mixed", "target")),
                        join(self.scenes_root, file.replace("mixed", "interferer")),
                        join(self.scenes_root, file),
                        join(self.scenes_root.replace("scenes", "lips"), file.replace("_mixed.wav", "_silent.mp4")),
                    )
                else:
                    files = (
                        join(self.scenes_root, file.replace("mixed", "target")),
                        join(self.scenes_root, file.replace("mixed", "interferer")),
                        join(self.scenes_root, file),
                        join(self.scenes_root, file.replace("_mixed.wav", "_silent.mp4")),
                    )
                if not self.test_set:
                    if all(isfile(f) for f in files):
                        files_list.append(files)
                else:
                    files_list.append(files)
        return files_list

    def _subsample_list(self, inp_list, sample_rate):
        random.shuffle(inp_list)
        return [inp_list[i] for i in range(int(len(inp_list) * sample_rate))]

    def __len__(self):
        return len(self.files_list)

    def __getitem__(self, idx):
        while True:
            try:
                data = {}
                if self.sample_items:
                    clean_file, noise_file, noisy_file, mp4_file = random.choice(self.files_list)
                else:
                    clean_file, noise_file, noisy_file, mp4_file = self.files_list[idx]
                data["noisy_audio"], data["clean"], data["video_frames"] = self._get_data(clean_file, noise_file,
                                                                                          noisy_file, mp4_file)
                data['scene'] = clean_file.replace(self.scenes_root, "").replace("_target.wav", "").replace("/", "")
                return data
            except Exception as e:
                logging.error(f"Error in loading data: {e}")

    def _load_wav(self, wav_path):
        return wavfile.read(wav_path)[1].astype(np.float32) / (2 ** 15)

    def _get_data(self, clean_file, noise_file, noisy_file, mp4_file):
        noisy = self._load_wav(noisy_file)
        vr = VideoReader(mp4_file, ctx=cpu(0))
        clean = self._load_wav(clean_file) if isfile(clean_file) else np.zeros_like(noisy)

        if self.clipped_batch:
            if clean.shape[0] > max_audio_len:
                clip_idx = random.randint(0, clean.shape[0] - max_audio_len)
                video_idx = int((clip_idx / sampling_rate) * frames_per_second)
                clean = clean[clip_idx:clip_idx + max_audio_len]
                noisy = noisy[clip_idx:clip_idx + max_audio_len]
                frames = vr.get_batch(list(range(video_idx, min(video_idx + max_frames, len(vr))))).asnumpy()
            else:
                clean = np.pad(clean, (0, max_audio_len - clean.shape[0]))
                noisy = np.pad(noisy, (0, max_audio_len - noisy.shape[0]))
                frames = vr.get_batch(list(range(min(len(vr), max_frames)))).asnumpy()
        else:
            frames = vr.get_batch(list(range(len(vr)))).asnumpy()

        if not self.rgb:
            frames = np.array([cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in frames], dtype=np.float32)
        else:
            frames = np.array(frames, dtype=np.float32)
        frames /= 255.0

        if len(frames) < max_frames:
            if self.rgb:
                padding = np.zeros((max_frames - len(frames), self.img_height, self.img_width, 3), dtype=frames.dtype)
            else:
                padding = np.zeros((max_frames - len(frames), self.img_height, self.img_width), dtype=frames.dtype)
            frames = np.concatenate([frames, padding], axis=0)

        return noisy, clean, frames.transpose(0, 3, 1, 2) if self.rgb else frames[np.newaxis, ...]


class AVSEDataModule(LightningDataModule):
    def __init__(self, batch_size=16, lips=False, rgb=False, num_workers=4):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.lips = lips
        self.rgb = rgb

        self.train_dataset = AVSEDataset(join(DATA_ROOT, "train/scenes"), lips=lips, rgb=rgb)
        self.val_dataset = AVSEDataset(join(DATA_ROOT, "dev/scenes"), lips=lips, rgb=rgb)
        self.test_dataset = AVSEDataset(join(DATA_ROOT, "eval/scenes"), lips=lips, rgb=rgb, test_set=True)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )


if __name__ == '__main__':
    datamodule = AVSEDataModule(batch_size=1, lips=False, rgb=False, num_workers=0)

    print("\n=== Testing Training Dataset ===")
    train_dataset = datamodule.train_dataset
    for i in tqdm(range(min(3, len(train_dataset))), desc="Training Samples"):
        data = train_dataset[i]
        print(f"\nSample {i}:")
        for k, v in data.items():
            if isinstance(v, np.ndarray):
                print(f"{k:>15}: shape={str(v.shape):<20} range=[{v.min():.4f}, {v.max():.4f}]")
            else:
                print(f"{k:>15}: {v}")

    print("\n=== Testing Validation Dataset ===")
    val_dataset = datamodule.val_dataset
    for i in tqdm(range(min(3, len(val_dataset))), desc="Validation Samples"):
        data = val_dataset[i]
        print(f"\nSample {i}:")
        for k, v in data.items():
            if isinstance(v, np.ndarray):
                print(f"{k:>15}: shape={str(v.shape):<20} range=[{v.min():.4f}, {v.max():.4f}]")
            else:
                print(f"{k:>15}: {v}")