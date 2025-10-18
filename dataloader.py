import os
import json
import random
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F


# -----------------------------
# Utility functions
# -----------------------------

def load_audio(path, sr=32000, mono=True):
    y, _ = librosa.load(path, sr=sr, mono=mono)
    return y

def to_mel(y, sr=32000, n_mels=128, hop_length=320, win_length=1024):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels,
                                         hop_length=hop_length, win_length=win_length)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db.astype(np.float32)

# -----------------------------
# 1. ShortAudioDataset
# -----------------------------

class ShortAudioDataset(Dataset):
    """
    Loads short audio clips stored in subfolders by class:
      root/classA/*.ogg, root/classB/*.ogg, ...
    Applies random augmentations (can be toggled).
    """
    def __init__(self, root_dirs, sr=32000, duration=5.0, augment=True,
                 n_mels=128, hop_length=320, win_length=1024, smooth_labels=True):
        """
        root_dirs: list of directories, each containing class subfolders
        """
        self.sr = sr
        self.duration = duration
        self.augment = augment
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.win_length = win_length
        self.smooth_labels_coef = 0 if not smooth_labels else 0.05

        self.files, self.labels, self.classes = self._collect_files(root_dirs)
        # Shuffle dataset
        combined = list(zip(self.files, self.labels))
        random.shuffle(combined)
        self.files[:], self.labels[:] = zip(*combined)

    def _collect_files(self, root_dirs):
        files, labels = [], []
        classes = sorted(
            [d for r in root_dirs for d in os.listdir(r)
             if os.path.isdir(os.path.join(r, d))]
        )
        class_to_idx = {cls: i for i, cls in enumerate(sorted(set(classes)))}

        for root in root_dirs:
            for cls in os.listdir(root):
                cdir = os.path.join(root, cls)
                if not os.path.isdir(cdir):
                    continue
                for f in os.listdir(cdir):
                    if f.lower().endswith((".ogg")):
                        files.append(os.path.join(cdir, f))
                        labels.append(class_to_idx[cls])
        return files, labels, list(class_to_idx.keys())

    # ----- Augmentations -----

    def _pitch_shift(self, y):
        n_steps = random.uniform(-1, 1)
        return librosa.effects.pitch_shift(y, sr=self.sr, n_steps=n_steps)

    def _add_noise(self, y):
        noise_amp = 5e-3 * np.random.uniform() * np.amax(y)
        return y + noise_amp * np.random.normal(size=y.shape[0])

    def _time_shift(self, y):
        shift = int(random.uniform(-0.2, 0.2) * len(y))
        return np.roll(y, shift)

    def _random_slice(self, y):
        target_len = int(self.duration * self.sr)
        if len(y) <= target_len:
            return y
        start = random.randint(0, len(y) - target_len)
        return y[start:start + target_len]

    def _merge_with_other(self, y):
        other_idx = random.choice(range(len(self.files)))
        other_path, other_label = self.files[other_idx], self.labels[other_idx]
        y2 = load_audio(other_path, sr=self.sr)
        min_len = min(len(y), len(y2))
        y, y2 = y[:min_len], y2[:min_len]
        alpha = random.uniform(0.3, 0.7)
        return alpha * y + (1 - alpha) * y2, other_label

    def _time_freq_mask(self, mel):
        mel = mel.copy()
        # time mask
        t = mel.shape[1]
        t_mask = random.randint(0, int(0.1 * t))
        t0 = random.randint(0, t - t_mask)
        mel[:, t0:t0 + t_mask] = mel.mean()
        # freq mask
        f = mel.shape[0]
        f_mask = random.randint(0, int(0.1 * f))
        f0 = random.randint(0, f - f_mask)
        mel[f0:f0 + f_mask, :] = mel.mean()
        return mel

    # -------------------------

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path, l = self.files[idx], self.labels[idx]
        label = torch.zeros(len(self.classes), dtype=torch.float32)
        label[l] = 1
        y = load_audio(path, sr=self.sr)
        y = self._random_slice(y)
        y_init = y.copy()

        if self.augment:
            if random.random() < 0.5: y = self._pitch_shift(y)
            if random.random() < 0.5: y = self._add_noise(y)
            if random.random() < 0.5: y = self._time_shift(y)
            if random.random() < 0.2:
                y, other_label = self._merge_with_other(y)
                label[other_label] = 1

        mel = to_mel(y, sr=self.sr, n_mels=self.n_mels,
                     hop_length=self.hop_length, win_length=self.win_length)
        mel_init = to_mel(y_init, sr=self.sr, n_mels=self.n_mels,
                          hop_length=self.hop_length, win_length=self.win_length)
        if self.augment and random.random() < 0.5:
            mel = self._time_freq_mask(mel)

        mel = torch.tensor(mel)
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)
        mel_init = torch.tensor(mel_init)
        mel_init = (mel_init - mel_init.mean()) / (mel_init.std() + 1e-6)
        return mel.unsqueeze(0), mel_init.unsqueeze(0), label


# -----------------------------
# 2. LongAudioSEDDataset
# -----------------------------

class LongAudioSEDDataset(Dataset):
    """
    For SED (frame-level) training:
      - loads long audios
      - reads JSON file with timestamps of target events
      - outputs (mel_tensor, label_tensor) for LSTM/attention head
    """
    def __init__(self, audio_dir, json_file, sr=32000,
                 n_mels=128, hop_length=320, win_length=1024,
                 window_size=10.0, step_size=5.0, duration=None):
        self.audio_dir = audio_dir
        self.json_file = json_file
        self.sr = sr
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.win_length = win_length
        self.window_size = window_size
        self.step_size = step_size
        self.duration = duration

        self.timestamps_dict = self._load_json(json_file)
        self.audio_files = [
            f for f in os.listdir(audio_dir)
            if f.lower().endswith((".ogg"))
        ]

    def _load_json(self, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        timestamps:dict = data["audios"]
        return {
            int(idx): data["timestamps"] for idx, data in timestamps.items()
        }

    def _frames_from_timestamps(self, y_len, timestamps):
        n_frames = 1 + int(y_len / self.hop_length)
        label = np.zeros(n_frames, dtype=np.float32)
        for (s, e) in timestamps:
            s_frame = int((s * self.sr) / self.hop_length)
            e_frame = int((e * self.sr) / self.hop_length)
            label[s_frame:e_frame] = 1.0
        return label

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        fname = self.audio_files[idx]
        y = load_audio(os.path.join(self.audio_dir, fname), sr=self.sr)

        mel = to_mel(y, sr=self.sr, n_mels=self.n_mels,
                     hop_length=self.hop_length, win_length=self.win_length)
        label = self._frames_from_timestamps(len(y), self.timestamps_dict.get(idx, []))
        # truncate/pad label to mel frames
        n_frames = mel.shape[1]
        label = label[:n_frames]
        if len(label) < n_frames:
            pad = n_frames - len(label)
            label = np.pad(label, (0, pad))

        # Normalize mel
        mel = torch.tensor((mel - mel.mean()) / (mel.std() + 1e-6)).unsqueeze(0)
        label = torch.tensor(label, dtype=torch.float32)

        return mel, label