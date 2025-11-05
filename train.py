#%%
import argparse
import sys
import glob
import json
import torch
import torchaudio
import torchaudio.functional as AF
import matplotlib.pyplot as plt
import time
import lightning.pytorch as pl
import numpy as np
import random
import torch.nn as nn
import einops
import torch.nn.functional as F
import tqdm
import wandb
from pathlib import Path
from lightning.pytorch.loggers import WandbLogger

class SMDataset(torch.utils.data.Dataset):
    def __init__(self, sm_files, cache_path=None):
        self.hop_size = 128
        self.n_fft = 256
        self.sample_rate = 16_000
        self.max_seq_len = 1000
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate, n_fft=self.n_fft, hop_length=self.hop_size
        )

        self.cache_path = Path(cache_path) if cache_path else None
        self.cache = []
        self.failed_to_load = 0
        self.too_short = 0
        self.loaded_from_disk = False

        if self.cache_path and self.cache_path.exists():
            payload = torch.load(self.cache_path, map_location="cpu")
            self.cache = payload.get("samples", [])
            self.failed_to_load = payload.get("failed_to_load", 0)
            self.too_short = payload.get("too_short", 0)
            self.loaded_from_disk = True
        else:
            iterator = tqdm.tqdm(sm_files, desc="Caching dataset", unit="file")
            for sm_file in iterator:
                try:
                    with open(sm_file, "r") as f:
                        data = json.load(f)
                except Exception as e:
                    print(f"Error reading {sm_file}: {e}")
                    self.failed_to_load += 1
                    continue

                charts = [chart for chart in data.get("charts", []) if chart.get("type") == "dance-single"]
                if not charts:
                    continue

                music_fp = data.get("music_fp")
                if not music_fp:
                    self.failed_to_load += 1
                    continue

                try:
                    audio, sr = torchaudio.load(music_fp)
                except Exception as e:
                    print(f"Error loading audio {music_fp}: {e}")
                    self.failed_to_load += 1
                    continue

                if audio.shape[0] > 1:
                    audio = audio.mean(dim=0)
                else:
                    audio = audio.squeeze(0)

                if sr != self.sample_rate:
                    audio = AF.resample(audio, sr, self.sample_rate)

                mel_spectrogram = self.mel_spectrogram(audio)
                mel_spectrogram = torch.log(mel_spectrogram + 1e-6)
                mel_spectrogram = torch.flip(mel_spectrogram, dims=[0])

                if mel_spectrogram.shape[1] < self.max_seq_len:
                    self.too_short += 1
                    continue

                self.cache.append({
                    "mel_spectrogram": mel_spectrogram,
                    "charts": charts,
                })

            if self.cache_path:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "samples": self.cache,
                        "failed_to_load": self.failed_to_load,
                        "too_short": self.too_short,
                    },
                    self.cache_path,
                )
                print(f"SMDataset cache saved to {self.cache_path}")

        summary_source = "loaded" if self.loaded_from_disk else "cached"
        print(
            f"SMDataset summary ({summary_source}): cached={len(self.cache)} failed_to_load={self.failed_to_load} too_short={self.too_short}"
        )

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, idx):
        data = self.cache[idx]
        mel_spectrogram = data["mel_spectrogram"]

        chart = random.choice(data['charts'])

        notes = [{"time": note[0], "action": note[1]} for note in chart['notes']]
        notes = [note for note in notes if note["action"] != "0000"]
        notes = [{"frame": int(note["time"] * self.sample_rate // self.hop_size), **note} for note in notes]

        activity_map = torch.zeros(mel_spectrogram.shape[1])
        for note in notes:
            if note["frame"] < mel_spectrogram.shape[1]:
                activity_map[note["frame"]] = 1

        crop_start = random.randint(0, mel_spectrogram.shape[1] - self.max_seq_len)
        crop_end = crop_start + self.max_seq_len
        mel_spectrogram = mel_spectrogram[:, crop_start:crop_end]
        activity_map = activity_map[crop_start:crop_end]
        difficulty = torch.tensor(int(chart['difficulty_fine']))

        return {"mel_spectrogram": mel_spectrogram.T, "activity_map": activity_map.T, "difficulty": difficulty}

class Model(pl.LightningModule):
    def __init__(self, input_size, hidden_size, output_size, n_layers, max_difficulty):
        super().__init__()
        # start with a input layer
        self.input_layer = nn.Linear(input_size, hidden_size)
        # now add n_layers of conv layers
        self.conv_layers = nn.ModuleList([nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1) for _ in range(n_layers)])
        # now add a output layer
        self.output_layer = nn.Linear(hidden_size, output_size)

        self.difficulty_embedding = nn.Embedding(max_difficulty, hidden_size)

    def forward(self, x, difficulty):
        x = self.input_layer(x)
        difficulty_z = self.difficulty_embedding(difficulty)
        x = x + difficulty_z[:, None, :]
        x = einops.rearrange(x, 'b t c -> b c t')
        for conv_layer in self.conv_layers:
            x = x + conv_layer(x)
        x = einops.rearrange(x, 'b c t -> b t c')
        x = self.output_layer(x)
        x = torch.sigmoid(x)
        return x

class TrainingWrapper(pl.LightningModule):
    def __init__(self, model, learning_rate=1e-3):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate

    def _step(self, batch, batch_idx):
        audio_features = batch['mel_spectrogram']
        activity_map = batch['activity_map']
        difficulty = batch['difficulty']
        # one hot encode difficulty
        # get predicted activity map
        predicted_activity_map = self.model(audio_features, difficulty)

        # calculate loss
        loss = F.binary_cross_entropy(predicted_activity_map[:, :, 0].flatten(), activity_map.flatten())
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self.log('train/loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # on first batch, log the predicted and true activity maps
        loss = self._step(batch, batch_idx)
        self.log('val/loss', loss, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the GrooveGuru model")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="artefacts/dataset_cache/",
        help="Directory where dataset caches will be stored/loaded.",
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    train_cache_path = cache_dir / "train.pt" if cache_dir else None
    val_cache_path = cache_dir / "val.pt" if cache_dir else None

    # wandb init
    wandb.init(project="ddr")

    # seed 
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    sm_files = glob.glob('./parsed_dataset/*.json')

    sm_files = random.sample(sm_files, len(sm_files))


    # keep 90 for dev and 10 for test
    dev_sm_files = sm_files[:int(len(sm_files)*0.9)]
    tst_sm_files = sm_files[int(len(sm_files)*0.9):]

    # keep 90 of dev for train and 10 for val
    trn_sm_files = dev_sm_files[:int(len(dev_sm_files)*0.9)]
    val_sm_files = dev_sm_files[int(len(dev_sm_files)*0.9):]

    trn_ds = SMDataset(trn_sm_files, cache_path=train_cache_path)
    val_ds = SMDataset(val_sm_files, cache_path=val_cache_path)

    BATCH_SIZE = 64

    trn_dl = torch.utils.data.DataLoader(trn_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, drop_last=True)


    model = Model(input_size=128, hidden_size=256, output_size=1, n_layers=6, max_difficulty=30)

    # create logger
    logger = WandbLogger(project="ddr")

    training_wrapper = TrainingWrapper(model)
    trainer = pl.Trainer(max_epochs=1000, logger=logger)
    trainer.fit(training_wrapper, trn_dl, val_dl)