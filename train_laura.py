#%%
import argparse
import sys
import glob
import json
from pkg_resources import safe_name
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
import os

def flip_chart(chart):
    flipped_notes = []
    for note in chart['notes']:
        time, action = note
        flipped_action = action[::-1]  # Reverse the action string
        flipped_notes.append([time, flipped_action])
    flipped_chart = chart.copy()
    flipped_chart['notes'] = flipped_notes
    return flipped_chart

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
                

                charts = [chart for chart in data.get("charts", []) if chart.get("type") == "dance-single" and chart.get("difficulty_fine") < 30]
                # add flipped charts
                charts += [flip_chart(chart) for chart in data.get("charts", []) if chart.get("type") == "dance-single" and chart.get("difficulty_fine") < 30]
                
                if not charts:
                    continue

                music_fp = data.get("music_fp")
                if not music_fp:
                    self.failed_to_load += 1
                    continue

                try:
                    audio, sr = torchaudio.load(music_fp)
                except Exception as e:
                    safe_name = os.fsencode(music_fp).decode('utf-8', 'replace')
                    print(f"Error loading audio {safe_name}: {e}")
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

        # 50% chance to crop around a note for better class balance
        if random.random() < 0.5 and len(notes) > 0:
            # Pick a random note
            note = random.choice(notes)
            center = note["frame"]
            # Crop around it (with bounds checking)
            crop_start = max(0, center - self.max_seq_len // 2)
            crop_start = min(crop_start, mel_spectrogram.shape[1] - self.max_seq_len)
        else:
            crop_start = random.randint(0, mel_spectrogram.shape[1] - self.max_seq_len)
        
        crop_end = crop_start + self.max_seq_len
        mel_spectrogram = mel_spectrogram[:, crop_start:crop_end]
        activity_map = activity_map[crop_start:crop_end]
        difficulty = torch.tensor(int(chart['difficulty_fine']))

        return {"mel_spectrogram": mel_spectrogram.T, "activity_map": activity_map.T, "difficulty": difficulty}

class Model(pl.LightningModule):
    def __init__(self, input_size, hidden_size, output_size, n_layers, max_difficulty, max_seq_len, n_heads):
        super().__init__()
        # start with a input layer
        self.input_layer = nn.Linear(input_size, hidden_size)
        # now add n_layers of transformer encoder layers
        self.main_block = nn.ModuleList([nn.TransformerEncoderLayer(d_model=hidden_size, nhead=n_heads, batch_first=True, dim_feedforward=hidden_size*4) for _ in range(n_layers)])
        # now add a output layer (no sigmoid - we'll use logits for BCE with logits)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.time_embedding = nn.Parameter(torch.randn(max_seq_len, hidden_size), requires_grad=True)

        self.difficulty_embedding = nn.Embedding(max_difficulty, hidden_size)

    def forward(self, x, difficulty):
        x = self.input_layer(x)
        difficulty_z = self.difficulty_embedding(difficulty)

        time_z = self.time_embedding[None,...]
        x = x + difficulty_z[:, None, :] + time_z
        for block in self.main_block:
            x = block(x)
        x = self.output_layer(x)
        # Return logits (no sigmoid)
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
        
        # Get predicted activity map (logits)
        predicted_activity_map = self.model(audio_features, difficulty)

        # Calculate positive weight to balance classes
        # This gives more weight to the minority class (notes)
        num_negatives = (activity_map == 0).sum().float()
        num_positives = (activity_map == 1).sum().float()
        pos_weight = num_negatives / num_positives.clamp(min=1.0)
        pos_weight = pos_weight.clamp(max=5.0)
        
        # Use binary_cross_entropy_with_logits with pos_weight
        loss = F.binary_cross_entropy_with_logits(
            predicted_activity_map[:, :, 0].flatten(),
            activity_map.flatten(),
            pos_weight=pos_weight
        )
        
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self.log('train/loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def _log_model_visualizations(self, batch, batch_idx):
        # first log the self similarity of difficulty embeddings
        difficulty_embeddings = self.model.difficulty_embedding.weight
        self_similarity = torch.matmul(difficulty_embeddings, difficulty_embeddings.T).cpu()
        wandb.log({"difficulty_self_similarity": wandb.Image(self_similarity.unsqueeze(0))}, step=self.global_step)
        # then log the self similarity of time embeddings
        time_embeddings = self.model.time_embedding
        self_similarity = torch.matmul(time_embeddings, time_embeddings.T).cpu()
        wandb.log({"time_self_similarity": wandb.Image(self_similarity.unsqueeze(0))}, step=self.global_step)
    
    def _log_predicted_and_true_activity_maps(self, batch, batch_idx):
        predicted_activity_map_logits = self.model(batch['mel_spectrogram'], batch['difficulty'])
        # Apply sigmoid to get probabilities
        predicted_activity_map = torch.sigmoid(predicted_activity_map_logits)
        true_activity_map = batch['activity_map']
        
        # log the melspec, predicted and true activity maps in one plot
        # take first example
        predicted_activity_map = predicted_activity_map[0].cpu().numpy()
        true_activity_map = true_activity_map[0].cpu().numpy()
        melspec = batch['mel_spectrogram'][0].cpu().numpy()

        # show with threshold  0.5 and 0.9, 0.95
        thresholds = [0.05, 0.1, 0.25, 0.5, 0.9, 0.95]
        # plot them with subplots
        fig, axs = plt.subplots(3+len(thresholds), 1, figsize=(10, 10))
        axs[0].imshow(melspec.T, aspect='auto')
        axs[0].set_title('Mel Spectrogram')
        axs[1].imshow(predicted_activity_map.T, aspect='auto', interpolation='none', vmin=0, vmax=1)
        axs[1].set_title('Predicted Activity (Probability)')
        axs[2].imshow(true_activity_map[None,...], aspect='auto', interpolation='none', vmin=0, vmax=1)
        axs[2].set_title('True Activity')
        for i, threshold in enumerate(thresholds):
            axs[3+i].imshow((predicted_activity_map.T > threshold).astype(float), aspect='auto', interpolation='none', vmin=0, vmax=1)
            axs[3+i].set_title(f'Threshold {threshold}')
        fig.tight_layout()
        wandb.log({"activity_map": wandb.Image(fig)}, step=self.global_step)
        plt.close(fig)
    
    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:
            self._log_predicted_and_true_activity_maps(batch, batch_idx)
            self._log_model_visualizations(batch, batch_idx)
        
        # Calculate loss
        loss = self._step(batch, batch_idx)
        self.log('val/loss', loss, prog_bar=True, on_epoch=True)
        
        # Calculate metrics for positive class
        predicted_activity_map_logits = self.model(batch['mel_spectrogram'], batch['difficulty'])
        predicted_activity_map = torch.sigmoid(predicted_activity_map_logits)
        pred_binary = (predicted_activity_map[:, :, 0] > 0.5).float()
        
        # Calculate precision, recall, F1 for positive class
        tp = ((pred_binary == 1) & (batch['activity_map'] == 1)).sum()
        fp = ((pred_binary == 1) & (batch['activity_map'] == 0)).sum()
        fn = ((pred_binary == 0) & (batch['activity_map'] == 1)).sum()
        tn = ((pred_binary == 0) & (batch['activity_map'] == 0)).sum()
        
        precision = tp / (tp + fp).clamp(min=1)
        recall = tp / (tp + fn).clamp(min=1)
        f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-8)
        accuracy = (tp + tn) / (tp + tn + fp + fn).clamp(min=1)
        
        self.log('val/precision', precision, prog_bar=True)
        self.log('val/recall', recall, prog_bar=True)
        self.log('val/f1', f1, prog_bar=True)
        self.log('val/accuracy', accuracy)
        
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

    # sm_files = random.sample(sm_files, len(sm_files))
    sm_files = random.sample(sm_files, 1000)

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
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, drop_last=False)

    model = Model(input_size=128, hidden_size=256, output_size=1, n_layers=3, max_difficulty=30, max_seq_len=1000, n_heads=8)

    # create logger
    logger = WandbLogger(project="ddr", log_model=False)

    training_wrapper = TrainingWrapper(model)
    trainer = pl.Trainer(max_epochs=1000, logger=logger)
    trainer.fit(training_wrapper, trn_dl, val_dl)