#%%
import torch
import sys
import glob
import json
import torch
import torchaudio
import matplotlib.pyplot as plt
import time
import lightning.pytorch as pl
import glob
import numpy as np
import random
import torch.nn as nn
import einops
import torch.nn.functional as F
import tqdm
import wandb
from lightning.pytorch.loggers import WandbLogger

class SMDataset(torch.utils.data.Dataset):
    def __init__(self, sm_files):
        self.hop_size = 128
        self.n_fft = 256
        self.sample_rate = 16_000
        self.max_seq_len = 1000
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=self.sample_rate, n_fft=self.n_fft, hop_length=self.hop_size)

        # load all sm files as json
        self.sm_files = [json.load(open(sm_file)) for sm_file in sm_files]
        # remove charts from sm files that have type other than dance-single
        self.sm_files = [{**file, "charts": [chart for chart in file["charts"] if chart["type"] == "dance-single"]} for file in self.sm_files]
        # remove sm files that have no charts
        self.sm_files = [file for file in self.sm_files if len(file["charts"]) > 0]

    def __len__(self):
        return len(self.sm_files)

    def __getitem__(self, idx):
        data = self.sm_files[idx]
        # get mel spectrogram from audio
        # print time to load audio
        try:
            audio, sr = torchaudio.load(data['music_fp'])
        except Exception as e:
            print(f"Error loading audio: {e}")
            return self.__getitem__(random.randint(0, len(self.sm_files) - 1))
        
        # if audio is stereo, convert to mono
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0)
        else:
            audio = audio[0]
            
        resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
        audio = resampler(audio)
        mel_spectrogram = self.mel_spectrogram(audio)
        mel_spectrogram = torch.log(mel_spectrogram + 1e-6)
        # mel_spectrogram = torch.flip(mel_spectrogram, dims=[0])
        
        # get random chart from sm file
        chart = random.choice(data['charts'])

        # convert notes to dict
        notes = [{"time": note[0], "action": note[1]} for note in chart['notes']]

        # deep copy notes
        # filter away all "0000" notes
        notes = [note for note in notes if note["action"] != "0000"]
        # add frame to notes
        notes = [{"frame": int(note["time"] * self.sample_rate // self.hop_size), **note} for note in notes]

        # add activity map to notes
        activity_map = torch.zeros(mel_spectrogram.shape[1])
        for note in notes:
            if note["frame"] < mel_spectrogram.shape[1]:
                activity_map[note["frame"]] = 1

        # flip mel spectrogram vertically
        mel_spectrogram = torch.flip(mel_spectrogram, dims=[0])

        # get random crop of mel spectrogram and activity map
        crop_start = random.randint(0, mel_spectrogram.shape[1] - self.max_seq_len)
        crop_end = crop_start + self.max_seq_len
        mel_spectrogram = mel_spectrogram[:, crop_start:crop_end]
        activity_map = activity_map[crop_start:crop_end]

        return {"mel_spectrogram": mel_spectrogram.T, "activity_map": activity_map.T}

class Model(pl.LightningModule):
    def __init__(self, input_size, hidden_size, output_size, n_layers):
        super().__init__()
        # start with a input layer
        self.input_layer = nn.Linear(input_size, hidden_size)
        # now add n_layers of conv layers
        self.conv_layers = nn.ModuleList([nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1) for _ in range(n_layers)])
        # now add a output layer
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.input_layer(x)

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

    def forward(self, x):
        return self.model(x)

    def _step(self, batch, batch_idx):
        audio_features = batch['mel_spectrogram']
        activity_map = batch['activity_map']

        # get predicted activity map
        predicted_activity_map = self(audio_features)

        # calculate loss
        loss = F.binary_cross_entropy(predicted_activity_map[:, :, 0].flatten(), activity_map.flatten())
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self.log('train/loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self.log('val/loss', loss, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

if __name__ == "__main__":

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

    trn_ds = SMDataset(trn_sm_files)
    val_ds = SMDataset(val_sm_files)

    trn_dl = torch.utils.data.DataLoader(trn_ds, batch_size=4, shuffle=True, num_workers=4, drop_last=True)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=4, drop_last=True)


    model = Model(input_size=128, hidden_size=16, output_size=1, n_layers=3)

    # create logger
    logger = WandbLogger(project="ddr")

    training_wrapper = TrainingWrapper(model)
    trainer = pl.Trainer(max_epochs=10, logger=logger)
    trainer.fit(training_wrapper, trn_dl, val_dl)