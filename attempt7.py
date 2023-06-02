# %% LET'S MESS UP WITH THE BPMs!! Adjusting all audios to be of 150bpms
# ALSO, let's use wandb to log the results

import sys
sys.path.insert(0, '/opt/homebrew/lib/python3.10/site-packages') # to make sure I get all the pip libraries

# Log in to your W&B account
import wandb
wandb.login()

import numpy as np
import librosa
import torch
import os
import laion_clap
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

clap_model = laion_clap.CLAP_Module(enable_fusion=False)
clap_model.load_ckpt()  # download the default pretrained checkpoint.

unique_tokens = []
with open('unique_tokens.txt', 'r') as f:
    for line in f:
        unique_tokens.append(line.strip())

token_to_index = {}
for i, token in enumerate(unique_tokens):
    token_to_index[token] = i

# do index_to_token as well
index_to_token = {}
for i, token in enumerate(unique_tokens):
    index_to_token[i] = token

PAD_IDX = token_to_index['pad']
max_output_size = 2048
# sr = 48000
sr = 4800
desired_bpm = 150

# %% define the dataset and dataloader

def preprocess_audio(audio_path, offset):
    audio_array, _ = librosa.load(audio_path, sr=sr)
    tempo, _ = librosa.beat.beat_track(audio_array, sr=sr)
    
    # if the offset is negative, add zeroes at the beginning
    if offset < 0:
        audio = audio_array[abs(offset):]
    # if the offset is positive, remove the first offset samples
    else:
        audio = np.pad(audio_array, (abs(offset), 0), 'constant')

    # adjust the tempo of the audio to the desired_bpm
    # audio = librosa.effects.time_stretch(audio, desired_bpm / tempo)
    audio_embed = clap_model.get_audio_embedding_from_data(x=[audio],
                                                           use_tensor=False)
    
    # add zeroes at the end of audio_array to make it 6000000 samples long
    audio_array = np.pad(audio_array, (0, int(sr*125) - len(audio_array)), 'constant')
    return audio_array, audio_embed

# Create new dataset and dataloader classes with audio_emb, SM1 and SM2
class DDRDataLoader(torch.utils.data.Dataset):

    def __init__(self, data_files):
        self.audio_array = []
        self.audio_emb = []
        self.SM = []
        self.SM_x = []
        self.SM_y = []
        self.token_to_index = token_to_index

        for file in data_files:
            data = np.load(file, allow_pickle=True)
            chart_metadata = data['chart_metadata'].item()
            if chart_metadata['offset'] is not None:
                offset = int(float(chart_metadata['offset']) * sr)
            else:
                offset = 0

            audio_array, input_data = preprocess_audio(data['audio_file'].item(), offset)
            # remove the batch dimension from input_data
            input_data = input_data.squeeze(0)
            target_aux = [
                self.token_to_index[token] for token in data['notes']
            ]
            target_data = np.array([target_aux[0]] +
                                   [token_to_index[chart_metadata['level']]] +
                                   target_aux[1:])
            # pad target data to max length with token_to_index[';']
            target_data = np.pad(target_data,
                                 (0, max_output_size - len(target_data)),
                                 'constant',
                                 constant_values=PAD_IDX)


            self.audio_array.append(audio_array)
            self.audio_emb.append(torch.from_numpy(input_data).squeeze(0))
            self.SM.append(torch.from_numpy(target_data).squeeze(0))

            self.SM_x.append(torch.from_numpy(target_data[:-1]))
            self.SM_y.append(torch.from_numpy(target_data[1:]))

    def __getitem__(self, index):
        audio_array = self.audio_array[index]
        audio_emb = self.audio_emb[index]
        SM_x = self.SM_x[index]
        SM_y = self.SM_y[index]
        return audio_array, audio_emb, SM_x, SM_y

    def __len__(self):
        return len(self.audio_emb)

# %% define the model

class TransformerDecoder(nn.Module):

    def __init__(self, vocab_size, output_size, hidden_size, num_layers,
                 num_heads):
        super(TransformerDecoder, self).__init__()

        # vocab_size is len of vocab of sm chart
        self.emb = nn.Embedding(vocab_size,
                                hidden_size,
                                padding_idx=PAD_IDX)
        # output_size is max len of sm chart
        self.pos = nn.Embedding(output_size, hidden_size)

        self.drop = nn.Dropout(0.1)

        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(hidden_size,
                                       num_heads,
                                       dim_feedforward=hidden_size),
            num_layers)
        self.fc1 = nn.Linear(hidden_size, vocab_size)

    def forward(self, x1, audio_emb):
        b, t = x1.size()

        pos = torch.arange(0, t, dtype=torch.long,
                           device=device).unsqueeze(0)  # shape (1, t)

        tok_emb = self.emb(x1)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.pos(pos)  # position embeddings of shape (1, t, n_embd)

        x = self.drop(tok_emb + pos_emb)

        # mask is masking upper triangular part of the attention matrix
        x = self.transformer(x, memory=audio_emb)

        output = self.fc1(x)

        return output

    def generate(self, audio_emb, level, max_len=100, temperature=1.0):
        # Initialize the input with the start token and the level
        start_token = torch.tensor([token_to_index['']],
                                   dtype=torch.long,
                                   device=device)
        level_token = torch.tensor([token_to_index[level]],
                                   dtype=torch.long,
                                   device=device)
        x = torch.cat((start_token, level_token),
                      dim=0).unsqueeze(0)  # shape (1, 2)

        output = []  # List to store the generated tokens

        with torch.no_grad():
            for _ in range(max_len):
                # copy audio_emb to memory
                memory = audio_emb.clone()
                # add channel dimension
                memory = memory.unsqueeze(1)
                memory = memory.repeat(1, x.shape[-1], 1)
                # Generate token embeddings and position embeddings
                b, t = x.size()
                pos = torch.arange(t, t + 1, dtype=torch.long,
                                   device=device).unsqueeze(0)
                tok_emb = self.emb(
                    x)  # Token embeddings of shape (b, t, n_embd)
                pos_emb = self.pos(
                    pos)  # Position embeddings of shape (1, t, n_embd)

                # Combine token embeddings and position embeddings
                x_input = self.drop(tok_emb + pos_emb)

                # Apply the transformer decoder
                x_output = self.transformer(x_input, memory=memory)

                # Apply the linear layer
                logits = self.fc1(
                    x_output)  # Logits of shape (b, t, vocab_size)

                # Apply temperature to the logits
                logits /= temperature

                # Sample the next token from the logits
                probs = F.softmax(logits[:, -1, :],
                                  dim=-1)  # Shape (b, vocab_size)
                next_token = torch.multinomial(probs,
                                               num_samples=1)  # Shape (b, 1)

                # Add the generated token to the output
                output.append(next_token.item())

                # Append the generated token to the input for the next step
                x = torch.cat((x, next_token), dim=1)

                # if next_token is the end token, break
                if next_token == token_to_index[';']:
                    break
        return x

# %% define the functions

def get_dataloader(data_files, batch_size):
    dataset = DDRDataLoader(data_files)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=True)
    return dataloader

def get_model(vocab_size, output_size, hidden_size, num_layers, num_heads):
    model = TransformerDecoder(vocab_size, output_size, hidden_size,
                               num_layers, num_heads)
    return model

def validate_model(model, val_dataloader, loss_fn, batch_idx=0):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for i, (audio_array, audio_emb, SM_x, SM_y) in enumerate(val_dataloader):
            audio_emb = audio_emb.to(device)
            SM_x = SM_x.to(device)
            SM_y = SM_y.to(device)

            audio_emb = audio_emb.unsqueeze(1)
            audio_emb = audio_emb.repeat(1, max_output_size - 1, 1)

            logits = model(SM_x, audio_emb)

            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), SM_y.reshape(-1))

            total_loss += loss.item()

            # log one batch of predictions
            if i == batch_idx:
                actual_chart = '\n'.join([index_to_token[i.item()] for i in SM_y[0]])
                generated_tokens = model.generate(audio_emb[0], level=actual_chart[0], max_len=max_output_size)
                generated_chart = '\n'.join([index_to_token[i.item()] for i in generated_tokens[0]])
                
                log_audio_chart_table(audio_array, 
                                        generated_chart,
                                        actual_chart,
                                        F.softmax(logits, dim=-1).max(-1)[0].cpu().numpy())
    return total_loss / len(val_dataloader.dataset)

def log_audio_chart_table(audio, predicted, labels, probs):
    # log the audio chart table
    table = wandb.Table(columns=["Audio", "Predicted", "Labels", "Probs"])
    for i in range(len(audio)):
        table.add_data(wandb.Audio(audio[i], sample_rate=sr), predicted[i], labels[i], probs[i])
    wandb.log({"Audio Chart": table})

# %%
# train

for _ in range(5):
    #initialize wandb run
    wandb.init(
        project="ddr",
        config={
            "epochs": 10,
            "batch_size": 16,
            "lr": 1e-3,
            "vocab_size": len(unique_tokens),
            "input_size": 512,
            "hidden_size": 512,
            "num_layers": 4,
            "num_heads": 16,
            "data_dir": "/Users/lcros/Documents/ddc/data_token_new_16",
            })
        
    # Copy your config 
    config = wandb.config

    # get the data files
    files = [
        os.path.join(config.data_dir, f) for f in os.listdir(config.data_dir)
        if f.endswith('.npz')
    ]

    train_files = files[:int(len(files) * 0.9)][:10]
    val_files = [files[int(len(files) * 0.1):][0]]

    train_dl = get_dataloader(train_files, config.batch_size)
    val_dl = get_dataloader(val_files, config.batch_size)
    n_steps_per_epoch = math.ceil(len(train_dl.dataset) / config.batch_size)

    model = get_model(config.vocab_size, max_output_size, config.hidden_size,
                        config.num_layers, config.num_heads)
    
    loss_fn = torch.nn.CrossEntropyLoss(weight=None, ignore_index=PAD_IDX)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    # Train the model
    example_ct = 0
    step_ct = 0
    for epoch in range(config.epochs):
        model.train()
        for step, (audio_array, audio_emb, SM_x, SM_y) in enumerate(train_dl):
            audio_emb = audio_emb.to(device)
            SM_x = SM_x.to(device)
            SM_y = SM_y.to(device)

            audio_emb = audio_emb.unsqueeze(1)
            audio_emb = audio_emb.repeat(1, max_output_size - 1, 1)

            logits = model(SM_x, audio_emb)

            train_loss = loss_fn(logits.reshape(-1, logits.shape[-1]), SM_y.reshape(-1))
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            example_ct += len(audio_array)
            metrics = {"train/train_loss": train_loss, 
                       "train/epoch": (step + 1 + (n_steps_per_epoch * epoch)) / n_steps_per_epoch, 
                       "train/example_ct": example_ct}

            if step + 1 < n_steps_per_epoch:
                # Log train metrics to wandb 
                wandb.log(metrics)
            
            step_ct += 1
        
        val_loss = validate_model(model, val_dl, loss_fn)

        val_metrics = {"val/val_loss": val_loss}

        wandb.log({**metrics, **val_metrics})
        
        print(f"Train Loss: {train_loss:.3f}, Valid Loss: {val_loss:3f}")  

    # Close your wandb run 
    wandb.finish()  
