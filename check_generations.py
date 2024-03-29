import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, RichProgressBar
import math
from torch import nn
from torch.nn import Transformer
from torch.utils.data import dataset
import torch.nn.functional as F
import random
import os
import datetime
import logging
import polars
import numpy as np


# from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
def get_positional_encoding(d_model: int, max_len: int = 5000):
    position = torch.arange(max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(max_len, 1, d_model)
    pe[:, 0, 0::2] = torch.sin(position * div_term)
    pe[:, 0, 1::2] = torch.cos(position * div_term)
    return pe.reshape(1, max_len, d_model)


class Model(pl.LightningModule):
    def __init__(
        self,
        hidden_size,
        n_heads,
        feed_forward_size,
        n_encoder_layers,
        encoder_ft_size,
        n_decoder_layers,
        decoder_vocab_size,
        max_seq_len,
        learning_rate,
    ):
        """
        seq_len: length of chart sequence (equal or longer to audio sequence)
        """
        super().__init__()
        self.save_hyperparameters()
        self.positional_encoding = get_positional_encoding(
            d_model=hidden_size, max_len=max_seq_len
        )
        self.decoder_embedding = nn.Embedding(decoder_vocab_size, hidden_size)
        # TODO: encoder embedding should be a cnn that takes in the spectrogram
        # self.encoder_embedding = nn.Conv1d(encoder_ft_size, hidden_size, kernel_size=1)
        self.encoder_embedding = nn.Linear(encoder_ft_size, hidden_size)
        self.transformer = Transformer(
            d_model=hidden_size,
            nhead=n_heads,
            num_encoder_layers=n_encoder_layers,
            num_decoder_layers=n_decoder_layers,
            dim_feedforward=feed_forward_size,
            dropout=0.1,
            batch_first=True,
        )
        self.decoder_output_layer = nn.Linear(hidden_size, decoder_vocab_size)

    def forward(self, encoder_fts, decoder_tokens):
        """
        encoder_ft: (batch_size, encoder_seq_len, encoder_ft_size)
        decoder_tokens: (batch_size, decoder_seq_len)
        """
        # embed encoder features
        ze = self.encoder_embedding(encoder_fts)
        # add positional encoding
        ze = ze + self.positional_encoding[:, : ze.shape[1], :].to(self.device)
        # embed decoder tokens
        zd = self.decoder_embedding(decoder_tokens)
        zd = zd + self.positional_encoding[:, : zd.shape[1], :].to(self.device)

        # pass through transformer
        zl = self.transformer(ze, zd)
        decoder_logits = self.decoder_output_layer(zl)
        return decoder_logits

    def generate(
        self, encoder_fts, decoder_prompt_tokens, temperature=1.0, max_len=1000
    ):
        """
        Does not use KV caching so it's slow
        """
        while decoder_prompt_tokens.shape[1] < max_len:
            decoder_logits = self(encoder_fts, decoder_prompt_tokens)[:, -1, :]
            decoder_probs = F.softmax(decoder_logits / temperature, dim=-1)
            sampled_token = torch.multinomial(decoder_probs, num_samples=1)
            decoder_prompt_tokens = torch.cat(
                [decoder_prompt_tokens, sampled_token], dim=-1
            )
        return decoder_prompt_tokens

    def step(self, batch, batch_idx):
        encoder_ft = batch["audio_fts"]
        decoder_tokens = batch["chart_tokens"]
        decoder_output_logits = self(encoder_ft, decoder_tokens)

        decoder_output_tokens = decoder_tokens[:, 1:]
        decoder_output_logits = decoder_output_logits[:, :-1]

        ce = F.cross_entropy(
            decoder_output_logits.reshape(-1, decoder_output_logits.shape[-1]),
            decoder_output_tokens.reshape(-1),
        )
        metrics = {}
        metrics["cross_entropy"] = ce
        # TODO: check that this code is correct
        with torch.no_grad():
            # get probability of the correct token
            decoder_output_probs = F.softmax(decoder_output_logits, dim=-1)
            probability = torch.gather(
                decoder_output_probs, dim=-1, index=decoder_output_tokens.unsqueeze(-1)
            ).squeeze(-1)
            metrics["probability"] = probability.mean()
            # sort yhat by probability
            decoder_output_probs_sort = torch.argsort(
                decoder_output_probs, dim=-1, descending=True
            )
            for k in [1, 2, 4]:
                metrics[f"accuracy@{k}"] = (
                    (
                        decoder_output_tokens.unsqueeze(-1)
                        == decoder_output_probs_sort[:, :, :k]
                    )
                    .any(dim=-1)
                    .float()
                    .mean()
                )
        return metrics

    def training_step(self, batch, batch_idx):
        metrics = self.step(batch, batch_idx)
        for metric in metrics:
            self.log(f"trn/{metric}", metrics[metric], prog_bar=True)
        loss = metrics["cross_entropy"]
        self.log("trn/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:
            generated = self.generate(
                batch["audio_fts"], batch["chart_tokens"], temperature=1.0, max_len=50
            )
        with torch.no_grad():
            metrics = self.step(batch, batch_idx)
        for metric in metrics:
            self.log(f"val/{metric}", metrics[metric], prog_bar=True)
        loss = metrics["cross_entropy"]
        self.log("val/loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


if __name__ == "__main__":
    SEED = 0
    torch.manual_seed(SEED)
    pl.seed_everything(SEED)
    random.seed(SEED)

    # os.environ["WANDB_SILENT"] = "true"

    class ChartTokenizer():
        def __init__(self, all_charts):
            all_charts_tokens = [chart.split("\n") for chart in all_charts]
            all_charts_flat = [item for sublist in all_charts_tokens for item in sublist]
            all_tokens = list(set(all_charts_flat))
            # sort 
            all_tokens = sorted(all_tokens)
            self.token_to_idx = {token: idx for idx, token in enumerate(all_tokens)}
            self.idx_to_token = all_tokens

        def transform(self, chart):
            chart_tokens = chart.split("\n")
            chart_tokens_idx = [self.token_to_idx[token] for token in chart_tokens]
            return chart_tokens_idx
    
    class DDRDataset(torch.utils.data.Dataset):
        def __init__(self, df_path, split_spectrogram_filenames_txt):
            # read df json with polars
            self.df = polars.read_json(df_path)

            split_spectrogram_filenames = set(open(split_spectrogram_filenames_txt).read().split("\n"))
            # remove empty string
            split_spectrogram_filenames = split_spectrogram_filenames - {''}
            # filter out charts that are not in split_spectrogram_filenames
            self.df = self.df.filter(self.df['SPECTROGRAM'].is_in(split_spectrogram_filenames))
            # SANITY CHECK: check if the resulting set(self.df['SPECTROGRAM']) is the same as split_spectrogram_filenames
            # print(set(self.df['SPECTROGRAM']) == split_spectrogram_filenames)

            chart_tokens = [ f'<sos>\n {row["NOTES_difficulty_fine"]}\n{row["NOTES_preproc"]}\n<eos>\n<pad>' for row in self.df.iter_rows(named=True)]
            self.df = self.df.with_columns(polars.Series(name="chart_tokens", values=chart_tokens)) 
            self.tokenizer = ChartTokenizer(self.df["chart_tokens"].to_list()) 
            self.chart_token_idx = [self.tokenizer.transform(chart) for chart in self.df["chart_tokens"].to_list()] 

            # TODO: don't store copies of the spectrograms that belong to the same chart
            # self.song_title2audio_ft = {
            #     row["SPECTROGRAM"]: np.load(row["SPECTROGRAM"])
            #     for row in self.df.filter(self.df['SPECTROGRAM'].is_duplicated()).iter_rows(named=True)
            # }
            self.song_title2audio_ft = {
                spec: np.load(spec)
                for spec in split_spectrogram_filenames
            }
            # SANITY CHECK: print the length of the song_title2audio_ft dict
            # print(len(self.song_title2audio_ft))

            # find longest chart and audio
            max_chart_len = max([len(chart) for chart in self.chart_token_idx])
            max_audio_seq_len = max([audio_ft.shape[1] for audio_ft in self.song_title2audio_ft.values()])

            max_len = max(max_chart_len, max_audio_seq_len)

            # pad all charts to the same length
            self.chart_token_idx = [chart + [self.tokenizer.token_to_idx["<pad>"]] * (max_len - len(chart)) for chart in self.chart_token_idx]

            # audio has shape (audio_ft_size, audio_seq_len), pad to (audio_ft_size, max_audio_seq_len)
            self.song_title2audio_ft = {song_title: np.pad(audio_ft, ((0, 0), (0, max_len - audio_ft.shape[1])), mode="constant") for song_title, audio_ft in self.song_title2audio_ft.items()}
            # SANITY CHECK: print the shape of the first and second audio ft
            # print(list(self.song_title2audio_ft.values())[0].shape, list(self.song_title2audio_ft.values())[1].shape)

            self.seq_len = max_len
            self.audio_ft_size = list(self.song_title2audio_ft.values())[0].shape[0]
            self.n_tokens = len(self.tokenizer.token_to_idx)

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            title = self.df.row(idx, named=True)["SPECTROGRAM"]
            audio_fts = torch.tensor(self.song_title2audio_ft[title], dtype=torch.float32) # (audio_ft_size, audio_seq_len)
            return {
                "audio_fts": audio_fts.T, # (audio_seq_len, audio_ft_size)
                "chart_tokens": torch.tensor(self.chart_token_idx[idx]),
            }

    # class MockDataset(torch.utils.data.Dataset):
    #     def __init__(self, n, audio_ft_size, n_tokens, seq_len):
    #         self.audio_ft_size = audio_ft_size
    #         self.n_tokens = n_tokens
    #         self.seq_len = seq_len
    #         self.n = n
    #         pass

    #     def __len__(self):
    #         return self.n

    #     def __getitem__(self, idx):
    #         return {
    #             "audio_fts": torch.randn(self.seq_len, self.audio_ft_size),
    #             "chart_tokens": torch.randint(0, self.n_tokens, (self.seq_len,)),
    #         }

    df_path = "./dataset/DDR_dataset.json"
    dev_ds = DDRDataset(df_path,"./dataset/dev.txt")

    BATCH_SIZE = 32

    seq_len = dev_ds.seq_len
    audio_ft_size = dev_ds.audio_ft_size
    n_tokens = dev_ds.n_tokens

    trn_ds, val_ds = torch.utils.data.random_split(dev_ds, [len(dev_ds) - 100, 100])

    trn_dl = torch.utils.data.DataLoader(
        trn_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
    )
    val_dl = torch.utils.data.DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )


    model = Model(
        hidden_size=32,
        n_heads=2,
        feed_forward_size=128,
        n_encoder_layers=2,
        encoder_ft_size=audio_ft_size,
        n_decoder_layers=2,
        decoder_vocab_size=n_tokens,
        max_seq_len=seq_len,
        learning_rate=1e-3,
    )

    
    # load latest checkpoint
    # run = 'treasured-vortex-59' # best prior to causal masking
    run = 'atomic-leaf-126' # best after causal masking
    model = Model.load_from_checkpoint("./checkpoints/"+run+"/last.ckpt")
    model.eval()
    model.freeze()
    model.cuda()

    # generate the charts for the validation set using the model validation_step

    batch = next(iter(val_dl))
    batch = {k: v.cuda() for k, v in batch.items()}

    generated = model.generate(
        batch["audio_fts"], batch["chart_tokens"], temperature=1.0, max_len=50
    )
    
    resolution = 32
    # save generated charts together with their spectrogram names in a txt file
    for i in range(20):
        # if folder 'generated_charts'/run does not exist, create it
        if not os.path.exists("generated_charts/"+run):
            os.makedirs("generated_charts/"+run)
        
        filename = f"generated_charts/{run}/{dev_ds.df['SPECTROGRAM'][i].split('/')[-1][:-4]}_{dev_ds.df['NOTES_difficulty_coarse'][i]}_{dev_ds.df['NOTES_difficulty_fine'][i]}.txt"
        
        with open(filename, "w") as f:
            for ii in range(len(generated)):
                tokens = [dev_ds.tokenizer.idx_to_token[idx] for idx in generated[ii]]
                tokens = [t for t in tokens if t not in ["<sos>", "<eos>", "<pad>"]]
                tokens = tokens[1:]
                downsample = 96//resolution
                tokens = tokens[::downsample]
                
                # write the chart to the txt file, with '\n,\n' every 96 tokens
                for jj in range(0, len(tokens), resolution):
                    f.write("\n".join(tokens[jj:jj+resolution]))
                    f.write("\n,\n")
