
#%% definitions and import
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

torch.set_float32_matmul_precision('medium')


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
        PAD_IDX,
        idx_to_token,
    ):
        """
        seq_len: length of chart sequence (equal or longer to audio sequence)
        """
        super().__init__()
        self.save_hyperparameters()

        self.PAD_IDX = PAD_IDX

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
            batch_first=True,
            dropout=0.1,
        )
        self.decoder_output_layer = nn.Linear(hidden_size, decoder_vocab_size)
        self.idx_to_token = idx_to_token


    # from pytorch tutorials on translation
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to(self.device)


    # def create_mask(self, src, tgt):
    #     src_seq_len = src.shape[0]
    #     tgt_seq_len = tgt.shape[0]

    #     tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len)
    #     src_mask = torch.zeros((src_seq_len, src_seq_len)).type(torch.bool)

    #     src_padding_mask = (src == self.PAD_IDX).transpose(0, 1)
    #     tgt_padding_mask = (tgt == self.PAD_IDX).transpose(0, 1)
    #     return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

    def forward(self, encoder_fts, decoder_tokens):
        """
        encoder_ft: (batch_size, encoder_seq_len, encoder_ft_size)
        decoder_tokens: (batch_size, decoder_seq_len)
        """
        # print(encoder_fts.shape)
        # print(decoder_tokens.shape)
        # embed encoder features
        ze = self.encoder_embedding(encoder_fts)
        # add positional encoding
        ze = ze + self.positional_encoding[:, : ze.shape[1], :].to(self.device)
        # embed decoder tokens
        zd = self.decoder_embedding(decoder_tokens)
        zd = zd + self.positional_encoding[:, : zd.shape[1], :].to(self.device)

        # src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = self.create_mask(ze, zd)
        tgt_mask = self.generate_square_subsequent_mask(decoder_tokens.shape[1])#.to(self.device)
        # src_padding_mask = (ze == self.PAD_IDX).transpose(0, 1)
        tgt_padding_mask = (decoder_tokens == self.PAD_IDX)#.transpose(0, 1)
        # make tgt_padding_mask be of zeroes and -inf
        # tgt_padding_mask = tgt_padding_mask.float().masked_fill(tgt_padding_mask == 1, float('-inf')).masked_fill(tgt_padding_mask == 0, float(0.0))

        # pass through transformer
        zl = self.transformer(ze, zd, 
                              tgt_mask=Transformer.generate_square_subsequent_mask(decoder_tokens.shape[1], device=self.device), 
                            #   src_key_padding_mask=src_padding_mask, 
                              tgt_key_padding_mask=tgt_padding_mask, 
                              tgt_is_causal=True
                              )
        decoder_logits = self.decoder_output_layer(zl)
        return decoder_logits

    @torch.no_grad()
    def generate(
        self, encoder_fts, decoder_prompt_tokens, temperature=1.0, max_len=10000
    ):
        """
        Does not use KV caching so it's slow
        """
        self.eval()
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
            decoder_output_tokens.reshape(-1), ignore_index=self.PAD_IDX,
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

        # generate the first chart in the batch
        generated = self.generate(
            batch["audio_fts"], batch["chart_tokens"], temperature=1.0, max_len=10_000
        )
        # turn into string
        generated = [self.idx_to_token[idx] for idx in generated[0].tolist()]
        generated = "\n".join(generated)
        true = [self.idx_to_token[idx] for idx in batch["chart_tokens"][0].tolist()]
        true = "\n".join(true)
        # log for pytorch lightning
        columns = ["generated", "true"]
        data = [[generated, true]]
        self.logger.log_table(key="generated", columns=columns, data=data, step=self.global_step)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
    
    # def validation_epoch_end(self, validation_step_outputs):
    #     pred = validation_step_outputs[0]
        
        



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
        def __init__(self, df_path, split_spectrogram_filenames_txt, max_len=10_000):
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
            # max_chart_len = max([len(chart) for chart in self.chart_token_idx])
            # print('max_chart_len', max_chart_len)
            # max_audio_seq_len = max([audio_ft.shape[1] for audio_ft in self.song_title2audio_ft.values()])
            # print('max_audio_seq_len', max_audio_seq_len)
            # max_len = max(max_chart_len, max_audio_seq_len)
            self.max_len = max_len

            # # avg chart and audio
            # mean_chart_len = np.mean([len(chart) for chart in self.chart_token_idx])
            # print('mean_chart_len', mean_chart_len)
            # mean_audio_seq_len = np.mean([audio_ft.shape[1] for audio_ft in self.song_title2audio_ft.values()])
            # print('mean_audio_seq_len', mean_audio_seq_len)
            

            # pad all charts to the same length, crop the ones that are too long
            self.chart_token_idx = [chart[:self.max_len] for chart in self.chart_token_idx]
            self.chart_token_idx = [chart + [self.tokenizer.token_to_idx["<pad>"]] * (self.max_len - len(chart)) for chart in self.chart_token_idx]

            # audio has shape (audio_ft_size, audio_seq_len), pad to (audio_ft_size, self.max_len) if necessary or crop if too long
            self.song_title2audio_ft = {title: np.pad(audio_ft, ((0, 0), (0, self.max_len - audio_ft.shape[1])), mode='constant') if audio_ft.shape[1] < self.max_len else audio_ft[:, :self.max_len] for title, audio_ft in self.song_title2audio_ft.items()}

            # SANITY CHECK: print the shape of the first and second audio ft
            # print(list(self.song_title2audio_ft.values())[0].shape, list(self.song_title2audio_ft.values())[1].shape)

            self.seq_len = self.max_len
            print('seq_len', self.seq_len)
            self.audio_ft_size = list(self.song_title2audio_ft.values())[0].shape[0]
            print('audio_ft_size', self.audio_ft_size)
            self.n_tokens = len(self.tokenizer.token_to_idx)
            print('n_tokens', self.n_tokens)

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

    #%%
    df_path = "./dataset/DDR_dataset.json"
    dev_ds = DDRDataset(df_path,"./dataset/dev.txt")

    BATCH_SIZE = 8

    seq_len = dev_ds.seq_len
    audio_ft_size = dev_ds.audio_ft_size
    n_tokens = dev_ds.n_tokens

    trn_ds, val_ds = torch.utils.data.random_split(dev_ds, [len(dev_ds) - 100, 100])

    trn_dl = torch.utils.data.DataLoader(
        trn_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=40
    )
    val_dl = torch.utils.data.DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=40
    )

    progress_bar_callback = RichProgressBar(refresh_rate=1)


    #%%
    model = Model(
        hidden_size=32,
        n_heads=2,
        feed_forward_size=64,
        n_encoder_layers=2,
        encoder_ft_size=audio_ft_size,
        n_decoder_layers=2,
        decoder_vocab_size=n_tokens,
        max_seq_len=seq_len,
        learning_rate=1e-4,
        PAD_IDX=dev_ds.tokenizer.token_to_idx["<pad>"],
        idx_to_token=dev_ds.tokenizer.idx_to_token,
    )

    wandb_logger = WandbLogger(log_model="all", project="DDR")
    # get name
    name = wandb_logger.experiment.name

    logger = logging.getLogger("wandb")
    logger.setLevel(logging.ERROR)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[0],
        max_epochs=20,
        precision='16-mixed',
        # val_check_interval=10,
        # accumulate_grad_batches=16,
        # max_epochs=100,
        callbacks=[
            progress_bar_callback,
            pl.callbacks.ModelCheckpoint(
                dirpath=f"./checkpoints/{name}/",
                monitor="val/loss",
                mode="min",
                save_top_k=3,
                save_last=True,
                filename="{epoch}-{step}-{val/loss:.2f}",
                train_time_interval=datetime.timedelta(minutes=2),
            ),
        ],
        logger=wandb_logger,
    )

    trainer.fit(model, trn_dl, val_dl)
