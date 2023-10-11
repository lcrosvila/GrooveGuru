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


# from https://pytorch.org/tutorials/beginner/transformer_tutorial.html 
def get_positional_encoding(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
    self.dropout = nn.Dropout(p=dropout)
    position = torch.arange(max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(max_len, 1, d_model)
    pe[:, 0, 0::2] = torch.sin(position * div_term)
    pe[:, 0, 1::2] = torch.cos(position * div_term)
    return pe

class Model(pl.LightningModule):
    def __init__(self, nheads, hidden_size, feed_forward_size, n_encoder_layers, n_decoder_layers, decoder_vocab_size, encoder_ft_size, max_seq_len):
        ''' 
        seq_len: length of chart sequence (equal or longer to audio sequence)
        '''
        super().__init__()
        self.save_hyperparameters()
        self.positional_encoding = get_positional_encoding(hidden_size, dropout=0.1,max_len=max_seq_len)
        self.decoder_embedding = nn.Embedding(decoder_vocab_size, hidden_size)
        self.encoder_embedding = nn.Linear(encoder_ft_size, hidden_size)
        self.transformer = Transformer(d_model=hidden_size, nhead=nheads, num_encoder_layers=n_encoder_layers, num_decoder_layers=n_decoder_layers, dim_feedforward=feed_forward_size, dropout=0.1)
        self.decoder_output_layer = nn.Linear(hidden_size, decoder_vocab_size)

    def forward(self, encoder_fts, decoder_tokens):
        '''
        encoder_ft: (batch_size, encoder_seq_len, encoder_ft_size)
        decoder_tokens: (batch_size, decoder_seq_len, 1)
        '''
        # embed encoder features
        ze = self.encoder_embedding(encoder_fts)
        # add positional encoding
        ze = ze + self.positional_encoding[:, :ze.shape[1], :]
        # embed decoder tokens
        zd = self.decoder_embedding(decoder_tokens)
        zd = zd + self.positional_encoding[:, :zd.shape[1], :]

        # pass through transformer
        zl = self.transformer(ze,zd, tgt_is_causal=True)
        decoder_logits = self.decoder_output_layer(zl)
        return decoder_logits
    
    def step(self, batch, batch_idx):
        '''
        batch: (x, y) where x is audio features and y is chart tokens
        '''
        x, y = batch
        logits = self(x, y)
        y_tgt = y[:,1:]
        logits_tgt = logits[:,:-1]
        loss = F.cross_entropy(logits_tgt.reshape(-1, logits_tgt.shape[-1]), y_tgt.reshape(-1))
        return loss
    
     
    def step(self, batch, batch_idx):
        encoder_ft, decoder_tokens = batch
        decoder_output_logits = self(encoder_ft, decoder_tokens)
        
        decoder_output_tokens = decoder_tokens[:, 1:]
        decoder_output_logits = decoder_output_logits[:, :-1]

        ce = F.cross_entropy(decoder_output_logits.reshape(-1, decoder_output_logits.shape[-1]), decoder_output_tokens.reshape(-1))
        metrics["cross_entropy"] = ce
        metrics = {}
        # TODO: check that this code is correct
        with torch.no_grad():
            # get probability of the correct token
            decoder_output_probs = F.softmax(decoder_output_logits, dim=-1)
            probability =  torch.gather(decoder_output_probs, dim=-1, index=decoder_output_tokens.unsqueeze(-1)).squeeze(-1)
            metrics["probability"] = probability.mean()
            # sort yhat by probability
            decoder_output_probs_sort = torch.argsort(decoder_output_probs, dim=-1, descending=True)
            for k in [1,2,4]:
                metrics[f"accuracy@{k}"] = (decoder_output_tokens.unsqueeze(-1) == decoder_output_probs_sort[:, :, :k]).any(dim=-1).float().mean()
        return metrics
   
    def training_step(self, batch, batch_idx):
        metrics = self.step(batch, batch_idx)
        for metric in metrics:
            self.log(f"trn/{metric}", metrics[metric], prog_bar=True)
        loss = metrics["cross_entropy"]
        self.log("trn/loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            metrics = self.step(batch, batch_idx)
        for metric in metrics:
            self.log(f"val/{metric}", metrics[metric], prog_bar=True)
        loss = metrics["cross_entropy"]
        self.log("val/loss", loss, prog_bar=True)
        return loss
    
    # def on_validation_start(self) -> None:
    #     with torch.no_grad():
    #         # generate a few samples
    #         print(sanitize_abc(self.generate(prompt="")))
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
    
if __name__ == "__main__":

    SEED = 0
    torch.manual_seed(SEED)
    pl.seed_everything(SEED)
    random.seed(SEED)

    os.environ["WANDB_SILENT"] = "true"

    BATCH_SIZE = 32
    
    trn_ds = DDSDataset(trn_data_tokenized,seq_len=SEQ_LEN)
    val_ds = DDSDataset(val_data_tokenized,seq_len=SEQ_LEN)

    trn_dl = torch.utils.data.DataLoader(trn_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    progress_bar_callback = RichProgressBar(refresh_rate=1)

    model = Model()

    wandb_logger = WandbLogger(log_model="all", project="llm_control")
    # get name
    name = wandb_logger.experiment.name

    logger = logging.getLogger("wandb")
    logger.setLevel(logging.ERROR)

    trainer = pl.Trainer(
    accelerator="gpu",
    devices=[6],
    precision=16,
    val_check_interval=100,
    callbacks=[progress_bar_callback,
            pl.callbacks.ModelCheckpoint(
            dirpath=f"./checkpoints/{name}/",
            monitor="val/loss",
            mode="min",
            save_top_k=3,
            save_last=True,
            filename="{epoch}-{step}-{val/loss:.2f}",
            train_time_interval = datetime.timedelta(minutes=2),)],
    logger=wandb_logger,
    )

    trainer.fit(model,
     trn_dl,
     val_dl)