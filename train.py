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


# fromhttps://pytorch.org/tutorials/beginner/transformer_tutorial.html 
def get_positional_encoding(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
    self.dropout = nn.Dropout(p=dropout)
    position = torch.arange(max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(max_len, 1, d_model)
    pe[:, 0, 0::2] = torch.sin(position * div_term)
    pe[:, 0, 1::2] = torch.cos(position * div_term)
    return pe

class Model(pl.LightningModule):
    def __init__(self,  audio_ft_size, vocab_size, seq_len, metadata_len, d_model, dim_ff, nheads, n_encoder_layers, n_decoder_layers):
        ''' 
        seq_len: length of chart sequence (equal or longer to audio sequence)
        '''
        super().__init__()
        self.save_hyperparameters()
        self.positional_encoding = get_positional_encoding(d_model, dropout=0.1,max_len=seq_len)
        self.chart_embedding = nn.Embedding(vocab_size, d_model)
        self.audio_embedding = nn.Linear(128, d_model)

        self.transformer = Transformer(d_model=d_model, nhead=nheads, num_encoder_layers=n_encoder_layers, num_decoder_layers=n_decoder_layers, dim_feedforward=dim_ff, dropout=0.1)
  
        self.output_layer = nn.Linear(d_model, 1)
        self.metadata_len = metadata_len

    def forward(self, x, y):
        '''
        x: (batch_size, seq_len, hidden_size), Audio features
        y: (batch_size, seq_len, hidden_size), Chart tokens
        '''
        # embed audio features
        zx = self.audio_embedding(x)
        # add positional encoding
        zx = zx + self.positional_encoding
        # embed chart tokens
        # add positional encoding to chart tokens
        zy = self.chart_embedding(y)
        # add positional encoding
        zy[:,self.metadata_len:] = zy[:,self.metadata_len:] + self.positional_encoding
        # pass through transformer
        zl = self.transformer(zx, zy, tgt_is_causal=True)
        logits = self.output_layer(zl)
        return logits
    
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
        x = batch
        y = x[:, 1:]
        y_hat = self(x)[:, :-1]

        ce = F.cross_entropy(y_hat.reshape(-1,self.tokenizer.vocab_size), y.reshape(-1))
        metrics["cross_entropy"] = ce
        metrics = {}
        with torch.no_grad():
            # get probability of the correct token
            y_hat_probs = F.softmax(y_hat, dim=-1)
            probability =  torch.gather(y_hat_probs, dim=-1, index=y.unsqueeze(-1))
            metrics["probability"] = probability.mean()
            # compute entropy
            entropy = -(y_hat_probs * torch.log(y_hat_probs)).sum(dim=-1)
            metrics["entropy"] = entropy.mean()
            # compute perplexity
            metrics["perplexity"] = (2**entropy).mean()

            # sort yhat by probability
            y_hat_sort = torch.argsort(y_hat_probs, dim=-1, descending=True)
            for k in [1,2,4]:
                metrics[f"accuracy@{k}"] = (y.unsqueeze(-1) == y_hat_sort[:, :, :k]).any(dim=-1).float().mean()
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
    
        
    def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
            return optimizer
    
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

    # use minimal precision

    progress_bar_callback = RichProgressBar(refresh_rate=1)

    model = Model()

    wandb_logger = WandbLogger(log_model="all", project="llm_control")
    # get name
    name = wandb_logger.experiment.name

    logger = logging.getLogger("wandb")
    logger.setLevel(logging.ERROR)

    trainer = pl.Trainer(accelerator="gpu",
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