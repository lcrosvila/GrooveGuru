# %%
# use the ngrams with the transformer encoder-decoder model

import torch
import torch.nn as nn
import torchaudio
from torch.nn import Transformer

# Define the Transformer model
class AudioTransformer(nn.Module):
    def __init__(self, input_dim, vocab_size, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6):
        super(AudioTransformer, self).__init__()

        self.embedding = nn.Embedding(input_dim, d_model)
        self.transformer = Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers
        )
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, src):
        src = self.embedding(src)
        output = self.transformer(src, src)
        output = self.fc(output)

        return output

# Define audio feature extraction function
def extract_audio_features(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)
    mfcc = torchaudio.transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=13,
        melkwargs={'n_fft': 400, 'hop_length': 160, 'n_mels': 23, 'center': False}
    )(waveform)

    return mfcc