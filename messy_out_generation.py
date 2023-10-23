# %%
import torch
import numpy as np
from audio_processing import process_audio
# from train import Model
from check_generations import Model

# %%
sm_file = '/root/GrooveGuru/outside_dataset/Video Game Step Pack /Death By Glamour/Death By Glamour.sm'
audio_file = '/root/GrooveGuru/outside_dataset/Video Game Step Pack /Death By Glamour/Undertale OST- 068 - Death by Glamour.mp3'

# %%
bpm = 1
offset = 0.0
S = process_audio(audio_file, bpm, offset, calculate_beat=True)

# %%
token_to_idx = torch.load('./dataset/token_to_idx.pt')
idx_to_token = torch.load('./dataset/idx_to_token.pt')
seq_len = torch.load('./dataset/seq_len.pt')
audio_ft_size = torch.load('./dataset/audio_ft_size.pt')
n_tokens = torch.load('./dataset/n_tokens.pt')

# %%
# crop audio to seq_len if necessary
S = S[:, :seq_len]
# %%
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
    # PAD_IDX=token_to_idx["<pad>"],
    # idx_to_token=idx_to_token,
)

# load latest checkpoint
run = 'treasured-vortex-59' # best prior to causal masking
# run = 'atomic-leaf-126' # best after causal masking
print(torch.device)
model = Model.load_from_checkpoint("./checkpoints/"+run+"/last.ckpt")
model.eval()
model.freeze()
try:
    model = model.to(torch.device('cuda'))
except:
    print(torch.cuda.is_available())

device = model.device
print(device)

# %%
# difficulty_fine = ' 5:'
# chart_tokens = f'<sos>\n{difficulty_fine}'
difficulty_coarse = 'Difficult:'
chart_tokens = f'<sos>\n{difficulty_coarse}'
chart_tokens = [token_to_idx[t] for t in chart_tokens.split('\n')]
chart_tokens = torch.tensor(chart_tokens).unsqueeze(0).to(device)
spec = torch.tensor(S.T, dtype=torch.float32).unsqueeze(0).to(device)
# %%
generated = model.generate(spec, chart_tokens, max_len=10000)

# %%
# save generated chart

resolution = 32
filename = "./generated_charts/"+run+"_"+str(sm_file.split('/')[-1])+".txt"
with open(filename, "w") as f:
    for ii in range(len(generated)):
        tokens = [idx_to_token[idx] for idx in generated[ii]]
        tokens = [t for t in tokens if t not in ["<sos>", "<eos>", "<pad>"]]
        tokens = tokens[1:]
        downsample = 96//resolution
        tokens = tokens[::downsample]
        
        # write the chart to the txt file, with '\n,\n' every 96 tokens
        for jj in range(0, len(tokens), resolution):
            f.write("\n".join(tokens[jj:jj+resolution]))
            f.write("\n,\n")

for t in tokens:
    print(t)