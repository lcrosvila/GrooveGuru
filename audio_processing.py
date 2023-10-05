# %%
# find all .sm files in the dataset folder
import os

sm_files = []
for root, dirs, files in os.walk('dataset'):
    for file in files:
        if file.endswith('.sm'):
            sm_files.append(os.path.join(root, file))

#select a random .sm file
import random

sm_path = random.choice(sm_files)
print(sm_path)
# check if there is a corresponding .ogg file otherwise try with .mp3
if os.path.isfile(sm_path[:-2] + 'ogg'):
    audio_path = sm_path[:-2] + 'ogg'
elif os.path.isfile(sm_path[:-2] + 'mp3'):
    audio_path = sm_path[:-2] + 'mp3'
else:
    raise Exception('No audio file found')
# audio_path = 'dataset/In The Groove/Anubis/Anubis.ogg'
# sm_path = 'dataset/In The Groove/Anubis/Anubis.sm'

# %%

with open(sm_path, 'r') as f:
    sm = f.read()

for line in sm.split('\n'):
    if 'bpm' in line.lower():
        if '=' in line:
            if ',' in line:
                bpms = [b.split('=') for b in line.split(':')[-1][:-1].split(',')]
                # flatten
                bpms = [float(b) for sublist in bpms for b in sublist]
                bpm = max(bpms)
            else:
                bpm = float(line.split('=')[-1][:-1])
        else:
            bpm = float(line.split(':')[-1][:-1])
    if 'offset' in line.lower():
        offset = float(line.split(':')[-1][:-1])

def get_charts(sm_txt):
    charts = []
    for c in sm_txt.split('//---------------dance')[1:]:
        if len(c) == 0:
            continue
        if 'single' in c.split(':')[0].lower():
            charts.append(c.split(':')[-1][:-2])

    return charts

charts = get_charts(sm)
import re
# clean up if there is measure
# Define the pattern to match
pattern = r'// measure \d+'
cleaned_chart = re.sub(pattern, '', charts[0])
# replace all \n by ' ' and make all multiple spaces into single spaces
cleaned_chart = re.sub(r'\n', ' ', cleaned_chart)
cleaned_chart = re.sub(r'\s+', ' ', cleaned_chart)

# replace all 'M' with '0'
cleaned_chart = re.sub(r'M', '0', cleaned_chart)
# replace all '4' with '2'
cleaned_chart = re.sub(r'4', '2', cleaned_chart)

measures = cleaned_chart.split(',')

num_measure = len(measures)
# %% NOTE: this is the part that is important (the rest we can forget about)

import librosa
import numpy as np

def process_audio(audio_path, bpm, offset, calculate_beat=False):
    print('offset:', offset)
    # load audio
    y, sr = librosa.load(audio_path, sr=None)

    # if offset is negative, add silence to the beginning
    if offset < 0:
        y = np.concatenate((np.zeros(int(-offset * sr)), y))
    # if offset is positive, trim from the beginning
    else:
        y = y[int(offset * sr):]

    # calculate beat
    print('metadata bpm:', bpm)
    estimated_bpm, _ = librosa.beat.beat_track(y=y, sr=sr)
    print('estimated bpm:', estimated_bpm)

    if calculate_beat:
        bpm = estimated_bpm
    
    # compute stft
    n_fft = 2048

    hop_length = int(np.ceil((60 / bpm) * (4 / 96) * sr))

    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    return S

# %%
S = process_audio(audio_path, bpm, offset, calculate_beat=False)
print('spectrogram shape:', S.shape)
print('measures:', num_measure)
print('measures * 96:', num_measure * 96)

# %%
# plot

import matplotlib.pyplot as plt
import librosa.display

plt.figure(figsize=(10, 4))
librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
                         y_axis='log', x_axis='time')
plt.title('Power spectrogram')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.show()