# %%
import os
import librosa
import numpy as np
from tqdm import tqdm

def process_audio(audio_path, bpm, offset, calculate_beat=False):
    # print('offset:', offset)
    # load audio
    y, sr = librosa.load(audio_path, sr=None)

    offset = -offset

    # if offset is negative, add silence to the beginning
    if offset < 0:
        y = np.concatenate((np.zeros(int(-offset * sr)), y))
    # if offset is positive, trim from the beginning
    else:
        y = y[int(offset * sr):]

    # calculate beat
    # print('metadata bpm:', bpm)
    # print('estimated measures:', (len(y) / (60 / bpm * sr)) / 4)

    samples_per_measure = np.ceil((60 * 4 * sr) / bpm)
    # print('samples per measure:', samples_per_measure)
    samples_total = np.ceil((len(y) / (60 / bpm * sr)) / 4) * samples_per_measure
    # print('samples total:', samples_total)
    # print('len y:', len(y))
    # print('samples total - len y:', samples_total - len(y))
    
    # if samples_total is greater than len(y), pad with zeros
    if samples_total > len(y):
        y = np.concatenate((y, np.zeros(int(samples_total - len(y)))))

    estimated_bpm, _ = librosa.beat.beat_track(y=y, sr=sr)
    # print('estimated bpm:', estimated_bpm)

    if calculate_beat:
        bpm = estimated_bpm
    
    # compute stft
    n_fft = 64

    hop_length = int(np.ceil((60 / bpm) * (4 / 96) * sr))
    # print('hop length:', hop_length)

    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    return S

# %%
import polars as pl
df = pl.read_json('dataset/DDR_dataset.json')

# %%
# for all tunes, get the bpm, offset and audio path and process the audio
# then save the spectrogram as a .npy file and the path to the .npy file in the dataframe

# initialize new column
df = df.with_columns(pl.Series(name="SPECTROGRAM", values=['']*len(df)))

for row in tqdm(df.iter_rows(named=True)):
    if not (row['#DISPLAYBPM'] == '' or row['#DISPLAYBPM'] == '*' or row['#DISPLAYBPM'] == '0.000'):
        bpm = float(row['#DISPLAYBPM'])
    else:
        if ',' in row['#BPMS']:
            continue
        else:
            bpm = float(row['#BPMS'].split('=')[-1][:-1])
            
    offset = float(row['#OFFSET'])
    audio_path = row['#PATH'] + row['#MUSIC']

    if not os.path.exists(audio_path):
        # try to see if there is an .ogg or .mp3 file in the same folder
        files_in_folder = os.listdir(row['#PATH'])
        for f in files_in_folder:
            if f.endswith('.ogg') or f.endswith('.mp3'):
                audio_path = row['#PATH'] + f
                break
        # otherwise skip
        continue

    # if both mp3 and ogg exist, take mp3
    if audio_path.endswith('.ogg'):
        audio_path = audio_path[:-3] + 'mp3'
        if not os.path.exists(audio_path):
            continue

    # check if npy file already exists
    if os.path.exists(audio_path[:-3] + 'npy'):
        row['SPECTROGRAM'] = audio_path[:-3] + 'npy'
        continue

    S = process_audio(audio_path, bpm, offset, calculate_beat=False)
    np.save(audio_path[:-3] + 'npy', S)

    row['SPECTROGRAM'] = audio_path[:-3] + 'npy'

df.write_json('dataset/DDR_dataset.json')
