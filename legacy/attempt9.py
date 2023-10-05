# %%
import json
import matplotlib.pyplot as plt
import soundfile as sf
import numpy as np

json_files = [pos_json for pos_json in os.listdir('dataset/parsed_files') if pos_json.endswith('.json')]
files_to_ignore = [line.rstrip('\n') for line in open('dataset/charts_to_ignore.txt')]
json_files = [file for file in json_files if file not in files_to_ignore]

# %%
import os
import librosa

for file in json_files[:20]:
    print(file)
    data = json.load(open('dataset/parsed_files/' + file))
    files_dir = '/'.join(data['sm_fp'].split('/')[:-1])
    audio_fp = data['music_fp']

    if not os.path.exists(audio_fp):
        type = '.ogg'
        # if there is no .ogg file or .mp3, continue
        if not any(file.endswith('.ogg') for file in files_dir):
            if any(file.endswith('.mp3') for file in files_dir):
                type = '.mp3'
            else:
                continue
        audio_fp = files_dir + [file for file in files_dir if file.endswith(type)][0]

    audio, sr = librosa.load(audio_fp, sr=None)  # Load audio, sr=None to preserve original sample rate

    # create an fft that we will filter to get peaks
    fft = np.fft.fft(audio)
    fft = np.abs(fft)
    fft = fft / np.max(fft)

    # filter fft to get peaks
    fft = fft[:len(fft) // 2]
    peaks = []
    for i in range(1, len(fft) - 1):
        if fft[i - 1] < fft[i] > fft[i + 1]:
            peaks.append(i)

    # get the bpm
    bpm = librosa.beat.tempo(audio, sr=sr)
    print(bpm)
    # get the time between beats
    time_between_beats = 60 / bpm
    print(time_between_beats)

    # get the time between peaks
    time_between_peaks = time_between_beats / 4
    print(time_between_peaks)
    