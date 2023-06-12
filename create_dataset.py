# %%

import os

json_files = [pos_json for pos_json in os.listdir('dataset/parsed_files') if pos_json.endswith('.json')]
files_to_ignore = [line.rstrip('\n') for line in open('dataset/charts_to_ignore.txt')]
json_files = [file for file in json_files if file not in files_to_ignore]

# %%
import librosa

def get_X_y(audio_fp, chart):
    # Compute the STFT of the audio signal
    audio, sr = librosa.load(audio_fp, sr=None)  # Load audio, sr=None to preserve original sample rate
    hop_length = int(0.003 * sr)  # 3ms hop length
    stft = librosa.stft(audio, hop_length=hop_length)

    # Convert time_values and step_values to STFT frames
    time_values = [entry[0] for entry in chart['notes']]
    step_values = [entry[1] for entry in chart['notes']]

    time_frames = librosa.time_to_frames(time_values, sr=sr, hop_length=hop_length)
    # Create a temporary label-to-index dictionary
    label_to_idx = {label: idx for idx, label in enumerate(step_values)}

    # Determine window size in frames for 3ms duration
    window_duration = 0.003  # 3ms
    window_size = int(window_duration * sr / hop_length)

    # Initialize aligned labels array
    aligned_labels = [None] * stft.shape[1]

    # Align labels with STFT frames
    for time_frame, label in zip(time_frames, step_values):
        label_idx = label_to_idx.get(label)
        if label_idx is not None:
            start_frame = max(0, time_frame)
            end_frame = min(len(aligned_labels), time_frame + window_size)
            aligned_labels[start_frame:end_frame] = [label] * (end_frame - start_frame)

    return stft, aligned_labels

# %%
import json
import numpy as np

if not os.path.exists('dataset/3ms_dataset'):
    os.makedirs('dataset/3ms_dataset')

# for all json files, get x and y and save it in dataset/3ms_dataset
for file in json_files:
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
        # update the json file
        data['music_fp'] = audio_fp
        with open('dataset/parsed_files/' + file, 'w') as outfile:
            json.dump(data, outfile)

    for chart in data['charts']:
        # if npz file exists, continue
        if os.path.exists('dataset/3ms_dataset/{}_{}_{}.npz'.format(file[:-5], chart['difficulty_coarse'], chart['difficulty_fine'])):
            continue
        # get x and y
        stft, aligned_labels = get_X_y(audio_fp, chart)
        # save it in dataset/3ms_dataset as an npz file
        np.savez_compressed('dataset/3ms_dataset/{}_{}_{}.npz'.format(file[:-5], chart['difficulty_coarse'], chart['difficulty_fine']), x=stft, y=aligned_labels)
        print('Saved {}_{}_{}.npz'.format(file[:-5], chart['difficulty_coarse'], chart['difficulty_fine']))