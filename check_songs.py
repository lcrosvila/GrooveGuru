# %%
import os
import librosa
import sys

sys.path.insert(0, '/opt/homebrew/lib/python3.10/site-packages')

import simfile
from simfile.timing import Beat, TimingData
from simfile.notes import NoteData
from simfile.notes.timed import time_notes
import numpy as np


# %%
# get the name of all .ogg files in the subfolders of '/Users/lcros/Documents/ddc/DDR Classics'
# and save them in a list, removing the ones that don't have a .sm file
song_list = []
for root, dirs, files in os.walk("/Users/lcros/Documents/ddc/DDR Classics"):
    for file in files:
        file_end = '.'+file.split('.')[-1]
        if file.endswith('.ogg') or file.endswith('.mp3'):
            # check if it has a .sm file
            chart_file = os.path.join(root, file).replace(file_end, '.sm')
            if os.path.exists(chart_file):
                song_list.append(os.path.join(root, file))

# %%
# check if the song has more than one bpm in their file, if so, remove from song_list
for song in song_list:
    file_end = '.'+song.split('.')[-1]
    chart_file = song.replace(file_end, '.sm')
    with open(chart_file, 'r') as f:
        chart = f.read()
        bpm = chart.split('#BPMS:')[1].split(';')[0].split(',')
        if len(bpm) > 1 or "Get Down Tonight" in song:
            song_list.remove(song)
            print(song, 'removed')

# %%
# read the song chart from its .sm file from the same folder and split it into sections based on the string '//-'

max_notes = 0
# unique_tokens = set()
# load the unique tokens from a file
with open('unique_tokens.txt', 'r') as f:
    unique_tokens = set(f.read().split('\n'))

save_dir = 'data_token_new_16'

for audio_file in song_list:
    print(audio_file)
    # audio_file = song_list[0]
    file_end = '.'+audio_file.split('.')[-1]
    chart_file = audio_file.replace(file_end, '.sm')
    with open(chart_file, 'r') as f:
        chart = f.read()
        chart = chart.split('//-')

    chart_metadata = {}

    for c in chart[0].split('#'):
        if c != '' and len(c.split(':')) > 1:
            key = c.split(':')[0].lower()
            value = c.split(':')[1].strip()
            # remove \n, ; and quotes from the value
            value = value.replace('\n', '').replace(';', '').replace('"', '')
            chart_metadata[key] = value

    if 'timesignatures' not in chart_metadata.keys():
        print(audio_file, 'removed (no timesignatures)')
        continue
    if chart_metadata['timesignatures'] != '0.000=4=4':
        print(audio_file, 'removed (not 4/4)')
        continue

    # audio, sr = librosa.load(audio_file, mono=True)

    # # # split the audio into 4 beat sections according to the bpm
    # bpm = int(float(chart_metadata['bpms'].split('=')[1]))

    # # estimate the bpm using librosa
    # tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)
    
    # # if the difference between the estimated tempo and the bpm is greater than 5, skip this chart
    # if abs(tempo - bpm) > 5:
    #     print(audio_file, 'removed (tempo)')
    #     continue


    # # adjust the offset

    # if chart_metadata['offset'] is not None:
    #     offset = int(float(chart_metadata['offset']) * sr)
    #     # if the offset is negative, add zeroes at the beginning
    #     if offset < 0:
    #         audio = audio[abs(offset):]
    #     # if the offset is positive, remove the first offset samples
    #     else:
    #         audio = np.pad(audio, (abs(offset), 0), 'constant')

    # # 60/bpm is the time between beats in seconds
    # # sr*60/bpm is the number of samples between beats
    # # sr*60/bpm*4 is the number of samples in each section
    # sections = []
    # i = 0
    # while i < len(audio):
    #     section = audio[i:i + int(sr * 60 / bpm * 4)]
    #     # if section is not full, pad it with zeros at the end
    #     if len(section) < int(sr * 60 / bpm * 4):
    #         section = np.pad(section,
    #                          (0, int(sr * 60 / bpm * 4) - len(section)),
    #                          'constant')
    #     sections.append(section)
    #     i += int(sr * 60 / bpm * 4)

    # X = np.array(sections[:-1])

    notes = {}

    for c in chart[1:]:
        if 'double' in c:
            continue
        difficulty = c.split(':')[3].lower()
        chart_metadata['difficulty'] = difficulty
        level = str(int(c.split(':')[4]))
        chart_metadata['level'] = level
        # remove '\n' and spaces from difficulty
        difficulty = difficulty.replace('\n', '').replace(' ', '')

        # if not exists, make a new folder called save_dir
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        file_name = os.path.join(
            save_dir,
            audio_file.split('/')[-1].replace(
                '.ogg', '_{}_{}.npz'.format(difficulty, level)))

        if file_name not in os.listdir(save_dir):      

            notes_aux = c.split(':')[-1].split('\n')
            # notes = c.split(':')[-1].split(',')

            notes = []
            # convert all notes that contain both ',' and 'measure' to just ',' (e.g. ',     measure 1' to ',')
            for n in notes_aux:
                if ',' in n and 'measure' in n:
                    notes.append(',')
                elif 'measure' in n and not ',' in n:
                    continue
                else:
                    notes.append(n)

            # check if 'measure' is in any of the notes
            # if so, remove it and the next note
            # for i in range(len(notes)):
            #     if 'measure' in notes[i]:
                    # # the new notes[i] is the all after the first '\n'
                    # notes[i] = '\n' + '\n'.join(notes[i].split('\n')[1:])
            
            # if the maximum characters of notes is greater than 4, skip this chart
            if max([len(note) for note in notes]) > 4:
                print(audio_file, 'removed (max characters)')
                continue

            # get the index where the notes are ';' and remove everything after it
            notes = notes[:notes.index(';')+1]
            
            # get the indices where the notes are ',' or '' or ';'
            indices = [i for i, x in enumerate(notes) if x == ',' or x == '' or x == ';']
            lengths = [indices[i+1] - indices[i] - 1 for i in range(len(indices)-1)]

            # if there is any length different from 2, 4, 8 or 16, skip the chart
            if any([length not in [2, 4, 8, 16] for length in lengths]):
                print(audio_file, 'removed (length)')
                continue

            zero_additions = [16 // (indices[i+1] - indices[i] - 1) for i in range(len(indices)-1)]

            print(zero_additions)
            j = 0
            addition = zero_additions[j] - 1
            new_notes = []
            for i in range(len(notes)):
                new_notes.append(notes[i])
                # if notes[i] == ';' exit the for loop
                if notes[i] == ';':                    
                    break
                if notes[i] == ',':
                    j += 1
                    addition = zero_additions[j] - 1
                elif notes[i] != '' and addition >= 0:
                    # add ['0000'] * addition after notes[i]
                    for _ in range(addition):
                        new_notes.append('0000')

            indices_new = [i for i, x in enumerate(new_notes[:-1]) if x == ',' or x == '' or x == ';']
            lengths_new = [indices_new[i+1] - indices_new[i] - 1 for i in range(len(indices_new)-1)]

            print(lengths_new)
            # if not all values un length_new are equal to 16, raise error
            if not all([l == 16 for l in lengths_new]):
                raise ValueError('not all lengths are 16')


            # add the unique tokens to the set
            unique_tokens.update(new_notes)
            unique_tokens.update([level])

            # update max_notes
            if len(new_notes) > max_notes:
                max_notes = len(new_notes)

            # if (len(X) == len(notes)):
            # save the chart metadata and notes to a .npz file
            np.savez_compressed(file_name,
                                audio_file=audio_file,
                                # X=X,
                                chart_metadata=chart_metadata,
                                bpm=bpm,
                                notes=new_notes)
                                # notes=c.split(':')[-1].split(','))

# # unique_tokens.update(['pad'])
print('max_notes', max_notes)
print('vocab', len(unique_tokens))
# save the unique tokens to a file
with open('unique_tokens.txt', 'w') as f:
    for token in unique_tokens:
        f.write(token + '\n')
# %%

import numpy as np

tempos = []
for audio_file in song_list:
    chart_file = audio_file.replace('.ogg', '.sm')
    # print(chart_file)
    data = simfile.open(chart_file)
    audio, sr = librosa.load(audio_file, mono=True)

    for chart in data.charts[:3]:
        if audio_file.replace(
                '.ogg', '_{}_{}.npz'.format(
                    chart.difficulty, chart.meter)) in os.listdir(
                        '/Users/lcros/Documents/ddc/DDR Classics/'):
            continue

        note_data = NoteData(chart)
        timing_data = TimingData(data, chart)

        # Calculate tempo (beats per minute)
        # tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        tempo = int(timing_data.bpms[0].value)

        # tempos.append(tempo)

        # split the audio file into sections based on the tempo
        # 60/tempo is the time between beats in seconds
        # sr*60/tempo is the number of samples between beats
        # sr*60/tempo/8 is the number of samples in each section
        sections = []
        i = 0
        while i < len(audio):
            section = audio[i:i + int(sr * 60 / tempo / 8)]
            # if section is not full, pad it with zeros at the end
            if len(section) < int(sr * 60 / tempo / 8):
                section = np.pad(section,
                                 (0, int(sr * 60 / tempo / 8) - len(section)),
                                 'constant')
            sections.append(section)
            i += int(sr * 60 / tempo / 8)

        X = np.array(sections)
        y = np.zeros((len(sections), 4))

        # check if the last note is in the last section

        for n, time in zip(note_data, time_notes(note_data, timing_data)):
            # print(n.beat, time.time, n.column, int(n.beat*8))
            # print('actual beat', int(time.time*tempo/60))
            beat = int(time.time * tempo / 60)
            y[int(beat * 8)][n.column] = 1

        # print(X.shape, y.shape, tempo, audio.shape, sr)
        # print(len(audio)/sr*tempo/60)

        # save X and y to a .npz file where the name ends with _{}.npz where {} is chart.difficulty
        # np.savez_compressed(audio_file.replace(
        #     '.ogg', '_{}_{}.npz'.format(chart.difficulty, chart.meter)),
        #                     X=X,
        #                     y=y,
        #                     tempo=tempo,
        #                     meter=chart.meter,
        #                     sr=sr)

# %%
# plot the tempos histogram
import matplotlib.pyplot as plt

plt.hist(tempos, bins=20)
plt.show()
# # %%
# print(X.shape)
# # %%
# # Split audio into sections based on onsets
# sections = []
# for i in range(len(onset_times) - 1):
#     start = int(onset_times[i] * sr)
#     end = int(onset_times[i + 1] * sr)
#     section = y[start:end]
#     sections.append(section)

# print(audio_file, len(sections))

# # read the song chart from its .sm file from the same folder and split it into sections based on the string '//-'
# chart_file = audio_file.replace('.ogg', '.sm')
# with open(chart_file, 'r') as f:
#     chart = f.read()
#     chart = chart.split('//-')

# chart_metadata = {}

# for c in chart[0].split('#'):
#     if c != '':
#         key = c.split(':')[0].lower()
#         value = c.split(':')[1].strip()
#         # remove \n, ; and quotes from the value
#         value = value.replace('\n', '').replace(';', '').replace('"', '')
#         chart_metadata[key] = value

# notes = {}

# for c in chart[1:]:
#     difficulty = c.split(':')[3].lower()
#     level = c.split(':')[4].lower()
#     print(c.split(':')[5].split(','))
#     print(len(c.split(':')[-1].split('\n')))

# #%%
# # try with simfile

# import simfile

# data = simfile.open(chart_file)
# print(data.keys())

# # %%
# print(data.charts[3].keys())

# # %%
# from simfile.notes import NoteData

# print(data.charts[3].radarvalues)
# for note, b in zip(NoteData(data.charts[3]), beats):
#     print(note.beat, b / sr)

# l = 0
# for _ in NoteData(data.charts[3]):
#     l += 1
# print(l, len(beats))

# #%%
# import simfile
# from simfile.timing import Beat, TimingData
# from simfile.notes import NoteData
# from simfile.notes.timed import time_notes

# print(chart_file)
# chart = data.charts[1]
# note_data = NoteData(chart)
# timing_data = TimingData(data, chart)
# print(timing_data.bpms)
# print(tempo)
# for timed_note, t in zip(time_notes(note_data, timing_data), onset_times):
#     print(timed_note, t)

# # %%
# from IPython.display import Audio

# Audio(audio_file)
