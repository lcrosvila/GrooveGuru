# %%
import json
import matplotlib.pyplot as plt
import soundfile as sf
import numpy as np

# Read the JSON file
with open('dataset/parsed_files/!.json') as file:
    data = json.load(file)

# %% Let's check the data

# get all json files
import os
json_files = [pos_json for pos_json in os.listdir('dataset/parsed_files') if pos_json.endswith('.json')]

files_lower_than_3ms = []
min_time_increment = 1000
for file in json_files:
    data_aux = json.load(open('dataset/parsed_files/' + file))
    for chart in data_aux['charts']:
        time_values = [entry[0] for entry in chart['notes']]
        
        # check the minimum time increment
        for i in range(1, len(time_values)):
            time_increment = time_values[i] - time_values[i-1]
            if time_increment < 0.003:
                files_lower_than_3ms.append(file)
            if time_increment < min_time_increment:
                min_time_increment = time_increment

print('Minimum time increment: {} s'.format(min_time_increment))
print(files_lower_than_3ms)
print(len(files_lower_than_3ms), len(json_files))
# %% Pretty plots
for chart in data['charts']:
    # Extract time and step values from the JSON data
    step_values = [entry[1] for entry in chart['notes']]

    # Convert the time values to float and step values to string
    time_values = list(map(float, time_values))
    step_values = list(map(str, step_values))

    audio_data, sample_rate = sf.read(data['music_fp'])
    
    # Create a time axis for the audio waveform
    audio_duration = len(audio_data) / sample_rate
    audio_time = np.linspace(0, audio_duration, len(audio_data))

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the audio waveform as the background
    ax.plot(audio_time, audio_data, color='lightgray', linewidth=0.5, zorder=1)

    # Plot the steps over time
    ax.scatter(time_values, step_values, color='blue', marker='o', s=30, zorder=2)

    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Steps', fontsize=12)
    ax.set_title('Steps Over Time with Audio Waveform', fontsize=14)
    ax.grid(True)

    plt.show()