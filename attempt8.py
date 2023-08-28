# %%
# This will take inspiration from: https://github.com/cpuguy96/StepCOVNet
# Or maybe: https://github.com/guillefix/transflower-lightning
import os
import numpy as np
import librosa
import json
import torch
import torch.nn as nn

json_files = [pos_json for pos_json in os.listdir('dataset/parsed_files') if pos_json.endswith('.json')]
files_to_ignore = [line.rstrip('\n') for line in open('dataset/charts_to_ignore.txt')]
json_files = [file for file in json_files if file not in files_to_ignore]

# %%
# get the vocabulary, going through all the charts
vocabulary = set()
for file in json_files:
    data = json.load(open('dataset/parsed_files/' + file))
    for chart in data['charts']:
        if "double" in chart['type']:
            continue
        for entry in chart['notes']:
            vocabulary.add(entry[1])

vocabulary.add('None')

# turn the vocabulary into a dictionary
vocabulary = list(vocabulary)
vocabulary_to_idx = {label: idx for idx, label in enumerate(vocabulary)}
idx_to_vocabulary = {idx: label for idx, label in enumerate(vocabulary)}
# %%
def get_X_y(audio, sr, chart, slide_window):
    # Compute the melspec of the audio signal
    hop_length = int(0.003 * sr)  # 3ms hop length
    melspec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=40)
    melspec = librosa.power_to_db(melspec, ref=np.max)

    # Convert time_values and step_values to melspec frames
    time_values = [entry[0] for entry in chart['notes']]
    step_values = [entry[1] for entry in chart['notes']]

    time_frames = librosa.time_to_frames(time_values, sr=sr, hop_length=hop_length)
    # Create a temporary label-to-index dictionary
    label_to_idx = {label: idx for idx, label in enumerate(step_values)}

    # Determine window size in frames for 3ms duration
    window_duration = 0.003  # 3ms
    window_size = int(window_duration * sr / hop_length)

    # Initialize aligned labels array
    aligned_labels = ['None'] * melspec.shape[1]

    # Align labels with melspec frames
    for time_frame, label in zip(time_frames, step_values):
        label_idx = label_to_idx.get(label)
        if label_idx is not None:
            start_frame = max(0, time_frame)
            end_frame = min(len(aligned_labels), time_frame + window_size)
            aligned_labels[start_frame:end_frame] = [label] * (end_frame - start_frame)
    
    # window the melspec and labels
    padded_spec = np.pad(melspec, ((0, 0), (slide_window // 2, slide_window // 2)), mode='constant')
    windowed_spec = []
    for i in range(slide_window // 2, padded_spec.shape[1] - (slide_window // 2)):
        window = padded_spec[:, i - (slide_window // 2):i + (slide_window // 2) + 1]
        windowed_spec.append(window)

    return windowed_spec, aligned_labels

# %%
X = []
y = []
difficulty = []
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
        # update the json file
        data['music_fp'] = audio_fp
        with open('dataset/parsed_files/' + file, 'w') as outfile:
            json.dump(data, outfile)

    audio, sr = librosa.load(audio_fp, sr=None)  # Load audio, sr=None to preserve original sample rate
    for chart in data['charts']:
        if "double" in chart['type']:
            continue
        # get x and y
        windowed_spec, aligned_labels = get_X_y(audio, sr, chart, slide_window=1)

        for x in windowed_spec:
            # turn into tensor and append
            X.append(torch.tensor(x).unsqueeze(0))
        for lab in aligned_labels:
            y.append(torch.tensor(vocabulary_to_idx[lab]).unsqueeze(0))
            difficulty.append(torch.tensor(chart['difficulty_fine']).unsqueeze(0))

# %%

class MelSpectrogramCNN(nn.Module):
    def __init__(self, num_classes):
        super(MelSpectrogramCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 5 + 1, 64)  # Additional input dimension
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, mel_spec, difficulty):
        x = self.conv1(mel_spec)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)
        # take out the last dimension of x
        x = x.view(-1)
        x = torch.cat((x, difficulty))  # Concatenate additional input
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x

class RNNModel(nn.Module):
    def __init__(self, mel_input_size, difficulty_size, hidden_size, num_classes):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        
        # Bidirectional LSTM layer
        self.lstm = nn.LSTM(mel_input_size + difficulty_size, hidden_size, bidirectional=True)
        
        # Linear layer to map LSTM output to classes
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    
    def forward(self, mel_input, difficulty):
        mel_input = mel_input.squeeze(dim=2)

        # Concatenate mel spectrogram and integer input
        combined_input = torch.cat((mel_input, difficulty.unsqueeze(0).repeat(mel_input.size(0), 1)), dim=1)
        
        # Set initial hidden and cell states
        batch_size = combined_input.size(0)
        h0 = torch.zeros(2, batch_size, self.hidden_size).to(combined_input.device)  # 2 for bidirectional
        c0 = torch.zeros(2, batch_size, self.hidden_size).to(combined_input.device)
        
        # Forward pass through LSTM layer
        output, _ = self.lstm(combined_input.unsqueeze(1), (h0, c0))

        # Reshape output for linear layer
        output = output.view(-1)
        
        # Fully connected layer
        output = self.fc1(output)
        output = self.relu(output)
        output = self.fc2(output)
        
        return output

# %%
# train the model
# model = MelSpectrogramCNN(len(vocabulary))
model = RNNModel(40, 1, 64, len(vocabulary))
weights = torch.tensor([1.0] * len(vocabulary))
weights[vocabulary_to_idx['None']] = 0.05
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# %%
for epoch in range(8):
    for i in range(len(X)):
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(X[i], difficulty[i])
        # turn y[i] into a one-hot vector
        y_onehot = torch.zeros(len(vocabulary))
        y_onehot[y[i]] = 1
        loss = criterion(outputs, y_onehot)
        loss.backward()
        optimizer.step()

        # print statistics
        if i % 500 == 0:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, loss.item()))

# %%
for idx in range(50):
    input_section = X[idx]
    # Forward pass
    output = model(input_section, difficulty[idx])

    # print the idx_to_vocabulary of the output
    print(idx_to_vocabulary[torch.argmax(output).item()], idx_to_vocabulary[y[idx][0].item()])