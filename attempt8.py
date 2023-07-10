# %%
# This will take inspiration from: https://github.com/cpuguy96/StepCOVNet
# Or maybe: https://github.com/guillefix/transflower-lightning

import numpy as np

data = np.load('dataset/3ms_dataset/XenomaX_dance-single_Hard_10.npz', allow_pickle=True)

# %%
def create_sliding_windows(x, y, window_size):
    padded_x = np.pad(x, ((0, 0), (window_size // 2, window_size // 2)), mode='constant')
    windows = []
    labels = []
    
    for i in range(window_size // 2, padded_x.shape[1] - (window_size // 2)):
        window = padded_x[:, i - (window_size // 2):i + (window_size // 2) + 1]
        windows.append(window)
        labels.append(y[i - (window_size // 2)])
    
    return np.array(windows), np.array(labels)

# %%
window_size = 5

windows, labels = create_sliding_windows(data['x'], data['y'], window_size)

print("Windows shape:", windows.shape)
print("Labels shape:", labels.shape)