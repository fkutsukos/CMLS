# %%
import librosa
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

# %% DEFINITION OF THE FEATURES

def compute_speccentr(spec):
    k_axis = np.arange(1, len(spec)+1)
    centr = sum(k_axis*abs(spec))/sum(abs(spec))
    return centr

def compute_specspread(spec):
    k_axis = np.arange(1, len(spec) + 1)
    centr = sum(k_axis * abs(spec)) / sum(abs(spec))

    spread = sqrt(sum((k_axis-centr)**2 * abs(spec)) / sum(abs(spec)))
    return spread

def compute_specskew(spec):
    k_axis = np.arange(1, len(spec) + 1)
    centr = sum(k_axis * abs(spec)) / sum(abs(spec))
    spread = sqrt(sum((k_axis - centr) ** 2 * abs(spec)) / sum(abs(spec)))

    m_th = sum((k_axis-centr)**3 * abs(spec)) / sum(abs(spec))
    skew = m_th / (spread)**3
    return skew

def compute_speckurt(spec):
    k_axis = np.arange(1, len(spec) + 1)
    centr = sum(k_axis * abs(spec)) / sum(abs(spec))
    spread = sqrt(sum((k_axis - centr) ** 2 * abs(spec)) / sum(abs(spec)))

    m_fo = sum((k_axis-centr)**4 * abs(spec)) / sum(abs(spec))
    kurt = m_fo / (spread)**4
    return kurt

def compute_rolloff(spec):
    ROE = 0.95 * sum(spec)
    E=0
    for i in np.arange(len(spec)):
        E = E + abs(spec[i])
        if E >= ROE
            break

    return i

def compute_slope(spec):
    k_axis = np.arange(1, len(spec) + 1)
    slope = (1 / sum(spec)) * (len(spec) * sum(k_axis*spec) - sum(k_axis)*sum(spec)) / (len(spec)*sum(k_axis**2) - (sum(k_axis))**2)
    return slope
# %%
# file path
file_path = 'data/Gitarre monophon/Samples/NoFX/G61-40100-1111-20593.wav'

# %%
# load and plot the audio file
x, Fs = librosa.load(file_path, sr=None)
time_axis = np.arange(x.shape[0]) / Fs
plt.figure(figsize=(16, 4))
plt.plot(time_axis, x)
plt.xlim([time_axis[0], time_axis[-1]])
plt.xlabel('Time (seconds)')
plt.ylabel('Waveform')

# %%
# windowing
win_length = int(np.floor(0.01 * Fs))
hop_size = int(np.floor(0.0075 * Fs))

window = sp.signal.get_window(window='hamming', Nx=win_length)

x_length = x.shape[0]
win_number = int(np.floor((x_length-win_length)/  hop_size))

for i in np.arange(win_number):
    frame = x[i * hop_size : i*hop_size + win_length]
    frame_wind = frame * window

    spec = np.fft.fft(frame_wind)
    nyquist = int(np.floor(spec.shape[0] / 2))

    spec = spec [1 : nyquist]

    #insert the computation of the feature



