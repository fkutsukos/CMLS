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

    spread = np.sqrt(sum((k_axis-centr)**2 * abs(spec)) / sum(abs(spec)))
    return spread

def compute_specskew(spec):
    k_axis = np.arange(1, len(spec) + 1)
    centr = sum(k_axis * abs(spec)) / sum(abs(spec))
    spread = np.sqrt(sum((k_axis - centr) ** 2 * abs(spec)) / sum(abs(spec)))

    m_th = sum((k_axis-centr)**3 * abs(spec)) / sum(abs(spec))
    skew = m_th / (spread)**3
    return skew

def compute_speckurt(spec):
    k_axis = np.arange(1, len(spec) + 1)
    centr = sum(k_axis * abs(spec)) / sum(abs(spec))
    spread = np.sqrt(sum((k_axis - centr) ** 2 * abs(spec)) / sum(abs(spec)))

    m_fo = sum((k_axis-centr)**4 * abs(spec)) / sum(abs(spec))
    kurt = m_fo / (spread)**4
    return kurt

def compute_rolloff(spec):
    ROE = 0.95 * sum(spec)
    E=0
    for i in np.arange(len(spec)):
        E = E + abs(spec[i])
        if E >= ROE:
            break
    return i

def compute_slope(spec):
    k_axis = np.arange(1, len(spec) + 1)
    slope = (1 / sum(np.abs(spec))) * (len(spec) * sum(k_axis*np.abs(spec)) - sum(k_axis)*sum(np.abs(spec))) / (len(spec)*sum(k_axis**2) - (sum(k_axis))**2)
    return slope

def compute_flux(win):
    spec_b = np.fft.fft(win[:-1])
    spec_a = np.fft.fft(win[1:])
    flux = np.sqrt(sum((spec_b - spec_a)**2))
    return flux

def compute_flatness(spec):
     flatness = (np.prod(np.abs(spec)))**(len(spec)-1) / (1/(len(spec)-1) * sum(np.abs(spec)))
     return flatness

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

'''
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
'''

# Hanning window shaping factor L = 4 , bass minimum frequency 40 Hz.
win_length = int(np.ceil((4 * Fs) / 40))
window = sp.signal.get_window(window='hanning', Nx=win_length)

# JND for a sound at 40 Hz is equal to 3Hz
JND = 3
# Peak localization
fft_length = int(np.ceil(Fs / (2 * JND)))

# FFT performance requires  N_FFT as a power of 2
fft_length = int(2 ** (np.ceil(np.log2(fft_length))))

#Hop size equal to 25% of the window length
hop_size = int(np.floor((win_length + 1) / 4))

frames_number = int(np.floor(x.shape[0]-fft_length)/hop_size)

#Compute Features
spectral_features = ['Spectral Centroid', 'Spectral Spread','Spectral Skewness','Spectral Kurtosis','Spectral Rolloff','Spectral Slope','Spectral Flatness']
train_feature = np.zeros((len(spectral_features),frames_number))
for i in np.arange(frames_number-1):

    frame = x[i * hop_size : i*hop_size + win_length]
    frame_wind = frame * window

    spec = np.fft.fft(frame_wind, n=fft_length)
    nyquist = int(np.floor(spec.shape[0] / 2))
    spec = spec[1: nyquist]
    if (sum(abs(spec)) != 0):
        train_feature[0, i] = compute_speccentr(spec)
        train_feature[1, i] = compute_specspread(spec)
        train_feature[2, i] = compute_specskew(spec)
        train_feature[3, i] = compute_speckurt(spec)
        train_feature[4, i] = compute_rolloff(spec)
        train_feature[5, i] = compute_slope(spec)
        train_feature[6, i] = compute_flatness(spec)

# Plotting Computed Features
feat_time_axis = np.arange(train_feature.shape[1]) * hop_size / Fs
for index, feature in enumerate(spectral_features):
    plt.figure(figsize=(16, 6))
    plt.title(feature)
    plt.plot(feat_time_axis, train_feature[index , :])
    plt.show()

