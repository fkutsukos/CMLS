import librosa
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

def compute_speccentr(spec):
    k_axis = np.arange(1, len(spec)+1) * (Fs/win_length)
    centr = sum(k_axis*abs(spec))/sum(abs(spec))
    return centr

def compute_spec_centroid_librosa(spec):
    spectral_centroid = librosa.feature.spectral_centroid(S=spec)
    return spectral_centroid

# file path
file_path = 'data/Gitarre monophon/Samples/NoFX/G61-40100-1111-20593.wav'

# %%
# load and plot the audio file
x, Fs = librosa.load(file_path, sr=None)
time_axis = np.arange(x.shape[0]) / Fs

# %%
# windowing
# win_length = int(np.floor(0.01 * Fs))
# hop_size = int(np.floor(0.0075 * Fs))

# Hanning window shaping factor L = 4 , bass minimum frequency 40 Hz.
win_length = int(np.ceil((4 * Fs) / 40))

# JND for a sound at 40 Hz is equal to 3Hz
JND = 3
# Peak localization
fft_length = int(np.ceil(Fs / (2 * JND)))
# FFT performance requires  N_FFT as a power of 2
fft_length = int(2 ** (np.ceil(np.log2(N_FFT))))

hop_size = int(np.floor((win_length + 1) / 4))


window = sp.signal.get_window(window='hamming', Nx=win_length)
x_length = x.shape[0]


# win_number = int(np.floor((x_length-win_length)/hop_size))
frames_number = int(np.floor(x_length-fft_length)/hop_size)

train_feature = np.zeros(frames_number)
for i in np.arange(frames_number-1):

    frame = x[i * hop_size : i*hop_size + win_length]
    frame_wind = frame * window

    spec = np.fft.fft(frame_wind, n=fft_length)
    nyquist = int(np.floor(spec.shape[0] / 2))
    spec = spec[1: nyquist]

    train_feature[i] = compute_speccentr(spec)


plt.figure(figsize=(16, 6))
plt.title('SPECTRAL CENTROID')
plt.plot(centr.T, label='Spectral Centroid')
plt.show()

spec = np.abs(librosa.stft(x, window='hamming', n_fft=win_length, hop_length=hop_size))
centr_librosa = librosa.feature.spectral_centroid(spec)

plt.figure(figsize=(16, 6))
plt.title('SPECTRAL LIBROSA')
plt.plot(centr_librosa.T, label='Spectral Centroid')
plt.show()
