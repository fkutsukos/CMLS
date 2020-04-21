import numpy as np
import librosa

# %%
# file path
file_path = 'data/Bass monophon/Samples/EQ/B21-50312-1122-12675.wav'

# %%
# load and plot the audio file
x, Fs = librosa.load(file_path, sr=None)
time_axis = np.arange(x.shape[0]) / Fs

harmonic_features = librosa.effects.harmonic(x)
spec = np.abs(np.fft.fft(harmonic_features))
nyquist = int(np.floor(spec.shape[0] / 2))

spec = spec[1:nyquist]

fundamental_index = np.argmax(spec)
fundamental_freq = fundamental_index * (Fs / 2) / spec.shape[0]

frequency_axis = np.arange(spec.shape[0]) * (Fs / 2) / spec.shape[0]