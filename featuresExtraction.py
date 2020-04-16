# %%
import librosa
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import librosa.display

# %%
# file path
file_path = ('data/Gitarre monophon/Samples/NoFX/G61-41101-1111-20594.wav')

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
# compute and plot the STFT
'''
# Hanning window shaping factor L = 4 , bass minimum frequency 40 Hz.
WIN_LEN = int(np.ceil((4 * Fs) / 40))
# JND for a sound at 40 Hz
JND = 3
# Peak localization
N_FFT = int(np.ceil(Fs // (2 * JND)))
# FFT performance requires  N_FFT as a power of 2
N_FFT = int(2 ** (np.ceil(np.log2(N_FFT))))
HOP_SIZE = int(np.floor((WIN_LEN + 1) / 8))
'''

WIN_LEN = int(np.floor(0.01 * Fs))
N_FFT = WIN_LEN
HOP_SIZE = int(np.floor(0.0075 * Fs))

X = np.abs(librosa.stft(x, window='hanning', n_fft=N_FFT, win_length=WIN_LEN, hop_length=HOP_SIZE))
Y = np.log(1 + 10 * np.abs(X))
indicesX = np.where(X == X.max())
indicesY = np.where(Y == Y.max())
# 10^ - 13/20 (13 db Attenuation)

time_axis = np.arange(X.shape[1]) / (Fs / HOP_SIZE)
frequency_axis = np.arange(X.shape[0]) / (N_FFT / Fs)

plt.figure(figsize=(10, 4))
x_ext = (time_axis[1] - time_axis[0]) / 2
y_ext = (frequency_axis[1] - frequency_axis[0]) / 2
image_extent = [time_axis[0] - x_ext, time_axis[-1] + x_ext, frequency_axis[0] - y_ext, frequency_axis[-1] + y_ext]
plt.imshow(Y, extent=image_extent, aspect='auto', origin='lower')
plt.xlabel('Time (seconds)')
plt.ylabel('Frequency (Hz)')
plt.show()


# %%
# Spectral Features

def compute_mfcc(audio, fs):
    mfcc = librosa.feature.mfcc(y=audio, sr=fs)
    return mfcc


def compute_rms(spec):
    rms = librosa.feature.rms(S=spec, frame_length=N_FFT)
    return rms


def compute_spec_centroid_librosa(spec):
    spectral_centroid = librosa.feature.spectral_centroid(S=spec, n_fft=N_FFT)
    return spectral_centroid


def compute_speccentr_clara(spec):
    k_axis = np.arange(1, spec.shape[0] + 1)
    centr = np.sum(np.transpose(k_axis) * np.abs(spec)) / np.sum(np.abs(spec))
    return centr


def compute_speccentr(spec):
    k_axis = np.arange(1, len(spec) + 1)
    centr = sum(k_axis * abs(spec)) / sum(abs(spec))
    return centr


# %%
# Compute and plot the features

mfcc = compute_mfcc(x, Fs)
plt.figure(figsize=(16, 6))
plt.title('MFCC')
plt.imshow(mfcc, origin='lower', aspect='auto')
plt.show()

rms = compute_rms(X)
plt.figure(figsize=(16, 6))
plt.title('RMS')
plt.semilogy(time_axis, rms.T, label='RMS Energy')
plt.show()

for i in np.arange()
cent = compute_speccentr(X)
plt.figure(figsize=(16, 6))
plt.title('SPECTRAL CENTROID')
plt.semilogy(time_axis, cent.T, label='Spectral Centroid')
plt.show()
# plt.xticks([])
# plt.xlim([0, rms_audio.shape[-1]])
# plt.legend()
# plt.subplot(2, 1, 2)
# librosa.display.specshow(librosa.amplitude_to_db(X, ref=np.max), y_axis='log', x_axis='time')
# plt.title('log Power spectrogram')
# plt.tight_layout()
# plt.show()
