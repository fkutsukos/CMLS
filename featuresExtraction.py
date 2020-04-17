# %%
import librosa
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import librosa.display




X = np.abs(librosa.stft(x, window='hanning', n_fft=N_FFT, win_length=WIN_LEN, hop_length=HOP_SIZE))
Y = np.log(1 + 10 * np.abs(X))

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
