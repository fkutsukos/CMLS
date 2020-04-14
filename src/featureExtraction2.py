# %%
import librosa
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

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

# compute the STFT using the following
N_FFT = 1024
HOP_SIZE = 512;

X = librosa.stft(x, window='hamming', n_fft=N_FFT, hop_length=HOP_SIZE)
Y = np.log(1 + 10 * np.abs(X))

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

def compute_mfcc(audio, fs):
    mfcc = librosa.feature.mfcc(y=audio, sr=fs)
    return mfcc


def compute_rms(spec):
    rms = librosa.feature.rms(S=spec, frame_length=N_FFT, hop_length=HOP_SIZE)
    return rms


def compute_spec_centroid(spec):
    spectral_centroid = librosa.feature.spectral_centroid(S=spec, n_fft=N_FFT, hop_length=HOP_SIZE)
    return spectral_centroid


# %%
rms_audio = compute_rms(X)

plt.figure()
plt.subplot(2, 1, 1)
plt.semilogy(rms_audio.T, label='RMS Energy')
plt.xticks([])
plt.xlim([0, rms_audio.shape[-1]])
plt.legend()
#plt.subplot(2, 1, 2)
#librosa.display.specshow(librosa.amplitude_to_db(X, ref=np.max), y_axis='log', x_axis='time')
#plt.title('log Power spectrogram')
#plt.tight_layout()

# %%
mfcc_audio = compute_mfcc(S)
