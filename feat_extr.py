# %%
import librosa
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp



# %% DEFINITION OF THE FEATURES

# Spectral Features
def compute_speccentr(spec):
    k_axis = np.arange(1, len(spec) + 1)
    centr = sum(k_axis * abs(spec)) / sum(abs(spec))
    return centr


def compute_specspread(spec):
    k_axis = np.arange(1, len(spec) + 1)
    centr = sum(k_axis * abs(spec)) / sum(abs(spec))

    spread = np.sqrt(sum((k_axis - centr) ** 2 * abs(spec)) / sum(abs(spec)))
    return spread


def compute_specskew(spec):
    k_axis = np.arange(1, len(spec) + 1)
    centr = sum(k_axis * abs(spec)) / sum(abs(spec))
    spread = np.sqrt(sum((k_axis - centr) ** 2 * abs(spec)) / sum(abs(spec)))

    m_th = sum((k_axis - centr) ** 3 * abs(spec)) / sum(abs(spec))
    skew = m_th / (spread) ** 3
    return skew


def compute_speckurt(spec):
    k_axis = np.arange(1, len(spec) + 1)
    centr = sum(k_axis * abs(spec)) / sum(abs(spec))
    spread = np.sqrt(sum((k_axis - centr) ** 2 * abs(spec)) / sum(abs(spec)))

    m_fo = sum((k_axis - centr) ** 4 * abs(spec)) / sum(abs(spec))
    kurt = m_fo / spread ** 4
    return kurt


def compute_rolloff(spec):
    ROE = 0.95 * (sum(np.abs(spec)**2))
    E = 0
    for k in np.arange(len(spec)):
        E = E + np.abs(spec[k])**2
        if E >= ROE:
            break
    return k*(Fs/2)/len(spec)


def compute_slope(spec):
    k_axis = np.arange(1, len(spec) + 1)
    slope = (1 / sum(np.abs(spec))) * (len(spec) * sum(k_axis * np.abs(spec)) - sum(k_axis) * sum(np.abs(spec))) / (
            len(spec) * sum(k_axis ** 2) - (sum(k_axis)) ** 2)
    return slope


def compute_flux(win):
    spec_b = np.fft.fft(win[:-1])
    spec_a = np.fft.fft(win[1:])
    flux = np.sqrt(sum((spec_b - spec_a) ** 2))
    return flux


def compute_flatness(spec):
    flatness = (np.prod(np.abs(spec))) ** (len(spec) - 1) / (1 / (len(spec) - 1) * sum(np.abs(spec)))
    return flatness


def compute_specdec(spec):
    mul_fact = 1 / np.sum(np.abs(spec[1:]))
    num = np.abs(spec[1:]) - np.tile(np.abs(spec[0]), len(spec) - 1)
    den = np.arange(1, len(spec))
    spectral_decrease = mul_fact * np.sum(num / den)
    return spectral_decrease


def compute_mfcc(audio, fs, n_mfcc):
    # Compute the spectrogram of the audio signal
    X = np.abs(librosa.stft(
        audio,
        window='hamming',
        n_fft=1024,
        hop_length=512, )
    )

    # Find the weights of the mel filters
    mel = librosa.filters.mel(
        sr=fs,
        n_fft=1024,
        n_mels=40,
        fmin=133.33,
        fmax=6853.8,
    )
    # Apply the filters to spectrogram
    melspectrogram = np.dot(mel, X)
    # Take the logarithm
    log_melspectrogram = np.log10(melspectrogram + 1e-16)

    # Apply the DCT to log melspectrogram to obtain the coefficients
    mfcc = sp.fftpack.dct(log_melspectrogram, axis=0, norm='ortho')[1:n_mfcc + 1]
    return mfcc


def compute_mfcc_librosa(audio, fs):
    mfcc = librosa.feature.mfcc(y=audio, sr=fs)
    return mfcc


# Time features
def compute_zcr(win, Fs):
    win_sign = np.sign(win)
    N = win.shape[0]
    sign_diff = np.abs(win_sign[:-1] - win_sign[1:])
    zcr = (Fs / (2 * N)) * np.sum(sign_diff)
    return zcr


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
'''
win_length = int(np.floor(0.10 * Fs))
hop_size = int(np.floor(0.050 * Fs))

window = sp.signal.get_window(window='hamming', Nx=win_length)

# JND for a sound at 40 Hz is equal to 3Hz
JND = 3
# Peak localization
fft_length = int(np.ceil(Fs / (2 * JND)))

# FFT performance requires  N_FFT as a power of 2
fft_length = int(2 ** (np.ceil(np.log2(fft_length))))
fft_length = win_length
'''
'''
x_length = x.shape[0]
win_number = int(np.floor((x_length-win_length)/hop_size))

for i in np.arange(win_number):
    frame = x[i * hop_size : i*hop_size + win_length]
    frame_wind = frame * window

    spec = np.fft.fft(frame_wind)
    nyquist = int(np.floor(spec.shape[0] / 2))

    spec = spec [1 : nyquist]

    #insert the computation of the feature
'''
# Hanning window shaping factor L = 4 , bass minimum frequency 40 Hz.
win_length = int(np.ceil((4 * Fs) / 50))

window = sp.signal.get_window(window='hanning', Nx=win_length)

# JND for a sound at 40 Hz is equal to 3Hz
JND = 3
# Peak localization
fft_length = int(np.ceil(Fs / (2 * JND)))

# FFT performance requires  N_FFT as a power of 2
fft_length = int(2 ** (np.ceil(np.log2(fft_length))))

# Hop size equal to 25% of the window length
hop_size = int(np.floor(win_length/8))

# Number of frames
frames_number = int(np.floor(x.shape[0] - win_length) / hop_size)

# Compute Spectral Features
spectral_features = ['Spectral Centroid', 'Spectral Spread', 'Spectral Skewness', 'Spectral Kurtosis',
                     'Spectral Rolloff', 'Spectral Slope', 'Spectral Flatness']
train_spec_feature = np.zeros((len(spectral_features), frames_number))
for i in np.arange(frames_number - 1):

    frame = x[i * hop_size: i * hop_size + win_length]
    frame_wind = frame * window

    spec = np.fft.fft(frame_wind, n=fft_length)
    nyquist = int(np.floor(spec.shape[0] / 2))
    spec = spec[1: nyquist]
    if (sum(abs(spec)) != 0):
        train_spec_feature[0, i] = compute_speccentr(spec)
        train_spec_feature[1, i] = compute_specspread(spec)
        train_spec_feature[2, i] = compute_specskew(spec)
        train_spec_feature[3, i] = compute_speckurt(spec)
        train_spec_feature[4, i] = compute_rolloff(spec)
        train_spec_feature[5, i] = compute_slope(spec)
        train_spec_feature[6, i] = compute_flatness(spec)

# Plotting Computed Features
feat_time_axis = np.arange(train_spec_feature.shape[1]) * hop_size / Fs
for index, feature in enumerate(spectral_features):
    plt.figure(figsize=(16, 6))
    plt.title(feature)
    plt.plot(feat_time_axis, train_spec_feature[index, :])
    plt.show()

# Compute Temporal Features

temporal_features = ['Zero-Crossing Rate', 'MFCCs']
train_temp_feature = np.zeros((len(temporal_features), frames_number))
mfcc = compute_mfcc(x, Fs, 13)