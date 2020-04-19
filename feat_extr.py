# %%
import librosa
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import os


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
    ROE = 0.95 * (sum(np.abs(spec) ** 2))
    E = 0
    for k in np.arange(len(spec)):
        E = E + np.abs(spec[k]) ** 2
        if E >= ROE:
            break
    return k * (Fs / 2) / len(spec)


def compute_slope(spec):
    k_axis = np.arange(1, len(spec) + 1)
    slope = (1 / sum(np.abs(spec))) * (len(spec) * sum(k_axis * np.abs(spec)) - sum(k_axis) * sum(np.abs(spec))) / (
            len(spec) * sum(k_axis ** 2) - (sum(k_axis)) ** 2)
    return slope


def compute_flux(win):
    spec_b = np.fft.fft(win[:-1])
    spec_a = np.fft.fft(win[1:])
    flux = np.sqrt(sum((np.abs(spec_b) - np.abs(spec_a)) ** 2))
    return flux


def compute_flatness(spec):
    flatness = librosa.feature.spectral_flatness(S=np.abs(spec))
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


# Harmonic Features
# Compute the Harmonic curves by:
# 1. computing pitch detection using frame-wise autocorrelation.
# 2. computing STFT on the harmonic curves
# 3. extract mean value, variance and standard deviation, maximum value and position, spectral centroid, spread, skewness and kurtosis
'''
                Fundamentals	Harmonics To	
4-string Bass	41Hz-392Hz	~4kHz-5kHz
5-string Bass	31Hz-392Hz	~4kHz-5kHz
6-string Bass	31Hz-523Hz	~4kHz-5kHz
'''


def pitch_detection(win, Fmin, Fmax, Fs):
    spec_length = spec.shape[0]
    xcorr = np.correlate(win, win, mode='full')

    # keep only the positive lags of autocorrelation
    rpos = xcorr[int(xcorr.size / 2):]

    # find maximum peak within the accepted range
    lagmin = int(np.floor(Fs / Fmax))
    lagmax = int(np.ceil(Fs / Fmin))
    rc = rpos[lagmin:lagmax]
    rc_maxv = np.max(rc)
    rc_maxi = np.argmax(rc)

    # get the real index of max i
    r_maxi = lagmin + rc_maxi
    r_maxv = rc_maxv

    pitch_lag = r_maxi
    pitch = Fs / pitch_lag

    return pitch


def compute_autocorr(win):
    result = np.correlate(win, win, mode='full')
    return result[int(result.size / 2):]


# %%
# file path
file_path = 'data/Bass monophon/Samples/NoFX/B11-42304-1111-00031.wav'

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
win_length = int(np.ceil((4 * Fs) / 40))

window = sp.signal.get_window(window='hanning', Nx=win_length)

# JND for a sound at 40 Hz is equal to 3Hz
JND = 3
# Peak localization
fft_length = int(np.ceil(Fs / (2 * JND)))

# FFT performance requires  N_FFT as a power of 2
fft_length = int(2 ** (np.ceil(np.log2(fft_length))))

# Hop size equal to 25% of the window length
hop_size = int(np.floor(win_length / 8))

# Number of frames
frames_number = int(np.floor(x.shape[0] - win_length) / hop_size)

# Training Features
features = ['Spectral Centroid', 'Spectral Spread', 'Spectral Skewness', 'Spectral Kurtosis',
            'Spectral Rolloff', 'Spectral Slope', 'Spectral Flatness', 'Spectral Flux', 'Zero Crossing Rate',
            'Auto-correlation', 'Pitch Detection']
train_features = np.zeros((len(features), frames_number))
for i in np.arange(frames_number - 1):

    frame = x[i * hop_size: i * hop_size + win_length]
    frame_wind = frame * window

    # compute spectrogram for every frame using zero-padding
    spec = np.fft.fft(frame_wind, n=fft_length)
    nyquist = int(np.floor(spec.shape[0] / 2))
    spec = spec[1: nyquist]

    # compute spec features
    if (sum(abs(spec)) != 0):
        train_features[0, i] = compute_speccentr(spec)
        train_features[1, i] = compute_specspread(spec)
        train_features[2, i] = compute_specskew(spec)
        train_features[3, i] = compute_speckurt(spec)
        train_features[4, i] = compute_rolloff(spec)
        train_features[5, i] = compute_slope(spec)
        train_features[6, i] = compute_flatness(spec)
        train_features[7, i] = compute_flux(frame_wind)

        # compute time features
        train_features[8, i] = compute_zcr(frame_wind, Fs)
        # train_features[9, i] = compute_autocorr(frame_wind)

    # compute harmonic features
    train_features[10, i] = pitch_detection(frame_wind, 40, 400, Fs)

    # compute mfcc features
    #    train_spec_feature[8, i] = compute_mfcc_librosa()

'''
# Plotting Computed Features
feat_time_axis = np.arange(train_features.shape[1]) * hop_size / Fs
for index, feature in enumerate(features):
    plt.figure(figsize=(16, 6))
    plt.title(feature)
    plt.plot(feat_time_axis, train_features[index, :])
    plt.show()
'''

classes = ['Distortion', 'NoFX', 'Tremolo']

dict_train_features = {'Distortion': [], 'NoFX': [], 'Tremolo': []}
'''
for c in classes:
    train_root = 'data/{}/training/'.format(c)
    class_train_files = [f for f in os.listdir(train_root) if f.endswith('.wav')]
    n_train_samples = len(class_train_files)

    train_features = np.zeros((n_train_samples, n_mfcc))
    for index, f in enumerate(class_train_files):
        audio, fs = librosa.load(os.path.join(train_root, f), sr=None)
        mfcc = compute_mfcc(audio, fs, n_mfcc)
        train_features[index, :] = np.mean(mfcc, axis=1)

    dict_train_features[c] = train_features
'''


'''
time_axis = np.arange(x.shape[0]) / Fs
plt.figure(figsize=(16, 4))
plt.plot(time_axis, x)
plt.xlim([time_axis[0], time_axis[-1]])
plt.xlabel('Time (seconds)')
plt.ylabel('Waveform')
'''
harmonic_features = librosa.effects.harmonic(x)
spec = np.abs(np.fft.fft(harmonic_features))
nyquist = int(np.floor(spec.shape[0] / 2))

spec = spec[1:nyquist]

frequency_axis = np.arange(spec.shape[0])*(Fs/(2*nyquist))

plt.figure(figsize=(10, 2))
plt.xlim([frequency_axis[0], frequency_axis[1000]])
plt.title('Harmonic Features')
plt.xlabel('frequencies')
plt.ylabel('Amplitude')
plt.plot(frequency_axis,spec)
plt.show()
'''
win_length = int(np.floor(0.01 * Fs))
hop_size = int(np.floor(0.0075 * Fs))
window = sp.signal.get_window(window='hanning', Nx=win_length)
harmonic_features_len = harmonic_features.shape[0]
harmonic_features_win_number = int(np.floor((harmonic_features_len - win_length) / hop_size))
fft_harmonics = np.zeros(len(harmonic_features_win_number))

for i in np.arange(x_win_number):
    frame = x[i * hop_size : i * hop_size + win_length]
    frame_wind = frame * window

    spec = np.fft.fft(frame_wind)
    nyquist = int(np.floor(spec.shape[0] / 2))

    spec = spec[1:nyquist]



freq_axis = np.arange(fft_x.shape[0]) / Fs
plt.plot(time_axis,fft_x)
plt.show()
'''