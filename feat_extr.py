# %%
import logging
import shutil
import librosa
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import os
from pathlib import Path

import datetime

from sklearn.preprocessing import Normalizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification

# ~ LOG_LEVEL = logging.INFO
LOG_LEVEL = logging.DEBUG
LOGFORMAT = "%(log_color)s%(message)s%(reset)s"
from colorlog import ColoredFormatter

logging.root.setLevel(LOG_LEVEL)
formatter = ColoredFormatter(LOGFORMAT)
stream = logging.StreamHandler()
stream.setLevel(LOG_LEVEL)
stream.setFormatter(formatter)
logger = logging.getLogger('colorlogger')
logger.setLevel(LOG_LEVEL)
logger.addHandler(stream)


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
    skew = m_th / spread ** 3
    return skew


def compute_speckurt(spec):
    k_axis = np.arange(1, len(spec) + 1)
    centr = sum(k_axis * abs(spec)) / sum(abs(spec))
    spread = np.sqrt(sum((k_axis - centr) ** 2 * abs(spec)) / sum(abs(spec)))

    m_fo = sum((k_axis - centr) ** 4 * abs(spec)) / sum(abs(spec))
    kurt = m_fo / spread ** 4
    return kurt


def compute_rolloff(spec, fs):
    ROE = 0.95 * (sum(np.abs(spec) ** 2))
    E = 0
    for k in np.arange(len(spec)):
        E = E + np.abs(spec[k]) ** 2
        if E >= ROE:
            break
    return k * (fs / 2) / len(spec)


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

The notes cover the common pitch range of a 4-string bass guitar from E1 (41.2 Hz) to G3 (196.0 Hz) or the common 
pitch range of a 6-string electric guitar from E2 (82.4 Hz) to E5 (659.3 Hz). 
'''


def pitch_detection(win, Fmin, Fmax, Fs):
    # spec_length = spec.shape[0]
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


'''
# %%
# file path
file_path = 'data/Bass monophon/Samples/EQ/B21-50312-1122-12675.wav'

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
'''
win_length = int(np.floor(0.10 * Fs))
hop_size = int(np.floor(0.075 * Fs))

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


# Training Features

def compute_features_frames(audio_file, fs, features):
    # Hanning window shaping factor L = 4 , bass minimum frequency 40 Hz.
    win_length = int(np.ceil((4 * fs) / 40))
    window = sp.signal.get_window(window='hanning', Nx=win_length)

    # JND for a sound at 40 Hz is equal to 3Hz
    JND = 3
    # Peak localization
    fft_length = int(np.ceil(fs / (2 * JND)))

    # FFT performance requires  N_FFT as a power of 2
    fft_length = int(2 ** (np.ceil(np.log2(fft_length))))

    # Hop size equal to 25% of the window length
    hop_size = int(np.floor(win_length / 8))

    # Number of frames
    frames_number = int(np.floor(audio_file.shape[0] - win_length) / hop_size)

    train_features_frames = np.zeros((len(features), frames_number))

    for i in np.arange(frames_number):

        frame = audio_file[i * hop_size: i * hop_size + win_length]
        frame_wind = frame * window

        # compute spectrogram for every frame using zero-padding
        spec = np.fft.fft(frame_wind, n=fft_length)
        nyquist = int(np.floor(spec.shape[0] / 2))
        spec = spec[1: nyquist]

        # compute spec features
        if (sum(abs(spec)) != 0):
            train_features_frames[0, i] = compute_speccentr(spec)
            train_features_frames[1, i] = compute_specspread(spec)
            train_features_frames[2, i] = compute_specskew(spec)
            train_features_frames[3, i] = compute_speckurt(spec)
            train_features_frames[4, i] = compute_rolloff(spec, fs)
            train_features_frames[5, i] = compute_slope(spec)
            train_features_frames[6, i] = compute_flatness(spec)
            train_features_frames[7, i] = compute_flux(frame_wind)

        # compute time features
        train_features_frames[8, i] = compute_zcr(frame_wind, fs)
        # train_features_frames[9, i] = compute_autocorr(frame_wind)

        # compute harmonic features
        # train_features_frames[9, i] = librosa.effects.harmonic(audio_file)

        # compute cepstrum features
        # train_features_frames[11, i] = compute_mfcc_librosa()

    return train_features_frames


'''
# Plotting Computed Features
feat_time_axis = np.arange(train_features.shape[1]) * hop_size / Fs
for index, feature in enumerate(features):
    plt.figure(figsize=(16, 6))
    plt.title(feature)
    plt.plot(feat_time_axis, train_features[index, :])
    plt.show()
'''


def compute_features_dataset(dataset, class_name):
    logger.info(str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) +
                ' Computing ' + str(dataset) + ' features for class: ' + str(class_name))
    files_root = 'data/{}/{}'.format(dataset, class_name)
    class_files = [f for f in os.listdir(files_root) if f.endswith('.wav')]
    n_files = len(class_files)
    files_features_mean = np.zeros((n_files, len(features)))
    for index, f in enumerate(class_files):
        # load the audio file f
        audio, fs = librosa.load(os.path.join(files_root, f), sr=None)
        # compute all the features for every frame of the audio file
        file_features_frames = compute_features_frames(audio, fs, features)
        # compute the mean value between frames of all the features of the audio file
        # and store it in the matrix that holds the features for all audio files of the class
        files_features_mean[index, :] = np.mean(file_features_frames, axis=1)
    return files_features_mean


datasets = ['Training', 'Test']
classes = ['Distortion', 'NoFX', 'Tremolo']
dict_features = {'Distortion': [], 'NoFX': [], 'Tremolo': []}
features = ['Spectral Centroid', 'Spectral Spread', 'Spectral Skewness', 'Spectral Kurtosis',
            'Spectral Rolloff', 'Spectral Slope', 'Spectral Flatness', 'Spectral Flux', 'Zero Crossing Rate',
            'Pitch Detection']

logger.info(str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ' Starting...')

# Building Labeled Features matrix for Training
X_train_Distortion = compute_features_dataset('Training', 'Distortion')
X_train_NoFX = compute_features_dataset('Training', 'NoFX')
X_train_Tremolo = compute_features_dataset('Training', 'Tremolo')
logger.info(str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ' Building Labeled Features matrix for Training')
X_train = np.concatenate((X_train_Distortion, X_train_NoFX, X_train_Tremolo), axis=0)


# Build the Ground Truth vector for Training
y_train_Distortion = np.zeros((X_train_Distortion.shape[0],))
y_train_NoFX = np.ones((X_train_NoFX.shape[0],))
y_train_Tremolo = np.ones((X_train_Tremolo.shape[0],)) * 2
logger.info(str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ' Building Ground Truth matrix for Training')
y_train = np.concatenate((y_train_Distortion, y_train_NoFX, y_train_Tremolo), axis=0)


# Building Labeled Features matrix for Test
X_test_Distortion = compute_features_dataset('Test', 'Distortion')
X_test_NoFX = compute_features_dataset('Test', 'NoFX')
X_test_Tremolo = compute_features_dataset('Test', 'Tremolo')
logger.info(str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ' Building Labeled Features matrix for Test')
X_test = np.concatenate((X_test_Distortion, X_test_NoFX, X_test_Tremolo), axis=0)


# Build the Ground Truth vector for Test
y_test_Distortion = np.zeros((X_test_Distortion.shape[0],))
y_test_NoFX = np.ones((X_test_NoFX.shape[0],))
y_test_Tremolo = np.ones((X_test_Tremolo.shape[0],)) * 2
logger.info(str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ' Building Ground Truth matrix for Test')
y_test = np.concatenate((y_test_Distortion, y_test_NoFX, y_test_Tremolo), axis=0)

'''
    if count == 0:
        X = train_features_mean[0:num_audio_files, :]

    else:
        X = np.concatenate((X, train_features_mean[0:num_audio_files, :]))

    if c == "Distortion":
        y[0:num_audio_files] = 0
    if c == "NoFX":
        y[num_audio_files:2*num_audio_files] = 1
    if c == "Tremolo":
        y[2*num_audio_files: 3*num_audio_files] = 2
    count = count + 1
'''

logger.info(str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ' Data Normalization in process...')
scaler = Normalizer().fit(X_train)
normalized_X = scaler.transform(X_train)
normalized_X_test = scaler.transform(X_test)

logger.info(str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ' Feature Selection in process...')
# define feature selection
selectedFeatures = SelectKBest(score_func=f_classif, k=3)
# apply feature selection
X_selected = selectedFeatures.fit_transform(X_train, y_train)
# print(X_selected.shape)


logger.info(str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ' Finished :)')

'''
harmonic_features = librosa.effects.harmonic(x)
spec = np.abs(np.fft.fft(harmonic_features))
nyquist = int(np.floor(spec.shape[0] / 2))

spec = spec[1:nyquist]

fundamental_index = np.argmax(spec)
fundamental_freq = fundamental_index * (Fs / 2) / spec.shape[0]

frequency_axis = np.arange(spec.shape[0]) * (Fs / 2) / spec.shape[0]

plt.figure(figsize=(10, 2))
plt.xlim([frequency_axis[0], frequency_axis[1000]])
plt.title('Harmonic Features')
plt.xlabel('frequencies')
plt.ylabel('Amplitude')
plt.plot(frequency_axis, spec)
plt.show()
'''


