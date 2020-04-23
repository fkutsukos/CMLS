# %%
import logging
import librosa
import numpy as np
import scipy as sp
import os


import datetime

from sklearn.preprocessing import Normalizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
import sklearn.svm
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
    mfcc_upper = mfcc[4:]
    return mfcc_upper


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
    n_mfcc = 13
    all_files_features_mean = np.zeros((n_files, len(features)))
    all_files_features_std = np.zeros((n_files, len(features)))
    all_files_features_var = np.zeros((n_files, len(features)))
    all_files_features_first_der_mean = np.zeros((n_files, len(features)))
    all_files_features_first_der_std = np.zeros((n_files, len(features)))
    all_files_features_first_der_var = np.zeros((n_files, len(features)))
    all_files_features_second_der_mean = np.zeros((n_files, len(features)))
    all_files_features_second_der_std = np.zeros((n_files, len(features)))
    all_files_features_second_der_var = np.zeros((n_files, len(features)))
    all_files_mfccs_mean = np.zeros((n_files, n_mfcc))
    all_files_mfccs_std = np.zeros((n_files, n_mfcc))
    all_files_mfccs_var = np.zeros((n_files, n_mfcc))
    all_files_mfccs_first_der_mean = np.zeros((n_files, n_mfcc))
    all_files_mfccs_first_der_std = np.zeros((n_files, n_mfcc))
    all_files_mfccs_first_der_var = np.zeros((n_files, n_mfcc))
    all_files_mfccs_second_der_mean = np.zeros((n_files, n_mfcc))
    all_files_mfccs_second_der_std = np.zeros((n_files, n_mfcc))
    all_files_mfccs_second_der_var = np.zeros((n_files, n_mfcc))

    for index, f in enumerate(class_files):
        # load the audio file f
        audio, fs = librosa.load(os.path.join(files_root, f), sr=None)
        # compute all the features for every frame of the audio file
        file_features_frames = compute_features_frames(audio, fs, features)

        # compute the first and second derivative of the features
        file_features_first_der_frames = np.diff(file_features_frames, axis=1)
        file_features_second_der_frames = np.diff(file_features_first_der_frames, axis=1)

        # compute the cepstrum coefficients
        file_mfccs = compute_mfcc(audio, fs, n_mfcc)

        # compute the first and second derivative of the cepstrum coefficients
        file_mfccs_first_der_frames = np.diff(file_mfccs, axis=1)
        file_mfccs_second_der_frames = np.diff(file_mfccs_first_der_frames, axis=1)

        # compute the mean value between frames of all the features of the audio file
        # and store it in the matrix that holds the features for all audio files of the class
        all_files_features_mean[index, :] = np.mean(file_features_frames, axis=1)
        all_files_features_std[index, :] = np.ndarray.std(file_features_frames, axis=1)
        all_files_features_var[index, :] = np.ndarray.var(file_features_frames, axis=1)

        all_files_features_first_der_mean[index, :] = np.mean(file_features_first_der_frames, axis=1)
        all_files_features_first_der_std[index, :] = np.ndarray.std(file_features_first_der_frames, axis=1)
        all_files_features_first_der_var[index, :] = np.ndarray.var(file_features_first_der_frames, axis=1)

        all_files_features_second_der_mean[index, :] = np.mean(file_features_second_der_frames, axis=1)
        all_files_features_second_der_std[index, :] = np.ndarray.std(file_features_second_der_frames, axis=1)
        all_files_features_second_der_var[index, :] = np.ndarray.var(file_features_second_der_frames, axis=1)

        all_files_mfccs_mean[index, :] = np.mean(file_mfccs, axis=1)
        all_files_mfccs_std[index, :] = np.ndarray.std(file_mfccs, axis=1)
        all_files_mfccs_var[index, :] = np.ndarray.var(file_mfccs, axis=1)

        all_files_mfccs_first_der_mean[index, :] = np.mean(file_mfccs_first_der_frames, axis=1)
        all_files_mfccs_first_der_std[index, :] = np.ndarray.std(file_mfccs_first_der_frames, axis=1)
        all_files_mfccs_first_der_var[index, :] = np.ndarray.var(file_mfccs_first_der_frames, axis=1)

        all_files_mfccs_second_der_mean[index, :] = np.mean(file_mfccs_second_der_frames, axis=1)
        all_files_mfccs_second_der_std[index, :] = np.ndarray.std(file_mfccs_second_der_frames, axis=1)
        all_files_mfccs_second_der_var[index, :] = np.ndarray.var(file_mfccs_second_der_frames, axis=1)

    all_features = np.concatenate((all_files_features_mean, all_files_features_std, all_files_features_var,
                                   all_files_features_first_der_mean, all_files_features_first_der_std,all_files_features_first_der_var,
                                   all_files_features_second_der_mean, all_files_features_second_der_std, all_files_features_second_der_var,
                                   all_files_mfccs_mean, all_files_mfccs_std, all_files_mfccs_var,
                                   all_files_mfccs_first_der_mean, all_files_mfccs_first_der_std, all_files_mfccs_first_der_var,
                                   all_files_mfccs_second_der_mean, all_files_mfccs_second_der_std, all_files_mfccs_second_der_var), axis=1)
    return all_features


datasets = ['Training', 'Test']
classes = ['Distortion', 'NoFX', 'Tremolo']
dict_features = {'Distortion': [], 'NoFX': [], 'Tremolo': []}
features = ['Spectral Centroid', 'Spectral Spread', 'Spectral Skewness', 'Spectral Kurtosis',
            'Spectral Rolloff', 'Spectral Slope', 'Spectral Flatness', 'Spectral Flux', 'Zero Crossing Rate']

logger.info(str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ' Starting...')

# Building Labeled Features matrix for Training
X_train_Distortion = compute_features_dataset('Training', 'Distortion')
X_train_NoFX = compute_features_dataset('Training', 'NoFX')
X_train_Tremolo = compute_features_dataset('Training', 'Tremolo')
logger.info(
    str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ' Building Labeled Features matrix for Training')
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


logger.info(str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ' Data Normalization in process...')
'''
scaler = Normalizer().fit(X_train)
normalized_X = scaler.transform(X_train)
normalized_X_test = scaler.transform(X_test)
'''
feat_max = np.max(X_train, axis=0)
feat_min = np.min(X_train, axis=0)
X_train_normalized = (X_train - feat_min) / (feat_max - feat_min)
X_test_normalized = (X_test - feat_min) / (feat_max - feat_min)


logger.info(str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ' Feature Selection in process...')
selectedFeatures = SelectKBest(score_func=f_classif, k=3)
X_train_selected = selectedFeatures.fit_transform(X_train_normalized, y_train)
X_test_selected = selectedFeatures.fit_transform(X_test_normalized, y_test)
# print(X_selected.shape)


logger.info(str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ' Classification in process...')
SVM_parameters = {
    'C': 1,
    'kernel': 'rbf',
}

def compute_metrics(gt_labels, predicted_labels):
    TP = np.sum(np.logical_and(predicted_labels == 1, gt_labels == 1))
    FP = np.sum(np.logical_and(predicted_labels == 1, gt_labels == 0))
    TN = np.sum(np.logical_and(predicted_labels == 0, gt_labels == 0))
    FN = np.sum(np.logical_and(predicted_labels == 0, gt_labels == 1))
    accuracy = (TP + TN) / (TP + FP + TN + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1_score = 2 * precision * recall / (precision + recall)
    print("Results : \n accuracy = {} \n precision = {} \n recall = {} \n F1 score = {}".format(
        accuracy, precision, recall, F1_score))

clf = sklearn.svm.SVC(**SVM_parameters)
clf.fit(X_train_selected, y_train)
y_predict = clf.predict(X_train_selected)
y_test_predict = clf.predict(X_test_selected)

compute_metrics(y_test, y_test_predict)

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
