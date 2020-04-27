# %%
import logging
import librosa
import numpy as np
import scipy as sp
import math
import os
import datetime
from numpy import savetxt

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
    # compute spec features
    if (sum(abs(spec)) != 0):
        k_axis = np.arange(1, len(spec) + 1)
        centr = sum(k_axis * abs(spec)) / sum(abs(spec))
        if math.isnan(centr):
            centr = 0
    else:
        centr = 0
    return centr


def compute_specspread(spec):
    if (sum(abs(spec)) != 0):
        k_axis = np.arange(1, len(spec) + 1)
        centr = sum(k_axis * abs(spec)) / sum(abs(spec))
        spread = np.sqrt(sum((k_axis - centr) ** 2 * abs(spec)) / sum(abs(spec)))
        if math.isnan(spread):
            spread = 0
    else:
        spread = 0
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


# Compute MFCCs
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


# Time features
def compute_zcr(win, fs):
    win_sign = np.sign(win)
    N = win.shape[0]
    sign_diff = np.abs(win_sign[:-1] - win_sign[1:])
    zcr = (fs / (2 * N)) * np.sum(sign_diff)
    return zcr


def pitch_detection(win, Fmin, Fmax, fs):
    # spec_length = spec.shape[0]
    xcorr = np.correlate(win, win, mode='full')

    # keep only the positive lags of autocorrelation
    rpos = xcorr[int(xcorr.size / 2):]

    # find maximum peak within the accepted range

    lagmin = int(np.floor(fs / Fmax))
    lagmax = int(np.ceil(fs / Fmin))
    rc = rpos[lagmin:lagmax]
    rc_maxv = np.max(rc)
    rc_maxi = np.argmax(rc)

    # get the real index of max i
    r_maxi = lagmin + rc_maxi
    r_maxv = rc_maxv

    pitch_lag = r_maxi
    pitch = fs / pitch_lag

    return pitch


def midi_to_freq(midi_num):
    g = 2 ** (1 / 12)
    f = lambda midi: 440 * g ** (midi - 69)
    freq = f(midi_num)
    return freq


def compute_autocorr(win):
    result = np.correlate(win, win, mode='full')
    return result[int(result.size / 2):]


#  Compute Spectral Features
def compute_features_frames(audio_file, fs, features, peaks_n, file_path):
    # STFT parameters for spectral features
    win_length = 1024
    window = sp.signal.get_window(window='hanning', Nx=win_length)
    hop_size = int(0.75 * win_length)
    frames_number = int(np.floor(audio_file.shape[0] - win_length) / hop_size)

    # initialize matrix to save features per frame
    train_features_frames = np.zeros((len(features), frames_number))

    # compute the spectral features for every frame using the corresponding window length
    for i in np.arange(frames_number):

        frame = audio_file[i * hop_size: i * hop_size + win_length]
        frame_wind = frame * window

        # compute spectrogram for every frame using zero-padding
        spec = np.fft.fft(frame_wind)
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


def compute_harmonics_curves_frame(win, peaks_n, file_path, fs):
    spec = np.abs(np.fft.fft(win))
    nyquist = int(np.floor(spec.shape[0] / 2))

    spec = spec[1:nyquist]

    # find frequency from file name
    file_name = os.path.basename(os.path.normpath(file_path))
    midi_num = int(file_name[4:6])
    freq = int(np.floor(midi_to_freq(midi_num)))
    fmin = freq - 10
    fmax = freq * 3

    fund_pitch = pitch_detection(win, fmin, fmax, fs)  # to do (change fmin fmax)

    fi_peaks = np.arange(1, peaks_n + 1) * fund_pitch

    k_peaks = np.round(fi_peaks * spec.shape[0] / (fs / 2))

    Hmax = np.zeros(len(k_peaks))
    for i in np.arange(len(k_peaks)):
        Hmax[i] = np.amax(spec[int(k_peaks[i] - k_peaks[0] / 10):int(k_peaks[i] + k_peaks[0] / 10)])

    Hpos = np.zeros(len(k_peaks))
    for i in np.arange(len(k_peaks)):
        Hpos[i] = (np.argmax(spec[int(k_peaks[i] - k_peaks[0] / 10):int(k_peaks[i] + k_peaks[0] / 2)]) + int(
            k_peaks[i] - k_peaks[0] / 10)) * (fs / 2) / spec.shape[0]

    Hen = np.zeros(len(k_peaks))
    for i in np.arange(len(k_peaks)):
        Hen[i] = np.mean(spec[int(k_peaks[i] - k_peaks[0] / 10):int(k_peaks[i] + k_peaks[0] / 10)])

    return Hmax, Hpos, Hen


def compute_harmonics_features(Hmax, Hpos, Hen, peaks_n):
    hp_s = 16
    wn_l = 64
    fr_n = int(np.floor(Hmax.shape[1] - wn_l) / hp_s)
    window = sp.signal.get_window(window='hanning', Nx=wn_l)
    Sr = 76  # frames number(152) / 2 sec
    Nyq = int(Sr / 2)
    temp = np.zeros(wn_l)
    # initialization
    spec_hmax_1 = np.zeros([Nyq - 1, fr_n])
    spec_hmax_2 = np.zeros([Nyq - 1, fr_n])
    spec_hmax_3 = np.zeros([Nyq - 1, fr_n])
    spec_hmax_4 = np.zeros([Nyq - 1, fr_n])
    spec_hmax_5 = np.zeros([Nyq - 1, fr_n])

    spec_hpos_1 = np.zeros([Nyq - 1, fr_n])
    spec_hpos_2 = np.zeros([Nyq - 1, fr_n])
    spec_hpos_3 = np.zeros([Nyq - 1, fr_n])
    spec_hpos_4 = np.zeros([Nyq - 1, fr_n])
    spec_hpos_5 = np.zeros([Nyq - 1, fr_n])

    spec_hen_1 = np.zeros([Nyq - 1, fr_n])
    spec_hen_2 = np.zeros([Nyq - 1, fr_n])
    spec_hen_3 = np.zeros([Nyq - 1, fr_n])
    spec_hen_4 = np.zeros([Nyq - 1, fr_n])
    spec_hen_5 = np.zeros([Nyq - 1, fr_n])

    for i in np.arange(fr_n):
        frame_hmax = Hmax[:, i * hp_s: i * hp_s + wn_l]
        frame_hpos = Hpos[:, i * hp_s: i * hp_s + wn_l]
        frame_hen = Hen[:, i * hp_s: i * hp_s + wn_l]

        # build spec_hmax's matrices
        temp = np.fft.fft(frame_hmax[0, :])
        spec_hmax_1[:, i] = np.abs(temp[1:Nyq])

        temp = np.fft.fft(frame_hmax[1, :])
        spec_hmax_2[:, i] = np.abs(temp[1:Nyq])

        temp = np.fft.fft(frame_hmax[2, :])
        spec_hmax_3[:, i] = np.abs(temp[1:Nyq])

        temp = np.fft.fft(frame_hmax[3, :])
        spec_hmax_4[:, i] = np.abs(temp[1:Nyq])

        temp = np.fft.fft(frame_hmax[4, :])
        spec_hmax_5[:, i] = np.abs(temp[1:Nyq])

        # build spec_hpos's matrices
        temp = np.fft.fft(frame_hpos[0, :])
        spec_hpos_1[:, i] = np.abs(temp[1:Nyq])

        temp = np.fft.fft(frame_hpos[1, :])
        spec_hpos_2[:, i] = np.abs(temp[1:Nyq])

        temp = np.fft.fft(frame_hpos[2, :])
        spec_hpos_3[:, i] = np.abs(temp[1:Nyq])

        temp = np.fft.fft(frame_hpos[3, :])
        spec_hpos_4[:, i] = np.abs(temp[1:Nyq])

        temp = np.fft.fft(frame_hpos[4, :])
        spec_hpos_5[:, i] = np.abs(temp[1:Nyq])

        # build spec_hen's matrices
        temp = np.fft.fft(frame_hen[0, :])
        spec_hen_1[:, i] = np.abs(temp[1:Nyq])

        temp = np.fft.fft(frame_hen[1, :])
        spec_hen_2[:, i] = np.abs(temp[1:Nyq])

        temp = np.fft.fft(frame_hen[2, :])
        spec_hen_3[:, i] = np.abs(temp[1:Nyq])

        temp = np.fft.fft(frame_hen[3, :])
        spec_hen_4[:, i] = np.abs(temp[1:Nyq])

        temp = np.fft.fft(frame_hen[4, :])
        spec_hen_5[:, i] = np.abs(temp[1:Nyq])

    spec_hmax_1 = np.abs(spec_hmax_1[0:23, :])
    spec_hmax_2 = np.abs(spec_hmax_2[0:23, :])
    spec_hmax_3 = np.abs(spec_hmax_3[0:23, :])
    spec_hmax_4 = np.abs(spec_hmax_4[0:23, :])
    spec_hmax_5 = np.abs(spec_hmax_5[0:23, :])

    spec_hpos_1 = np.abs(spec_hpos_1[0:23, :])
    spec_hpos_2 = np.abs(spec_hpos_2[0:23, :])
    spec_hpos_3 = np.abs(spec_hpos_3[0:23, :])
    spec_hpos_4 = np.abs(spec_hpos_4[0:23, :])
    spec_hpos_5 = np.abs(spec_hpos_5[0:23, :])

    spec_hen_1 = np.abs(spec_hen_1[0:23, :])
    spec_hen_2 = np.abs(spec_hen_2[0:23, :])
    spec_hen_3 = np.abs(spec_hen_3[0:23, :])
    spec_hen_4 = np.abs(spec_hen_4[0:23, :])
    spec_hen_5 = np.abs(spec_hen_5[0:23, :])

    dic_spec_hmax = {1: spec_hmax_1, 2: spec_hmax_2, 3: spec_hmax_3, 4: spec_hmax_4, 5: spec_hmax_5}
    dic_spec_hpos = {1: spec_hpos_1, 2: spec_hpos_2, 3: spec_hpos_3, 4: spec_hpos_4, 5: spec_hpos_5}
    dic_spec_hen = {1: spec_hen_1, 2: spec_hen_2, 3: spec_hen_3, 4: spec_hen_4, 5: spec_hen_5}

    hf_means = np.zeros(120)
    hf_stds = np.zeros(120)
    temp2 = np.zeros(fr_n)
    temp3 = np.zeros((spec_hmax_1.shape[0], spec_hmax_1.shape[1]))

    # compute harmonic features for Hmax
    for i in np.arange(1, peaks_n + 1):
        temp3 = dic_spec_hmax[i]

        temp2 = np.mean(dic_spec_hmax[i], axis=0)
        hf_means[8 * (i - 1)] = np.mean(temp2)
        hf_stds[8 * (i - 1)] = np.std(temp2)

        temp2 = np.std(dic_spec_hmax[i], axis=0)
        hf_means[8 * (i - 1) + 1] = np.mean(temp2)
        hf_stds[8 * (i - 1) + 1] = np.std(temp2)

        temp2 = np.amax(dic_spec_hmax[i], axis=0)
        hf_means[8 * (i - 1) + 2] = np.mean(temp2)
        hf_stds[8 * (i - 1) + 2] = np.std(temp2)

        temp2 = np.argmax(dic_spec_hmax[i], axis=0)
        hf_means[8 * (i - 1) + 3] = np.mean(temp2)
        hf_stds[8 * (i - 1) + 3] = np.std(temp2)

        for j in np.arange(fr_n):
            temp2[j] = compute_speccentr(temp3[:, j])
        hf_means[8 * (i - 1) + 4] = np.mean(temp2)
        hf_stds[8 * (i - 1) + 4] = np.std(temp2)

        for j in np.arange(fr_n):
            temp2[j] = compute_specspread(temp3[:, j])
        hf_means[8 * (i - 1) + 5] = np.mean(temp2)
        hf_stds[8 * (i - 1) + 5] = np.std(temp2)

        temp2 = sp.stats.skew(dic_spec_hmax[i], axis=0)
        hf_means[8 * (i - 1) + 6] = np.mean(temp2)
        hf_stds[8 * (i - 1) + 6] = np.std(temp2)

        temp2 = sp.stats.kurtosis(dic_spec_hmax[i], axis=0)
        hf_means[8 * (i - 1) + 7] = np.mean(temp2)
        hf_stds[8 * (i - 1) + 7] = np.std(temp2)

    # compute harmonic features for hpos
    for i in np.arange(1, peaks_n + 1):
        temp3 = dic_spec_hpos[i]

        temp2 = np.mean(dic_spec_hpos[i], axis=0)
        hf_means[40 + 8 * (i - 1)] = np.mean(temp2)
        hf_stds[40 + 8 * (i - 1)] = np.std(temp2)

        temp2 = np.std(dic_spec_hpos[i], axis=0)
        hf_means[40 + 8 * (i - 1) + 1] = np.mean(temp2)
        hf_stds[40 + 8 * (i - 1) + 1] = np.std(temp2)

        temp2 = np.amax(dic_spec_hpos[i], axis=0)
        hf_means[40 + 8 * (i - 1) + 2] = np.mean(temp2)
        hf_stds[40 + 8 * (i - 1) + 2] = np.std(temp2)

        temp2 = np.argmax(dic_spec_hpos[i], axis=0)
        hf_means[40 + 8 * (i - 1) + 3] = np.mean(temp2)
        hf_stds[40 + 8 * (i - 1) + 3] = np.std(temp2)

        for j in np.arange(fr_n):
            temp2[j] = compute_speccentr(temp3[:, j])
        hf_means[40 + 8 * (i - 1) + 4] = np.mean(temp2)
        hf_stds[40 + 8 * (i - 1) + 4] = np.std(temp2)

        for j in np.arange(fr_n):
            temp2[j] = compute_specspread(temp3[:, j])
        hf_means[40 + 8 * (i - 1) + 5] = np.mean(temp2)
        hf_stds[40 + 8 * (i - 1) + 5] = np.std(temp2)

        temp2 = sp.stats.skew(dic_spec_hpos[i], axis=0)
        hf_means[40 + 8 * (i - 1) + 6] = np.mean(temp2)
        hf_stds[40 + 8 * (i - 1) + 6] = np.std(temp2)

        temp2 = sp.stats.kurtosis(dic_spec_hpos[i], axis=0)
        hf_means[40 + 8 * (i - 1) + 7] = np.mean(temp2)
        hf_stds[40 + 8 * (i - 1) + 7] = np.std(temp2)

        # compute harm features for hen
    for i in np.arange(1, peaks_n + 1):
        temp3 = dic_spec_hen[i]

        temp2 = np.mean(dic_spec_hen[i], axis=0)
        hf_means[80 + 8 * (i - 1)] = np.mean(temp2)
        hf_stds[80 + 8 * (i - 1)] = np.std(temp2)

        temp2 = np.std(dic_spec_hen[i], axis=0)
        hf_means[80 + 8 * (i - 1) + 1] = np.mean(temp2)
        hf_stds[80 + 8 * (i - 1) + 1] = np.std(temp2)

        temp2 = np.amax(dic_spec_hen[i], axis=0)
        hf_means[80 + 8 * (i - 1) + 2] = np.mean(temp2)
        hf_stds[80 + 8 * (i - 1) + 2] = np.std(temp2)

        temp2 = np.argmax(dic_spec_hen[i], axis=0)
        hf_means[80 + 8 * (i - 1) + 3] = np.mean(temp2)
        hf_stds[80 + 8 * (i - 1) + 3] = np.std(temp2)

        for j in np.arange(fr_n):
            temp2[j] = compute_speccentr(temp3[:, j])
        hf_means[80 + 8 * (i - 1) + 4] = np.mean(temp2)
        hf_stds[80 + 8 * (i - 1) + 4] = np.std(temp2)

        for j in np.arange(fr_n):
            temp2[j] = compute_specspread(temp3[:, j])
        hf_means[80 + 8 * (i - 1) + 5] = np.mean(temp2)
        hf_stds[80 + 8 * (i - 1) + 5] = np.std(temp2)

        temp2 = sp.stats.skew(dic_spec_hen[i], axis=0)
        hf_means[80 + 8 * (i - 1) + 6] = np.mean(temp2)
        hf_stds[80 + 8 * (i - 1) + 6] = np.std(temp2)

        temp2 = sp.stats.kurtosis(dic_spec_hen[i], axis=0)
        hf_means[80 + 8 * (i - 1) + 7] = np.mean(temp2)
        hf_stds[80 + 8 * (i - 1) + 7] = np.std(temp2)

    harm_features = np.concatenate((hf_means, hf_stds))
    return harm_features


def compute_harmonic_curves(audio_file, fs, peaks_n, file_path):
    # calculate harmonic features with respective window and fft length
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

    # initialize matrices to save harmonic curves for every feature Max, Pos, Energy and for every frame
    Hmax = np.zeros((peaks_n, frames_number))
    Hpos = np.zeros((peaks_n, frames_number))
    Hen = np.zeros((peaks_n, frames_number))

    # compute the harmonic curves
    for i in np.arange(frames_number):
        frame = audio_file[i * hop_size: i * hop_size + win_length]
        frame_wind = frame * window

        spec = np.fft.fft(frame_wind, n=fft_length)
        nyquist = int(np.floor(spec.shape[0] / 2))

        spec = spec[1: nyquist]

        Hmax[:, i], Hpos[:, i], Hen[:, i] = compute_harmonics_curves_frame(frame_wind, peaks_n, file_path, fs)

    return Hmax, Hpos, Hen


def compute_features_dataset(dataset, class_name):
    logger.info(str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) +
                ' Computing ' + str(dataset) + ' features for class: ' + str(class_name))

    # data = genfromtxt('data/{}_{}.csv'.format(dataset, class_name), delimiter=',')
    # df = pd.read_csv('data/{}_{}.csv'.format(dataset, class_name))

    files_root = 'data/{}/{}'.format(dataset, class_name)
    class_files = [f for f in os.listdir(files_root) if f.endswith('.wav')]

    n_files = len(class_files)
    n_mfcc = 10
    n_harmonics = 5
    n_harmonic_curves = 3

    all_files_features_mean = np.zeros((n_files, len(features)))
    all_files_features_std = np.zeros((n_files, len(features)))
    all_files_features_kurt = np.zeros((n_files, len(features)))
    all_files_features_skew = np.zeros((n_files, len(features)))

    all_files_features_first_der_mean = np.zeros((n_files, len(features)))
    all_files_features_first_der_std = np.zeros((n_files, len(features)))
    all_files_features_first_der_kurt = np.zeros((n_files, len(features)))
    all_files_features_first_der_skew = np.zeros((n_files, len(features)))

    all_files_features_second_der_mean = np.zeros((n_files, len(features)))
    all_files_features_second_der_std = np.zeros((n_files, len(features)))
    all_files_features_second_der_kurt = np.zeros((n_files, len(features)))
    all_files_features_second_der_skew = np.zeros((n_files, len(features)))

    all_files_mfccs_mean = np.zeros((n_files, n_mfcc))
    all_files_mfccs_max = np.zeros((n_files, n_mfcc))
    all_files_mfccs_std = np.zeros((n_files, n_mfcc))

    all_files_mfccs_first_der_mean = np.zeros((n_files, n_mfcc))
    all_files_mfccs_first_der_std = np.zeros((n_files, n_mfcc))

    all_files_mfccs_second_der_mean = np.zeros((n_files, n_mfcc))
    all_files_mfccs_second_der_std = np.zeros((n_files, n_mfcc))

    all_files_harmonics_features = np.zeros((n_files, n_harmonic_curves * n_harmonics * 2 * 8))

    for index, f in enumerate(class_files):
        # load the audio file f
        print("File analysed : " + str(f))
        audio, fs = librosa.load(os.path.join(files_root, f), sr=None)
        # compute the spectral features for every frame
        file_features_frames = compute_features_frames(audio, fs, features, n_harmonics, f)

        # compute the first and second derivative of the features
        file_features_first_der_frames = np.diff(file_features_frames, axis=1)
        file_features_second_der_frames = np.diff(file_features_first_der_frames, axis=1)

        # compute the cepstrum coefficients
        file_mfccs = compute_mfcc(audio, fs, n_mfcc)

        # compute the first and second derivative of the cepstrum coefficients
        file_mfccs_first_der_frames = np.diff(file_mfccs, axis=1)
        file_mfccs_second_der_frames = np.diff(file_mfccs_first_der_frames, axis=1)

        # compute harmonic curves
        Hmax, Hpos, Hen = compute_harmonic_curves(audio, fs, n_harmonics, f)

        # compute the mean value between frames of all the features of the audio file
        # and store it in the matrix that holds the features for all audio files of the class
        all_files_features_mean[index, :] = np.mean(file_features_frames, axis=1)
        all_files_features_std[index, :] = np.ndarray.std(file_features_frames, axis=1)
        all_files_features_kurt[index, :] = sp.stats.kurtosis(file_features_frames, axis=1)
        all_files_features_skew[index, :] = sp.stats.skew(file_features_frames, axis=1)

        all_files_features_first_der_mean[index, :] = np.mean(file_features_first_der_frames, axis=1)
        all_files_features_first_der_std[index, :] = np.ndarray.std(file_features_first_der_frames, axis=1)
        all_files_features_first_der_kurt[index, :] = sp.stats.kurtosis(file_features_first_der_frames, axis=1)
        all_files_features_first_der_skew[index, :] = sp.stats.skew(file_features_first_der_frames, axis=1)

        all_files_features_second_der_mean[index, :] = np.mean(file_features_second_der_frames, axis=1)
        all_files_features_second_der_std[index, :] = np.ndarray.std(file_features_second_der_frames, axis=1)
        all_files_features_second_der_kurt[index, :] = sp.stats.kurtosis(file_features_second_der_frames, axis=1)
        all_files_features_second_der_skew[index, :] = sp.stats.skew(file_features_second_der_frames, axis=1)

        all_files_mfccs_mean[index, :] = np.mean(file_mfccs, axis=1)
        all_files_mfccs_max[index, :] = np.amax(file_mfccs, axis=1)
        all_files_mfccs_std[index, :] = np.ndarray.std(file_mfccs, axis=1)

        all_files_mfccs_first_der_mean[index, :] = np.mean(file_mfccs_first_der_frames, axis=1)
        all_files_mfccs_first_der_std[index, :] = np.ndarray.std(file_mfccs_first_der_frames, axis=1)

        all_files_mfccs_second_der_mean[index, :] = np.mean(file_mfccs_second_der_frames, axis=1)
        all_files_mfccs_second_der_std[index, :] = np.ndarray.std(file_mfccs_second_der_frames, axis=1)

        # compute harmonic features using harmonic curves
        all_files_harmonics_features[index, :] = compute_harmonics_features(Hmax, Hpos, Hen, n_harmonics)

    all_features = np.concatenate(
        (all_files_features_mean, all_files_features_std, all_files_features_kurt, all_files_features_skew,

         all_files_features_first_der_mean, all_files_features_first_der_std, all_files_features_first_der_kurt,
         all_files_features_first_der_skew,

         all_files_features_second_der_mean, all_files_features_second_der_std, all_files_features_second_der_kurt,
         all_files_features_second_der_skew,

         all_files_mfccs_mean, all_files_mfccs_max, all_files_mfccs_std,

         all_files_mfccs_first_der_mean, all_files_mfccs_first_der_std,

         all_files_mfccs_second_der_mean, all_files_mfccs_second_der_std,

         all_files_harmonics_features), axis=1)

    return all_features


datasets = ['Training', 'Test']
classes = ['Distortion', 'NoFX', 'Tremolo']
dict_features = {'Distortion': [], 'NoFX': [], 'Tremolo': []}
features = ['Spectral Centroid', 'Spectral Spread', 'Spectral Skewness', 'Spectral Kurtosis',
            'Spectral Rolloff', 'Spectral Slope', 'Spectral Flatness', 'Spectral Flux', 'Zero Crossing Rate']

logger.info(str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ' Starting Feature Extraction...')

# Building Labeled Features matrix for Training
X_train_Distortion = compute_features_dataset(datasets[0], classes[0])
X_train_NoFX = compute_features_dataset(datasets[0], classes[1])
X_train_Tremolo = compute_features_dataset(datasets[0], classes[2])

logger.info(str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ' Saving Training Features...')

# save train features to csv file
savetxt('results/X_train_Distortion.csv', X_train_Distortion, delimiter=',')
savetxt('results/X_train_NoFX.csv', X_train_NoFX, delimiter=',')
savetxt('results/X_train_Tremolo.csv', X_train_Tremolo, delimiter=',')

# Building Labeled Features matrix for Test
X_test_Distortion = compute_features_dataset(datasets[1], classes[0])
X_test_NoFX = compute_features_dataset(datasets[1], classes[1])
X_test_Tremolo = compute_features_dataset(datasets[1], classes[2])

logger.info(str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ' Saving Test Features...')

# save train features to csv file
savetxt('results/X_test_Distortion.csv', X_test_Distortion, delimiter=',')
savetxt('results/X_test_NoFX.csv', X_test_NoFX, delimiter=',')
savetxt('results/X_test_Tremolo.csv', X_test_Tremolo, delimiter=',')

logger.info(str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ' Ended Feature Extraction...')
