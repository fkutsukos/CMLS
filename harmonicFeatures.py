'''
import numpy as np
import librosa
import scipy as sp
%matplotlib inline

import matplotlib.pyplot as plt
import IPython.display as ipd
from sklearn.mixture import BayesianGaussianMixture
import scipy.stats
import seaborn as sns

# %%
# file path
file_path = '/content/B11-28100-1111-00001.wav'

# %%
# load and plot the audio file
x, Fs = librosa.load(file_path, sr=None)
time_axis = np.arange(x.shape[0]) / Fs

# %%
#   defnition of the three armonic features
def compute_harmonics(win):
    harmonic_features = librosa.effects.harmonic(win)
    spec = np.abs(np.fft.fft(harmonic_features))
    nyquist = int(np.floor(spec.shape[0] / 2))

    spec = spec[1:nyquist]

    fundamental_index = np.argmax(spec)
    fundamental_freq = fundamental_index * (Fs / 2) / spec.shape[0]

    frequency_axis = np.arange(spec.shape[0]) * (Fs / 2) / spec.shape[0]


    plt.figure(figsize=(16, 5))

    plt.plot(frequency_axis[1:1000], spec[1:1000])
    plt.grid(True)
    plt.title('Spectrum of harmonic part')


    peaks,_ = scipy.signal.find_peaks(spec, prominence = 100)

    for i,_ in enumerate(peaks):
      if i < len(peaks)-1:
        if spec[peaks[i]] < spec[peaks[i + 1]]:
            peaks = np.delete(peaks, i)
            i = i-1

    peaks = peaks[:5]

    Hmax = np.zeros(len(peaks))
    for i in np.arange(len(peaks)):
        Hmax[i] = spec[peaks[i]]

    Hpos = peaks * (Fs / 2) / spec.shape[0]
    Hen = np.zeros(len(peaks))
    for i in np.arange(len(peaks)):
        Hen[i] = np.mean(spec[int(peaks[i]-peaks[1]/10):int(peaks[i]+peaks[1]/10)])
    return Hmax, Hpos, Hen

#%%
#computig the features

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

Hmax = np.zeros((5,frames_number))
Hpos = np.zeros((5,frames_number))
Hen = np.zeros((5,frames_number))


#compute the harmonic features
for i in np.arange(frames_number):
    frame = x[i * hop_size : i*hop_size + win_length]
    frame_wind = frame * window

    spec = np.fft.fft(frame_wind)
    nyquist = int(np.floor(spec.shape[0] / 2))

    spec = spec [1 : nyquist]

    Hmax[:,i],Hmax[:,i],Hmax[:,i] = compute_harmonics(frame_wind)

#%%
#plot it

plt.figure(figsize=(16, 5))
for i in np.arange(5):
    plt.plot(Hmax[i,:])
plt.title('Hmax Curves')
plt.xlabel('frames')
plt.grid(True)

plt.figure(figsize=(16, 5))
for i in np.arange(5):
    plt.plot(Hpos[i,:])
plt.title('Hpos Curves')
plt.xlabel('frames')
plt.grid(True)

plt.figure(figsize=(16, 5))
for i in np.arange(5):
    plt.plot(Hen[i,:])
plt.title('Hen Curves')
plt.xlabel('frames')
plt.grid(True)
'''
import numpy as np
import librosa
import scipy as sp
% matplotlib
inline

import matplotlib.pyplot as plt
import IPython.display as ipd
from sklearn.mixture import BayesianGaussianMixture
import scipy.stats
import seaborn as sns

# %%
# file path
file_path = '/content/B11-30102-4413-07491.wav'


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


# %%
# load and plot the audio file
x, Fs = librosa.load(file_path, sr=None)
time_axis = np.arange(x.shape[0]) / Fs


# %%
#   defnition of the three armonic features
def compute_harmonics(win, peaks_n):
    harmonic_features = librosa.effects.harmonic(win)
    spec = np.abs(np.fft.fft(harmonic_features))
    #spec = np.abs(np.fft.fft(win))
    nyquist = int(np.floor(spec.shape[0] / 2))

    spec = spec[1:nyquist]
    fund_pitch = pitch_detection(win, 40, 1200, Fs)

    fi_peaks = np.arange(1, peaks_n + 1) * fund_pitch
    k_peaks = np.round(fi_peaks * spec.shape[0] / (Fs / 2))

    Hmax = np.zeros(len(k_peaks))
    for i in np.arange(len(k_peaks)):
        Hmax[i] = np.amax(spec[int(k_peaks[i] - k_peaks[0] / 10):int(k_peaks[i] + k_peaks[0] / 10)])

    Hpos = np.zeros(len(k_peaks))
    for i in np.arange(len(k_peaks)):
        Hpos[i] = np.argmax(spec[int(k_peaks[i] - k_peaks[0] / 10):int(k_peaks[i] + k_peaks[0] / 2)]) * (Fs / 10) / \
                  spec.shape[0]

    Hen = np.zeros(len(k_peaks))
    for i in np.arange(len(k_peaks)):
        Hen[i] = np.mean(spec[int(k_peaks[i] - k_peaks[0] / 10):int(k_peaks[i] + k_peaks[0] / 10)])

    return Hmax, Hpos, Hen

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

peaks_n = 5

Hmax = np.zeros((peaks_n,frames_number))
Hpos = np.zeros((peaks_n,frames_number))
Hen = np.zeros((peaks_n,frames_number))


#compute the harmonic features
for i in np.arange(frames_number):
    frame = x[i * hop_size : i*hop_size + win_length]
    frame_wind = frame * window

    spec = np.fft.fft(frame_wind)
    nyquist = int(np.floor(spec.shape[0] / 2))

    spec = spec [1 : nyquist]

    Hmax[:,i], Hpos[:,i], Hen[:,i] = compute_harmonics(frame_wind, peaks_n)

plt.figure(figsize=(16, 5))
for i in np.arange(5):
    plt.plot(Hmax[i,:])
plt.title('Hmax Curves')
plt.xlabel('frames')
plt.grid(True)

plt.figure(figsize=(16, 5))
for i in np.arange(5):
    plt.plot(Hpos[i,:])
plt.title('Hpos Curves')
plt.xlabel('frames')
plt.grid(True)

plt.figure(figsize=(16, 5))
for i in np.arange(5):
    plt.plot(Hen[i,:])
plt.title('Hen Curves')
plt.xlabel('frames')
plt.grid(True)