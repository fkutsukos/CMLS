import numpy as np
import pandas as pd
import librosa
import os
import sys
import scipy as sp

def compute_ap(win):
    n = win.shape[0]
    s_m = win**2
    ap = sum(s_m)/n
    return ap

folder = ['Test', 'Training'];
classes=['Distortion', 'NoFX', 'Tremolo'];
dict_train_ap = {'NoFX': [], 'Distortion': [], 'Tremolo': []};

Fs = 44100 #for all audio files;
win_length = int(np.floor(0.01 * Fs));
hop_size = win_length;
window = sp.signal.get_window(window='boxcar', Nx=win_length);

files_information = pd.DataFrame(columns = ['name', 'effect', 'attack'])
all_files = pd.DataFrame()

from pathlib import Path

for i in folder:
    print(i)
    for c in classes:
        print(c)

        ap_root = Path('Data/data/{}/{}'.format(i, c))

        #ap_root = '\CMLS\Prova\{}\{}'.format(i,c)
        files = [f for f in os.listdir(ap_root) if f.endswith('.wav')];

        for index, f in enumerate(files):
            audio, fs = librosa.load(os.path.join(ap_root, f), sr=None);
            audio_win_number = int(np.floor((audio.shape[0] - win_length) / hop_size));

            ap_audio = np.zeros((1,audio_win_number));

            for j in np.arange(audio_win_number):
                frame = audio[j * hop_size : j * hop_size + win_length];
                frame_wind = frame*window;

                en_frame = compute_ap(frame_wind);

                ap_audio[0,j] = en_frame;

                max_ap = np.amax(ap_audio);

            for j in np.arange(audio_win_number):
                if ap_audio[0,j] > 0.2*max_ap:
                   attack_frame = j;
                   attack_sample = j*win_length;
                   break

            #store useful data in pandas DataFrames

            np.set_printoptions(threshold=sys.maxsize)

            audio = pd.Series(audio)
            all_files = all_files.append(audio, ignore_index=True)

            files_information = files_information.append({'name': f, 'effect': c, 'attack': attack_sample}, ignore_index=True)

            files_information.to_csv('files_information.csv', index = False)
