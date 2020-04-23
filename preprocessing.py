import numpy as np
import pandas as pd
import librosa
import os
import sys
import matplotlib.pyplot as plt
import IPython.display as ipd
import scipy as sp
import wget as wget


!wget --no-check-certificate -r "https://drive.google.com/uc?export=download&id=13gk4s3f0ULwee24Xb1gAA6Mm4YqveA3Q" -O "Prova.zip"


!unzip Prova.zip

def compute_ap(win):
    n = win.shape[0]
    s_m = win**2
    ap = sum(s_m)/n
    return ap


instrument = ['Bass monophon', 'Gitarre monophon'];
classes=['Distortion', 'Tremolo', 'NoFX'];
dict_train_ap = {'NoFX': [], 'Distortion': [], 'Tremolo': []};

Fs = 44100 #for all audio files;
win_length = int(np.floor(0.01 * Fs));
hop_size = win_length;
window = sp.signal.get_window(window='boxcar', Nx=win_length);

files_information = pd.DataFrame(columns = ['name', 'instrument', 'effect', 'index'])
processed_files = pd.DataFrame()
#dataset = pd.DataFrame(columns=['Audios','vec'])

counter = 0



for i in instrument:
    print(i)
    for c in classes:
        print(c)
        ap_root = '/content/Prova/{}/{}/'.format(i,c);
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

            processed_audio = audio[attack_sample:];

            #store processed_audio in csv file

            np.set_printoptions(threshold=sys.maxsize)

            processed_audio = pd.Series(processed_audio)

            files_information = files_information.append({'name': f, 'instrument': i, 'effect': c, 'index': counter}, ignore_index=True)
            files_information.to_csv('informations.csv')

            processed_files = processed_files.append(processed_audio, ignore_index=True)
            processed_files.to_csv('processed_files.csv')

            counter = counter + 1




            #dataset=dataset.append({'Audios':[f],'vec': processed_audio}, ignore_index= True)

#np.set_printoptions(threshold=sys.maxsize)
#dataset.to_csv('dataset.csv')




audio_test = processed_files.loc[3]

plt.figure(figsize=(16, 16))
time_axis = np.arange(audio_test.shape[0])
plt.plot(time_axis, audio_test)
plt.grid(True)
plt.title('attack audio')

ipd.Audio(audio_test, rate=Fs)