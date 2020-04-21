'''
# Moving Files from Test to Train
for c in classes:
    print(1)
    train_root = Path('data/Training/{}'.format(c))
    test_root = Path('data/Test/{}'.format(c))
    class_train_files = [f for f in os.listdir(train_root) if f.endswith('.wav')]
    class_test_files = [f for f in os.listdir(test_root) if f.endswith('.wav')]
    n_train_audio_files = len(class_train_files)
    n_test_audio_files = len(class_test_files)
    print('Train path , total files', train_root, ', ', n_train_audio_files)
    print('Train path , total files', test_root, ', ', n_test_audio_files)
    for index, f in enumerate(class_test_files):
        if index > int(n_test_audio_files*0.9):
            break
        shutil.move(test_root / f, train_root / f)
'''

'''
for c in classes:
    logger.info(str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) +
                ' Computing features for class: ' + str(c))
    train_root = 'data/Training/{}'.format(c)
    class_train_files = [f for f in os.listdir(train_root) if f.endswith('.wav')]
    n_audio_files = len(class_train_files)
    train_features_mean = np.zeros((n_audio_files, len(features)))
    for index, f in enumerate(class_train_files):
        # testing limit of 10 files
        # if index == num_audio_files:
        #    break
        # load the audio file f
        audio, fs = librosa.load(os.path.join(train_root, f), sr=None)
        # compute all the features for every frame of the audio file
        train_features_frames = compute_features_frames(audio, fs, features)
        # compute the mean value between frames of all the features of the audio file
        # and store it in the matrix that holds the features for all audio files of the class
        train_features_mean[index, :] = np.mean(train_features_frames, axis=1)
    # add into the dictionary of class c the matrix train_features_mean
    dict_features[c] = train_features_mean
'''

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