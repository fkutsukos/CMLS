# CMLS 
CMLS Assignment 3 - Audio Effect Classification by students 


A system of consisted of 3 python programs that combined are able
to predict the audio effect used in recordings of electric guitar and bass.

The 3 classes to discriminated are :
- distortion
- tremolo
- no effect

The dataset is a set of around 11k monophonic guitar and bass sample sounds plus 4k polyphonic sounds.

The system is composed of three python programs that can be executed independently:

1. preprocessing.py: This program produces as output the files_information.csv table which contains the sample of the 
audio file at the beginning of the sustain phase.

2. featureExtraction.py: Based on the files_information.csv and the /data audio files this program computes the features
 for every audio file in the data folder and produces the csv feature vectors for every type of effect and sound
  (monophonic/polyphonic) in the /results folder
    - X_Distortion_poly.csv
    - X_Distortion.csv
    - X_NoFX.csv
    - X_NoFX_poly.csv
    - X_Tremolo.csv
    - X_Tremolo_poly.csv
    
3.classify.py: implements the estimator and produces the metrics and the confusion matrix result
