# CMLS 
CMLS  Homework 1 , Group 2, Assignment 3 - Audio Effect Classification by:
- 10577360 
- 10650559 
- 10574134 
- 10532582

A system of consisted of 3 python programs that combined are able
to predict the audio effect used in recordings of electric guitar and bass.

The 3 classes to discriminated are :
- distortion
- tremolo
- no effect

The dataset is a set of around 11k monophonic guitar and bass sample sounds plus 4k polyphonic sounds.

The system is composed of three python programs that can be executed independently:

1. preprocessing.py: this script reads the audio files and produces as output the files_information.csv table, which contains the index of the audio sample in which the attack phase
ends.

2. featureExtraction.py: based on the files_information.csv this program computes the
low-level features for every audio file of the data folder and produces the results folder csv
tables for every effect (Tremolo, Distortion, NoFX):
– X_Tremolo.csv
– X_Distortion.csv
– X_NoFX.csv

3. classify.py: this script implements the estimator and produces the metrics and the confusion matrix result.

