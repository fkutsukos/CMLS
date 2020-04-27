from numpy import loadtxt
import logging
import numpy as np
import datetime
import sklearn.svm

# ~ LOG_LEVEL = logging.INFO
from sklearn.feature_selection import SelectKBest, f_classif

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

X_train_Distortion = loadtxt('results/X_train_Distortion.csv',  delimiter=',')
X_train_NoFX = loadtxt('results/X_train_NoFX.csv', delimiter=',')
X_train_Tremolo = loadtxt('results/X_train_Tremolo.csv', delimiter=',')

# Build the Ground Truth vector for Training
y_train_Distortion = np.zeros((X_train_Distortion.shape[0],))
y_train_NoFX = np.ones((X_train_NoFX.shape[0],))
y_train_Tremolo = np.ones((X_train_Tremolo.shape[0],)) * 2

X_test_Distortion = loadtxt('results/X_test_Distortion.csv', delimiter=',')
X_test_NoFX = loadtxt('results/X_test_NoFX.csv', delimiter=',')
X_test_Tremolo = loadtxt('results/X_test_Tremolo.csv', delimiter=',')

# Build the Ground Truth vector for Test
y_test_Distortion = np.zeros((X_test_Distortion.shape[0],))
y_test_NoFX = np.ones((X_test_NoFX.shape[0],))
y_test_Tremolo = np.ones((X_test_Tremolo.shape[0],)) * 2

y_test = np.concatenate((y_test_Distortion, y_test_NoFX, y_test_Tremolo), axis=0)

# Normalization
logger.info(str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ' Data Normalization in process...')

feat_max = np.max(np.concatenate((X_train_Distortion, X_train_NoFX, X_train_Tremolo), axis=0), axis=0)
feat_min = np.min(np.concatenate((X_train_Distortion, X_train_NoFX, X_train_Tremolo), axis=0), axis=0)

# Train Matrices Normalised
X_train_Distortion_normalized = (X_train_Distortion - feat_min) / (feat_max - feat_min)
X_train_NoFX_normalized = (X_train_NoFX - feat_min) / (feat_max - feat_min)
X_train_Tremolo_normalized = (X_train_Tremolo - feat_min) / (feat_max - feat_min)

# Test Matrices Normalised
X_test_Distortion_normalized = (X_test_Distortion - feat_min) / (feat_max - feat_min)
X_test_NoFX_normalized = (X_test_NoFX - feat_min) / (feat_max - feat_min)
X_test_Tremolo_normalized = (X_test_Tremolo - feat_min) / (feat_max - feat_min)

X_test_normalized = np.concatenate((X_test_Distortion_normalized, X_test_NoFX_normalized, X_test_Tremolo_normalized),
                                   axis=0)

# Train Matrices Feature Selection
k = 90
logger.info(str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ' Feature Selection in process with k = ' +str(k))
selectedFeatures = SelectKBest(score_func=f_classif, k=k)

selected = selectedFeatures.fit(np.concatenate((X_train_Distortion_normalized, X_train_NoFX_normalized, X_train_Tremolo_normalized), axis=0),
                                                  np.concatenate((y_train_Distortion, y_train_NoFX, y_train_Tremolo), axis=0))
X_train_selected_normalised = selectedFeatures.fit_transform(np.concatenate((X_train_Distortion_normalized, X_train_NoFX_normalized, X_train_Tremolo_normalized), axis=0),
                                                  np.concatenate((y_train_Distortion, y_train_NoFX, y_train_Tremolo), axis=0))

# Getting the mask of selected features X_Train and apply it on X_Test
feature_index = selected.get_support(True)
X_test_normalized_selected = X_test_normalized[:, feature_index]

# Splitting the concatenated X_train_selected and normalized matrix per each class
X_train_Distortion_selected_normalized = X_train_selected_normalised[: int(X_train_selected_normalised.shape[0]/3), :]
X_train_NoFX_selected_normalized = X_train_selected_normalised[int(X_train_selected_normalised.shape[0]/3) : int(X_train_selected_normalised.shape[0]*2/3), :]
X_train_Tremolo_selected_normalized = X_train_selected_normalised[int(X_train_selected_normalised.shape[0]*2/3) :, :]

logger.info(str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ' Classification in process...')
# Setting SVM Parameters
SVM_parameters = {
    'C': 1,
    'kernel': 'rbf',
}

clf_01 = sklearn.svm.SVC(**SVM_parameters, probability=True)
clf_02 = sklearn.svm.SVC(**SVM_parameters, probability=True)
clf_12 = sklearn.svm.SVC(**SVM_parameters, probability=True)


# Training multiclass SVM

clf_01.fit(np.concatenate((X_train_Distortion_selected_normalized, X_train_NoFX_selected_normalized), axis=0),
           np.concatenate((y_train_Distortion, y_train_NoFX), axis=0))

clf_02.fit(np.concatenate((X_train_Distortion_selected_normalized, X_train_Tremolo_selected_normalized), axis=0),
           np.concatenate((y_train_Distortion, y_train_Tremolo), axis=0))

clf_12.fit(np.concatenate((X_train_NoFX_selected_normalized, X_train_Tremolo_selected_normalized), axis=0),
           np.concatenate((y_train_NoFX, y_train_Tremolo), axis=0))


# Fitting multiclass SVM
y_test_predicted_DN = clf_01.predict(X_test_normalized_selected).reshape(-1, 1)
y_test_predicted_DT = clf_02.predict(X_test_normalized_selected).reshape(-1, 1)
y_test_predicted_NT = clf_12.predict(X_test_normalized_selected).reshape(-1, 1)

y_test_predicted = np.concatenate((y_test_predicted_DN, y_test_predicted_DT, y_test_predicted_NT), axis=1)
y_test_predicted = np.array(y_test_predicted, dtype=np.int)

# SVM majority voting
y_test_predicted_mv = np.zeros((y_test_predicted.shape[0],))
for i, e in enumerate(y_test_predicted):
    y_test_predicted_mv[i] = np.bincount(e).argmax()


def compute_cm_multiclass(gt, predicted):
    classes = np.unique(gt)

    CM = np.zeros((len(classes), len(classes)))

    for i in np.arange(len(classes)):
        pred_class = predicted[gt == i]

        for j in np.arange(len(pred_class)):
            CM[i, int(pred_class[j])] = CM[i, int(pred_class[j])] + 1
    print(CM)


compute_cm_multiclass(y_test, y_test_predicted_mv)


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

#compute_metrics(y_test, y_test_predicted_mv)
logger.info(str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ' Finished :)')


