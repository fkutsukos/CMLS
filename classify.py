from numpy import loadtxt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix
import logging
import numpy as np
import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix

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

# enable this parameter to include polyphonic sound in the classification
classifyPoly = True

logger.info(
    str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ' Reading and concatenating feature files ... ')

X_Distortion = loadtxt('results/X_Distortion_mono.csv', delimiter=',')
X_NoFX = loadtxt('results/X_NoFX_mono.csv', delimiter=',')
X_Tremolo = loadtxt('results/X_Tremolo_mono.csv', delimiter=',')

if classifyPoly == True:
    logger.info(
        str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ' Polyphonic sounds included ... ')
    X_Distortion_poly = loadtxt('results/X_Distortion_poly.csv', delimiter=',')
    X_NoFX_poly = loadtxt('results/X_NoFX_poly.csv', delimiter=',')
    X_Tremolo_poly = loadtxt('results/X_Tremolo_poly.csv', delimiter=',')
    X_Distortion = np.concatenate((X_Distortion, X_Distortion_poly), axis=0)
    X_NoFX = np.concatenate((X_NoFX, X_NoFX_poly), axis=0)
    X_Tremolo = np.concatenate((X_Tremolo, X_Tremolo_poly), axis=0)

X = np.concatenate((X_Distortion, X_NoFX, X_Tremolo), axis=0)

# Build the Ground Truth vector for Mono
y_Distortion = np.zeros((X_Distortion.shape[0],))
y_NoFX = np.ones((X_NoFX.shape[0],))
y_Tremolo = np.ones((X_Tremolo.shape[0],)) * 2

y = np.concatenate((y_Distortion, y_NoFX, y_Tremolo), axis=0)

# Sample 3 training sets while holding out 40% of the data for testing (evaluating) our classifiers
logger.info(
    str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ' Splitting into random train and test subsets..')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, stratify=y)

pipeline = Pipeline([
    ('scale', StandardScaler()),
    ('variance_thresh', VarianceThreshold(threshold=(.8 * (1 - .8)))),
    ('select', SelectKBest(score_func=f_classif)),
    ('clf', SVC())]
)

param_grid = {'select__k': [10, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 418],
              'clf__C': [0.1, 1, 2, 5, 10, 100],
              'clf__kernel': ['rbf']}

logger.info(
    str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ' Computing Grid-search CV ... ')

grid_search = GridSearchCV(pipeline, param_grid, scoring='accuracy', cv=5, n_jobs=5)

logger.info(
    str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ' Fitting grid search with data train... ')

grid_search.fit(X_train, y_train)

logger.info(
    str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ' Grid Search Fitted!! ')

# computing scores
print("Best parameters set found on development set:")
print()
print(grid_search.best_params_)
print()
print("Grid scores on development set:")
print()
means = grid_search.cv_results_['mean_test_score']
stds = grid_search.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, grid_search.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))
print()

print("Detailed classification report:")
print()
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()
y_true, y_pred = y_test, grid_search.predict(X_test)
print(classification_report(y_true, y_pred))
print()

np.set_printoptions(precision=2)

class_names = ['Distortion', 'NoFX', 'Tremolo']
# Plot non-normalized confusion matrix
titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(grid_search, X_test, y_test,
                                 display_labels=class_names,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

    plt.show()
