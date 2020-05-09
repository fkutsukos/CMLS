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

X_Distortion = loadtxt('results/X_Distortion.csv', delimiter=',')
X_NoFX = loadtxt('results/X_NoFX.csv', delimiter=',')
X_Tremolo = loadtxt('results/X_Tremolo.csv', delimiter=',')

X = np.concatenate((X_Distortion, X_NoFX, X_Tremolo), axis=0)


# Build the Ground Truth vector for Training
y_Distortion = np.zeros((X_Distortion.shape[0],))
y_NoFX = np.ones((X_NoFX.shape[0],))
y_Tremolo = np.ones((X_Tremolo.shape[0],)) * 2

y = np.concatenate((y_Distortion, y_NoFX, y_Tremolo), axis=0)

# Sample 3 training sets while holding out 40% of the data for testing (evaluating) our classifiers
logger.info(
    str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ' Splitting into random train and test subsets..')

X_train , X_test, y_train, y_test= train_test_split(
    X, y, test_size=0.4, stratify=y)

pipeline = Pipeline([
    ('scale', StandardScaler()),
    ('variance_thresh', VarianceThreshold(threshold=(.8 * (1 - .8)))),
    ('select', SelectKBest(score_func=f_classif)),
    ('clf', SVC())]
)

param_grid = {'select__k': [1, 5],
              'clf__C': [1],
              'clf__kernel': ['rbf' , 'linear']}

grid_search = GridSearchCV(pipeline, param_grid, scoring='accuracy', cv=5, n_jobs=5)
grid_search.fit(X_train, y_train)

print(grid_search.best_params_)

np.set_printoptions(precision=2)

class_names = ['Distortion', 'NoFX','Tremolo']
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
