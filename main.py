# Testing 1NN, Logistic Regression, Decision Trees on Iris dataset
# Python code by Corban Kenyon (adapted from Dr. Kursun's Google Collaborate Examples)
# Project 1 AI with Dr. Kursun
# Project groupmates Tiffany Podlogar and Nathan Van Aalsburg

import numpy as np

np.set_printoptions(precision=2)

from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import classification_report


# dataset retrieval and random seeding
iris = datasets.load_iris()
X = iris.data
y = iris.target
randomStates = [1, 11, 21, 31, 41]

# results storage since we are checking performance over multiple dataset train/test random splits
masterResults = {'kNN': {'acc (f1)': 0,
                         'lr_TP': 0,
                         'lr_FN': 0,
                         'dt_TP': 0,
                         'dt_FN': 0},
                 'LR': {'acc (f1)': 0,
                        'kNN_TP': 0,
                        'kNN_FN': 0,
                        'dt_TP': 0,
                        'dt_FN': 0},
                 'DT': {'acc (f1)': 0,
                        'lr_TP': 0,
                        'lr_FN': 0,
                        'kNN_TP': 0,
                        'kNN_FN': 0}}


def testClassifiers(state):
    # split our dataset into test/train partitions
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.5, random_state=state, stratify=y)

    # --kNN classification model--
    clf_kNN = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
    clf_kNN.fit(X_train, y_train)
    predictions_kNN = clf_kNN.predict(X_test)
    masterResults['kNN']['acc (f1)'] += f1_score(predictions_kNN, y_test, average='micro')

    # --Logistic Regression modeling--
    clf_logReg = LogisticRegression(random_state=state)
    clf_logReg.fit(X_train, y_train)
    predictions_LR = clf_logReg.predict(X_test)
    masterResults['LR']['acc (f1)'] += f1_score(predictions_LR, y_test, average='micro')

    # --Decision Tree modeling--
    clf_dTree = DecisionTreeClassifier(random_state=state)
    clf_dTree.fit(X_train, y_train)
    predictions_DT = clf_dTree.predict(X_test)
    masterResults['DT']['acc (f1)'] += f1_score(predictions_DT, y_test, average='micro')

    # kNN and LR comparisons
    overlapTP = np.mean(np.logical_and(predictions_kNN == predictions_LR, predictions_kNN == y_test))
    masterResults['kNN']['lr_TP'] += overlapTP
    masterResults['LR']['kNN_TP'] += overlapTP
    overlapFN = np.mean(np.logical_and(predictions_kNN == predictions_LR, predictions_kNN != y_test))
    masterResults['kNN']['lr_FN'] += overlapFN
    masterResults['LR']['kNN_FN'] += overlapFN

    # DT and LR comparisons
    overlapTP = np.mean(np.logical_and(predictions_DT == predictions_LR, predictions_DT == y_test))
    masterResults['DT']['lr_TP'] += overlapTP
    masterResults['LR']['dt_TP'] += overlapTP
    overlapFN = np.mean(np.logical_and(predictions_DT == predictions_LR, predictions_DT != y_test))
    masterResults['DT']['lr_FN'] += overlapFN
    masterResults['LR']['dt_FN'] += overlapFN

    # kNN and DT comparisons
    overlapTP = np.mean(np.logical_and(predictions_kNN == predictions_DT, predictions_kNN == y_test))
    masterResults['kNN']['dt_TP'] += overlapTP
    masterResults['DT']['kNN_TP'] += overlapTP
    overlapFN = np.mean(np.logical_and(predictions_kNN == predictions_DT, predictions_kNN != y_test))
    masterResults['kNN']['dt_FN'] += overlapFN
    masterResults['DT']['kNN_FN'] += overlapFN

    return None


# reclassify models over several randomState seeds
for state in randomStates:
    testClassifiers(state)

# show template for how output is formatted in console
print("--Classifier--\naccuracy (f1 score): <value>\n<other classifier>_TruePositiveOverlap: <value>\n<other "
      "classifier>_FalseNegativeOverlap: <value>\n<other classifier>_TruePositiveOverlap: <value>\n<other "
      "classifier>_FalseNegativeOverlap: <value>")

# average our results and output them
for model, attributes in masterResults.items():
    print("\n--" + model + "--")
    for attribute, discard in attributes.items():

        # average all values over the amount of randomState samples
        masterResults[model][attribute] = round((masterResults[model][attribute] / len(randomStates)), 3)
        # output our performances for each metric
        print(attribute + ": " + str(masterResults[model][attribute]))

# The below performance was a result 5 iterations of dataset split with seeds 1, 11, 21, 31, 41
# accuracies (f1) for each classifier ranged between a 95% and 97% average
# logistic regression performed the best with the highest f1 score, the highest True Positive classifier overlap...
    # ... and least False Negative classifier overlap
