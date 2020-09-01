# Testing 1NN, Logistic Regression, Decision Trees on Iris dataset
# Python code by Corban Kenyon (adapted from Dr. Kursun's Google Collaborate Examples)
# Project 1 AI with Dr. Kursun
# Project groupmates Tiffany Podlogar and Nathan Van Aalsburg

import numpy as np
np.set_printoptions(precision=2)

from sklearn import datasets
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


# dataset retrieval and random seeding
iris = datasets.load_iris()
X = iris.data
y = iris.target
randomStates = list(range(1, 100))

# results storage since we are checking performance over multiple dataset train/test random splits
masterResults = {'kNN': {'acc': 0,
                         'f1': 0,
                         'LR_kjS': 0,
                         'LR_TP': 0,
                         'LR_FN': 0,
                         'DT_kjS': 0,
                         'DT_TP': 0,
                         'DT_FN': 0},
                 'LR': {'acc': 0,
                        'f1': 0,
                        'kNN_kjS': 0,
                        'kNN_TP': 0,
                        'kNN_FN': 0,
                        'DT_kjS': 0,
                        'DT_TP': 0,
                        'DT_FN': 0},
                 'DT': {'acc': 0,
                        'f1': 0,
                        'LR_kjS': 0,
                        'LR_TP': 0,
                        'LR_FN': 0,
                        'kNN_kjS': 0,
                        'kNN_TP': 0,
                        'kNN_FN': 0}}


def testClassifiers(state):
    # split our dataset into test/train partitions
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.5, random_state=state, stratify=y)

    # --kNN classification model--
    clf_kNN = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
    clf_kNN.fit(X_train, y_train)
    predictions_kNN = clf_kNN.predict(X_test)
    # tracking accuracy and f1 score
    masterResults['kNN']['acc'] += np.mean(predictions_kNN == y_test)
    masterResults['kNN']['f1'] += f1_score(predictions_kNN, y_test, average='weighted')  # using micro causes f1=acc

    # --Logistic Regression modeling--
    # doubled default max iterations to avoid convergence issues on some seeds
    clf_logReg = LogisticRegression(random_state=state, max_iter=200)
    clf_logReg.fit(X_train, y_train)
    predictions_LR = clf_logReg.predict(X_test)
    masterResults['LR']['acc'] += np.mean(predictions_LR == y_test)
    masterResults['LR']['f1'] += f1_score(predictions_LR, y_test, average='weighted')

    # --Decision Tree modeling--
    clf_dTree = DecisionTreeClassifier(random_state=state)  # max features not used given amount of features in dataset
    clf_dTree.fit(X_train, y_train)
    predictions_DT = clf_dTree.predict(X_test)
    masterResults['DT']['acc'] += np.mean(predictions_DT == y_test)
    masterResults['DT']['f1'] += f1_score(predictions_DT, y_test, average='weighted')

    # kNN and LR comparisons
    compareClassifiers(predictions_kNN, 'kNN', predictions_LR, 'LR', y_test)
    # DT and LR comparisons
    compareClassifiers(predictions_DT, 'DT', predictions_LR, 'LR', y_test)
    # kNN and DT comparisons
    compareClassifiers(predictions_kNN, 'kNN', predictions_DT, 'DT', y_test)

    return None


def compareClassifiers(pred1, classifier1, pred2, classifier2, y_test):
    # comparing classifiers will use a different calculation for mis-classifications than used for DT python from class
    # these models have greater accuracy, less randomness, less radical predictions meaning that in the 100 iterations
    # ... 2 models oddly only ever selected the identical incorrect response when both mis-classifying

    # Kursun/Jaccard similarity
    both_correct = sum(np.logical_and(pred1 == y_test, pred2 == y_test))
    at_least_one_correct = sum(np.logical_or(pred1 == y_test, pred2 == y_test))
    kjSimilarity = both_correct / at_least_one_correct
    # above is equivalent to: (overlapTP / (1 - overlapFN)) since there are no disagreements on mis-classification
    masterResults[classifier1][(classifier2 + '_kjS')] += kjSimilarity
    masterResults[classifier2][(classifier1 + '_kjS')] += kjSimilarity

    # how often were these classifiers both correct?
    overlapTP = np.mean(np.logical_and(pred1 == pred2, pred1 == y_test))
    masterResults[classifier1][(classifier2 + '_TP')] += overlapTP
    masterResults[classifier2][(classifier1 + '_TP')] += overlapTP

    # how often were the classifiers both incorrect (classifiers agreed on output in all cases of mis-classification)
    overlapFN = np.mean(np.logical_and(pred1 == pred2, pred1 != y_test))
    masterResults[classifier1][(classifier2 + '_FN')] += overlapFN
    masterResults[classifier2][(classifier1 + '_FN')] += overlapFN

    return None


# retrain models over several randomState seeds
for state in randomStates:
    testClassifiers(state)


# show template for how output is formatted in console
print("--Classifier--\n"
      "accuracy: <value>\n"
      "f1 score: <value>\n"
      "<other classifier1>_Jaccard Similarity: <value>\n"
      "<other classifier1>_True Positive Overlap: <value>\n"
      "<other classifier1>_False Negative Overlap: <value>\n"
      "<other classifier2>_Jaccard Similarity: <value>\n"
      "<other classifier2>_True Positive Overlap: <value>\n"
      "<other classifier2>_False Negative Overlap: <value>")


# average our results and output them
for model, attributes in masterResults.items():
    print("\n--" + model + "--")
    for attribute, discard in attributes.items():
        # average all values over the amount of randomState samples
        masterResults[model][attribute] = round((masterResults[model][attribute] / len(randomStates)), 4)
        # output our performances for each metric
        print(attribute + ": " + str(masterResults[model][attribute]))

# The below performance was a result 100 iterations of dataset split with seeds 1:100
# Average accuracies for each classifier ranged between 94.4% and 96%
# Logistic regression performed the best with the highest accuracy as well as greatest overlap between classifiers
# of note is that 1NN had the least agreement when mis-classifying

# f1 scores for each classifier were minimally different from accuracies <= .02%
