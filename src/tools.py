import sys, os.path, shutil
import numpy as np
import scipy.stats

# helper functions for moving files:
def ensureFolder(relativePath):
    if not os.path.exists(relativePath):
        os.makedirs(relativePath)
    return relativePath

def moveFolder(source, destination):
    shutil.move(source, destination)

def overwritePath(source, target):
    if os.path.exists(target):
        shutil.rmtree(target)
    shutil.move(source, target)

def clearFolder(path):
    shutil.rmtree(path)

def fileExists(path):
    return os.path.exists(path)

# functions to shuffle and save the shuffle for replicability:
def shuffledArray(arr, save=None):
    if not save or not fileExists(save):
        sieve = np.random.permutation(len(arr))
        if save:
            np.save(save, sieve)
    else:
        sieve = np.load(save)

    return arr[sieve]

def shuffledUnison(a, b, save=None):
    assert len(a) == len(b)
    if not save or not fileExists(save):
        sieve = np.random.permutation(len(a))
        if save:
            np.save(save, sieve)
    else:
        sieve = np.load(save)
    return a[sieve], b[sieve]

# functions to show accuracy metrics:
def printConfusionStats(Y, Yorig):
    matrix = calculateConfusionMatrix(Y, Yorig)
    np.set_printoptions(precision=6, suppress=True)
    precision, recall = precisionAndRecall(matrix)
    totalCorrect = np.sum(np.array(np.diag(matrix), dtype=float))
    totalEntries = np.sum(matrix)

    print("Predicted\Real\n{}".format(matrix))
    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))
    print("Accuracy: {}".format(totalCorrect/totalEntries))

def precisionAndRecall(matrix):
    totalTrue = np.sum(matrix, axis=0)
    totalPredicted = np.sum(matrix, axis=1)
    tp = np.array(np.diag(matrix), dtype=float)

    precision = np.divide(tp, totalPredicted)
    recall = np.divide(tp, totalTrue)
    return precision, recall

def calculateConfusionMatrix(Y, Yorig):
    nclasses = len(Y[0])
    matrix = np.zeros((nclasses, nclasses), dtype=int)
    for n in range(len(Y)):
        originalClass = np.argmax((Yorig[n]))
        predictedClass = np.argmax((Y[n]))
        matrix[predictedClass, originalClass] += 1
    return matrix

def pearsonCorrelation(gt, pred):
    return np.squeeze(scipy.stats.pearsonr(gt, pred))[0]

def printPearsonCorrelation(gt, pred):
    for cn in range(len(gt[0])):
        print('Class {}, Pearson r: {}'.format(cn, pearsonCorrelation(gt[:, cn], pred[:, cn])))