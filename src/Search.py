import pickle
from src.optimise import optimise
from src.train import crossvalidate
from src.Models import Ensemble
from src.tools import *
from sklearn.metrics import f1_score, mean_squared_error

param_ranges = dict([
        ('batchSize', range(4, 10)),
        ('dense', range(0, 128, 32)),
        ('dropout', [x / 10.0 for x in range(0, 10)]),
        ('filters', range(32, 1024, 32)),
        ('length', range(6, 24, 1))
    ])

def completeParameters(params):
    params["epochs"] = 300
    params["stopPatience"] = 10
    params["batchSize"] = int(2 ** params["batchSize"])
    params["filters"] = int(params["filters"])
    params["length"] = int(params["length"])
    params["dense"] = int(params["dense"])
    return params

def hyperPathForName(name):
    return 'hyper/'+name+'.pkl'

def bestHyperparameters(name):
    def toDict(list):
        keys = sorted(param_ranges.keys())
        params = dict()
        for i in range(len(list)):
            params[keys[i]] = list[i]
        return params
    path = hyperPathForName(name)
    if fileExists(path):
        results = pickle.load(open(path, "rb"))
        params = results[0]
        scores = results[1]
        order = np.argsort(np.squeeze(scores))
        return completeParameters(toDict(params[order[0]]))
    else:
        return None

def classificationScore(trueY, predY):
    def oneHotToCategorical(array):
        categorical = np.zeros(len(array))
        for ind, n in enumerate(array):
            categorical[ind] = np.argmax(array[ind, :])
        return categorical
    return 1.0 - f1_score(oneHotToCategorical(trueY), oneHotToCategorical(predY), average='weighted')

def regressionScore(trueY, predY):
    return mean_squared_error(trueY, predY)

def findHyperparameters(name, X, Y, vX, vY, regression, maxIterations=20):

    tempFolder = ensureFolder('hyper/temp/')

    def train(**params):
        clearFolder(tempFolder)
        crossvalidate(tempFolder, X, Y, completeParameters(params), regression=regression, verbose=False)
        ensemble = Ensemble(tempFolder)
        predY = ensemble.predictMeanInPlace(vX)
        if regression:
            return regressionScore(vY, predY)
        else:
            return classificationScore(vY, predY)

    path = hyperPathForName(name)
    if fileExists(path):
        results = pickle.load(open(path, "rb"))
        startFrom = len(results[0])+1
    else:
        results = None
        startFrom = 1

    for iter in range(startFrom, maxIterations+1):
        X_opt, Y_opt, T_eval, T_opt, failed_opt = optimise(train, param_ranges, n_initial=10,
                                                           max_iter=iter, previous_results=results)

        results = (X_opt, Y_opt, T_eval, T_opt, failed_opt)
        pickle.dump(results, open(path, "wb"))

    return bestHyperparameters(name)

