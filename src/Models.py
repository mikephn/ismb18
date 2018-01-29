import os, json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.models import model_from_json
from src.tools import *
from src.Saliency import *
from src.Mutagenesis import *
from src.train import *

batchSizeForPrediction = 512

def modelPath(name):
    if name == "Meis RPKM":
        return "models/Meis-RPKM/"

    elif name == "Meis downbinding":
        return "models/Meis-down/"

    elif name == "Meis upbinding":
        return "models/Meis-up/"

    elif name == "Meis Hoxa2 cobinding":
        return "models/Acbg-Meis-Hoxa2/"
    else:
        print('Please specify model name and path in Models.py.')
        return None

def loadModel(name):
    if ensembleExists(modelPath(name)):
        return Ensemble(modelPath(name))
    else:
        return None

def trainModel(name, X, Y, params, regression):
    crossvalidate(modelPath(name), X, Y, params, regression=regression)
    print('Training complete.')
    return Ensemble(modelPath(name))

def loadDefaultModel(fromFolder, reshape=0):
    return loadSavedModel(fromFolder, "weights.hdf5", reshape=reshape)

def reshapeModel(json, width):
    import re
    format = r"\"batch_input_shape\": \[null, \d+,"
    replacement = "\"batch_input_shape\": [null, {},".format(width)
    return re.sub(format, replacement, json)

def modelExists(path):
    return (fileExists(path+'model.json') and fileExists(path+"weights.hdf5"))

def loadSavedModel(folder, weightsPath, reshape=0):
    jsonPath = folder + 'model.json'
    weightsPath = folder + weightsPath
    with open(jsonPath, 'r') as jsonFile:
        contents = json.loads(jsonFile.read())
        if reshape > 0:
            contents = reshapeModel(contents, reshape)
        model = model_from_json(contents)
        model.load_weights(weightsPath)
        return model

def ensembleExists(folder, length=5):
    return fileExists(folder+"f-{}/model.json".format(length-1))

class Ensemble:
    def __init__(self, folder, nFolds=5, reshape=0):
        self.nets = []
        self.reshapedTo = reshape
        for nf in range(nFolds):
            model = loadSavedModel(folder+"f-{}/".format(nf), "weights.hdf5", reshape=reshape)
            loadSaliencyFunctions(model)
            self.nets.append(model)

    def predict(self, X):
        Ys = []
        for n in range(len(self.nets)):
            model = self.nets[n]
            Y = model.predict(np.array(X), batch_size=batchSizeForPrediction, verbose=2)
            Ys.append(Y)
        return np.average(np.array(Ys), axis=0)

    def mutagenesis(self, X, target, width, step=1, salient=True, bs=2048, regression=False):
        results = []
        for ni, net in enumerate(self.nets):
            res = slidingWindowMutagenesis(X, target, net, width, step, salient, bs, regression)
            results.append(res)
        ensembleMean = np.mean(results, axis=0)
        return ensembleMean