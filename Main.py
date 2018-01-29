# Script to recreate the results from the paper:
# 'Differential and Cooperative Binding Feature Discovery with
# Convolutional Neural Networks'

# src/Data.py defines loading and caching the data from .bed files
# src/Search.py defines hyper-parameter optimisation
# src/Models.py implements model loading including defining paths to saved models
# src/Mutagenesis.py implements obtaining salient mutagenesis scores
# src/Visualize.py implements plot generation

from src.Visualize import *
from src.Data import *
from src.Models import *
from src.Search import *

# This function loads a trained model,
# train if the model is not saved,
# perform optimisation if hyper-parameters are not saved.
def model(experimentName, trX, trY, regression):
    model = loadModel(experimentName)
    if model is None:
        params = bestHyperparameters(experimentName)
        if params is None:
            print('Performing hyper-parameter search for experiment: {}'.format(experimentName))
            opTX, opVX = splitData(trX, 0.2)
            opTY, opVY = splitData(trY, 0.2)
            params = findHyperparameters(experimentName, opTX, opTY, opVX, opVY, regression, maxIterations=20)
        print('Training model for experiment: {}'.format(experimentName))
        model = trainModel(experimentName, trX, trY, params, regression=regression)
    return model

# Function to obtain salient mutagenesis scores
# plot multiple features given model, peak and features information:
def generatePlots(model, peak, features, folder, filePrefix, window, stride, regression=False):
    # each feature tuple is defined as (feature name, neuron number, color)
    peak.tracks = []
    for f_tuple in features:
        # obtain salient mutagenesis scores from ensemble for a given output
        # note that function can accept a list of inputs for batch processing
        scores = model.mutagenesis([peak.x], target=f_tuple[1], width=window, step=stride, regression=regression)[0]
        # add the resulting track to plot
        peak.tracks.append(PlotTrack(scores, f_tuple[0], f_tuple[2], width=1))
    saveTrackLine(peak, folder, filePrefix=filePrefix)

# Script to recreate the models and label peaks shown in the paper:
if __name__ == "__main__":

    # RPKM regression for synergistic features contributing to Meis occupancy.
    trX, trY, hX, hY, hP = meisRPKMData(testProp=0.2)
    meisRPKMModel = model("Meis RPKM", trX, trY, regression=True)
    predY = meisRPKMModel.predict(hX)
    printPearsonCorrelation(hY, predY)

    # Cooperative binding classification for separating co-bound Meis and Hoxa2.
    trX, trY, hX, hY, hP = CoopMeisHoxa2Data(nTestPerClass=500)
    meisHoxCoopModel = model("Meis Hoxa2 cobinding", trX, trY, regression=False)
    predY = meisHoxCoopModel.predict(hX)
    printConfusionStats(predY, hY)

    # Differential Meis classification for tissue-specific features.
    trX, trY, hX, hY, hP = meisClassLowData(testProp=0.2)
    meisDiffModel = model("Meis downbinding", trX, trY, regression=False)
    predY = meisDiffModel.predict(hX)
    printConfusionStats(predY, hY)

    trX, trY, hX, hY, hP = meisClassHighData(testProp=0.2)
    meisDiffModel = model("Meis upbinding", trX, trY, regression=False)
    predY = meisDiffModel.predict(hX)
    printConfusionStats(predY, hY)

    # Plot peaks presented in the paper:

    peak1 = Peak('chr6', 15378458, 15378758)
    peak2 = Peak('chr8', 26304170, 26306170)

    # (feature name, neuron number, color)
    features = [('Meis (BA1)', 0, '#2ac400'),
                ('Meis (BA2)', 1, '#005cf2'),
                ('Meis (PBA)', 2, '#7f01ad')]
    generatePlots(meisRPKMModel, peak1, features, 'plots/chr6/', 'Meis-RPKM-', window=50, stride=10, regression=True)
    generatePlots(meisRPKMModel, peak2, features, 'plots/chr8/', 'Meis-RPKM-', window=200, stride=50, regression=True)

    features = [('Meis only (BA2)', 1, '#005cf2'),
                ('Hoxa2 and Meis (BA2)', 2, '#ff0c0c')]
    generatePlots(meisHoxCoopModel, peak1, features, 'plots/chr6/', 'Coop-', window=50, stride=10)
    generatePlots(meisHoxCoopModel, peak2, features, 'plots/chr8/', 'Coop-', window=200, stride=50)

    features = [('Meis downbinding (BA1)', 0, '#ff0c0c'),
                ('Meis downbinding (BA2)', 1, '#01ad31'),
                ('Meis downbinding (PBA)', 2, '#e57302')]
    generatePlots(meisDiffModel, peak1, features, 'plots/chr6/', 'Diff-', window=50, stride=10)
    generatePlots(meisDiffModel, peak2, features, 'plots/chr8/', 'Diff-', window=200, stride=50)

