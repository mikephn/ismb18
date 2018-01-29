import copy, math
import numpy as np
from src.Saliency import saliencyMap

# functions to create and merge sliding window sub-sequences:
def splitSequence(x, width, stride=1):
    count = int(math.floor((len(x) - width) / stride) + 1)
    if count <1:
        print("Region shorter than window.")
    if count == 1:
        return [x]
    else:
        subs = []
        for origin in range(count):
            subs.append(x[origin*stride:origin*stride + width])
        return subs

def mergeSlidingSplit(origLength, subScores, width, step):
    ntScores = np.zeros(origLength)
    count = np.zeros(origLength)
    for ind, res in enumerate(subScores):
        ntScores[ind * step:ind * step + width] += res
        count[ind * step:ind * step + width] += 1
    return ntScores/count

# functions to create and combine scores of nucleotide mutants:
def expandNucleotide(x, loc):
    examples = []
    for nbase in range(len(x[loc]) - 1):
        mutated = copy.copy(x)
        orig = mutated[loc, :]
        mutated[loc, :] = np.roll(orig, nbase + 1)
        examples.append(mutated)
    return examples

def combineMutationScores(mutScores):
    nMutPerSub = 3
    scores = []
    for ind in range(len(mutScores)/nMutPerSub):
        subMutScores = np.mean(mutScores[ind * nMutPerSub:(ind + 1) * nMutPerSub])
        scores.append(subMutScores)
    return np.array(scores)

# implementation of a sliding window mutagenesis (can be equally used with full region width)
def slidingWindowMutagenesis(X, target, model, width, step, salient=True, bs=32, regression=False):
    nX = []
    origLen = 0
    subCount = 0
    for x in X:
        subs = splitSequence(x, width, step)
        nX.extend(subs)
        if origLen == 0:
            origLen = len(x)
            subCount = len(subs)
    if salient:
        scores = positiveMutagenesis(nX, target, model, bs, regression)
    else:
        scores = fullMutagenesis(nX, target, model, bs)
    mergedScores = []
    for ind, x in enumerate(X):
        subScores = scores[ind*subCount:(ind+1)*subCount]
        merged = mergeSlidingSplit(origLen, subScores, width, step)
        mergedScores.append(merged)
    return np.array(mergedScores)

# below function is used when slidingWindowMutagenesis is called with salient=True:
def positiveMutagenesis(X, target, model, bs=32, regression=False):
    saliencies = []
    for x in X:
        sal = saliencyMap(model, x, n_out=target)
        itg = np.sum(x * sal, axis=1)
        saliencies.append(itg)
    saliencies = np.array(saliencies)

    mutMap = saliencies > 0
    mutants = []
    for pn in range(len(mutMap)):
        for loc in range(len(mutMap[pn])):
            if mutMap[pn][loc]:
                mutants.extend(expandNucleotide(X[pn], loc))

    stacked = np.concatenate((X, mutants))
    predMut = model.predict(stacked, batch_size=bs, verbose=0)[:, target]
    origScores = predMut[:len(X)]
    mutScores = combineMutationScores(predMut[len(X):])

    smScores = saliencies.clip(0)
    mutInd = 0
    for pn in range(len(mutMap)):
        origScore = origScores[pn]
        for loc in range(len(mutMap[pn])):
            if mutMap[pn][loc]:
                mutScore = origScore - mutScores[mutInd]
                smScores[pn][loc] *= mutScore
                mutInd += 1
        if not regression:
            # weigh by prediction confidence
            smScores[pn] *= origScores[pn]

    return np.array(smScores)

# below function is used when slidingWindowMutagenesis is called with salient=False
# (resulting in saturated mutagenesis):
def fullMutagenesis(X, target, model, bs=32):
    mutants = []
    for pn in range(len(X)):
        for loc in range(len(X[pn])):
            mutants.extend(expandNucleotide(X[pn], loc))

    stacked = np.concatenate((X, mutants))
    predMut = model.predict(stacked, batch_size=bs, verbose=0)[:, target]
    origScores = predMut[:len(X)]
    mutScores = combineMutationScores(predMut[len(X):])

    scores = np.zeros((len(X), len(X[0])))
    mutInd = 0
    for pn in range(len(X)):
        origScore = origScores[pn]
        for loc in range(len(X[pn])):
            scores[pn][loc] = origScore - mutScores[mutInd]
            mutInd += 1
    return scores