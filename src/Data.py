from src.Peaks import *
from src.tools import *
import numpy as np

# location og bed files:
dataFolder = ensureFolder("data/")
# folder for storing expanded data and shuffle permutations:
cacheDataFolder = ensureFolder("data/cache/")

# functions defining data loading and caching:

def meisRPKMData(testProp=0.2):
    peaks = BedFile(dataFolder+'Meis_RPKM.bed', ignoreFirstLine=True, ensureWidth=1000).peaks
    for p in peaks:
        p.value = [p.value[1], p.value[5], p.value[8]] # select values from BA1 rep1, BA2 rep2, and PBA rep2

    genome = mm10Genome()
    cacheAndLoadPeaks(genome, peaks, 'meisRPKM')
    peaks = shuffledArray(peaks, cacheDataFolder+'meisRPKM-shuffle.npy')
    tP, hP = splitData(peaks, testProp)
    trX, trY = arraysFromPeaks(tP)
    hX, hY = arraysFromPeaks(hP)
    return trX, trY, hX, hY, hP


def meisClassHighData(testProp=0.2):
    ba1hi = BedFile(dataFolder+'Meis-classes/meis-ba1-hi.bed', ignoreFirstLine=True)
    ba1hi.setAllValues([1, 0, 0, 0])
    ba2hi = BedFile(dataFolder+'Meis-classes/meis-ba2-hi.bed', ignoreFirstLine=True)
    ba2hi.setAllValues([0, 1, 0, 0])
    pbahi = BedFile(dataFolder+'Meis-classes/meis-pba-hi.bed', ignoreFirstLine=True)
    pbahi.setAllValues([0, 0, 1, 0])
    background = BedFile(dataFolder+'Meis-classes/meis-bg.bed', ignoreFirstLine=True)
    background.setAllValues([0, 0, 0, 1])
    bg = shuffledArray(background.peaks)[:20000]

    allPeaks = np.concatenate((ba1hi.peaks, ba2hi.peaks, pbahi.peaks, bg))
    reshapePeaks(allPeaks, 200)
    genome = mm10Genome()
    cacheAndLoadPeaks(genome, allPeaks, "meis-high")

    peaks = shuffledArray(allPeaks, cacheDataFolder+'meisHi-shuffle.npy')
    tP, hP = splitData(peaks, testProp)
    trX, trY = arraysFromPeaks(tP)
    hX, hY = arraysFromPeaks(hP)
    return trX, trY, hX, hY, hP

def meisClassLowData(testProp=0.2):

    ba1low = BedFile(dataFolder+'Meis-classes/meis-ba1-low.bed', ignoreFirstLine=True)
    ba1low.setAllValues([1, 0, 0, 0])
    ba2low = BedFile(dataFolder+'Meis-classes/meis-ba2-low.bed', ignoreFirstLine=True)
    ba2low.setAllValues([0, 1, 0, 0])
    pbalow = BedFile(dataFolder+'Meis-classes/meis-pba-low.bed', ignoreFirstLine=True)
    pbalow.setAllValues([0, 0, 1, 0])
    background = BedFile(dataFolder+'Meis-classes/meis-bg.bed', ignoreFirstLine=True)
    background.setAllValues([0, 0, 0, 1])
    bg = shuffledArray(background.peaks)[:20000]

    allPeaks = np.concatenate((ba1low.peaks, ba2low.peaks, pbalow.peaks, bg))
    reshapePeaks(allPeaks, 200)
    genome = mm10Genome()
    cacheAndLoadPeaks(genome, allPeaks, "meis-low")

    peaks = shuffledArray(allPeaks, cacheDataFolder+'meisLow-shuffle.npy')
    tP, hP = splitData(peaks, testProp)
    trX, trY = arraysFromPeaks(tP)
    hX, hY = arraysFromPeaks(hP)

    return trX, trY, hX, hY, hP


def backgroundPeaks(name, width, genome):
    fileNames = {"ac": "Diffbind_H3K27ac_nonMeis_regions.bed",
                 "condor": "NonMeisHox_CONDOR_CNEs_regions.bed",
                 "ctcfdb": "NonMeisHox_CTCFDB_DB_regions_reduced.bed",
                 "ucne": "NonMeisHox_UCNE_orthologs_regions.bed"}
    fileName = dataFolder+"Background-sets/" + fileNames[name]

    bgPeaks = BedFile(fileName, ignoreFirstLine=False).peaks
    sufficientWidth = [p for p in bgPeaks if p.width >= width]
    reshapePeaks(sufficientWidth, width)
    cacheAndLoadPeaks(genome, sufficientWidth, "bg-"+name)
    return np.array(sufficientWidth)


def CoopMeisHoxa2Data(width=200, limitExPerClass=5000, nTestPerClass=100, bgtype="ac", bgprop=1.5):
    meisFile = BedFile(dataFolder+'Meis_Hoxa2_FE.bed', ignoreFirstLine=True)
    meisPeaks = []
    for p in meisFile.peaks:
        if p.value[2] == 1 and p.value[6] == 0:  # select peaks with Meis in BA2 without Hoxa2
            p.enrichment = p.value[3]  # save Meis enrichment
            p.value = [0, 1, 0]  # set label for classification
            meisPeaks.append(p)

    hoxFile = BedFile(dataFolder+'Hoxa2_FE.bed', ignoreFirstLine=True)
    hoxMeisPeaks = []
    for p in hoxFile.peaks:
        if p.value[1] == 1:  # select Hoxa2 peaks intersecting Meis
            p.enrichment = p.value[0]  # save Meis enrichment
            p.value = [0, 0, 1]  # set label for classification
            hoxMeisPeaks.append(p)

    meisPeaks = sortByEnrichment(meisPeaks)[:limitExPerClass]
    hoxMeisPeaks = sortByEnrichment(hoxMeisPeaks)[:limitExPerClass]

    allPeaks = np.concatenate((meisPeaks, hoxMeisPeaks))
    reshapePeaks(allPeaks, 200)
    genome = mm10Genome()
    cacheAndLoadPeaks(genome, allPeaks, "hox-meis-ba2")

    # hold out every other high FE peak for testing
    mT, mH = holdOutAlternating(meisPeaks, nTestPerClass)
    hT, hH = holdOutAlternating(hoxMeisPeaks, nTestPerClass)

    trainPeaks = []
    holdOutPeaks = []
    trainPeaks.extend(mT)
    trainPeaks.extend(hT)
    holdOutPeaks.extend(mH)
    holdOutPeaks.extend(hH)

    if bgtype is "shuffle":
        xToShuffle = np.array([peak.x for peak in allPeaks])
        bgX = dishuffledNegatives(shuffledArray(xToShuffle), 'ba2-shuffled')[:int(limitExPerClass*bgprop)]
    else:
        bgPeaks = backgroundPeaks(bgtype, width, genome)
        bgPeaks = shuffledArray(bgPeaks)
        bgX = [p.x for p in bgPeaks][:int(limitExPerClass*bgprop)]

    tX, tY = arraysFromPeaks(trainPeaks)
    hX, hY = arraysFromPeaks(holdOutPeaks)

    bgLabel = [1, 0, 0]
    bgH = bgX[:nTestPerClass]
    bgT = bgX[nTestPerClass:]

    hX = np.concatenate((hX, bgH))
    hY = np.concatenate((hY, [bgLabel] * len(bgH)))

    tX = np.concatenate((tX, bgT))
    tY = np.concatenate((tY, [bgLabel] * len(bgT)))

    tX, tY = shuffledUnison(tX, tY, cacheDataFolder+'coop-train-shuffle.npy')

    return tX, tY, hX, hY, holdOutPeaks

# for generating di-shuffled nucleotide background set:
def dishuffled(X):
    exn = len(X)
    exl = len(X[0])
    exc = len(X[0][0])

    flatX = np.reshape(X, (exn*exl, exc))
    dinucCount = int(math.floor(len(flatX) / 2))
    dinucleotides = np.reshape(flatX[:dinucCount*2], (dinucCount, 2, exc))
    np.random.shuffle(dinucleotides)
    flatSh = np.reshape(dinucleotides, (dinucCount*2, exc))
    if len(flatSh) < len(flatX):
        flatSh = np.append((flatSh, flatX[-1]))
    shuffledSeqs = np.reshape(flatSh, X.shape)
    return shuffledSeqs


def dishuffledNegatives(X, name, type=bool):
    filePath = cacheDataFolder + name + ".npy"
    if fileExists(filePath):
        return np.load(filePath)
    else:
        print("Generating negative samples: " + str(len(X)))
        nX = dishuffled(X)
        np.save(filePath, np.array(nX, dtype=type))
        return nX

# for splitting data for testing:
def splitData(input, testProp):
    testCount = int(round(len(input) * testProp))
    trainCount = len(input)-testCount
    return input[:trainCount], input[trainCount:]


def holdOutAlternating(X, howMany):
    tX = []
    hX = []
    taken = 0
    for exn, ex in enumerate(X):
        if exn % 2 and taken < howMany:
            hX.append(ex)
            taken += 1
        else:
            tX.append(ex)
    return np.array(tX), np.array(hX)

# caching sequences from the genome:
def saveTrainingFile(genomeTwoBit, peaks, name):
    width = int(peaks[0].width)
    filePath = cacheDataFolder + name + "-{}.npy".format(width)
    if fileExists(filePath):
        print("File exists: {}".format(filePath))
        return filePath
    sequences = []
    for peak in peaks:
        chr = peak.chromosome
        sequence = genomeTwoBit[chr][peak.start - 1:peak.end - 1].lower()
        if len(sequence):
            sequences.append(oneHotVector(sequence))
        else:
            print("No sequence for: " + chr + " " + str(peak.summit))
    np.save(filePath, np.array(sequences, dtype=bool))
    print("Saved {}: {} examples.".format(filePath, len(sequences)))
    return filePath


def cacheAndLoadPeaks(genome, peaks, name):
    X = np.load(saveTrainingFile(genome, peaks, name))
    for ind, p in enumerate(peaks):
        p.x = X[ind]


class BedFile:
    def __init__(self, path, ignoreFirstLine=True, ensureWidth=0):
        self.peaks = []
        self.widMax = 0
        self.widAvg = 0
        self.ensWid = ensureWidth
        self.dropped = 0
        file = open(path, 'r')
        lines = file.readlines()
        if ignoreFirstLine:
            lines = lines[1:]

        for l in lines:
            self.peaksFromLine(l)

        self.peaks = np.array(self.peaks)

        self.widAvg = self.widAvg / (len(self.peaks))
        print("Peaks:" + str(len(self.peaks)) + " avg width: " + str(self.widAvg) + " max: " + str(
            self.widMax) + " dropped: " + str(self.dropped))

    def peaksFromLine(self, line):
        peak = Peak()
        comps = line.split()
        peak.chromosome = comps[0]
        peak.start = int(comps[1])
        peak.end = int(comps[2])
        peak.width = peak.end - peak.start
        if peak.width > self.widMax:
            self.widMax = peak.width
        self.widAvg += peak.width
        peak.value = np.array(comps[3:], dtype=float)
        if self.ensWid > 0:
            if peak.width == self.ensWid:
                self.peaks.append(peak)
            else:
                self.dropped += 1
        else:
            self.peaks.append(peak)

    def setAllValues(self, value):
        for p in self.peaks:
            p.value = value
