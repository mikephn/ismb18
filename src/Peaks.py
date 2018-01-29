import numpy as np
from twobitreader import *
import os.path
import shutil, math

# load and cache mm10 for .bed processing:
def mm10Genome():
    mm10source = "http://hgdownload-test.cse.ucsc.edu/goldenPath/mm10/bigZips/mm10.2bit"
    mm10Path = "mm10.2bit"
    if not os.path.exists(mm10Path):
        print("Downloading mm10...")
        try:
            import urllib
            mm10 = urllib.URLopener()
            mm10.retrieve(mm10source, mm10Path)
        except AttributeError:
            import urllib.request
            with urllib.request.urlopen(mm10source) as response, open(mm10Path, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)


    genome = TwoBitFile(mm10Path)
    print("Loaded mm10.")
    return genome

mm10 = mm10Genome()

oneHotNucleotides = {"n": [0,0,0,0],
                    "c": [1,0,0,0],
                    "g": [0,1,0,0],
                    "t": [0,0,1,0],
                    "a": [0,0,0,1]}

def oneHotVector(string):
    list = []
    for n in string:
        list.append(oneHotNucleotides[n])
    return np.array(list, dtype=bool)

def oneHotToString(array):
    string = ""
    for n in array:
        if n[0] == 1:
            string = string + 'c'
        elif n[1] == 1:
            string = string + 'g'
        elif n[2] == 1:
            string = string + 't'
        elif n[3] == 1:
            string = string + 'a'
        else:
            string = string + 'n'
    return string

class Peak:
    def __init__(self, chr=None, start=None, end=None):
        if chr:
            self.chromosome = chr
            self.start = start
            self.end = end
            self.width = self.end - self.start
            self.load(mm10)
        self.tracks = []

    def __str__(self):
        return self.chromosome + " w: " + str(self.width) + " start: " + str(self.start) + " end: " + str(self.end) \
               + "{value}".format(value=" value: {}".format(self.value) if hasattr(self, 'value') else "") \
               + "{enrichment}".format(enrichment=" enrichment: {}".format(self.enrichment) if hasattr(self, 'enrichment') else "")
    def load(self, genome):
        self.sequence = genome[self.chromosome][self.start - 1:self.end - 1].lower()
        self.x = oneHotVector(self.sequence)

    def subpeak(self, start, end):
        peak = Peak()
        peak.chromosome = self.chromosome
        peak.start = self.start+start
        peak.end = peak.start + end - start
        peak.width = peak.end - peak.start
        if hasattr(self, 'x'):
            peak.x = self.x[start:end]
        if hasattr(self, 'sequence'):
            peak.sequence = self.sequence[start:end]
        return peak

class classValueMask:
    def __init__(self, required, forbidden):
        self.required = required
        self.forbidden = forbidden

def filterPeaks(peaks, classMask):
    filtered = []
    requiredLen = np.sum(classMask.required)
    for peak in peaks:
        required = np.dot(peak.value, classMask.required)
        forbidden = np.dot(peak.value, classMask.forbidden)
        if required == requiredLen and forbidden == 0:
            filtered.append(peak)
    return filtered

def reshapePeaks(peaks, width):
    for peak in peaks:
        centre = peak.start + peak.width / 2.0
        peak.start = int(math.floor(centre - width / 2.0))
        peak.end = int(round(peak.start + width))
        peak.width = width

def sortByEnrichment(peaks):
    values = [p.enrichment for p in peaks]
    order = np.argsort(values)[::-1]
    return np.array(peaks)[order]

def arraysFromPeaks(peaks):
    X = np.array([p.x for p in peaks], dtype=bool)
    Y = np.array([p.value for p in peaks], dtype=float)
    return X, Y