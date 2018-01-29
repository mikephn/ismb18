import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.tools import *
from src.Peaks import *

class PlotTrack:
    def __init__(self, value, name, color, width=1):
        self.value = value
        self.name = name
        self.color = color
        self.width = width

def saveTrackLine(peak, outFolder, titlePrefix="", filePrefix="", fill=False, normTogether=False):
    ensureFolder(outFolder)
    title = titlePrefix + peak.chromosome + " " + str(peak.start) + "-" + str(peak.end) + " "
    fileName = filePrefix + peak.chromosome + "-" + str(peak.start) + "-" + str(peak.end) + ".png"
    print("Saving {}".format(title))
    X = np.arange(peak.end - peak.start)
    if fill:
        plt.figure(figsize=(float(len(X)) / 9, 3))
    else:
        plt.figure(figsize=(32, 2))

    if normTogether:
        normFactor = 0.000001
        for track in peak.tracks:
            normFactor = max(normFactor, np.max(np.abs(track.value), axis=0))

    handles = []
    colors = ['b', 'y', 'r', 'g']
    sequenceMap = ["c", "g", "t", "a"]

    for ind, track in enumerate(peak.tracks):
        if not normTogether:
            normFactor = np.max(np.abs(track.value), axis=0)
        if fill:
            for ind, plotNtd in enumerate(sequenceMap):
                filledVals = np.zeros(len(peak.sequence))
                for loc, nt in enumerate(peak.sequence):
                    if nt == plotNtd:
                        filledVals[loc] = (track.value/normFactor)[loc]
                plt.bar(X, filledVals, edgecolor="none", align='center', color=colors[ind], linewidth=0, width=1)
        else:
            h, = plt.plot(track.value/normFactor, color=track.color, label=track.name, linewidth=track.width)
            handles.append(h)

    if not fill:
        plt.legend(handles=handles)

    if len(X) < 300:
        plt.xticks(X, peak.sequence)
    plt.axis([-1, len(peak.sequence), -0.1, 1.1])
    plt.yticks([])
    plt.title(title)

    plt.savefig(outFolder + "/" + fileName, bbox_inches='tight', dpi=100)  # save the figure to file
    print("Saved {}".format(fileName))
    plt.close()