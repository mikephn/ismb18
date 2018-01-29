import json, math
from src.tools import *
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Sequential
from keras.layers import *

# function building a CNN model:
def convnet(inputShape, outputLength, params, outActivation='relu', average=False):
    nFilters = params["filters"]
    lFilter = params["length"]
    dropout = params["dropout"]
    dense = params["dense"]

    model = Sequential()
    model.add(Convolution1D(filters=nFilters, kernel_size=lFilter, padding='same', strides=1, input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(Dropout(dropout))
    if average:
        model.add(GlobalAveragePooling1D())
    else:
        model.add(GlobalMaxPooling1D())
    if dense > 0:
        model.add(Dense(dense, activation='relu'))
    model.add(Dense(outputLength, activation=outActivation))
    return model

# for classification classes are upsampled to match the count of the most frequent one:
def upsampleClass(x, y):
    nclasses = len(y[0])
    cx = []
    for n in range(nclasses):
        cx.append([])
    for ind, lx in enumerate(y):
        classn = np.argmax(lx)
        cx[classn].append(x[ind])
    upsampleTo = np.max([len(exs) for exs in cx])
    uX = []
    uY = []
    for ind, exs in enumerate(cx):
        if len(exs) == 0:
            continue
        label = np.zeros(nclasses)
        label[ind] = 1
        upsampledX = []
        ncopies = int(math.floor(float(upsampleTo)/len(exs)))
        for n in range(ncopies):
            upsampledX.extend(exs)
        left = upsampleTo - ncopies*len(exs)
        for n in range(left):
            rand = int(np.random.uniform(0, len(exs)))
            upsampledX.append(exs[rand])
        uX.extend(upsampledX)
        uY.extend([label]*len(upsampledX))
    return shuffledUnison(np.array(uX), np.array(uY))

# alternative to upsampling (used when training is called with upsampling=False):
def equalizedClassWeights(Y):
    nclasses = len(Y[0])
    counts = np.sum(Y, axis=0)
    total = float(len(Y))
    dictionary = {}
    norm = 0
    for cn in range(nclasses):
        norm += total / counts[cn]
    norm /= nclasses
    for cn in range(nclasses):
        dictionary[cn] = total / counts[cn] / norm
    return dictionary

def crossvalidationFold(foldN, outOf, X, Y):
    countPerFold = float(len(X)) / outOf
    vstart = int(round(countPerFold * foldN))
    vend = int(round(countPerFold * (foldN + 1)))
    vX = X[vstart:vend]
    vY = Y[vstart:vend]
    tX = np.vstack((X[:vstart], X[vend:]))
    tY = np.vstack((Y[:vstart], Y[vend:]))
    return tX, tY, vX, vY

def printLabelCount(Y):
    count = np.sum(Y, axis=0)
    if type(count) is not np.ndarray:
        return
    if np.sum(count) == len(Y):
        print("{}, total: {}".format(count, len(Y)))
    else:
        clc = {}
        for lab in Y:
            lab = tuple(lab)
            if lab in clc:
                clc[lab] += 1
            else:
                clc[lab] = 1
        print("Label count: {}".format(clc))

def saveModelToJson(model, filePath):
    with open(filePath, 'w') as outfile:
        json.dump(model.to_json(), outfile)

def train(runFolder, X, Y, vX, vY, params={}, upsample=True, regression=False, verbose=True):

    outputLength = len(Y[0])
    firstExampleX = X[0]
    inputDim = len(firstExampleX[0])
    inputShape = (None, inputDim)

    if regression:
        activation = 'linear'
    else:
        activation = 'softmax'

    model = convnet(inputShape, outputLength, params, outActivation=activation, average=regression)
    optimizer = Adam()

    if regression:
        model.compile(optimizer=optimizer,
                      loss='mse',
                      metrics=['mse', 'mae'])
    else:
        model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy',
                      metrics=['categorical_accuracy'])

    callbacks = []

    callbacks.append(
        EarlyStopping(monitor='val_loss', min_delta=0, patience=params["stopPatience"], verbose=0, mode='auto'))

    if runFolder is not None:
        clearFolder(runFolder)
        ensureFolder(runFolder)
        weightsFilePath = runFolder + "weights.hdf5"
        callbacks.append(ModelCheckpoint(monitor='val_loss', filepath=weightsFilePath,
                                         verbose=0, save_best_only=True, save_weights_only=True))
        saveModelToJson(model, runFolder + "model.json")

    if verbose:
        model.summary()

    classWeights = params.get("classWeights", None)
    if classWeights is None and not regression:
        if upsample:
            if verbose:
                print("Upsampling classes...")
            X, Y = upsampleClass(X, Y)
            vX, vY = upsampleClass(vX, vY)
        else:
            classWeights = equalizedClassWeights(Y)
            if verbose:
                print("Weighting loss function: {}".format(classWeights))

    vb = 0
    if verbose:
        vb = 2

    model.fit(X, Y,
                validation_data=(vX, vY),
                shuffle=0,
                batch_size=params["batchSize"],
                epochs=params["epochs"],
                class_weight=classWeights,
                verbose=vb,
                callbacks=callbacks)

# used to train an ensemble of networks, each on a different fold of the data:
def crossvalidate(folder, X, Y, params, folds=5, upsample=True, regression=True, verbose=True):

    for fn in range(folds):
        foldFolder = folder + "f-{}/".format(fn)
        ensureFolder(foldFolder)
        tX, tY, vX, vY = crossvalidationFold(foldN=fn, outOf=folds, X=X, Y=Y)

        if verbose:
            print("\nFold {} / {}".format(fn+1, folds))
            if not regression:
                print("Fold training data distribution:")
                printLabelCount(tY)
                print("Fold validation data distribution:")
                printLabelCount(vY)

        train(foldFolder, tX, tY, vX, vY, params, upsample, regression, verbose)
