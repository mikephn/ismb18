from keras import backend as K
import numpy as np
import tensorflow as tf

# functions to obtain saliency maps from a Keras model
def saliencyFunction(model, n_output=-1):
    if n_output == -1:
        saliency = K.gradients(K.max(model.output), model.input)
    else:
        saliency = K.gradients(tf.slice(model.output, [0, n_output], [-1, 1]), model.input)
    return K.function([model.input, K.learning_phase()], saliency)

def loadSaliencyFunctions(model):
    model.saliencyFunctions = []
    for out_n in range(model.output.shape[1]):
        model.saliencyFunctions.append(saliencyFunction(model, out_n))

def saliencyMap(model, input, n_out=-1):
    example = np.reshape(input, (1, input.shape[0], input.shape[1]))
    sf = model.saliencyFunctions[n_out]
    saliencyMap = np.squeeze(sf([example, 0]))
    return saliencyMap