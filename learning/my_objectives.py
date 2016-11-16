import tensorflow as tf
import keras.backend.tensorflow_backend as TB
from keras import backend as K
import scipy as sp
import numpy as np


def binary_crossentropy(y_true, y_pred):
    return K.mean(original_binary_crossentropy(y_pred, y_true), axis=-1)


def original_binary_crossentropy(output, target, from_logits=False):
    '''Binary crossentropy between an output tensor and a target tensor.
    '''
    # Note: tf.nn.softmax_cross_entropy_with_logits
    # expects logits, Keras expects probabilities.
    if not from_logits:
        # transform back to logits
        epsilon = TB._to_tensor(TB._EPSILON, output.dtype.base_dtype)
        output = tf.clip_by_value(output, epsilon, 1 - epsilon)
        output = tf.log(output / (1 - output))
    return tf.nn.sigmoid_cross_entropy_with_logits(output, target)


def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll


def main():

    real = [0, 1, 1, 0, 1, 0, 1]
    pred = [0, 0, 0, 0, 1, 0, 1]

    size = len(real)

    y_true = np.zeros((1, size), dtype=np.uint8)
    y_true[0, :] = real


    y_pred = np.zeros((1, size), dtype=np.uint8)
    y_pred[0, :] = pred


    resultado_ori = binary_crossentropy(y_true, y_pred)
    resultado_outro = logloss(y_true, y_pred)

    print 'Resultado original = ', resultado_ori
    print 'Resultado outro = ', resultado_outro

    # TODO precisa standarizar como no model.fit do keras - da erro assim


if __name__ == '__main__':
    main()