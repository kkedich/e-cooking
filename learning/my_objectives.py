import tensorflow as tf
import keras.backend.tensorflow_backend as TB
from keras import backend as K
import scipy as sp
import numpy as np
from keras.engine.training import standardize_input_data


def binary_crossentropy(y_true, y_pred):
    resul = original_binary_crossentropy(y_pred, y_true)
    # print resul
    print resul.eval(session=tf.Session())
    return K.mean(resul, axis=-1)


def original_binary_crossentropy(output, target, from_logits=False):
    '''Binary crossentropy between an output tensor and a target tensor.
    '''
    # Note: tf.nn.softmax_cross_entropy_with_logits
    # expects logits, Keras expects probabilities.
    if not from_logits:
        # transform back to logits
        # print "aquiiiiiiiiiiiiiii ", output.dtype.base
        epsilon = TB._to_tensor(TB._EPSILON, output.dtype.base)
        output = tf.clip_by_value(output, epsilon, 1 - epsilon)
        output = tf.log(output / (1 - output))
    return tf.nn.sigmoid_cross_entropy_with_logits(output, target)


def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act * sp.log(pred) + sp.subtract(1,act) * sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll

def _standardize_user_data(y, sample_weight=None, class_weight=None,
                           check_batch_dim=True, batch_size=None):
    output_shapes = [None]
    output_names = []
    y = standardize_input_data(y, output_names,
                               output_shapes,
                               check_batch_dim=False,
                               exception_prefix='model target')
    return y



def my_standardize_input_data(data):
    if type(data) is dict:
        print 'dict'
    elif type(data) is list:
        print 'list'
    else:
        print 'else'
        if not hasattr(data, 'shape'):
            print 'hasattr'

    arrays = [np.asarray(data)]
    # arrays = [data]
    print 'tamanho vetor={}, quantos elementos={}'.format(len(data), len(data[0]))
    print 'tamanho vetor arrays={}'.format(len(arrays))

    # make arrays at least 2D
    for i in range(len(data)) :           #range(len(names)):
        array = arrays[i]
        if len(array.shape) == 1:
            array = np.expand_dims(array, 1)
            arrays[i] = array

    print arrays
    return arrays

def main():



    real = [0, 1, 1, 0, 1, 0, 1]
    pred_ruim = [0, 0, 0, 0, 1, 0, 1]
    pred_certo = [1, 0, 0, 1, 0, 0, 1]

    size = len(real)

    y_true = np.zeros((1, size), dtype=np.float32)
    y_true[0, :] = real


    y_pred = np.zeros((1, size), dtype=np.float32)
    y_pred[0, :] = pred_certo


    # y = my_standardize_input_data(y_true)
    # y2 = _standardize_user_data(y_true)


    resultado_ori = binary_crossentropy(y_true, y_pred)
    # resultado_outro = logloss(y_true, y_pred)

    print 'Resultado original = ', resultado_ori
    print resultado_ori.eval(session=tf.Session())
    # print 'Resultado outro = ', resultado_outro

    # TODO precisa standarizar como no model.fit do keras - da erro assim


if __name__ == '__main__':
    main()