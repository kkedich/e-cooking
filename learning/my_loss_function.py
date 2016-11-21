import tensorflow as tf

import keras.backend.tensorflow_backend as TB
from keras import backend as K
import scipy as sp
import numpy as np


from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops


# from tensorflow TB._EPSILON
_EPSILON = 10e-8
POS_WEIGHT = 1.5


def weighted_binary_crossentropy(y_true, y_pred):
    return K.mean(my_weighted_binary_crossentropy(y_pred, y_true), axis=-1)


def tf_count(t, value):
    """Count the number of val elements in tensor t
       t: tensor with a numpy array
       value: value to be counted

       tensor with the array [0, 1, 0, 1, 0]
       Ex: tf_count(t, 0) returns 3
           tf_count(t, 1) returns 2
    """
    elements_equal_to_value = tf.equal(t, value)
    as_ints = tf.cast(elements_equal_to_value, tf.float32)
    count = tf.reduce_sum(as_ints, 1, keep_dims=True)
    return count


def my_weighted_binary_crossentropy(output, target, from_logits=False):
    '''
      Note: this implementation is based on the tensorflow implementation.
      Binary crossentropy between an output tensor and a target tensor.
    '''
    # Note: tf.nn.softmax_cross_entropy_with_logits
    # expects logits, Keras expects probabilities.
    if not from_logits:
        # transform back to logits. Avoid numerical instability with _EPSILON clipping
        epsilon = TB._to_tensor(_EPSILON, output.dtype.base_dtype)  # output.dtype.base_dtype, in docker output.dtype.base
        output = tf.clip_by_value(output, epsilon, 1 - epsilon)
        output = tf.log(output / (1 - output))

    print 'dentro1=', target.get_shape()
    print 'dentro2=', output.get_shape()

    nb_input, nb_values = output.get_shape()

    zeros =  tf_count(target, 0)
    # print 'zeros', zeros
    # print 'zeros=', zeros.eval(session=tf.Session())

    non_zeros = tf_count(target, 1)
    # print 'non_zeros', non_zeros
    # print 'non_zeros=', non_zeros.eval(session=tf.Session())

    weight =  tf.div(zeros, non_zeros)
    # print 'weight', weight
    # print 'weight=', weight.eval(session=tf.Session())

    value = my_weighted_cross_entropy_with_logits(output, target, pos_weight=weight)
    normalized = tf.div(value, zeros)  # value / zeros -> value / (nb_values-non_zeros)
    # print 'normalized ', normalized.eval(session=tf.Session())

    # Multiply the result by a constant -> nb_values to avoid very small values (<1.0)
    # print nb_values
    nb_values_float = tf.to_float(nb_values)
    tensor_nb_values = tf.convert_to_tensor(nb_values_float, dtype=tf.float32)  # convert value to tensor
    final_value = tf.mul(normalized, tensor_nb_values)
    # print 'final_value', final_value.eval(session=tf.Session())
    return final_value


def my_weighted_cross_entropy_with_logits(logits, targets, pos_weight, name=None):
    with ops.name_scope(name, "logistic_loss", [logits, targets]) as name:
        logits = ops.convert_to_tensor(logits, name="logits")
        targets = ops.convert_to_tensor(targets, name="targets")
        try:
          targets.get_shape().merge_with(logits.get_shape())
        except ValueError:
          raise ValueError("logits and targets must have the same shape (%s vs %s)"
                           % (logits.get_shape(), targets.get_shape()))

        print 'dentro3=', targets.get_shape()

        # The logistic loss formula from above is
        #   (1 - z) * x + (1 + (q - 1) * z) * log(1 + exp(-x))
        # For x < 0, a more numerically stable formula is
        #   (1 - z) * x + (1 + (q - 1) * z) * log(1 + exp(x)) - l * x
        # To avoid branching, we use the combined version
        #   (1 - z) * x + l * (log(1 + exp(-abs(x))) + max(-x, 0))
        log_weight = 1 + (pos_weight - 1) * targets
        print 'dentro4=', log_weight
        # print log_weight.eval(session=tf.Session())

        # print quebra_isso_aqui
        return math_ops.add(
            (1 - targets) * logits,
            log_weight * (math_ops.log(1 + math_ops.exp(-math_ops.abs(logits))) +
                          nn_ops.relu(-logits)),
            name=name)


def binary_crossentropy(y_true, y_pred):
    resul = original_binary_crossentropy(y_pred, y_true)
    return K.mean(resul, axis=-1)


def original_binary_crossentropy(output, target, from_logits=False):
    '''Binary crossentropy between an output tensor and a target tensor.
    '''
    # Note: tf.nn.softmax_cross_entropy_with_logits
    # expects logits, Keras expects probabilities.
    if not from_logits:
        # transform back to logits
        # avoid numerical instability with _EPSILON clipping
        epsilon = TB._to_tensor(_EPSILON, output.dtype.base)
        output = tf.clip_by_value(output, epsilon, 1 - epsilon)
        output = tf.log(output / (1 - output))
    return tf.nn.sigmoid_cross_entropy_with_logits(output, target)


# https://www.kaggle.com/wiki/LogarithmicLoss
def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act * sp.log(pred) + sp.subtract(1,act) * sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll


def main():

    real = [0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    pred = [1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    pred_certo = [0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # pred_certo = [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    size = len(real)

    y_true = np.zeros((2, size), dtype=np.float32)
    y_true[0, :] = real
    y_true[1, :] = real

    y_pred = np.zeros((2, size), dtype=np.float32)
    y_pred[0, :] = pred_certo
    y_pred[1, :] = pred_certo

    y_pred2 = np.zeros((2, size), dtype=np.float32)
    y_pred2[0, :] = pred
    y_pred2[1, :] = pred

    tf.initialize_all_variables()

    y_true_new = tf.convert_to_tensor(y_true, dtype=y_true.dtype)
    y_pred_new = tf.convert_to_tensor(y_pred, dtype=y_true.dtype)

    y_pred_new2 = tf.convert_to_tensor(y_pred2, dtype=y_true.dtype)

    resultado_ori = weighted_binary_crossentropy(y_true_new, y_pred_new)
    resultado_ori2 = weighted_binary_crossentropy(y_true_new, y_pred_new2)
    res = resultado_ori.eval(session=tf.Session())
    res2 = resultado_ori2.eval(session=tf.Session())
    print res
    print res2

    resultado_ori_bin = binary_crossentropy(y_true, y_pred)
    resultado_ori_errado = binary_crossentropy(y_true, y_pred2)
    #
    # resultado_ori = weighted_binary_crossentropy(y_true, y_pred)
    # resultado_ori2 = weighted_binary_crossentropy(y_true, y_pred2)
    # # resultado_outro = logloss(y_true, y_pred)
    #
    original = resultado_ori_bin.eval(session=tf.Session())
    print original
    original_errado = resultado_ori_errado.eval(session=tf.Session())
    print original_errado
    # um = resultado_ori.eval(session=tf.Session())
    # print '------------------------------------------------'
    # dois = resultado_ori2.eval(session=tf.Session())
    # um_novo = np.float32(um[0])
    # dois_novo = np.float32(dois[0])
    #
    # original_novo = np.float32(original[0])
    # original_errado_novo = np.float32(original_errado[0])
    #
    # print 'Resultado original (certo) = {0:.16f}'.format(um_novo)
    # print 'resultado2 = {0:.16f}'.format(dois_novo)
    # print '(Bin)Resultado original (certo) = {0:.16f}'.format(original_novo)
    # print '(bin)resultado2 = {0:.16f}'.format(original_errado_novo)
    #
    # if um_novo > dois_novo:
    #     print 'errado: um maior que dois'
    # elif um_novo == dois_novo:
    #     print 'iguais'
    # else:
    #     print 'certo: dois maior que um'

    # print 'multi -> {0:.16f}'.format(um_novo*len(real))
    # print 'multi -> {0:.16f}'.format(dois_novo * len(real))
    # print resultado_ori.eval(session=tf.Session())
    # print 'Resultado outro = ', resultado_outro



if __name__ == '__main__':
    main()