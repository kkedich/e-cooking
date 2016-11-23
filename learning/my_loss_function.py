
import numpy as np
np.random.seed(0)

from keras.objectives import binary_crossentropy
import keras.backend.tensorflow_backend as TB
from keras import backend as K

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import array_ops


# from tensorflow TB._EPSILON
_EPSILON = K.epsilon() #10e-8
print _EPSILON

def weighted_binary_crossentropy(y_true, y_pred):
    return K.mean(my_weighted_binary_crossentropy(y_pred, y_true), axis=-1)

# def loss_fn(peso):
def new(y_true, y_pred):
    zeros = tf_count(y_true, 0)
    print 'zeros', zeros

    non_zeros = tf_count(y_true, 1)
    print 'non_zeros', non_zeros

    weight = tf.div(zeros, non_zeros)
    print 'weight', weight

    print 'y_true', y_true
    print y_true.get_shape()
    print y_pred.get_shape()

    # peso = np.ones((32,1))

    y_pred = K.clip(y_pred, _EPSILON, 1.0 - _EPSILON)
    # out = -((y_true * K.log(y_pred) )/non_zeros + ((1.0 - y_true) * K.log(1.0 - y_pred))/zeros )
    out = -((y_true * K.log(y_pred)) * weight + ((1.0 - y_true) * K.log(1.0 - y_pred)) )
    return K.mean(out, axis=-1)
    # return new

def tf_count(t, value):
    """Count the number of val elements in tensor t
       t: tensor with a numpy array
       value: value to be counted

       tensor with the array [0, 1, 0, 1, 0]
       Ex: tf_count(t, 0) returns 3
           tf_count(t, 1) returns 2
    """
    elements_equal_to_value = K.tf.equal(t, value)
    as_ints = K.tf.cast(elements_equal_to_value, tf.float32)
    count = K.tf.reduce_sum(as_ints, 1, keep_dims=True)
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

    zeros =  tf_count(target, 0)
    print 'zeros', zeros
    # # print 'zeros=', zeros.eval(session=tf.Session())

    non_zeros = tf_count(target, 1)
    print 'non_zeros', non_zeros
    # # print 'non_zeros=', non_zeros.eval(session=tf.Session())

    weight = tf.div(non_zeros, non_zeros)
    print 'weight', weight
    # # print 'weight=\n', weight.eval(session=tf.Session())

    # value = my_weighted_cross_entropy_with_logits(output, target, pos_weight=1.0)
    value = tf.nn.weighted_cross_entropy_with_logits(output, target, pos_weight=weight)
    # print 'value ', value.eval(session=tf.Session())

    # value / zeros -> value / (nb_values-non_zeros)
    # print 'normalized ', normalized.eval(session=tf.Session())

    # tf.Print(normalized, [normalized], "normalized")
    # tf.Print(value, [value], "normalized")
    # Multiply the result by a constant -> nb_values to avoid very small values (<1.0)
    # print nb_values
    # nb_values_float = tf.to_float(nb_values)
    # tensor_nb_values = tf.convert_to_tensor(nb_values_float, dtype=tf.float32)  # convert value to tensor
    # final_value = tf.mul(value, tensor_nb_values)
    #
    # normalized = tf.div(final_value, zeros)
    # print 'final_value', final_value.eval(session=tf.Session())
    return value


def my_weighted_cross_entropy_with_logits(logits, targets, pos_weight, name=None):
    with ops.name_scope(name, "logistic_loss", [logits, targets]) as name:
        logits = ops.convert_to_tensor(logits, name="logits")
        targets = ops.convert_to_tensor(targets, name="targets")
        try:
          targets.get_shape().merge_with(logits.get_shape())
        except ValueError:
          raise ValueError("logits and targets must have the same shape (%s vs %s)"
                           % (logits.get_shape(), targets.get_shape()))

        # The logistic loss formula from above is
        #   (1 - z) * x + (1 + (q - 1) * z) * log(1 + exp(-x))
        # For x < 0, a more numerically stable formula is
        #   (1 - z) * x + (1 + (q - 1) * z) * log(1 + exp(x)) - l * x
        # To avoid branching, we use the combined version
        #   (1 - z) * x + l * (log(1 + exp(-abs(x))) + max(-x, 0))

        log_weight = 1 + (pos_weight - 1) * targets
        # log_weight = 1 + (pos_weight * targets) - targets

        # first_part = (1 - targets) * logits
        # # print 'first_part\n', first_part.eval(session=tf.Session())
        # second_part = log_weight * (math_ops.log(1 + math_ops.exp(-math_ops.abs(logits))) + nn_ops.relu(logits))
        # value = math_ops.add(first_part, second_part, name=name)
        # log_weight = 1 + (pos_weight - 1) * targets
        # return math_ops.add(
        #     (1 - targets) * logits,
        #     log_weight * (math_ops.log(1 + math_ops.exp(-math_ops.abs(logits))) +
        #                   nn_ops.relu(-logits)),
        #     name=name)

        # another test
        zeros = array_ops.zeros_like(logits, dtype=logits.dtype)
        cond = (logits >= zeros)
        relu_logits = math_ops.select(cond, logits, zeros)
        neg_abs_logits = math_ops.select(cond, -logits, logits)

        primeira_parte = relu_logits - logits * targets
        segunda_parte = log_weight * (math_ops.log(1 + math_ops.exp(neg_abs_logits)) + nn_ops.relu(-logits))
        final2 = math_ops.add(primeira_parte, segunda_parte, name=name)

        return final2




# def sigmoid_cross_entropy_with_logits(logits, targets, name=None):
#   with ops.name_scope(name, "logistic_loss", [logits, targets]) as name:
#     logits = ops.convert_to_tensor(logits, name="logits")
#     targets = ops.convert_to_tensor(targets, name="targets")
#     try:
#       targets.get_shape().merge_with(logits.get_shape())
#     except ValueError:
#       raise ValueError("logits and targets must have the same shape (%s vs %s)"
#                        % (logits.get_shape(), targets.get_shape()))
#
#     # The logistic loss formula from above is
#     #   x - x * z + log(1 + exp(-x))
#     # For x < 0, a more numerically stable formula is
#     #   -x * z + log(1 + exp(x))
#     # Note that these two expressions can be combined into the following:
#     #   max(x, 0) - x * z + log(1 + exp(-abs(x)))
#     # To allow computing gradients at zero, we define custom versions of max and
#     # abs functions.
#     zeros = array_ops.zeros_like(logits, dtype=logits.dtype)
#     cond = (logits >= zeros)
#     relu_logits = math_ops.select(cond, logits, zeros)
#     neg_abs_logits = math_ops.select(cond, -logits, logits)
#     return math_ops.add(relu_logits - logits * targets,
#                         math_ops.log(1 + math_ops.exp(neg_abs_logits)),
#                         name=name)


def main():

    real = [0, 1, 0, 0, 1, 0] #, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    pred = [1, 0, 1, 1, 0, 1] #, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    pred_certo = [0, 1, 0, 0, 1, 0]#, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # pred_certo = [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    real2 = [1, 0, 0, 0, 0, 0]

    size = len(real)

    y_true = np.zeros((2, size), dtype=np.float32)
    y_true[0, :] = real
    y_true[1, :] = real

    y_pred2 = np.zeros((2, size), dtype=np.float32)
    y_pred2[0, :] = pred_certo
    y_pred2[1, :] = pred

    tf.initialize_all_variables()

    y_true_new = tf.convert_to_tensor(y_true, dtype=y_true.dtype)
    # y_pred_new = tf.convert_to_tensor(y_pred, dtype=y_true.dtype)

    y_pred_new2 = tf.convert_to_tensor(y_pred2, dtype=y_true.dtype)

    resultado_ori = binary_crossentropy(y_true_new, y_pred_new2)
    resultado_ori2 = weighted_binary_crossentropy(y_true_new, y_pred_new2)
    res = resultado_ori.eval(session=tf.Session())
    res2 = resultado_ori2.eval(session=tf.Session())
    print res
    print res2
    print np.float32(res2)
    print 'certo {0:.16f}'.format(np.float32(res2)[0])
    print 'errado {0:.16f}'.format(np.float32(res2)[1])

    print np.float32(res)
    print '(Bin)certo {0:.16f}'.format(np.float32(res)[0])
    print '(Bin)errado {0:.16f}'.format(np.float32(res)[1])



if __name__ == '__main__':
    main()