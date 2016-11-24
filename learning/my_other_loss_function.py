import numpy as np
np.random.seed(0)

from keras import backend as K

_EPSILON = K.epsilon()


def weighted_loss(class_weight, nb_ingredients):
    def new_loss(y_true, y_pred):
        """New loss function where the class is weighted and the ingredients present in the ground truth are
           assigned with high weights.
        """
        non_zeros = K.sum(y_true, axis=1, keepdims=True)
        zeros = nb_ingredients - non_zeros
        weight_for_ones = zeros / non_zeros

        y_pred = K.clip(y_pred, _EPSILON, 1.0-_EPSILON)
        out = -((y_true * weight_for_ones) * K.log(y_pred) + (1.0 - y_true) * K.log(1.0 - y_pred))
        final = out * class_weight
        return K.mean(final, axis=-1)
    return new_loss

# def _loss_np(y_true, y_pred):
#     y_pred = np.clip(y_pred, _EPSILON, 1.0-_EPSILON)
#     out = -(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred))
#     return np.mean(out, axis=-1)


def weighted_binary_crossentropy_numpy(y_true, y_pred, class_weight, nb_ingredients):
    epsilon = 10e-8
    y_pred = np.maximum(epsilon, y_pred)
    y_pred = np.minimum(1 - epsilon, y_pred)

    print 'y_true\n',y_true
    print 'y_pred\n',y_pred
    # Generate weight
    non_zeros = np.sum(y_true, axis=1, keepdims=True)
    zeros = nb_ingredients - non_zeros
    weight_for_ones = zeros / non_zeros

    # ponderacao = teste * peso e depois  ponderacao * result_for_keras['weight']

    # y_pred = K.clip(y_pred, _EPSILON, 1.0 - _EPSILON)
    # out = -((y_true * K.log(y_pred) )/non_zeros + ((1.0 - y_true) * K.log(1.0 - y_pred))/zeros )
    out = -(((y_true * weight_for_ones) * np.log(y_pred)) + ((1.0 - y_true) * np.log(1.0 - y_pred)))
    final = out * class_weight
    print final.shape
    print final
    return np.mean(final, axis=-1)

# https://github.com/fchollet/keras/issues/2662
def binary_crossentropy_numpy(y_true, y_pred):
    epsilon = 10e-8
    y_pred = np.maximum(epsilon, y_pred)
    y_pred = np.minimum(1 - epsilon, y_pred)

    # y_pred = K.clip(y_pred, _EPSILON, 1.0 - _EPSILON)
    # out = -((y_true * K.log(y_pred) )/non_zeros + ((1.0 - y_true) * K.log(1.0 - y_pred))/zeros )
    out = -((y_true * np.log(y_pred)) + ((1.0 - y_true) * np.log(1.0 - y_pred)))
    return np.mean(out, axis=-1)



def main():

    class_weight = [1, 2, 3, 4, 5, 6]

    real = [0, 1, 0, 0, 1, 0]
    pred = [1, 0, 1, 1, 0, 1]
    pred_certo = [0, 1, 0, 0, 1, 0]


    size = len(real)
    y_true = np.zeros((2, size), dtype=np.float32)
    y_true[0, :] = real
    y_true[1, :] = real

    y_pred = np.zeros((2, size), dtype=np.float32)
    y_pred[0, :] = pred_certo
    y_pred[1, :] = pred

    print 'entrei'
    result = weighted_binary_crossentropy_numpy(y_true, y_pred, class_weight, len(class_weight))
    print 'pred=[certo, errado]'
    print 'result=\n', result

    binary_result = binary_crossentropy_numpy(y_true, y_pred)
    print 'pred=[certo, errado]'
    print 'binary_result=\n', binary_result

if __name__ == '__main__':
    main()

