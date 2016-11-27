import numpy as np
np.random.seed(0)

from keras import optimizers
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Dropout, Flatten, Dense
from keras import backend as K

from learning import result
from learning.my_other_loss_function import weighted_loss

import os
import h5py

# path to the model weights files. https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3
weights_path = 'vgg16_weights.h5'


def acc2(y_true, y_pred):
    return K.mean(K.equal(y_true, K.round(y_pred)))


def fine_tuning(top_model_weights_path, final_vgg16_model,
                img_width, img_height,
                train_data, train_ingredients,
                validation_data, val_ingredients,
                nb_epoch, batch_size, #validation_split,
                class_weight,
                dropout=0.5, neurons_last_layer=256,
                custom_loss=None):

    # Determine proper input shape
    if K.image_dim_ordering() == 'th':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)

    # build the VGG16 network
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=input_shape))


    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))


    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))


    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))


    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))


    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))


    # load the weights of the VGG16 networks
    # (trained on ImageNet, won the ILSVRC competition in 2014)
    # note: when there is a complete match between your model definition
    # and your weight savefile, you can simply call model.load_weights(filename)
    assert os.path.exists(weights_path), 'Model weights not found (see "weights_path" variable in script).'
    f = h5py.File(weights_path)
    for k in range(f.attrs['nb_layers']):
        if k >= len(model.layers):
            # we don't look at the last (fully-connected) layers in the savefile
            break
        g = f['layer_{}'.format(k)]
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        model.layers[k].set_weights(weights)
    f.close()
    print('Model loaded.')

    nb_images, nb_ingredients = train_ingredients.shape
    nb_images_val, nb_ingredients_val = val_ingredients.shape
    print 'Train: number of ingredients={}, input_size={}, input_shape={}'.format(nb_ingredients, nb_images,
                                                                                  train_data.shape[1:])
    print 'Validation: number of ingredients={}, input_size={}, input_shape={}'.format(nb_ingredients_val,
                                                                                       nb_images_val,
                                                                                       validation_data.shape[1:])
    print 'model.output_shape=', model.output_shape[1:]

    # build a classifier model to put on top of the convolutional model
    top_model = Sequential()
    top_model.add(Flatten(input_shape=model.output_shape[1:]))
    top_model.add(Dense(neurons_last_layer, activation='relu'))  # before was 256
    top_model.add(Dropout(dropout))
    top_model.add(Dense(nb_ingredients, activation='sigmoid')) #, name='ingredients'

    # note that it is necessary to start with a fully-trained classifier, including the top classifier,
    # in order to successfully do fine-tuning
    top_model.load_weights(top_model_weights_path)

    # add the model on top of the convolutional base
    model.add(top_model)

    # set the first 25 layers (up to the last conv block) to non-trainable (weights will not be updated)
    for layer in model.layers[:25]:
        layer.trainable = False

    # compile the model with a SGD/momentum optimizer and a very slow learning rate.
    # model.compile(loss='binary_crossentropy',
    #               optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
    #               metrics=['accuracy'])

    # for layer in model.layers:
    #     print layer.get_config()

    print '\n'
    model.summary()

    history = None
    if custom_loss is None:
        model.compile(optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),  #SGD(lr=0.001)
                      loss='binary_crossentropy',  # loss={'ingredients': 'binary_crossentropy'},
                      metrics=['accuracy', acc2])


        history = model.fit(train_data, y=train_ingredients,  # y={'ingredients': ingredients}
                            nb_epoch=nb_epoch, batch_size=batch_size,
                            validation_data=(validation_data, val_ingredients),
                            class_weight=class_weight,  # validation_split=validation_split,
                            verbose=2)

    elif custom_loss == 'weighted_binary_crossentropy':
        print 'custom_loss=', custom_loss

        # compile the model with a SGD/momentum optimizer and a very slow learning rate.
        model.compile(optimizer=optimizers.SGD(lr=1e-4, momentum=0.9), #optimizers.Adam(lr=1e-4),
                      loss=weighted_loss(class_weight, nb_ingredients),
                      metrics=['accuracy', acc2])

        history = model.fit(train_data, y=train_ingredients,  # y={'ingredients': ingredients}
                            nb_epoch=nb_epoch, batch_size=batch_size,
                            validation_data=(validation_data, val_ingredients),
                            verbose=2)

    else:
        print 'No loss function defined. Returning...'
        return []


    model.save(final_vgg16_model)

    result.process(history, '../e-cooking/')
    # print history
    # np.save(open('history.npy', 'w'), history)

