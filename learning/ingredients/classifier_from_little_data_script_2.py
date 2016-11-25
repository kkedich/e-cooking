import numpy as np
np.random.seed(0)

import os
import h5py

# from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
# from keras.objectives import binary_crossentropy
from keras.optimizers import SGD, Adam, RMSprop
# from keras import optimizers

# from learning.my_loss_function import weighted_binary_crossentropy
from learning.my_other_loss_function import weighted_loss

# Path to the model weights file. https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3
weights_path = 'vgg16_weights.h5'

# def weighted_binary_crossentropy(y_true, y_pred):
#     return K.mean(K.binary_crossentropy(y_pred, y_true), axis=-1)

def acc3(y_true, y_pred):
    return K.mean(K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)))


def acc2(y_true, y_pred):
    return K.mean(K.equal(y_true, K.round(y_pred)))



def save_bottlebeck_features(file_bottleneck_features_train, file_bottleneck_features_validation,
                             input_data_train, input_data_validation,
                             batch_size,
                             img_width, img_height):
    # TODO first without image augmentation
    # datagen = ImageDataGenerator(rescale=1./255)

    # Determine proper input shape
    if K.image_dim_ordering() == 'th':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)

    # Build the VGG16 network
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

    # generator = datagen.flow_from_directory(
    #         constants.TRAIN_DATA_DIR,
    #         target_size=(constants.IMG_WIDTH, constants.IMG_HEIGHT),
    #         batch_size=constants.BATCH_SIZE,
    #         class_mode=None,
    #         shuffle=False)
    # bottleneck_features_train = model.predict_generator(generator, constants.NB_TRAIN_SAMPLES)
    print 'Predict (train): generating bottleneck features.'
    bottleneck_features_train = model.predict(input_data_train, batch_size, verbose=1)
    np.save(open(file_bottleneck_features_train, 'w'), bottleneck_features_train)
    print 'Bottleneck features (train) saved.'

    # generator = datagen.flow_from_directory(
    #         constants.VALIDATION_DATA_DIR,
    #         target_size=(img_width, img_height),
    #         batch_size=constants.BATCH_SIZE,
    #         class_mode=None,
    #         shuffle=False)
    # bottleneck_features_validation = model.predict_generator(generator, constants.NB_VALIDATION_SAMPLES)
    print 'Predict (validation): generating bottleneck features.'
    bottleneck_features_validation = model.predict(input_data_validation, batch_size, verbose=1)
    np.save(open(file_bottleneck_features_validation, 'w'), bottleneck_features_validation)
    print 'Bottleneck features (validation) saved.'



def train_top_model(file_bottleneck_features_train, file_bottleneck_features_validation,
                    top_model_weights_path,
                    nb_epoch, batch_size, # validation_split,
                    train_ingredients, val_ingredients,
                    dropout=0.5, neurons_last_layer=256,
                    custom_loss=None,
                    class_weight=None):

    train_data = np.load(open(file_bottleneck_features_train))
    # train_labels = np.array([0] * (constants.NB_TRAIN_SAMPLES / 2) + [1] * (constants.NB_TRAIN_SAMPLES / 2))

    validation_data = np.load(open(file_bottleneck_features_validation))
    # validation_labels = np.array([0] * (constants.NB_VALIDATION_SAMPLES / 2) + [1] * (constants.NB_VALIDATION_SAMPLES / 2))

    nb_images, nb_ingredients = train_ingredients.shape
    nb_images_val, nb_ingredients_val = val_ingredients.shape
    print 'Train: number of ingredients={}, input_size={}, input_shape={}'.format(nb_ingredients, nb_images,
                                                                                  train_data.shape[1:])
    print 'Validation: number of ingredients={}, input_size={}, input_shape={}'.format(nb_ingredients_val, nb_images_val,
                                                                                       validation_data.shape[1:])

    assert (train_data.shape[1:] == validation_data.shape[1:]), 'Incorrect shape: train and validation have different shapes.'
    assert nb_ingredients == nb_ingredients_val, 'Incorrect number of ingredients in train {} and validation {}.'.format(nb_ingredients, nb_ingredients_val)


    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(neurons_last_layer, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(nb_ingredients, activation='sigmoid')) #, name='ingredients'

    # Define which loss function will be used
    if custom_loss is None:
        model.compile(optimizer='sgd',
                      loss='binary_crossentropy',
                      metrics=['accuracy', acc2])


        model.fit(train_data, y=train_ingredients,
                  nb_epoch=nb_epoch, batch_size=batch_size,
                  validation_data=(validation_data, val_ingredients),
                  class_weight=class_weight)

    elif custom_loss == 'weighted_binary_crossentropy':
        # model.compile(optimizer='adam', #optimizer=optimizers.Adam(lr=1e-4), sgd = SGD(lr=0.001, decay=1e-6, momentum=0.5, nesterov=True)
        #               loss=weighted_binary_crossentropy,
        #               metrics=['accuracy', acc2, acc3])

        model.compile(optimizer='sgd',
                      loss=weighted_loss(class_weight, nb_ingredients),
                      metrics=['accuracy', acc2, acc3])


        model.fit(train_data, y=train_ingredients,
                  nb_epoch=nb_epoch, batch_size=batch_size,
                  validation_data=(validation_data, val_ingredients))
    else:
        print 'Something is wrong. Returning...'
        return []


    print 'Saving model weights...'
    model.save_weights(top_model_weights_path)