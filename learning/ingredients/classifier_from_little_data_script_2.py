import os
import h5py
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense

# Path to the model weights file.
weights_path = 'vgg16_weights.h5'


def save_bottlebeck_features(file_bottleneck_features_train, file_bottleneck_features_validation,
                             input_data_train, input_data_val,
                             batch_size,
                             img_width, img_height):
    # TODO first without image augmentation
    # datagen = ImageDataGenerator(rescale=1./255)

    # Build the VGG16 network
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3, img_width , img_height)))

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
    bottleneck_features_train = model.predict(input_data_train, batch_size)
    np.save(open(file_bottleneck_features_train, 'w'), bottleneck_features_train)

    # generator = datagen.flow_from_directory(
    #         constants.VALIDATION_DATA_DIR,
    #         target_size=(img_width, img_height),
    #         batch_size=constants.BATCH_SIZE,
    #         class_mode=None,
    #         shuffle=False)
    # bottleneck_features_validation = model.predict_generator(generator, constants.NB_VALIDATION_SAMPLES)
    bottleneck_features_validation = model.predict(input_data_val, batch_size)
    np.save(open(file_bottleneck_features_validation, 'w'), bottleneck_features_validation)


def train_top_model(file_bottleneck_features_train, file_bottleneck_features_validation,
                    top_model_weights_path,
                    nb_epoch, batch_size,
                    ingredients):

    train_data = np.load(open(file_bottleneck_features_train))
    # train_labels = np.array([0] * (constants.NB_TRAIN_SAMPLES / 2) + [1] * (constants.NB_TRAIN_SAMPLES / 2))

    validation_data = np.load(open(file_bottleneck_features_validation))
    # validation_labels = np.array([0] * (constants.NB_VALIDATION_SAMPLES / 2) + [1] * (constants.NB_VALIDATION_SAMPLES / 2))

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(ingredients), activation='sigmoid', name='ingredients'))

    model.compile(optimizer='adam',
                  loss={'ingredients': 'binary_crossentropy'}, metrics=['accuracy'])

    # model.fit(train_data, train_labels,
    #           nb_epoch=nb_epoch, batch_size=batch_size,
    #           validation_data=(validation_data, validation_labels))

    # TODO ver como fazer com validation, ja que nao tem labels e sim vetor de ingredientes
    model.fit(train_data,
              y={'ingredients': ingredients},
              nb_epoch=nb_epoch, batch_size=batch_size)

    model.save_weights(top_model_weights_path)