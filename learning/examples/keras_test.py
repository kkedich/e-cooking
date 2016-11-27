import numpy as np
np.random.seed(0)

from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import Flatten


# categorical_accuracy from keras.metrics
def acc3(y_true, y_pred):
    return K.mean(K.equal(K.argmax(y_true, axis=-1),
                  K.argmax(y_pred, axis=-1)))

# binary_accuracy from keras.metrics
def acc2(y_true, y_pred):
    return K.mean(K.equal(y_true, K.round(y_pred)))


def binary_crossentropy(y_true, y_pred):
      return K.mean(K.binary_crossentropy(y_pred, y_true), axis=-1)

#minha funcao, repare que e a mesma coisa
def my_binary_crossentropy(y_true, y_pred):
      return K.mean(K.binary_crossentropy(y_pred, y_true), axis=-1)


# if K.image_dim_ordering() == 'th':
#       input_shape = (3, img_width, img_height)
# else:
      # input_shape = (img_width, img_height, 3)

N = 10
N_INGREDIENTS = 20
images = np.random.normal(size=[N, 3, 32, 32])
ingredientes = np.random.randint(low=0, high=2, size=[N, N_INGREDIENTS])

input_image = Input(shape=[3, 32, 32])


x = Convolution2D(20, 3, 3)(input_image)
x = Activation('relu')(x)
x = Convolution2D(20, 3, 3)(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.25)(x)
out_conv = Flatten()(x)


x = Dense(10)(out_conv)
x = Activation('relu')(x)
x = Dropout(0.25)(x)
out_ingredientes = Dense(N_INGREDIENTS, activation='sigmoid', name='ingredientes')(x)


model = Model(input=input_image, output=out_ingredientes)

# model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.compile(loss=my_binary_crossentropy, optimizer='sgd', metrics=['accuracy', acc2, acc3])

model.fit(x=images, y=ingredientes, nb_epoch=1, batch_size=32)