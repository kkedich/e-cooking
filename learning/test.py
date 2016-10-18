import numpy as np
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import Flatten


N = 10
N_INGRADIENTS = 100
N_CATEGORIAS = 50

images = np.random.normal(size=[N, 32, 32, 3])
ingredientes = np.random.uniform(low=0, high=1, size=[N, N_INGRADIENTS])
categorias = np.random.uniform(low=0, high=1, size=[N, N_CATEGORIAS])

input_image = Input(shape=[32, 32, 3])

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
out_ingredientes = Dense(N_INGRADIENTS, activation='sigmoid', name='ingredientes')(x)


x = Dense(10)(out_conv)
x = Activation('relu')(x)
x = Dropout(0.25)(x)
out_categorias = Dense(N_CATEGORIAS, activation='softmax', name='categorias')(x)

model = Model(input=input_image, output=[out_ingredientes, out_categorias])

model.compile(loss={'ingredientes': 'binary_crossentropy',
                    'categorias': 'categorical_crossentropy'},
              optimizer='adam',
              metrics={'categorias': 'accuracy'})

model.fit(x=images, y={'ingredientes': ingredientes,
                       'categorias': categorias}, nb_epoch=100, batch_size=100)
