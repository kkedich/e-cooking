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
N_INGREDIENTS = 100
N_CATEGORIES = 50

from keras import backend as K
K.set_image_dim_ordering('th')


images = np.random.normal(size=[N, 32, 32, 3])
ingredients = np.random.uniform(low=0, high=1, size=[N, N_INGREDIENTS])
categories = np.random.uniform(low=0, high=1, size=[N, N_CATEGORIES])

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
out_ingredients = Dense(N_INGREDIENTS, activation='sigmoid', name='ingredients')(x)


x = Dense(10)(out_conv)
x = Activation('relu')(x)
x = Dropout(0.25)(x)
out_categories = Dense(N_CATEGORIES, activation='softmax', name='categories')(x)

model = Model(input=input_image, output=[out_ingredients, out_categories])

model.compile(loss={'ingredients': 'binary_crossentropy',
                    'categories': 'categorical_crossentropy'},
              optimizer='adam',
              metrics={'categories': 'accuracy'})

model.fit(x=images, y={'ingredients': ingredients,
                       'categories': categories}, nb_epoch=100, batch_size=100)
