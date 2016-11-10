import numpy as np
from learning.ingredients import classifier_from_little_data_script_2 as classifier2
from learning.ingredients import classifier_from_little_data_script_3 as classifier3
import learning.ingredients.constants as C
# from utils import data
from keras import backend as K
from keras.models import load_model
import utils.data as data
import os


def predict(model_file, dir_images, list_of_all_ingredients):
    """Predict the ingredients for the images in the directory dir_images.
       model: .h5 model of the net.
       dir_images: directory containing the images"""
    input_data, list_images = data.load_images(dir_images, img_height=C.IMG_HEIGHT, img_width=C.IMG_WIDTH)

    # Returns a compiled model identical to model.h5
    model = load_model(model_file)
    predictions = model.predict(input_data)

    # Round predictions by threshold
    rounded = []
    for prediction in predictions:
        current_array = np.zeros(len(prediction), dtype=np.uint8)

        for index in range(0, len(prediction)):
            if prediction[index] >= C.MIN_VALUE:
                current_array[index] = 1  # the ingredient is there
            else:
                current_array[index] = 0
        rounded.append(current_array)

    for index in range(0, len(list_images)):
        ingredients = data.list_ingredients(rounded[index], list_of_all_ingredients)
        print '\nThe image: <{}> has the following ingredients:\n{}'.format(list_images[index], ingredients)


# def evaluate TODO com a base de teste


def main():
    K.set_image_dim_ordering('th')
    override = False

    # Generate data for training and test
    train_path, test_path, data_train, data_test = data.split_data('pre-processed-recipes-ctc.json', './data/recipes-ctc/', train=0.2)
    # Load images and ingredients array
    input_tensor, input_ingredients = data.load(data_train, train_path, img_width=C.IMG_WIDTH, img_height=C.IMG_HEIGHT,
                                                file_ingredients='./data/ingredients.txt')

    nb_epoch = 1
    validation_split = 0.1  # 10 % of train data for validation
    dropout = 0.5
    neurons_last_layer = 256 # 256, 4096


    # if not os.path.exists(C.file_bottleneck_features_train) or override:
    #     classifier2.save_bottlebeck_features(C.file_bottleneck_features_train, C.file_bottleneck_features_validation,
    #                                          img_width=C.IMG_WIDTH, img_height=C.IMG_HEIGHT,
    #                                          input_data_train=input_tensor,
    #                                          batch_size=C.BATCH_SIZE)
    #
    # if not os.path.exists(C.top_model_weights_path) or override:
    #     classifier2.train_top_model(C.file_bottleneck_features_train, C.file_bottleneck_features_validation,
    #                                 C.top_model_weights_path,
    #                                 nb_epoch=nb_epoch, batch_size=C.BATCH_SIZE, validation_split=validation_split,
    #                                 ingredients=input_ingredients, dropout=dropout, neurons_last_layer=neurons_last_layer)
    #
    #
    # classifier3.fine_tuning(C.top_model_weights_path, final_vgg16_model=C.final_vgg16_model,
    #                         img_width=C.IMG_WIDTH, img_height=C.IMG_HEIGHT,
    #                         batch_size=C.BATCH_SIZE, nb_epoch=nb_epoch,
    #                         ingredients=input_ingredients,
    #                         train_data=input_tensor, validation_split=validation_split,
    #                         dropout=dropout, neurons_last_layer=neurons_last_layer)


if __name__ == '__main__':
    main()