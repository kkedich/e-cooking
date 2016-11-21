import numpy as np
import os

from learning.ingredients import classifier_from_little_data_script_2 as classifier2
from learning.ingredients import classifier_from_little_data_script_3 as classifier3
import learning.ingredients.constants as C
import utils.data as data
# from utils.data_analysis import absence_of_ingredients

from keras import backend as K
from keras.models import load_model
import keras.backend.tensorflow_backend as TB
# import tensorflow as tf


def predict(sample_data, model_file, dir_images, list_of_all_ingredients):
    """Predict the ingredients for the images in the directory dir_images.
       model: .h5 model of the net.
       dir_images: directory containing the images"""
    print 'Image dir: ', dir_images
    input_data, list_images = data.load_images(dir_images, img_height=C.IMG_HEIGHT, img_width=C.IMG_WIDTH)

    # Returns a compiled model identical to model.h5
    model = load_model(model_file)
    predictions = model.predict(input_data, verbose=1)
    print predictions   
    print 'len=', len(predictions)
    
    # Round predictions by threshold
    rounded = []
    for prediction in predictions:
        current_array = np.zeros(len(prediction), dtype=np.uint8)
        print 'Original prediction: ', prediction
    
        for index in range(0, len(prediction)):
            if prediction[index] >= C.MIN_VALUE:
                current_array[index] = 1  # the ingredient is there
            else:
                current_array[index] = 0
        rounded.append(current_array)
   
    for index in range(0, len(list_images)):
        ingredients = data.list_ingredients(rounded[index], list_of_all_ingredients)
        print '\nThe image: <{}> has the following ingredients:\n{}'.format(list_images[index], ingredients)
        print 'Prediction: ', predictions[index]

    for recipe in sample_data:
        ingredients = sample_data[recipe]['ingredients']
        ingredients_array = data.ingredients_vector(ingredients, list_of_all_ingredients)
        ingredients_gt = data.list_ingredients(ingredients_array[0], list_of_all_ingredients)
        print 'Image: ', sample_data[recipe]['file_image']
        print 'GT ingredients: \n{}'.format(ingredients_gt)
        print 'GT: ', ingredients_array


def predict_ingredients():

    sample_data, dir_sample = data.sample('pre-processed-full-recipes-dataset.json', './data/full-recipes-dataset/')
    list_of_all_ingredients = data.load_all_ingredients(file='./data/ingredients.txt')

    predict(sample_data, C.final_vgg16_model, dir_sample, list_of_all_ingredients)

# def evaluate TODO com a base de teste


def main():
    K.set_image_dim_ordering('th')
    override = False


    nb_epoch = 2  # 100
    validation_split = 0.05  # 10 % of train data for validation, the last % of the data is used for validation
    dropout = 0.5
    neurons_last_layer = 4096  # 256, 4096
    my_batch_size = 32
    custom_loss = 'weighted_binary_crossentropy'  # or None for binary_crossentropy


    # Predict ingredients for some sample (model already trained)
    # predict_ingredients()


    # Generate data for training and test
    # data.split_data('pre-processed-recipes-ctc.json', './data/recipes-ctc/', train=0.9)
    # train_path, test_path, data_train, data_test = data.split_data('pre-processed-full-recipes-dataset-v2.json', './data/full-recipes-dataset/', train=0.9)
    train_path, test_path, data_train, data_test = data.split_data('pre-processed-recipes-ctc.json', './data/recipes-ctc/', train=0.15)

    # Load images and ingredients array
    input_tensor, input_ingredients = data.load(data_train, train_path, img_width=C.IMG_WIDTH, img_height=C.IMG_HEIGHT,
                                                file_ingredients='./data/ingredients.txt')

    C.NB_INPUT, C.NB_INGREDIENTS = input_ingredients.shape
    print 'nb_input={}, nb_ingredients={}'.format(C.NB_INPUT, C.NB_INGREDIENTS)
    # mean_zeros = absence_of_ingredients(input_ingredients)


    # Define which gpu we are going to use
    # with TB.tf.device('/gpu:1'):
    if not os.path.exists(C.file_bottleneck_features_train) or override:
        classifier2.save_bottlebeck_features(C.file_bottleneck_features_train, C.file_bottleneck_features_validation,
                                             img_width=C.IMG_WIDTH, img_height=C.IMG_HEIGHT,
                                             input_data_train=input_tensor,
                                             batch_size=my_batch_size)

    if not os.path.exists(C.top_model_weights_path) or override:
        classifier2.train_top_model(C.file_bottleneck_features_train, C.file_bottleneck_features_validation,
                                    C.top_model_weights_path,
                                    nb_epoch=nb_epoch, batch_size=my_batch_size, validation_split=validation_split,
                                    ingredients=input_ingredients, dropout=dropout, neurons_last_layer=neurons_last_layer,
                                    custom_loss=custom_loss)
#
#     classifier3.fine_tuning(C.top_model_weights_path, final_vgg16_model=C.final_vgg16_model,
#                             img_width=C.IMG_WIDTH, img_height=C.IMG_HEIGHT,
#                             batch_size=my_batch_size, nb_epoch=nb_epoch,
#                             ingredients=input_ingredients,
#                             train_data=input_tensor, validation_split=validation_split,
#                             dropout=dropout, neurons_last_layer=neurons_last_layer,
#                             custom_loss=custom_loss)


if __name__ == '__main__':
    main()
