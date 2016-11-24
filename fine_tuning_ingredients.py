import numpy as np
np.random.seed(0)
import os

from learning.ingredients import classifier_from_little_data_script_2 as classifier2
from learning.ingredients import classifier_from_little_data_script_3 as classifier3
import learning.ingredients.constants as C
import utils.data as data
from utils.data_analysis import dist_samples_per_ingredient

from keras.models import load_model
# from keras import backend as K
# import keras.backend.tensorflow_backend as TB
# import tensorflow as tf


def predict_this(image_file, file_ingredients='./data/ingredients.txt'):
    """Predict the ingredients of an image with the last available model
       return: list of strings with the ingredients.
       Ex: predict_this('food.jpg')
           returns ['salt', 'oil', 'pepper', 'oliv']
       """
    print 'Generating ingredients for <{}>'.format(image_file)
    input_image = data.load_image(image_file, C.IMG_HEIGHT, C.IMG_WIDTH)
    # Dictionary of ingredients for the model
    list_of_all_ingredients = data.load_all_ingredients(file=file_ingredients)

    # Returns a compiled model identical to model.h5
    assert os.path.exists(C.final_vgg16_model), 'File for the model <{}> not found.'.format(C.final_vgg16_model)

    model = load_model(C.final_vgg16_model)
    prediction = model.predict(input_image)

    rounded_pred = np.zeros(prediction.shape[1], dtype=np.uint8)
    # print rounded_pred.shape
    # print prediction.shape
    print 'Original prediction: ', prediction
    for index in range(0, prediction.shape[1]):
        if prediction[0][index] >= C.MIN_VALUE:
            rounded_pred[index] = 1  # the ingredient is here
        else:
            rounded_pred[index] = 0

    ingredients = data.list_ingredients(rounded_pred, list_of_all_ingredients)
    print 'Ingredients:', ingredients

    return ingredients


def predict(sample_data, model_file, dir_images, list_of_all_ingredients):
    """Predict the ingredients for the images in the directory dir_images.
       model: .h5 model of the net.
       dir_images: directory containing the images"""
    print 'Image dir: ', dir_images
    input_data, list_images = data.load_images(dir_images, img_height=C.IMG_HEIGHT, img_width=C.IMG_WIDTH)

    # Returns a compiled model identical to model.h5
    assert os.path.exists(model_file), 'File for the model <{}> not found.'.format(model_file)

    model = load_model(model_file)
    predictions = model.predict(input_data, verbose=1)
    print predictions   
    print 'len=', len(predictions)
    
    # Round predictions by threshold
    # TODO checar isso, predict_this foi alterado porque tinha erro no indice do vetor
    rounded = []
    for prediction in predictions:
        current_array = np.zeros(len(prediction), dtype=np.uint8)
        print 'Original prediction: ', prediction
    
        for index in range(0, len(prediction)):
            if prediction[index] >= C.MIN_VALUE:
                current_array[index] = 1  # the ingredient is here
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
    """Predict ingredients for some sample (model already trained)"""
    sample_data, dir_sample = data.sample('pre-processed-full-recipes-dataset.json', './data/full-recipes-dataset/')
    list_of_all_ingredients = data.load_all_ingredients(file='./data/ingredients.txt')

    predict(sample_data, C.final_vgg16_model, dir_sample, list_of_all_ingredients)


def evaluate(data_test, path_images, model_file, output_file_accuracy='test_accuracy.npy'):
    """Evaluate ou final model with the test data.
       (only execute in the final model)"""
    print 'Evaluating test data...'
    input_images_test, input_ingredients_test = data.load(data_test, path_images,
                                                          img_width=C.IMG_WIDTH, img_height=C.IMG_HEIGHT,
                                                          file_ingredients='./data/ingredients.txt')
    # Returns a compiled model identical to model.h5
    assert os.path.exists(model_file), 'File for the model <{}> not found.'.format(model_file)

    model = load_model(model_file)
    score = model.evaluate(x=input_images_test, y=input_ingredients_test,
                           batch_size=32, verbose=1, sample_weight=None)

    np.save(open(output_file_accuracy, 'w'), score)

    print 'loss={}, accuracy={}'.format(score[0], score[1]*100)
    print score


def main():
    # K.set_image_dim_ordering('th')
    override = False
    evaluate_model = True

    # validation_split = 0.05  # 10 % of train data for validation, the last % of the data is used for validation
    nb_epoch = 100  # 100
    dropout = 0.5
    neurons_last_layer = 256  # 256, 4096
    my_batch_size = 32
    custom_loss = None #'weighted_binary_crossentropy'  # or None for binary_crossentropy

    file_dist_ingredients_dict = 'inverse_distribution_ingredients_dict.npy'
    file_dist_ingredients_array = 'inverse_distribution_ingredients_array.npy'


    # Generate data for training and test
    # # data.split_data('pre-processed-full-recipes-dataset-v2.json', './data/full-recipes-dataset/', train=0.9)
    # train_path, val_path, test_path, data_train, data_val, data_test  = data.split_data('pre-processed-recipes-ctc.json', './data/recipes-ctc/',
    #                                                                                     train=0.2, validation_split=0.1)
    train_path, val_path, test_path, data_train, data_val, data_test = data.split_data('pre-processed-full-recipes-dataset-v2.json',
                                                                                       './data/full-recipes-dataset/',
                                                                                        train=0.9, validation_split=0.1)


    # Load images and ingredients array. First for training and then for validation
    input_images_train, input_ingredients_train = data.load(data_train, train_path,
                                                            img_width=C.IMG_WIDTH, img_height=C.IMG_HEIGHT,
                                                            file_ingredients='./data/ingredients.txt')

    input_images_val, input_ingredients_val = data.load(data_val, val_path,
                                                        img_width=C.IMG_WIDTH, img_height=C.IMG_HEIGHT,
                                                        file_ingredients='./data/ingredients.txt')


    # Calculate the distribution of each ingredient in the data set for training. This distribution will be used
    # as a weight in the loss fuction, frequent ingredients will be assigned small weights.
    # https://github.com/fchollet/keras/pull/188
    ingredients_weight_dict = None
    ingredients_weight_array = None
    if not os.path.exists(file_dist_ingredients_dict) or override:
        ingredients_weight_dict, ingredients_weight_array = dist_samples_per_ingredient(data=data_train, file_ingredients='./data/ingredients.txt',
                                                         generate_figure=True, image_file='dist_ingredients_train.png')
        np.save(open(file_dist_ingredients_dict, 'w'), ingredients_weight_dict)
        np.save(open(file_dist_ingredients_array, 'w'), ingredients_weight_array)
    else:
        ingredients_weight_dict = np.load(open(file_dist_ingredients_dict))
        ingredients_weight_array = np.load(open(file_dist_ingredients_array))
        print 'Loaded file {}'.format(file_dist_ingredients_dict)
        print 'Loaded file {}'.format(file_dist_ingredients_array)
    print ingredients_weight_dict
    print ingredients_weight_array

    class_weight = None
    if custom_loss is None:
        class_weight = ingredients_weight_dict
    elif custom_loss == 'weighted_binary_crossentropy':
        class_weight = ingredients_weight_array


    # Define which gpu we are going to use
    # with TB.tf.device('/gpu:1'):
    if not os.path.exists(C.file_bottleneck_features_train) or override:
        classifier2.save_bottlebeck_features(C.file_bottleneck_features_train, C.file_bottleneck_features_validation,
                                             img_width=C.IMG_WIDTH, img_height=C.IMG_HEIGHT,
                                             input_data_train=input_images_train, input_data_validation=input_images_val,
                                             batch_size=my_batch_size)

    if not os.path.exists(C.top_model_weights_path) or override:
        classifier2.train_top_model(C.file_bottleneck_features_train, C.file_bottleneck_features_validation,
                                    C.top_model_weights_path,
                                    nb_epoch=nb_epoch, batch_size=my_batch_size, dropout=dropout,
                                    neurons_last_layer=neurons_last_layer,
                                    train_ingredients=input_ingredients_train, val_ingredients=input_ingredients_val,
                                    custom_loss=custom_loss,
                                    class_weight=class_weight)

    classifier3.fine_tuning(C.top_model_weights_path, final_vgg16_model=C.final_vgg16_model,
                            img_width=C.IMG_WIDTH, img_height=C.IMG_HEIGHT,
                            batch_size=my_batch_size, nb_epoch=nb_epoch,
                            train_ingredients=input_ingredients_train, val_ingredients=input_ingredients_val,
                            train_data=input_images_train, validation_data=input_images_val, # validation_split=validation_split,
                            class_weight=class_weight,
                            dropout=dropout, neurons_last_layer=neurons_last_layer,
                            custom_loss=custom_loss)


    # Evaluate test data with the final model
    if evaluate_model:
        assert os.path.exists(C.final_vgg16_model), 'File for the model <{}> not found.'.format(C.final_vgg16_model)
        evaluate(data_test, test_path, C.final_vgg16_model)


if __name__ == '__main__':
    main()
