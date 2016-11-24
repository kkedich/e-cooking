import numpy as np
np.random.seed(0)

import math
import random
from PIL import ImageFile

from keras import backend as K
from keras.applications import vgg16
from keras.preprocessing.image import load_img, img_to_array
from sklearn.feature_extraction.text import CountVectorizer

import ingredients_utils
import myutils

ImageFile.LOAD_TRUNCATED_IMAGES = True


def recipes_ids(json_file):
    """Get the ids of all recipes in a list.
       Format of Json file: each recipe is a dictionary."""
    data = myutils.load_json(json_file)

    ids = []
    for id_recipe in data:
        ids.append(id_recipe)

    return ids, data


def copy_images(source, destination, data):
    for id_recipe in data:
        image = data[id_recipe]['file_image']
        myutils.copy(image, source, destination)

    print 'Copy files ({}) from <{}> to <{}>: OK'.format(len(data), source, destination)


def sample(json_file, data_dir, number_samples=5, images_dir='images'):
    """Get sample: first <number_samples> of recipes dataset"""
    images_path = data_dir + images_dir + '/'
    new_images_path = data_dir + 'sample-images/'

    # Loading test and train data
    full_data = myutils.load_json(data_dir + json_file)

    # Get first recipes for the sample
    sample = {}
    count = 0
    for recipe in full_data:
        if count == number_samples:
            break

        sample[recipe] = full_data[recipe]
        count += 1

    print 'Sample: {} recipes'.format(count)
    myutils.save_json(data_dir + 'sample-{}.json'.format(str(count)), sample)

    print 'Copying image files...'
    copy_images(images_path, new_images_path, sample)

    return sample, new_images_path


def split_data(json_file, data_dir, images_dir='images',
               train=0.9, validation_split=0.1, revert=False):
    """ Split the dataset into train, validation and test
       train: float value from 0 to 1 (test will be 1.0 - train - validation = test) specifying the amount of data for training
       and test
       validation_split: float value from 0 to 1 specifying the amount of data from training for validation.
                         Example: train=0.9, validation_split=0.1
                                  test will be 0.1 of the total data and validation will be 0.1 of the train data.
       revert: if True merge the folders 'train' and 'test' of images_path.
    """
    random.seed(100)  # Random number

    ids, data = recipes_ids(data_dir + json_file)

    images_path = data_dir + images_dir + '/'
    train_path = data_dir + 'train/'
    test_path = data_dir + 'test/'
    val_path = data_dir + 'val/'

    if revert:
        print 'TODO Reverting...'
    else:
        if myutils.directory_exists(train_path) or myutils.directory_exists(test_path):
            print 'Train or/and test folder already there. Returning...'

            # Loading test and train data
            data_train = myutils.load_json(data_dir + 'train.json')
            data_test = myutils.load_json(data_dir + 'test.json')
            data_val = myutils.load_json(data_dir + 'validation.json')

            return train_path, val_path, test_path, data_train, data_val, data_test

        data_train = {}
        data_test = {}
        data_val = {}

        size_dataset = len(data)
        samples_train = int(math.ceil(train * size_dataset))
        samples_val = int(math.ceil(validation_split * samples_train))
        samples_train = samples_train - samples_val
        samples_test = size_dataset - samples_train - samples_val

        print 'Total dataset={}, train={}, val={}, test={}'.format(size_dataset, samples_train, samples_val, samples_test)

        # Shuffle data to get random order of recipes
        random.shuffle(ids)

        # Get first samples for training, then validation, and the rest for test
        for index in range(0, samples_train):
            id_recipe = ids[index]

            data_train[id_recipe] = data[id_recipe]
            data.pop(id_recipe)  # Removes the recipe

        # validation
        for index in range(samples_train, (samples_train + samples_val)):
            id_recipe = ids[index]

            data_val[id_recipe] = data[id_recipe]
            data.pop(id_recipe)  # Removes the recipe

        data_test = data

        print 'Split data: {} for training (request={}), {} for validation (request={}),' \
              ' and {} for test (request={})'.format(len(data_train), samples_train,
                                                     len(data_val), samples_val,
                                                     len(data_test), samples_test)

        myutils.save_json(data_dir + 'train.json', data_train)
        myutils.save_json(data_dir + 'test.json', data_test)
        myutils.save_json(data_dir + 'validation.json', data_val)

        # print 'Copying image files...'
        copy_images(images_path, train_path, data_train)
        copy_images(images_path, test_path, data_test)
        copy_images(images_path, val_path, data_val)

        return train_path, val_path, test_path, data_train, data_val, data_test


def list_ingredients(array, list_of_all_ingredients):
    """Returns a list of ingredients (names) in array.
       array: of 0s and 1s indicating the presence or not of a particular ingredient
       list_of_all_ingredients: list with all names of ingredients."""
    names = []
    for index in range(0, len(array)):
        if array[index] == 1:
            names.append(list_of_all_ingredients[index])

    return names


def ingredients_vector(current_ingredients, list_all_ingredients, random_values=False, nb_ingredients=100, size=1):
    """ Returns a vector of size len(ingredients_list) that indicate the presence or not of each
        ingredient of ingredients_list in recipe_ingredients.
        current_ingredients: ingredients of the recipe
        list_all_ingredients: all ingredients used to create our vector (all possible ingredients considering all the
        dataset)
        random: if True returns a random ingredient vector of size N"""
    if random_values:
        return np.random.randint(2, size=nb_ingredients)
    else:
        ingr_word_complete = []
        ingredient_content = ""

        for ingredient in current_ingredients:
            ingredient_item = ingredients_utils.clean(ingredient)
            ingredient_content = ingredient_content + ' ' + ingredient_item

            ingr_word_array = ingredient_content.split()
            ingr_word_array = ingredients_utils.remove_stop_words(ingr_word_array) # Stop words
            ingr_word_array = ingredients_utils.stem_words(ingr_word_array) # Stemming

            # TODO quando arrumar a lista de ingredientes, fazer o mesmo pre-processamento aqui

            ingredient_content = " ".join(ingr_word_array)

        ingr_word_complete.append(ingredient_content)

        cv = CountVectorizer(vocabulary=list_all_ingredients)
        count = cv.fit_transform(ingr_word_complete).toarray()

        for vector in count:
            vector[vector > 0] = 1

        return np.array(count)


def load_all_ingredients(file='../data/ingredients.txt'):

    with open(file, 'r') as f:
        ingredients_count = [line.rstrip('\n') for line in f]

    ingredients = []

    for i in ingredients_count:
        ingredients.append(i.split(';')[0])

    print 'Loaded {} ingredients to generate our vector.'.format(len(ingredients))
    return ingredients


# Util function to open, resize and format pictures into appropriate tensors
# (from an example in the keras documentation)
def preprocess_image(image_path, img_height=224, img_width=224):
    img = load_img(image_path, target_size=(img_height, img_width))  # target_size=(img_nrows, img_ncols)
    img = img_to_array(img)
    # img = np.expand_dims(img, axis=0)  # parameter for the function is an array
    # img = vgg16.preprocess_input(img)  # parameter for the function is an array
    return img


def preprocess_image_array(array_img):
    """Pre-processing for the VGG16 pre-trained model.
       First the data is zero-centered from the mean values of each color channel obtained from ImageNet.
       Then, the image is converted to BGR, since the pre-trained model is generated with BGR images.

       Use this function for training, validation, and test.
       Note: from keras imagenet_utils
    """
    # img = np.expand_dims(array_img, axis=0)  # parameter for the function is an array
    img = vgg16.preprocess_input(array_img)  # parameter for the function is an array

    return img


def load_images(dir_images, img_height, img_width):
    """Load only images in dir_images. Returns a Keras tensor combining all images."""
    images = myutils.my_list_pictures(dir_images)
    print 'Loading images ({}): {}'.format(len(images), images)

    input_images = None
    # Determine proper input shape
    if K.image_dim_ordering() == 'th':
        # input_shape = (3, img_width, img_height)
        input_images = np.zeros((len(images), 3, img_width, img_height), dtype=np.float32)
    else:
        # input_shape = (img_width, img_height, 3)
        input_images = np.zeros((len(images), img_width, img_height, 3), dtype=np.float32)

    index = 0
    for image in images:
        input_images[index,:,:,:] = preprocess_image(image, img_height, img_width)
        index += 1

    input_images = preprocess_image_array(input_images)
    print 'Shape after pre-process(vgg16): {}'.format(input_images.shape)

    return input_images, images


def load(data, dir_images, img_height, img_width, file_ingredients):
    """ Load images from dir (directory) and ingredients of all recipes in data.
        Return a tensor combining all tensor for each image, and a numpy list of ingredients"""
    print 'Loading ingredients dictionary...'
    list_of_all_ingredients = load_all_ingredients(file_ingredients)

    assert len(list_of_all_ingredients) > 0, 'Something is wrong, no data for the ingredients list: {}.'.format(len(list_of_all_ingredients))

    input_ingredients = np.zeros((len(data), len(list_of_all_ingredients)), dtype=np.float32)
    # input_ingredients = np.zeros((len(data), 100), dtype=np.uint8) # for random values

    input_images = None
    # Determine proper input shape
    if K.image_dim_ordering() == 'th':
        input_images = np.zeros((len(data), 3, img_width, img_height), dtype=np.float32) #(3, img_width, img_height)
    else:
        input_images = np.zeros((len(data), img_width, img_height, 3), dtype=np.float32) #(img_width, img_height, 3)

    print 'Loading data ({} images)...'.format(len(data))
    index = 0
    for id_recipe in data:
        image = data[id_recipe]['file_image']
        input_images[index,:,:,:] = preprocess_image(dir_images + image, img_height, img_width)

        # Get ingredients_input
        ingredients = data[id_recipe]['ingredients']
        input_ingredients[index, :] = ingredients_vector(ingredients, list_of_all_ingredients)

        index += 1

    input_images = preprocess_image_array(input_images)
    print 'Shape after pre-process(vgg16): {}'.format(input_images.shape)

    return input_images, input_ingredients


# def main():
#     # train_path, test_path, data_train, data_test = split_data('pre-processed-recipes-ctc.json', '../data/recipes-ctc/', train=0.8)
#     #
#     # load(data_train, train_path, 224, 224)
#     #
#     load_images('../data/img_teste', 224, 224)


      # list_of_all_ingredients = load_all_ingredients()
      # print len(list_of_all_ingredients)
      #
      # ingredients = ['appl', 'orang']
      # teste = ingredients_vector(ingredients, list_of_all_ingredients)
      # print 'len=', len(teste), '\n', teste
#
# if __name__ == '__main__':
#     main()
