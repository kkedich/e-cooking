import myutils
import numpy as np
import math
import random

from keras.preprocessing.image import load_img, img_to_array
from keras.applications import vgg16
from keras import backend as K


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


def split_data(json_file, data_dir, images_dir='images',
               train=0.9, revert=False):
    """ Split the dataset into train and test
       train: float value from 0 to 1 (test will be 1.0 - train = test) specifying the amount of data for training
       and test
       revert: if True merge the folders 'train' and 'test' of images_path.
    """
    random.seed(100)  # Random number

    ids, data = recipes_ids(data_dir + json_file)

    images_path = data_dir + images_dir + '/'
    train_path = data_dir + 'train/'
    test_path = data_dir + 'test/'

    if (revert):
        print 'TODO Reverting...'
    else:
        if myutils.directory_exists(train_path) or myutils.directory_exists(test_path):
            print 'Train or/and test folder already there. Returning...'
            return train_path, test_path

        data_train = {}
        data_test = {}

        size_dataset = len(data)
        samples_train = int(math.ceil(train * size_dataset))
        samples_test = size_dataset - samples_train

        # Shuffle data to get random order of recipes
        random.shuffle(ids)

        # Get first samples for training and the rest for test
        for index in range(0, samples_train):
            id_recipe = ids[index]

            data_train[id_recipe] = data[id_recipe]
            data.pop(id_recipe)  # Removes the recipe

        data_test = data

        print 'Split data: {} for training (request={}) and {} for test (request={})'.format(len(data_train),
               samples_train, len(data_test), samples_test)

        myutils.save_json(data_dir + 'train.json', data_train)
        myutils.save_json(data_dir + 'test.json', data_test)

        print 'Copying image files...'
        copy_images(images_path, train_path, data_train)
        copy_images(images_path, test_path, data_test)

        return train_path, test_path, data_train, data_test


def ingredients_vector(recipe_ingredients, ingredients_list,
                       random=False, nb_ingredients=100, size=1):
    """ Returns a vector of size len(ingredients_list) that indicate the presence or not of each
        ingredient of ingredients_list in recipe_ingredients.
        recipe_ingredients: ingredients of the recipe
        ingredients_list: all ingredients used to create our vector (all possible ingredients considering all the
        dataset)
        random: if True returns a random ingredient vector of size N"""
    if random:
        return np.random.random_integers(low=0, high=1, size=(size, nb_ingredients))
    else:
        # TODO quando terminar pre-processamento ingredients
        our_ing_vector = np.zeros(len(ingredients_list), dtype=np.uint8)

        return our_ing_vector


# Util function to open, resize and format pictures into appropriate tensors
# (from an example in the keras documentation)
def preprocess_image(image_path, img_height=224, img_width=224):
    img = load_img(image_path, target_size=(img_height, img_width))  # target_size=(img_nrows, img_ncols)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg16.preprocess_input(img)
    return img


def load(data, dir_images, img_height, img_width):
    """ Load images from dir (directory) and ingredients of all recipes in data.
        Return a tensor combining all tensor for each image, and a numpy list of ingredients"""
    list_images = []
    list_of_all_ingredients = []  # TODO carregar quando terminar pre-processamento ingredientes

    # input_ingredients = np.zeros((len(data), len(list_of_all_ingredients)), dtype=np.uint8)  correto depois dos TODOs
    input_ingredients = np.zeros((len(data), 100), dtype=np.uint8)

    index = 0
    for id_recipe in data:
        # Get tensor representations of our images
        image = data[id_recipe]['file_image']
        current_image = K.variable(preprocess_image(dir_images + image, img_height, img_width))
        list_images.append(current_image)

        # Get ingredients_input
        list_ingredients = data[id_recipe]['ingredients']
        input_ingredients[index, :] = ingredients_vector(list_ingredients, list_of_all_ingredients, random=True)  #TODO

        index += 1

    # combine the 3 images into a single Keras tensor
    # input_tensor = K.concatenate([base_image, style_reference_image], axis=0)
    input_tensor = K.concatenate(list_images, axis=0)

    return input_tensor, input_ingredients


def main():
    train_path, test_path, data_train, data_test = split_data('recipes-ctc.json', '../data/recipes-ctc/', train=0.8)

    load(data_train, train_path, 224, 224)

if __name__ == '__main__':
    main()
