import os
import myutils
import numpy as np
import math
import random


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
    train_path = data_dir  + 'train/'
    test_path = data_dir  + 'test/'

    if (revert):
       print 'TODO Reverting...'
    else:
        if myutils.directory_exists(train_path) or myutils.directory_exists(test_path):
            print 'Train or/and test folder already there. Returning...'
            return train_path, test_path

        data_train = {}
        data_test  = {}

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


def load(dir):
    """ Load images from dir (directory)"""

    return 1



def main():
    split_data('recipes-ctc.json', '../data/recipes-ctc/', train=0.8)


if __name__ == '__main__':
    main()
