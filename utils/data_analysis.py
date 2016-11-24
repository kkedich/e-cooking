import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from data import load_all_ingredients, ingredients_vector
from myutils import load_json


def dist_samples_per_ingredient(data, file_ingredients, json_file=None, values=None,
                                generate_figure=True, horizontal=True, percentage=True,
                                image_file='images-per-ingredient.png'):
    """Returns the inverse class frequencies (distribution) of the ingredients in the data set.

       Let number_ingredients be the number of entries in file_ingredients.
       The distribution obeys the ordering in the file <file_ingredients>.
       return dictionary with number_of_samples (or percentage) per_ingredient (number_ingredients, 1)
    """
    print 'Distribution of samples per ingredient...'

    list_of_all_ingredients = load_all_ingredients(file_ingredients)
    # x = np.arange(1, (len(list_of_all_ingredients))*2, 2)
    x_values = np.arange(1, len(list_of_all_ingredients) + 1)

    # Set the type of my list
    my_dtype = [('samples', np.float32), ('ingredient', 'S17')]
    result = np.zeros(len(list_of_all_ingredients), dtype=my_dtype)

    result['ingredient'] = list_of_all_ingredients   # Add list of ingredients

    if values is None:
        if len(data) == 0:
            print 'Loading data...'
            data = load_json(json_file)

        samples_per_ingredient = np.zeros(len(list_of_all_ingredients), dtype=np.float32)

        for id_recipe in data:
            ingredients = data[id_recipe]['ingredients']  # Get ingredients_input
            current = samples_per_ingredient
            new_sample = ingredients_vector(ingredients, list_of_all_ingredients)
            samples_per_ingredient = current + new_sample
        # print 'Samples per ingredient:\n', samples_per_ingredient
    else:
        samples_per_ingredient = values

    if percentage:
        samples_per_ingredient = samples_per_ingredient / len(data)

    result['samples'] = samples_per_ingredient  # Add number of samples per ingredient

    print 'Shape samples_per_ing={}, result.shape={}'.format(samples_per_ingredient.shape, result.shape)
    print 'Samples per ingredient:\n', result  #samples_per_ingredient

    if generate_figure:
        # Sort the list by the number of images
        result_sorted = np.sort(result, order='samples')
        # print result_sorted['samples']

        if horizontal:
            # horizontal bar
            plt.barh(x_values - 0.4, result_sorted['samples'], align='center', height=0.7)
            plt.title('Images per ingredient')
            plt.ylabel('ingredient')
            plt.xlabel('number of images')
            plt.yticks(x_values - 0.4, result_sorted['ingredient'])
            plt.grid(True)

            fig = matplotlib.pyplot.gcf()
            fig.set_size_inches(13.0, 16.5)
            plt.savefig(image_file)
        else:
            # vertical bar
            plt.bar(x_values-0.4, result_sorted['samples'], align='center', width=0.5)
            plt.title('Images per ingredient')
            plt.ylabel('number of images')
            plt.xlabel('ingredient')
            plt.xticks(x_values-0.4, result_sorted['ingredient'], rotation=90)

            fig = matplotlib.pyplot.gcf()
            fig.set_size_inches(18.5, 12.0)
            plt.savefig(image_file)


    # Define the dictionary used for the class-weight of keras. Mapping class indices (integers) to a weight (float)
    # shape (classes, weights)
    # high_weight = 100
    # +class_weight = {0:1,1:1,2:1,3:1,4:1,5:1,6:1,7:1,8:1,9:high_weight}
    # my_dtype_keras = [('indices', np.int32), ('weight', np.float32)]
    # result_for_keras = np.zeros(len(list_of_all_ingredients), dtype=my_dtype_keras)
    # result_for_keras['indices'] = np.arange(1, len(list_of_all_ingredients) + 1)
    # Inverse class frequencies
    # result_for_keras['weight'] = 1.0 - samples_per_ingredient  # Ordering of file_ingredients
    index = 0
    inverse_dist_ingredients = {}
    for index in range(0, len(list_of_all_ingredients)):
        inverse_dist_ingredients[index] = 1.0 - samples_per_ingredient[0][index]

    # result_for_keras['weight']
    return inverse_dist_ingredients


def fig_ingredients_per_recipe(json_file, file_ingredients, values=None, image_file='ingredients_per_recipe.png'):
    """Returns number of recipes(images) per number of ingredients.
       Example: we have 1000 recipes that have 2 ingredients...
    """
    list_of_all_ingredients = load_all_ingredients(file_ingredients)
    x_values = np.arange(1, len(list_of_all_ingredients) + 1)

    if values is None:
        print 'Loading data...'
        data = load_json(json_file)

        recipes_per_ingredients = np.zeros(len(list_of_all_ingredients), dtype=np.uint8)

        for id_recipe in data:
            ingredients = data[id_recipe]['ingredients']  # Get ingredients_input
            current_sum = np.sum(ingredients_vector(ingredients, list_of_all_ingredients))
            recipes_per_ingredients[current_sum] += 1

        print 'Number of recipes per number of ingredients:\n', recipes_per_ingredients
    else:
        recipes_per_ingredients = values

    plt.bar(x_values, recipes_per_ingredients, align='center', width=0.5)
    plt.title('Number of recipes(images) per number of ingredients')
    plt.ylabel('number of recipes')
    plt.xlabel('number of ingredients')
    plt.grid(True)
    # plt.xticks(x_values)

    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(18.5, 12.0)
    plt.savefig(image_file)

    return recipes_per_ingredients


def absence_of_ingredients(input_ingredients):
    """Returns the mean of the number of ingredients not present in the recipes.
       input_ingredients: numpy array (x,y), where x is the number of observations, and y the total number of
       ingredients in our dataset.
    """
    nb_recipes, nb_ingredients = input_ingredients.shape

    if nb_recipes == 0 or nb_ingredients == 0:
        print 'Empty array.'
        return 0.0

    index = 0
    mean = 0.0
    for i in range(nb_recipes):
        current_recipe = input_ingredients[i][:]

        non_zero = np.count_nonzero(current_recipe)
        zeros = nb_ingredients - non_zero
        mean += zeros

    # Calculate the mean
    mean = mean / nb_recipes
    print mean

    return mean


def main():

      file_recipes = '../data/full-recipes-dataset/pre-processed-full-recipes-dataset-v3-without-html.json'
      file_ingredients = '../data/ingredients.txt'

      samples = [3100, 5409, 5194, 70, 1567, 1179, 2751, 2306, 1231, 1057, 2140, 2774,
                                 12890, 2258, 28252, 2197, 1538, 2275, 1126, 2874, 2664, 994, 1604, 972,
                                 1143, 999, 3662, 776, 6946, 2141, 1344, 19750, 1892, 6053, 9536, 1649,
                                 21005, 1330, 1550, 1281, 13071, 3324, 3124, 1773, 2656, 1685, 13546, 997,
                                 2615, 2315, 3206, 11580, 929, 1023, 2363, 2343, 13081, 5053, 5732, 5610,
                                 1720, 2288, 1663, 2354, 10393, 70, 1843, 1184, 1825, 2244, 12997, 5699,
                                 1263, 995, 3442, 1088, 1170, 1609, 1237, 2072, 2034, 1184, 941, 2115,
                                 1456, 1364, 2691, 3539, 6537, 2115, 2649, 3430, 1084, 7713, 3411, 1188,
                                 1758, 1008, 1685, 2150]

      # fig_samples_per_ingredient(data=[], json_file=file_recipes, file_ingredients=file_ingredients, values=values)


      # Number of recipes per number of ingredients:
      values = [93, 151, 113, 129, 205,  72,   8,  89, 122, 238,  60,  66, 141,  98, 251,  41, 155, 125,
       125, 181,  57, 224, 151,  76,  51,  26,  27,  15,   4,   3,   3,   4,   1,   2,   0,   1,
       0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
       0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
       0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
       0,   0,   0,   0,   0,   0,   0,   0,   0,   0]

      fig_ingredients_per_recipe(json_file=file_recipes, file_ingredients=file_ingredients, values=values)


if __name__ == '__main__':
    main()