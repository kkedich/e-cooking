import os
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer

import ingredients_utils
import myutils


# def list_ingredients(file='../data/full-recipes-dataset/pre-processed-full-recipes-dataset.json'):
#     list_ingredients = []
#
#     config = myutils.load_json(file)
#
#     for recipe in config.keys():
#         list_ingredients.append(config[recipe]["ingredients"])
#
#     return list_ingredients


def read_cached_ingredients_words(file='../data/full-recipes-dataset/pre-processed-full-recipes-dataset.json',
                                  file_words='../data/words.txt'):
    """ Returns a list with all words from all ingredients from all recipes of the dataset."""
    config = myutils.load_json(file)

    ingr_word_list = []
    if not os.path.exists(file_words):
        with open(file_words, 'w') as f:
            
            for recipe in config.keys():

                for ingredient in config[recipe]["ingredients"]:
                    # ingredient = ingredients_utils.clean_html(ingredient)
                    ingredient = ingredients_utils.clean(ingredient)
                    ingredient = ingredients_utils.clean_recipes_terms(ingredient)

                    word_list = ingredient.split()
                    for w in word_list:
                        f.write(w + '\n')
                    
    with open(file_words, 'r') as f:
        ingr_word_list = [line.rstrip('\n') for line in f]

    return ingr_word_list


def cache_counts(counts, file_ingredients='../data/ingredients.txt', frequency_threshold=750):
    """ Saves the values to a file for human analysis """

    with open(file_ingredients, 'w') as f:
        for more_freq in counts.keys():
            if counts[more_freq] > frequency_threshold:
                f.write(more_freq + ';' + str(counts[more_freq]) + '\n')


def count_words_of_ingredients_list():
    """ Counts the number of occurrences of each word in dataset, after removing the most
    common stopwords and performing stemming """

    # vectorizer = CountVectorizer(stop_words="english")
    ingr_word_list = read_cached_ingredients_words()

    filtered_words = ingredients_utils.remove_stop_words(ingr_word_list)
    filtered_words[:] = ingredients_utils.stem_words(filtered_words)

    counts = Counter(filtered_words)
    print len(counts)
    print  counts
    # cache_counts(counts)

    return counts


# def main():
#    # recipe_ingredients = data.load_all_ingredients()
#    # ingredients_list = list_ingredients()
#    # data.ingredients_vector(recipe_ingredients, ingredients_list)
#    # print("---end")
#
#    count_words_of_ingredients_list()
#
#    # ingr_word_list = ['extra-virgin', 'oliv', 'oil', 'plus', 'for', 'the', 'pan', 's', 'flour', 'plus', 'for', 'dust',
#    #                   'kosher', 'salt', 'dri', 'yeast', 'ice-cold', 'waterpizza']
#    # filtered_words = ingredients_utils.remove_stop_words(ingr_word_list)
#    #
#    # print filtered_words
#
# if __name__ == '__main__':
#     main()
