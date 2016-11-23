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
                                  file_words='../data/words-teste.txt'):
    """ Returns a list with all words from all ingredients from all recipes of the dataset."""
    print 'Reading ingredients of all recipes'
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
                        if len(w) == 1: # Removing words with just one letter.
                            continue
                        f.write(w + '\n')

    print 'Saving words...'
    with open(file_words, 'r') as f:
        ingr_word_list = [line.rstrip('\n') for line in f]

    return ingr_word_list


def cache_counts(counts, file_ingredients='../data/ingredients-teste.txt', frequency_threshold=30): #frequency_threshold=750
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
    print 'Full {} words.'.format(len(ingr_word_list))

    print 'Removing stop words.'
    filtered_words = ingredients_utils.remove_stop_words(ingr_word_list)
    print len(filtered_words)

    print 'Removing verbs and adverbs.'
    filtered_words = ingredients_utils.remove_speech_tags(filtered_words)
    print len(filtered_words)

    # filtered_words[:] = ingredients_utils.stem_words(filtered_words)
    print 'Running lemmatizer.'
    filtered_words = ingredients_utils.lemmatize(filtered_words)
    print len(filtered_words)


    counts = Counter(filtered_words)
    print len(counts)
    print counts
    cache_counts(counts)

    return counts


def main():
   # recipe_ingredients = data.load_all_ingredients()
   # ingredients_list = list_ingredients()
   # data.ingredients_vector(recipe_ingredients, ingredients_list)
   # print("---end")

   count_words_of_ingredients_list()

   # words = ['loving', 'sugar', 'onions', 'dying', 'driving', 'reduced']   # onions
   # words = ['fixing', 'tied', 'dozen']
   # # print words
   # words = ingredients_utils.lemmatize(words)
   # print words
   # words = ingredients_utils.remove_speech_tags(words)
   # print words
   #
   # # words = ingredients_utils.lemmatize(words, 'v')
   # # print words
   # # words = ingredients_utils.remove_verbs(words)
   # #
   # # words = ingredients_utils.stem_words(words)
   # words.sort()
   # print words
   # #


   # ingr_word_list = ['extra-virgin', 'oliv', 'oil', 'plus', 'for', 'the', 'pan', 's', 'flour', 'plus', 'for', 'dust',
   #                   'kosher', 'salt', 'dri', 'yeast', 'ice-cold', 'waterpizza']
   # filtered_words = ingredients_utils.remove_stop_words(ingr_word_list)
   #
   # print filtered_words

if __name__ == '__main__':
    main()
