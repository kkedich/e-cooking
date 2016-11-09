import json
import glob, os
import numpy as np
import myutils
import ingredients_utils
import data
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from collections import OrderedDict

def list_ingredients():
    list_ingredients = []

    config = json.loads(open('pre-processed-full-recipes-dataset.json').read())

    for recipe in config.keys():
        list_ingredients.append(config[recipe]["ingredients"])

    return list_ingredients


def read_cached_ingredients_words():
    """ Returns a list with all words from all ingredients from all recipes of dataset """

    config = json.loads(open('pre-processed-full-recipes-dataset.json').read())

    words = ""
    ingr_word_list = []

    if not os.path.exists('words.txt'):
        with open("words.txt", 'w') as f:
            
            for recipe in config.keys():

                for ingredient in config[recipe]["ingredients"]:
                    ingredient = ingredients_utils.cleanhtml(ingredient) 
                    ingredient = ingredients_utils.clean_recipes_terms(ingredient)
                    word_list = ingredient.split()
                    for w in word_list:
                        f.write(w + '\n')
                    
    with open("words.txt", 'r') as f:
        ingr_word_list = [line.rstrip('\n') for line in f]

    return ingr_word_list



def cache_counts(counts):
    """ Saves the values to a file for human analysis """

    with open("ingredients.txt", 'w') as f:
        for more_freq in counts.keys():
            if counts[more_freq] > 750:
                f.write(more_freq+';'+str(counts[more_freq])+'\n')


def count_words_of_ingredients_list():
    """ Counts the number of ocurrencies of each word in dataset, after removing the most
    common stopwords and performing stemming """

    vectorizer = CountVectorizer(stop_words="english")

    ingr_word_list = read_cached_ingredients_words()

    filtered_words = [word for word in ingr_word_list if word not in stopwords.words('english')]
    filtered_words[:] = ingredients_utils.stem_words(filtered_words)
    counts = Counter(filtered_words)

    cache_counts(counts)

    return counts


#def main():
#    recipe_ingredients = data.load_all_ingredients()
#    ingredients_list = list_ingredients()
#    data.ingredients_vector(recipe_ingredients, ingredients_list)
#    print("---end")
    

#if __name__ == '__main__':
    #main()
