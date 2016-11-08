import json
import glob, os
import numpy as np
import re
import myutils
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from collections import OrderedDict
import sys

reload(sys)
sys.setdefaultencoding("utf-8")

def cleanhtml(raw_html):
  """ Remove html tags from a text """
  cleanr = re.compile('<.*?>')
  cleantext = re.sub(cleanr, '', raw_html)

  return cleantext

def clean_recipes_terms(ingredient):
  """ Remove common terms in ingredients of recipes. """
  ingredient = re.sub("\d+", "", ingredient)
  ingredient = re.sub("/", "", ingredient)
  ingredient = re.sub(",", "", ingredient)
  ingredient = re.sub("ml", "", ingredient)
  ingredient = re.sub("cups", "", ingredient)
  ingredient = re.sub("cup", "", ingredient)
  ingredient = re.sub("tablespoons", "", ingredient)
  ingredient = re.sub("tablespoon", "", ingredient)
  ingredient = re.sub("teaspoons", "", ingredient)
  ingredient = re.sub("teaspoon", "", ingredient)
  ingredient = re.sub("spoons", "", ingredient)
  ingredient = re.sub("spoon", "", ingredient)
  ingredient = re.sub("finely", "", ingredient)
  ingredient = re.sub("pounded", "", ingredient)
  ingredient = re.sub("pound", "", ingredient)
  ingredient = re.sub("diced", "", ingredient)
  ingredient = re.sub("minced", "", ingredient)
  ingredient = re.sub("freshly", "", ingredient)
  ingredient = re.sub("fresh", "", ingredient)
  ingredient = re.sub("handful", "", ingredient)
  ingredient = re.sub("chopped", "", ingredient)
  ingredient = re.sub("canned", "", ingredient)
  ingredient = re.sub("soft", "", ingredient)
  ingredient = re.sub("large", "", ingredient)
  ingredient = re.sub("bunch", "", ingredient)
  ingredient = re.sub("cut", "", ingredient)
  ingredient = re.sub("ounces", "", ingredient)
  ingredient = re.sub("ounce", "", ingredient)
  ingredient = re.sub("crumbled", "", ingredient)
  ingredient = re.sub("chilled", "", ingredient)
  ingredient = re.sub("sliced", "", ingredient)
  ingredient = re.sub("wedges", "", ingredient)
  ingredient = re.sub("wedge", "", ingredient)
  ingredient = re.sub("small", "", ingredient)
  ingredient = re.sub("generous", "", ingredient)
  ingredient = re.sub("finely", "", ingredient)
  ingredient = re.sub("stems", "", ingredient)
  ingredient = re.sub("removed", "", ingredient)
  ingredient = re.sub("serving", "", ingredient)
  ingredient = re.sub("spreading", "", ingredient)
  ingredient = re.sub("cubed", "", ingredient)
  ingredient = re.sub("to taste", "", ingredient)
  ingredient = re.sub("unsalted", "", ingredient)
  ingredient = re.sub("grated", "", ingredient)
  ingredient = re.sub("boneless", "", ingredient)
  ingredient = re.sub("skinless", "", ingredient)
  ingredient = re.sub("inch", "", ingredient)
  ingredient = re.sub("thick", "", ingredient)
  ingredient = re.sub("ground", "", ingredient)
  ingredient = re.sub("crushed", "", ingredient)
  ingredient = re.sub("package", "", ingredient)
  ingredient = re.sub("pack", "", ingredient)
  ingredient = re.sub("scraped", "", ingredient)
  ingredient = re.sub("medium", "", ingredient)
  ingredient = re.sub("all-purpose", "", ingredient)
  ingredient = re.sub("heads", "", ingredient)
  ingredient = re.sub("head", "", ingredient)

  return ingredient


def read_cached_ingredients_words():
    """ Returns a list with all words from all ingredients from all recipes of dataset """

    config = json.loads(open('pre-processed-full-recipes-dataset.json').read())

    words = ""
    ingr_word_list = []

    if not os.path.exists('words.txt'):
        with open("words.txt", 'w') as f:
            
            for recipe in config.keys():

                for ingredient in config[recipe]["ingredients"]:
                    ingredient = cleanhtml(ingredient) 
                    ingredient = clean_recipes_terms(ingredient)
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

    stemmer = SnowballStemmer("english")
    vectorizer = CountVectorizer(stop_words="english")

    ingr_word_list = read_cached_ingredients_words()

    filtered_words = [word for word in ingr_word_list if word not in stopwords.words('english')]
    filtered_words[:] = [stemmer.stem(word) for word in filtered_words]
    counts = Counter(filtered_words)

    cache_counts(counts)

    return counts


def main():
    count_words_of_ingredients_list()
    

if __name__ == '__main__':
    main()
