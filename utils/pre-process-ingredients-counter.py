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


file_names = []

# get a list of all files
file_names = glob.glob("images/*.*")

# check if the base was already indexed
if not os.path.exists('recipes.txt'):
    with open("recipes.txt", 'w') as f:
        for s in file_names:
            s = s.split("--")[1]
            s = s.split(".")[0]
            f.write(s + '\n')

with open("recipes.txt", 'r') as f:
    recipes_code = [line.rstrip('\n') for line in f]

config = json.loads(open('pre-processed-recipes-ctc.json').read())

words = ""
ingr_word_list = []

if not os.path.exists('words.txt'):
    with open("words.txt", 'w') as f:
        
        for recipe in recipes_code:

            print(config[recipe]["name"])

            for ingredient in config[recipe]["ingredients"]:
                ingredient = myutils.cleanhtml(ingredient) 
                
                ingredient = myutils.clean_recipes_terms(ingredient)
                word_list = ingredient.split()
                for w in word_list:
                    f.write(w + '\n')
                
with open("words.txt", 'r') as f:
    ingr_word_list = [line.rstrip('\n') for line in f]

stemmer = SnowballStemmer("english")
vectorizer = CountVectorizer(stop_words="english")

filtered_words = [word for word in ingr_word_list if word not in stopwords.words('english')]
filtered_words[:] = [stemmer.stem(word) for word in filtered_words]
counts = Counter(filtered_words)

with open("ingredients.txt", 'w') as f:
    for more_freq in counts.keys():
        if counts[more_freq] > 150:
            f.write(more_freq+';'+str(counts[more_freq])+'\n')
