import re
import htmlentitydefs

from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
# from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
# from nltk.corpus import wordnet as wn
from nltk import pos_tag


def clean(text):
    """ Remove html tags and special characters from a text """
    # Clean html tags
    cleantext = clean_html(text.lower())

    # Clean special characters. Also removes -, which can remove the ingredients composed name.
    # Example: extra-virgin will be extra virgin
    cleantext = re.sub('[^A-Za-z0-9]+', ' ', cleantext)  # If you do not want to remove .(dot), consider [^A-Za-z0-9\.]

    # Clean some recipe terms
    cleantext = clean_recipes_terms(cleantext)

    return cleantext


def clean_htmlentities(text):
    """Removes HTML or XML character references and entities from a text string.
       From: http://stackoverflow.com/questions/57708/convert-xml-html-entities-into-unicode-string-in-python
       @param text The HTML (or XML) source text.
       @return The plain text, as a Unicode string, if necessary.
    """
    def fixup(m):
        text = m.group(0)
        if text[:2] == "&#":
            # character reference
            try:
                if text[:3] == "&#x":
                    return unichr(int(text[3:-1], 16))
                else:
                    return unichr(int(text[2:-1]))
            except ValueError:
                pass
        else:
            # named entity
            try:
                text = unichr(htmlentitydefs.name2codepoint[text[1:-1]])
            except KeyError:
                pass
        return text  # leave as is

    return re.sub("&#?\w+;", fixup, text)


def clean_html(raw_html):
    """ Remove html tags from a text """
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)

    cleantext = clean_htmlentities(cleantext)

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
    ingredient = re.sub("scoop", "", ingredient)
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
    ingredient = re.sub("recipe", "", ingredient)
    ingredient = re.sub("cooking", "", ingredient)


    # other terms found in the recipes
    ingredient = re.sub("juice", "", ingredient)
    ingredient = re.sub("cocktail", "", ingredient)
    ingredient = re.sub("vegan", "", ingredient)
    ingredient = re.sub("veggie", "", ingredient)
    ingredient = re.sub("lukewarm", "", ingredient)
    ingredient = re.sub("soaked", "", ingredient)
    ingredient = re.sub("betty", "", ingredient)
    ingredient = re.sub("gluten", "", ingredient)
    ingredient = re.sub("pale", "", ingredient)
    ingredient = re.sub("tequila", "", ingredient)
    ingredient = re.sub("dice", "", ingredient)
    ingredient = re.sub("stock", "", ingredient)
    ingredient = re.sub("piece", "", ingredient)
    ingredient = re.sub("dry", "", ingredient)
    ingredient = re.sub("cow", "", ingredient)
    ingredient = re.sub("crumbles", "", ingredient)
    ingredient = re.sub("meal", "", ingredient)
    ingredient = re.sub("winter", "", ingredient)
    ingredient = re.sub("farmer", "", ingredient)
    ingredient = re.sub("dessert", "", ingredient)
    ingredient = re.sub("candied", "", ingredient)
    ingredient = re.sub("campari", "", ingredient)
    ingredient = re.sub("blanco", "", ingredient)
    ingredient = re.sub("hershey", "", ingredient)
    ingredient = re.sub("confit", "", ingredient)
    ingredient = re.sub("may", "", ingredient)

    return ingredient


def remove_words_from_size(words, size):
    """Remove words of smaller sizes than parameter <size>"""
    assert size > 0, 'Size must be: size > 0'

    filtered_words = [word for word in words if len(word) > size]
    return filtered_words


def stem_words(words):
    stemmer = SnowballStemmer("english")
    words[:] = [stemmer.stem(word) for word in words]

    return words


def remove_stop_words(words):
    filtered_words = [word for word in words if word not in stopwords.words('english')]

    return filtered_words


def lemmatize(words, pos='n'):
    """Lemmatize verbs and nouns.
       Update: lemmatize of verbs is not done, since in the forward steps the words are considered nouns.
       Example: loving ->> lemmatizer ->> love. Love is a noun.
       ADJ, ADJ_SAT, ADV, NOUN, VERB = 'a', 's', 'r', 'n', 'v'"""
    wnl = WordNetLemmatizer()
    filtered_words = [wnl.lemmatize(word, pos) for word in words]  # lemmatize nouns
    # words[:] = [wnl.lemmatize(word, 'v') for word in words]  # lemmatize verbs

    return filtered_words


# http://stackoverflow.com/questions/25534214/nltk-wordnet-lemmatizer-shouldnt-it-lemmatize-all-inflections-of-a-word
def is_noun(tag):
    return tag in ['NN', 'NNS', 'NNP', 'NNPS']


def is_verb(tag):
    return tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']


def is_adverb(tag):
    return tag in ['RB', 'RBR', 'RBS']


def is_adjective(tag):
    return tag in ['JJ', 'JJR', 'JJS']


def remove_speech_tags(words):
    """Remove verbs and adverbs
       Note: some verbs still pass this pre-processing.
    """
    words_with_tags = pos_tag(words)
    # print words_with_tags

    filtered_words = []
    for current_word in words_with_tags:
        # if current_word[0] == "reduced":
        #     print 'Problema aqui <{}> tag: <{}>'.format(current_word[0], current_word[1])

        if not is_verb(current_word[1]) and not is_adverb(current_word[1]):
            filtered_words.append(current_word[0])

    # [current for current in words_with_tags if current[1] != 'NN']
    # print filtered_words
    return filtered_words


# def remove_adverbs(words):
#     words_with_tags = pos_tag(words)
#
#     filtered_words = []
#     for current_word in words_with_tags:
#         if not is_adverb(current_word[1]):
#             filtered_words.append(current_word[0])
#
#     return filtered_words

# def penn_to_wn(tag):
#     if is_adjective(tag):
#         return wn.ADJ
#     elif is_noun(tag):
#         return wn.NOUN
#     elif is_adverb(tag):
#         return wn.ADV
#     elif is_verb(tag):
#         return wn.VERB
#     return None
