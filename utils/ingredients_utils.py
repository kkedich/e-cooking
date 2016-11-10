import re
from nltk.stem.snowball import SnowballStemmer


def clean(text):
    """ Remove html tags and special characters from a text """
    # Clean html tags
    cleantext = clean_html(text.lower())

    # Clean special characters
    # cleantext = re.sub('[^A-Za-z0-9\.]+', ' ', cleantext)

    cleantext = clean_recipes_terms(cleantext)

    return cleantext


def clean_html(raw_html):
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


def stem_words(words):
    stemmer = SnowballStemmer("english")
    words[:] = [stemmer.stem(word) for word in words]

    return words
