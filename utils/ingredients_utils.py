import re
import htmlentitydefs

from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords


def clean(text):
    """ Remove html tags and special characters from a text """
    # Clean html tags
    cleantext = clean_html(text.lower())

    # Clean special characters. Also removes -, which can remove the ingredients composed name.
    # Example: extra-virgin will be extra virgin
    cleantext = re.sub('[^A-Za-z0-9]+', ' ', cleantext)  # If you do not want to remove .(dot), consider [^A-Za-z0-9\.]

    # Clean some recipe terms
    cleantext = clean_recipes_terms(cleantext)

    # TODO ainda sobra umas palavras de uma letra apenas, nao sei se isso eh erro no processo de limpeza anteriores
    # ou se eh algo que sobrou mesmo das palavras

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

    ingredient = re.sub("yes", "", ingredient)
    ingredient = re.sub("no", "", ingredient)

    return ingredient


def stem_words(words):
    stemmer = SnowballStemmer("english")
    words[:] = [stemmer.stem(word) for word in words]

    return words


def remove_stop_words(words):
    filtered_words = [word for word in words if word not in stopwords.words('english')]

    return filtered_words
