import json
import os.path
import shutil
import re


def save_json(filename, data):
    with open(filename, 'w') as outfile:
        json.dump(data, outfile)


def load_json(filename):
    data = {}
    with open(filename) as json_data:
        data = json.load(json_data)

    return data


def save_file(filename, data_list):
    output_file = open(filename, "w")
    output_file.write(len(data_list).__str__() + "\n")
    output_file.write("\n".join(data_list))
    output_file.close()


def load_file(filename):
    data_list = []

    output_file = open(filename, 'r')

    output_file.readline()  # Number of elements
    for line in output_file:
        data_list.append(line.split('\n')[0])

    output_file.close()

    return data_list


def save_number(filename, number):
    output_file = open(filename, "w")
    output_file.write(str(number))
    output_file.close()


def load_number(filename):
    number = -1

    if os.path.isfile(filename):
        output_file = open(filename, 'r')

        number = int(output_file.readline())
        output_file.close()

    return number


def directory_exists(name):
    directory = os.path.dirname(name)
    return os.path.exists(directory)


def create_directory(name):
    directory = os.path.dirname(name)
    if not os.path.exists(directory):
        os.makedirs(directory)


def move(file, source, destination):
    create_directory(destination)
    shutil.move(source + file, destination + file)


def copy(file, source, destination):
    create_directory(destination)
    shutil.copy(source + file, destination + file)

def cleanhtml(raw_html):
  cleanr = re.compile('<.*?>')
  cleantext = re.sub(cleanr, '', raw_html)

  return cleantext

def clean_recipes_terms(ingredient):
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

def my_list_pictures(directory, ext='jpg|jpeg|bmp|png'):
    """My version of list_pictures of image.py from Keras. Keras function doesn't match with .JPG.
       Lower case added."""
    return [os.path.join(directory, f) for f in sorted(os.listdir(directory))

            if os.path.isfile(os.path.join(directory, f)) and re.match('([\w]+\.(?:' + ext + '))', f.lower())]

