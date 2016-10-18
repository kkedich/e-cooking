#!/usr/bin/bash
#-*- coding: utf-8 -*-
"""
Get recipes with the ingredients list, description and image from Cooking TV Channel (CTC)
"""

import math
import requests
import time
from string import ascii_lowercase

from utils import myutils as utils

PATH_IMAGES = './data/recipes-ctc/images/'
PATH_RECIPES = './data/recipes-ctc/'
NAME_IMAGE_PREFIX = 'CTC-'
ID_STARTER = 300000
ENTRIES_PER_PAGE = 150.0  # Note: 150 entries per page


def get_term_id(line):
    assert isinstance(line, unicode)
    return line.split('href="')[1].split('"')[0]


def get_content_of(url):
    print 'Get content of: {}'.format(url)

    response = requests.get(url, allow_redirects=False)
    if response.status_code == 302:
        print '[ERROR] Redirection'
        return []
    elif response.status_code == 301:
        print '[ERROR] Redirection, but there is another url for the recipe which probably is in the list already.'
        return []
    elif response.status_code == 404:
        print '[ERROR] Page not found'
        return []

    if len(response.content) == 0:
        print '[ERROR] Content not found'
        return []

    html = response.content
    html = html.decode('latin1')
    html = html.replace('\r\n', ' ')
    lines = html.split('\n')
    print 'Get content of: {}: OK'.format(url)

    return lines


def find(lines, what_to_find, startswith=False):
    if startswith:
        return filter(lambda line: line.startswith(what_to_find) != False, lines)
    else:
        return filter(lambda line: line.find(what_to_find) != -1, lines)


def find_between(lines, first_match, second_match):
    # second = False
    first = False
    result = []

    for line in lines:
        if first:
            if line.find(second_match) != -1:  # Second match ok, end list
                break
            else:
                result.append(line.lstrip().replace('\r', ' '))

        if not first and line.find(first_match) != -1:  #First match ok
            first = True

    return result


def save_image(url, output_file):
    response = requests.get(url)

    output = open(output_file, "wb")
    output.write(response.content)
    output.close()


def get_recipes_from(link, urls):
    prefix = 'http://www.cookingchanneltv.com'

    content = get_content_of(link)

    spaces_24 = '                        '  # 24 spaces for the first column
    spaces_16 = '                '  # 16 spaces for the second column
    lines_recipes_col1 = find(content, '{}<li><a href="/recipes/'.format(spaces_24), startswith=True)
    lines_recipes_col2 = find(content, '{}<li><a href="/recipes/'.format(spaces_16), startswith=True)

    lines_recipes = lines_recipes_col1 + lines_recipes_col2
    print len(lines_recipes)

    ids = map(get_term_id, lines_recipes)

    for i in range(0, len(ids)):
        id_recipe = prefix + ids[i]
        urls.append(id_recipe)
        # print id_recipe


def get_list_recipes(letter, list_recipes):
    letter = letter.upper()
    first_link = 'http://www.cookingchanneltv.com/recipes/a-z.{}.0.html'.format(letter)

    content_first_link = get_content_of(first_link)
    line_number_recipes = find(content_first_link, 'recipes beginning with')

    print line_number_recipes
    number_recipes = line_number_recipes[0].lstrip().split(' ')[5]  # number of entries (position 5)
    print number_recipes

    number_pages = int(math.ceil(int(number_recipes)/ENTRIES_PER_PAGE))
    print 'Number pages=', number_pages

    # Get the first page and iterate over the other pages
    get_recipes_from(first_link, list_recipes)

    for page in range(1, number_pages):
        current_page = 'http://www.cookingchanneltv.com/recipes/a-z.{}.{}.html'.format(letter, str(page))
        get_recipes_from(current_page, list_recipes)

        if page % 10 == 0:
            time.sleep(5)

    print 'Size list of recipes =', len(list_recipes)


def list_recipes_urls(load=False):
    list_recipes = []
    file_name = PATH_RECIPES + 'list_of_recipes.txt'

    if load:
        list_recipes = utils.load_file(file_name)
        print 'Number of recipes loaded=', len(list_recipes)

    else:
        # Get recipes from 123, and A to Z
        first_set = '123'
        get_list_recipes(first_set, list_recipes)

        for letter in ascii_lowercase:
            final_letter = ('xyz' if letter == chr(ord('x')) else letter)
            print u'Downloading list of recipes of <{}>'.format(final_letter)

            get_list_recipes(final_letter, list_recipes)

            if letter == chr(ord('x')):
                break

            # Save list of recipes (till current letter)
            tmp_file_name = PATH_RECIPES + final_letter.upper() + '_list_of_recipes.txt'
            utils.save_file(tmp_file_name, list_recipes)

            time.sleep(10)

        # Save full list of recipes
        utils.save_file(file_name, list_recipes)

    return list_recipes


# TODO tem receitas repetidas, deixei tudo ai, para depois pre-processar isso.
def get_recipes_data(output_dataset_file, output_notfound_file, file_last_recipe, list_recipes):
    recipes_data = {}
    recipes_not_found = []

    prefix = 'http://www.cookingchanneltv.com'
    tmp_json = PATH_RECIPES + 'temporary-recipes-ctc.json'
    tmp_notfound = PATH_RECIPES + 'temporary-NOT_FOUND-recipes-ctc.txt'

    # Define start_by by the last recipe saved
    start_by = utils.load_number(file_last_recipe) + 1
    print 'start_by=', start_by

    index_recipe = ID_STARTER + start_by

    # Load recipes data from the last point saved
    if start_by > 0:
        recipes_data = utils.load_json(tmp_json)  # Recipes data
        recipes_not_found = utils.load_file(tmp_notfound)  # Not found

        index_recipe -= len(recipes_not_found)
        print u'\nLoaded data. Recipes = {} and Recipes_CTC = {}. ID_STARTER = {}\n'.format(len(recipes_data), len(recipes_not_found), index_recipe)

    for index in range(start_by, len(list_recipes)):
        url_recipe = list_recipes[index]
        # url_recipe = 'http://www.cookingchanneltv.com/recipes/adult-shamrock-mint-shake-the-irish-pirate.html' #list_recipes[index]
        # http://www.cookingchanneltv.com/recipes/adult-shamrock-mint-shake-the-irish-pirate.html

        content_recipe = get_content_of(url_recipe)

        if len(content_recipe) == 0:
            recipes_not_found.append(url_recipe)
            continue

        # Check if the recipe is an food network recipe or not
        is_from_fn = find(content_recipe, 'Originally from FoodNetwork.com')
        # print is_from_fn
        if len(is_from_fn) > 0:
            print '[ERROR] Recipe from FN.com\n'
            recipes_not_found.append(url_recipe)
            continue

        lines = find(content_recipe, '<span class="rTitle fn">')
        title = lines[0].split('>')[1].split('<')[0]
        # print title

        print u'Downloading information about the recipe <{}> ID-{} index={}'.format(title, index_recipe, index)

        # Image of current recipe.
        url_image = None
        line_image = find(content_recipe, 'class="img-enlarge"')
        if len(line_image) > 0:
            url_image = prefix + line_image[0].split('href="')[1].split('"')[0]
        else:
            # Test if there is a video: without example
            # Test if there is only a small image 266x354
            line_image = find(content_recipe, 'class="photo"')
            if len(line_image) > 0:
                url_image = line_image[0].split('src="')[1].split('"')[0]
            else:
                print '[ERROR] Image NOT FOUND for <{}>\n'.format(title)
                recipes_not_found.append(url_recipe)
                continue

        # Ingredients
        ingredients_rawdata = find(content_recipe, 'li class="ingredient"')
        if len(ingredients_rawdata) == 0:
            print '[ERROR] Ingredients NOT FOUND for <{}>\n'.format(title)
            recipes_not_found.append(url_recipe)
            continue

        ingredients = ingredients_rawdata[0].split('<li class="ingredient">')

        list_ingredients = []
        for i in range(1, len(ingredients)):  # First position is ignored (not ingredient)
            element = ingredients[i]
            element = element.replace('</li>', '').replace('</ul>','').replace('</span>','').replace('<ul class="col2">', '')
            list_ingredients.append(element)

        # Directions/description
        directions_rawdata = find_between(content_recipe, 'itemprop="recipeInstructions"', '</div>')
        if len(directions_rawdata) == 0:
            directions_rawdata = ''  # Add recipe even without directions, since for now this information is not required
        else:
            directions_rawdata = directions_rawdata[0]
        # print directions_rawdata

        # Save the data. Image and Ingredients: OK.
        image_file = NAME_IMAGE_PREFIX + '-' + str(index_recipe) + '.jpeg'
        path_image_file = PATH_IMAGES + image_file
        save_image(url_image, path_image_file)

        recipes_data[index_recipe] = {}  # url_recipe   recipes_data[url_recipe] = {}
        recipes_data[index_recipe]['_id'] = index_recipe
        recipes_data[index_recipe]['name'] = title
        recipes_data[index_recipe]['ingredients'] = list_ingredients
        recipes_data[index_recipe]['url'] = url_recipe
        recipes_data[index_recipe]['image'] = url_image
        recipes_data[index_recipe]['file_image'] = image_file
        recipes_data[index_recipe]['description'] = directions_rawdata

        print u'Downloaded recipe <{}> ID-{} index={}: OK\n'.format(title, index_recipe, index)
        index_recipe += 1

        # Saves data at each time when one recipe is retrieved
        utils.save_json(tmp_json, recipes_data)
        utils.save_file(tmp_notfound, recipes_not_found)
        utils.save_number(file_last_recipe, index)

        if index % 100 == 0 or index % 50 == 0 or index % 60 == 0:
            print 'Sleeping 20\n'
            time.sleep(20)
        elif index % 5 == 0:
            print 'Sleeping 5\n'
            time.sleep(5)

        # if index == start_by + 1:
        #     break

    # Saves all recipes and broken urls
    utils.save_json(output_dataset_file, recipes_data)
    utils.save_file(output_notfound_file, recipes_not_found)

    return recipes_not_found, recipes_data


def main():
    # Get list of recipes
    recipes_urls = list_recipes_urls(True)

    # Get the data of all recipes
    output_dataset = PATH_RECIPES + 'recipes-ctc.json'
    output_notfound = PATH_RECIPES + 'NOT_FOUND-recipes-ctc.txt'

    file_last_recipe = PATH_RECIPES + 'last-index-recipe-saved-ctc.txt'
    recipes_notfound, recipes_data = get_recipes_data(output_dataset, output_notfound, file_last_recipe, recipes_urls)

    print 'Total recipes not found =', len(recipes_notfound)
    print 'Total recipes retrieved =', len(recipes_data)


if __name__ == '__main__':
    main()
