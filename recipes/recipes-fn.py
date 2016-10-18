#!/usr/bin/bash
#-*- coding: utf-8 -*-
"""
Get recipes with the ingredients list, description and image.
"""

import math
import requests
import time
from string import ascii_lowercase

from utils import myutils as utils

PATH_IMAGES = './data/recipes-fn/images/'
PATH_RECIPES = './data/recipes-fn/'
NAME_IMAGE_PREFIX = 'FN-'
ID_STARTER = 200000
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
        # response = requests.get(url)
        return []
    elif response.status_code == 404:
        print '[ERROR] Page not found'
        return []

    if len(response.content) == 0:
        print '[ERROR] Content not found'
        return []

    html = response.content
    html = html.decode('latin1')
    # print 'tamanho html antes do replace=',  len(html)
    html = html.replace('\r\n', ' ')
    # print 'tamanho html depois do replace=', len(html)
    lines = html.split('\n')
    # print len(lines)

    return lines


def find(lines, what_to_find):
    return filter(lambda line: line.find(what_to_find) != -1, lines)


def save_image(url, output_file):
    response = requests.get(url)
    # response.status_code

    output = open(output_file, "wb")
    output.write(response.content)
    output.close()


def get_recipes_from(link, urls):
    prefix = 'http://www.foodnetwork.com'

    content = get_content_of(link)
    lines_recipes = find(content, 'span class="arrow"')
    ids = map(get_term_id, lines_recipes)

    # Ignore last line '/recipes/photos/fall-desserts-for-dinner-parties.html'
    for i in range(0, len(ids)-1):
        id_recipe = prefix + ids[i]
        urls.append(id_recipe)
        # print id_recipe


def get_list_recipes(letter, list_recipes):
    # http://www.foodnetwork.com/recipes/a-z.123.1.html
    letter = letter.upper()
    first_link = 'http://www.foodnetwork.com/recipes/a-z.{}.1.html'.format(letter)

    content_first_link = get_content_of(first_link)
    line_number_recipes = find(content_first_link, 'entries beginning with')

    print line_number_recipes
    number_recipes = line_number_recipes[0].split(' ')[4]  # number of entries (position 6)
    print number_recipes

    number_pages = int(math.ceil(int(number_recipes)/ENTRIES_PER_PAGE))

    # Get the first page and iterate over the other pages
    get_recipes_from(first_link, list_recipes)

    for page in range(2, number_pages + 1):
        current_page = 'http://www.foodnetwork.com/recipes/a-z.{}.{}.html'.format(letter, str(page))
        get_recipes_from(current_page, list_recipes)

        if page % 10 == 0:
            time.sleep(5)

    print 'Size list of recipes =', len(list_recipes)


# TODO tem receitas repetidas, deixei tudo ai, para depois pre-processar isso.
def get_recipes_data(output_dataset_file, output_notfound_file, file_last_recipe, list_recipes):
    recipes_data = {}
    recipes_not_found = []

    tmp_json = PATH_RECIPES + 'temporary-recipes-fn.json'
    tmp_notfound = PATH_RECIPES + 'temporary-NOT_FOUND-recipes.txt'

    # Define start_by by the last recipe saved
    start_by = utils.load_number(file_last_recipe) + 1
    print 'start_by=', start_by

    index_recipe = ID_STARTER + start_by

    # Load recipes data from the last point saved
    if start_by > 0:
        recipes_data = utils.load_json(tmp_json)  # Recipes data
        recipes_not_found = utils.load_file(tmp_notfound)  # Not found

        index_recipe -= len(recipes_not_found)
        print u'\nLoaded data. Recipes = {} and Recipes_NF = {}. ID_STARTER = {}\n'.format(len(recipes_data), len(recipes_not_found), index_recipe)


    for index in range(start_by, len(list_recipes)):
        url_recipe = list_recipes[index]

        content_recipe = get_content_of(url_recipe)

        if len(content_recipe) == 0:
            recipes_not_found.append(url_recipe)
            continue

        lines = find(content_recipe, 'h1 itemprop="name"')
        title = lines[0].split('>')[1].split('<')[0]
        # Used for the image and video search (eliminating ascii characters TM)
        tmp_title = title.encode('ascii', 'xmlcharrefreplace').replace('&#226;&#132;&#162;', '&trade;')
        title = title.encode("ascii", "ignore")

        print u'Downloading information about the recipe <{}> ID-{} index={}'.format(tmp_title, index_recipe, index)

        # Image of current recipe. Always in position 2. First position gets an inferior (resolution) image.
        url_image = None
        line_image = find(content_recipe, 'title="{}" itemprop="image"'.format(tmp_title))
        if len(line_image) > 0:
            url_image = line_image[1].split('src="')[1].split('"')[0]
        else:
            # Test if there is a video.
            line_image_video = find(content_recipe, 'alt="Recipe Video" itemprop="video"'.format(tmp_title))

            if len(line_image_video) > 0:
                url_image = line_image_video[0].split('img src="')[1].split('" width=')[0]
            else:
                print '[ERROR] Image NOT FOUND for <{}>\n'.format(tmp_title)
                recipes_not_found.append(url_recipe)
                continue


        # Ingredients
        ingredients_rawdata = find(content_recipe, 'li itemprop="ingredients"')
        if len(ingredients_rawdata) == 0:
            print '[ERROR] Ingredients NOT FOUND for <{}>\n'.format(title)
            recipes_not_found.append(url_recipe)
            continue
        ingredients = [ingredient.split('>')[1].split('<')[0] for ingredient in ingredients_rawdata]

        list_ingredients = []
        for element in ingredients:
            list_ingredients.append(element)

        # Directions/description
        directions_rawdata = find(content_recipe, 'ul class="recipe-directions-list"')
        directions_rawdata = [directions.lstrip() for directions in directions_rawdata]
        # print 'directions=', len(directions_rawdata), ' ', directions_rawdata

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

        # if index % 100 == 0:
        #     print 'Sleeping\n'
        #     time.sleep(10)
        if index % 100 == 0 or index % 50 == 0 or index % 60 == 0:
            print 'Sleeping 14\n'
            time.sleep(14)
        elif index % 5 == 0:
            print 'Sleeping 2\n'
            time.sleep(2)

        # if index == start_by + 1:
        #     break

    # Saves all recipes and broken urls
    utils.save_json(output_dataset_file, recipes_data)
    utils.save_file(output_notfound_file, recipes_not_found)

    return recipes_not_found, recipes_data


def list_recipes_urls(load=False):
    list_recipes = []
    file_name = PATH_RECIPES + 'list_of_recipes.txt'

    if load:
        list_recipes = utils.load_file(file_name)
        print 'Number of recipes loaded=', len(list_recipes)

    else:
        # Get recipes from 123, and A to Z
        first_set = '123'  # http://www.foodnetwork.com/recipes/a-z.123.1.html
        get_list_recipes(first_set, list_recipes)

        for letter in ascii_lowercase:
            final_letter = ('xyz' if letter == chr(ord('x')) else letter)
            # final_letter = 'D'
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


def main():
    # Get list of recipes
    recipes_urls = list_recipes_urls(True)

    # Get the data of all recipes
    output_dataset = PATH_RECIPES + 'recipes-fn.json'
    output_notfound = PATH_RECIPES + 'NOT_FOUND-recipes.txt'

    file_last_recipe = PATH_RECIPES + 'last-index-recipe-saved.txt'
    recipes_notfound, recipes_data = get_recipes_data(output_dataset, output_notfound, file_last_recipe, recipes_urls)

    print 'Total recipes not found =', len(recipes_notfound)
    print 'Total recipes retrieved =', len(recipes_data)

    # URL(s) that generated problems, but were already corrected.
    # url_recipe = 'http://www.foodnetwork.com/recipes/melissa-darabian/affogato-recipe2.html' another type of image
    # url_recipe = 'http://www.foodnetwork.com/recipes/alaskan-sushi-recipe.html' redirect
    # http://www.foodnetwork.com/recipes/jeff-mauro/1-smore-for-the-road-and-kiddie-smores-recipe.html
    # http://www.foodnetwork.com/recipes/michael-chiarello/melissa-darabian/affogato-recipe.html without valid link
    # http://www.foodnetwork.com/recipes/ina-garten/herb-mesa/agave-kettle-corn-recipe.html without content
    # http://www.foodnetwork.com/recipes/adam-and-eve-on-a-raftpoached-eggs-with-roasted-tomatoes-mushrooms-and-ham-recipe.html
    # http: // www.foodnetwork.com / recipes / boars - head - ichiban - teriyakiandtrade - style - chicken - and -avocado - bites - on - rice - crackers.html



if __name__ == '__main__':
    main()
