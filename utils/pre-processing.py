import myutils
import hashlib
import os


def remove_recipe(recipes_to_be_removed, json_data, path_data):
    if len(json_data) == 0:
        print 'Error: empty file.'
    else:
        if len(recipes_to_be_removed) == 0:
            print 'No recipe to be removed.'

        # Save the recipes to be removed in another file
        myutils.save_file(path_data + 'removed-recipes.txt', recipes_to_be_removed)
        # Create directory for the images that will be removed
        dir_images = path_data + 'images-removed/'

        for id_recipe in recipes_to_be_removed:
            print '\nRemoving recipe with id={}'.format(id_recipe)

            # Moves recipe image to folder 'images-removed'
            current_image = json_data[id_recipe]['file_image']
            print 'Moving image <{}> to <{}>'.format(path_data + 'images/' + current_image, dir_images + current_image)
            myutils.move(current_image, path_data + 'images/', dir_images)

            # Removes recipe from json file
            json_data.pop(id_recipe)


def remove_duplicates(file_name, path_data='../data/', folder='recipes-ctc'):
    ''' Remove recipes that have (image) duplicates in the dataset.'''

    path_images = path_data + folder + '/images/'
    path_json_file = path_data + folder + '/' + file_name
    path_output_json_file = path_data + folder + '/pre-processed-' + file_name

    data = myutils.load_json(path_json_file)
    recipes_to_be_removed = []
    ignore_list = []

    if len(data) == 0:
        print 'Error: empty file.'
    else:
        print 'Total of {} recipes'.format(len(data))

        # Compute all hashes first.
        recipes_hash = {}
        for recipe in data:
            current_file = open(path_images + data[recipe]['file_image']).read()

            hash_image = hashlib.md5(current_file).hexdigest()
            size_image = os.path.getsize(path_images + data[recipe]['file_image'])

            recipes_hash[recipe] = {}
            recipes_hash[recipe]['hash'] = hash_image
            recipes_hash[recipe]['size'] = size_image

        print 'All hashes were computed. :D'


        # Verifies if there are duplicates
        count = 0
        for dict_index in data:
            if dict_index in recipes_to_be_removed or dict_index in ignore_list:
                continue
            print '{} Checking: {}, URL: {}'.format(count, data[dict_index]['file_image'], data[dict_index]['url'])

            list_entries = []
            list_urls = []

            # Compares with all other recipes
            index_achieved = False
            for dict_index_search in data:
                # Iterates till dict_index is achieved
                while not index_achieved and dict_index_search != dict_index:
                    index_achieved = True
                    continue

                # Ignores same index
                if (dict_index_search == dict_index) or (dict_index_search in recipes_to_be_removed)\
                        or (dict_index_search in ignore_list):
                    continue

                # Ignore file with different sizes. Maybe we can delete this line,
                # since there is already the hash comparison
                if recipes_hash[dict_index]['size'] != recipes_hash[dict_index_search]['size']:
                    continue

                if recipes_hash[dict_index]['hash'] == recipes_hash[dict_index_search]['hash']:
                    print '--- Found duplicate: {}'.format(path_images + data[dict_index_search]['file_image'])

                    list_entries.append(dict_index_search)
                    list_urls.append(data[dict_index_search]['url'])

            count += 1
            if len(list_urls) == 0:
                continue

            # User determines which recipe delete
            for url in list_urls:
                print url

            user_input = raw_input("Which recipe do I remove? (None, ID or list of IDs separated by ,): ")
            print 'user input = ', user_input.split(',')

            if user_input.lower() == 'none':
                print 'No recipe will be removed'
                ignore_list.append(dict_index)
                for id_recipe in list_entries:
                    ignore_list.append(id_recipe)
            else:
                ids_to_be_removed = user_input.split(',')

                for id_recipe in ids_to_be_removed:
                    id_recipe = id_recipe.lstrip().rstrip()
                    recipes_to_be_removed.append(id_recipe)
                    print 'Included id={} to be removed'.format(id_recipe)


        # Remove recipes
        remove_recipe(recipes_to_be_removed, data, (path_data + folder + '/'))

        # Save the new json file without duplicates
        print 'Saving data...'
        myutils.save_json(path_output_json_file, data)

#
# def main():
#
#     # Remove duplicates
#     # remove_duplicates('pre-processed-v1-recipes-ctc.json', folder='recipes-ctc')
#
#     # Recipes fn
#     # remove_duplicates('recipes-fn.json', folder='recipes-fn')
#
#     # data = myutils.load_json('../data/recipes-fn/pre-processed-recipes-fn.json')
#     #
#     # list = ['200133', '232752' ]
#     # remove_recipe(list, data, '../data/recipes-fn/')
#     #
#     # myutils.save_json('../data/recipes-fn/final-pre-processed-recipes-fn.json', data)
#     # data = myutils.load_json('../data/recipes-fn/final-pre-processed-recipes-fn.json')
#     # print 'len = {}'.format(len(data))
#
# if __name__ == '__main__':
#     main()