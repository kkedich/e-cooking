
from utils import myutils
import hashlib


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
            print 'Moving image <{}> to <{}>'.format(path_data + 'images/'+ current_image, dir_images + current_image)
            myutils.move(current_image, path_data + 'images/', dir_images)

            # Removes recipe from json file
            json_data.pop(id_recipe)



def remove_duplicates(file_name, path_data='../data/', folder='recipes-ctc'):
    ''' Remove recipes that have (image) duplicates in the dataset.
    '''
    path_images = path_data + folder + '/images/'
    path_json_file = path_data + folder + '/' + file_name
    path_output_json_file = path_data + folder + '/pre-processed-' + file_name

    data = myutils.load_json(path_json_file)
    recipes_to_be_removed = []

    if len(data) == 0:
        print 'Error: empty file.'
    else:
        print 'Total of {} recipes'.format(len(data))

        count = 0
        for dict_index in data:
            if dict_index in recipes_to_be_removed:
                continue
            print '{} Checking: {}, URL: {}'.format(count, data[dict_index]['file_image'], data[dict_index]['url'])

            image_file = path_images + data[dict_index]['file_image']
            current_file = open(image_file).read()
            hash_image = hashlib.md5(current_file).hexdigest()

            list_entries = []
            list_urls = []

            index_achieved = False
            for dict_index_search in data:
                # Iterates till dict_index is achieved
                while (not index_achieved and dict_index_search != dict_index):
                    index_achieved = True
                    continue
                # Ignores same index
                if (dict_index_search == dict_index) or (dict_index_search in recipes_to_be_removed):
                    continue

                image_file_search = path_images + data[dict_index_search]['file_image']
                current_file_search = open(image_file_search).read()
                hash_image_search = hashlib.md5(current_file_search).hexdigest()

                if hash_image == hash_image_search:
                    print '--- Founded duplicate: {}'.format(image_file_search)

                    list_entries.append(dict_index_search)
                    list_urls.append(data[dict_index_search]['url'])

            count += 1

            # if count == 100:
            #     break
            if len(list_urls) == 0:
                continue

            # User determines which recipe delete
            for url in list_urls:
                print url, ' size=', len(url)

            user_input = raw_input("Which recipe do I remove? (None, ID or list of IDs separated by ,): ")
            print 'user input = ', user_input.split(',')

            if user_input.lower() == 'none':
                print 'No recipe will be removed'
            else:
                ids_to_be_removed = user_input.split(',')

                for id in ids_to_be_removed:
                    recipes_to_be_removed.append(id)
                    print 'Included id={} to be removed'.format(id)

        # Remove recipes
        remove_recipe(recipes_to_be_removed, data, (path_data + folder + '/'))

        # Save the new json file without duplicates
        print 'Saving data...'
        myutils.save_json(path_output_json_file, data)


def main():

    # Remove duplicates
    remove_duplicates('teste.json', folder='recipes-ctc')


if __name__ == '__main__':
    main()