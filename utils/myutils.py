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


def merge_json_files(file1, file2, output_file):
    data1 = load_json(file1)
    data2 = load_json(file2)

    # Concatenate data2 to merge_data (data1)
    merge_data = {}
    for dict in data1:
        merge_data[dict] = data1[dict]

    for dict in data2:
        merge_data[dict] = data2[dict]

    save_json(output_file, merge_data)

    print 'File1={}, file2={}, merge file={}.'.format(len(data1), len(data2), len(merge_data))


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


def my_list_pictures(directory, ext='jpg|jpeg|bmp|png'):
    """My version of list_pictures of image.py from Keras. Keras function doesn't match with .JPG.
       Lower case added and re.match (search only at start) replaced by re.search"""
    sorted_list = sorted(os.listdir(directory))
    pictures = [os.path.join(directory, f) for f in sorted_list if os.path.isfile(os.path.join(directory, f)) and re.search('([\w]+\.(?:' + ext + '))', f.lower())]
    return pictures
