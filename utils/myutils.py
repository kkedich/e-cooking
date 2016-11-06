
import json
import os.path
import shutil


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