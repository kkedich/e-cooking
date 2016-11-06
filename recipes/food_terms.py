import math
import urllib2
from string import ascii_lowercase
from utils import myutils as utils


def get_term_id(line):
    assert isinstance(line, unicode)
    return line.split('href="')[1].split('"')[0]


def get_terms_from(link, terms):
    prefix = 'http://www.foodterms.com'
    what_to_find = 'class="arrow"'

    response = urllib2.urlopen(link)
    html = response.read()
    html = html.decode('latin1')

    lines = html.split('\n')
    lines_terms = filter(lambda line: line.find(what_to_find) != -1, lines)
    ids = map(get_term_id, lines_terms)

    for id_term in ids:
        parts_term = id_term.split('/')
        term_and_link = parts_term[2] + ' ' + prefix + id_term
        print term_and_link

        terms.append(term_and_link)


def get_terms(letter):
    first_link = 'http://www.foodterms.com/encyclopedia/{}/index.html'.format(letter)
    print first_link

    # Note 150 entries per page
    entries_per_page = 150.0
    entries = 'p class="result-count"'

    response = urllib2.urlopen(first_link)
    html = response.read()
    html = html.decode('latin1')

    lines = html.split('\n')
    line_number_terms = filter(lambda line: line.find(entries) != -1, lines)

    print line_number_terms
    number_terms = line_number_terms[0].split(' ')[6]  # number of entries (position 6)
    print number_terms

    terms = []  # list of terms

    # get the first page and iterate over the other pages
    get_terms_from(first_link, terms)

    if number_terms >= entries_per_page:
        number_pages = int(math.ceil(int(number_terms)/entries_per_page))

        for page in range(2, number_pages + 1):
            suffix = '' + str(page) + ',00.html'
            current_page = 'http://www.foodterms.com/ftrm/encyclopedia/index/0,1001382,FTRM_44547_347419_{}-'.format(letter.upper()) + suffix
            # print current_page

            get_terms_from(current_page, terms)

    print len(terms)
    return terms


def generate_list(load=False):
    if load:
        # Load files with food terms and generate one file with all terms.
        # Each letter has a specific file.
        final_file_name = './data/food-terms/all_food_terms_FN.json'
        food_terms = {}
        id_starter = 0

        for letter in ascii_lowercase:
            final_letter = ('xyz' if letter == chr(ord('x')) else letter)

            # Read terms
            file_name = './data/food-terms/food_terms_{}.txt'.format(final_letter.upper())
            current_file = open(file_name, 'r')

            number_terms = current_file.readline()  # Number of elements
            print 'Reading {} terms - {}'.format(int(number_terms), final_letter)
            for line in current_file:
                # print line.replace('\n', '')
                # First position is the term itself, second position is the url that explains
                line = line.replace('\n', '').split(' ')

                food_terms[id_starter] = {}
                food_terms[id_starter]['_id'] = id_starter
                food_terms[id_starter]['name'] = line[0]
                food_terms[id_starter]['url'] = line[1]
                id_starter += 1

            current_file.close()

            if letter == chr(ord('x')):
                break

        print 'Total terms founded =', len(food_terms)
        # Save all terms loaded
        utils.save_json(final_file_name, food_terms)
    else:
        # Generate list of terms: A to Z
        for letter in ascii_lowercase:
            final_letter = ('xyz' if letter == chr(ord('x')) else letter)
            terms_for_letter = get_terms(final_letter)

            # Save terms
            file_name = './data/food-terms/food_terms_{}.txt'.format(final_letter.upper())
            current_file = open(file_name, "w")
            current_file.write(len(terms_for_letter).__str__() + "\n")
            current_file.write("\n".join(terms_for_letter))
            current_file.close()

            if letter == chr(ord('x')):
                break


def main():

    # Generate list of (food) terms
    generate_list(load=True)


if __name__ == '__main__':
    main()
