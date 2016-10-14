from string import ascii_lowercase
import urllib2
import math


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


def main():
    # A to Z list of terms
    for letter in ascii_lowercase:
        final_letter = ('xyz' if letter == chr(ord('x')) else letter)
        terms_for_letter = get_terms(final_letter)

        # Save terms in a file
        file_name = './data/food_terms_{}.txt'.format(final_letter.upper())
        current_file = open(file_name, "w")
        current_file.write(len(terms_for_letter).__str__() + "\n")
        current_file.write("\n".join(terms_for_letter))

        if letter == chr(ord('x')):
            break


if __name__ == '__main__':
    main()
