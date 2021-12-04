import pickle


def convert_file_to_dict(file, name_new_file):
    '''
    :param file: Name of the file to convert into a dictionary
    :param name_new_file: Name of the new file where the dictionary is going to be stored
    :return: Void
    '''
    a_dictionary = {}

    a_file = open(file)

    for line in a_file:
        key, value = line.split()
        a_dictionary[key] = value
    new_file = open(name_new_file, "wb")
    pickle.dump(a_dictionary, new_file)
convert_file_to_dict("en_full.txt", "en_full_dict.txt")