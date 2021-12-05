import csv
import pickle

import numpy as np
from nltk.corpus import wordnet
import spacy
nlp = spacy.load("en_core_web_sm")
from nltk import pos_tag
def total_number_of_hyperonyms(token):
    '''
    :param token: expression to be evaluated
    :return: number of hyperonyms obtained from all the definitions of the token in wordnet
    '''
    definitions = wordnet.synsets(token)
    number_of_hypernyms = 0
    for definit in definitions:
        number_of_hypernyms = number_of_hypernyms + len(definit.hypernyms())
    return number_of_hypernyms
def obtain_number_hyperonyms_first_def(token):
    '''
    :param token: expression to be evaluated
    :return: number of hyperonyms obtained from the first definition of the token in wordnet
    '''
    if len(wordnet.synsets(token)) >= 1:
        first_def = wordnet.synsets(token)[0]
        return len(first_def.hypernyms())
    else:
        return 0
def total_number_of_hyponyms(token):
    '''
    :param token: expression to be evaluated
    :return: number of hyponyms obtained from all the definitions of the token in wordnet
    '''
    definitions = wordnet.synsets(token)
    number_of_hyponyms = 0
    for definit in definitions:
        number_of_hyponyms = number_of_hyponyms + len(definit.hyponyms())
    return number_of_hyponyms
def obtain_number_hyponyms_first_def(token):
    '''
    :param token: expression to be evaluated
    :return: number of hyponyms obtained from the first definition of the token in wordnet
    '''
    if len(wordnet.synsets(token)) >= 1:
        first_def = wordnet.synsets(token)[0]
        return len(first_def.hyponyms())
    else:
        return 0
def obtain_number_synsets(token):
    '''
    :param token: expression to be evaluated
    :return: number of synsets obtained from wordnet with the token
    '''
    return len(wordnet.synsets(token))
def obtain_subject_iobject_dobject(text):
    '''
    :param text: context sentence to analyze, it contains the token
    :return: 0 if the expression is the subject of the sentence
            1 if the expression is the indirect object of the sentence
            2 if the expression is the direct object of the sentence
            3 is the default returning value
    '''
    parsed_text = nlp(text)
    flag_subject = 0
    flag_iobject = 0
    flag_dobject = 0
    for text in parsed_text:
        if text.dep_ == "nsubj":
            flag_subject = 1
        if text.dep_ == "iobj":
            flag_iobject = 1
        if text.dep_ == "dobj":
            flag_dobject = 1
    return flag_subject + (flag_iobject * 2) + (flag_dobject * 3)
def obtain_subject_object(text, token):
    '''
    :param text: context sentence to analyze, it contains the token
    :param token: expression to be evaluated
    :return: 0 if the expression is the subject of the sentence
            1 if the expression is the indirect object of the sentence
            2 if the expression is the direct object of the sentence
            3 is the default returning value
    '''
    parsed_text = nlp(text)
    for text in parsed_text:
        if text.dep_ == "nsubj":
            if text.orth_ == token:
                return 0
        if text.dep_ == "iobj":
            if text.orth_ == token:
                return 1
        if text.dep_ == "dobj":
            if text.orth_ == token:
                return 2
    return 3
def syllable_count(word):
    '''
    :param word: the token which complexity is going to be evaluated
    :return: number of syllables of the token
    '''
    word = word.lower()
    count = 0
    vowels = "aeiouy"
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
    if word.endswith("e"):
        count -= 1
    if count == 0:
        count += 1
    return count
def obtain_features(path_to_file, file_to_save):
    '''
    :param path_to_file: path to the file where is the corpus to extract the features from
    :param file_to_save: path to the file where the features extracted are going to be saved
    :return: array of features of the sentences and array of the target value of the sentences
    '''
    tsv_file = open(path_to_file)
    read_tsv = csv.reader(tsv_file, delimiter="\t")
    a_file = open("en_full_dict.txt", "rb")
    dictionary_frequency_words = pickle.load(a_file)
    x_train = []
    y_train = []
    for row in read_tsv:
        features = []
        #length of the token
        features.append(len(row[3]))
        list = [x[1] for x in pos_tag(row[2].split())]
        #number of nouns in the sentence
        features.append(list.count('NN')+list.count('NNS')+list.count('NNP'))
        #number of verbs in the sentence
        features.append(list.count('VB') + list.count('VBP') + list.count('VBZ')+ list.count('VBG')+ list.count('VBN') + list.count('VBD'))
        #number of words of the sentence
        features.append(len(list))
        #number of syllables of the token
        features.append(syllable_count(row[3]))
        #analyze if the token is subject or object
        features.append(obtain_subject_object(row[2],row[3]))
        #analyze if the sentence has subject, direct object and indirect object
        features.append(obtain_subject_iobject_dobject(row[2]))#Está en el _2 de las features
        #obtain number of synsets in wordnet of the token
        features.append(obtain_number_synsets(row[3]))#Está en el _3 de las features
        #obtain number of hyponyms from the first definition of the token in wordnet
        features.append(obtain_number_hyponyms_first_def(row[3]))#Está en el _4 de las features
        #obtain number of hyponyms of all the definitions of the token in wordnet
        features.append(total_number_of_hyponyms(row[3]))#Está en el _4 de las features
        # obtain number of hyperonyms from the first definition of the token in wordnet
        features.append(obtain_number_hyperonyms_first_def(row[3]))  # Está en el _5 de las features
        # obtain number of hyperonyms of all the definitions of the token in wordnet
        features.append(total_number_of_hyperonyms(row[3]))  # Está en el _5 de las features
        #obtain frequency of occurrence of the token
        if row[3] in dictionary_frequency_words.keys():
            features.append(int(dictionary_frequency_words[row[3]]))
        else:
            features.append(0)
        if row[4] != 'complexity':
            y_train.append(float(row[4]))
        x_train.append(features)
    x_train = x_train[1:]
    np.savetxt(file_to_save+".txt", x_train, fmt='%d')
    np.savetxt(file_to_save+"_target.txt", y_train, fmt='%.20f')
    return x_train, y_train
obtain_features("lcp_single_test.tsv", "test_features_5")