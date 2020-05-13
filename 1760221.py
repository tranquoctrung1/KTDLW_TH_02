# import statement
import os
import re
import ntpath
from bs4 import BeautifulSoup
import nltk
from string import punctuation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from nltk.tokenize import sent_tokenize
from nltk.stem import PorterStemmer
import tkinter as tk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import spatial

# set stopwords
my_stopwords = set(stopwords.words('english') + list(punctuation))

# excuted function
def get_text(file):
    read_file = open(file, "r")
    text = read_file.readlines()
    text  =  ' '.join(text);
    return text

def clear_html(text):
    soup = BeautifulSoup(text, 'html.parser')
    return soup.get_text()

def remove_special_character(text):
    string = re.sub('[^\w\s]', '',text)
    string = re.sub('\s+', ' ', string)
    string = string.strip()
    return string

# write output file to output directory
def write_file(string_file, content):
    dir_name_file = './output/' + string_file

    os.makedirs(os.path.dirname(dir_name_file), exist_ok=True)
    with open(dir_name_file, "w") as f:
        f.write(content)

def appendListPath(path):
    for root, dirs, files in os.walk(path):
        for file in files:
            list_path.append(root+"/"+file)

def choice_inputDirectory():
    root = tk.Tk()
    root.directory = tk.filedialog.askdirectory(title = "Choose input directory (double click to choose)")
    root.destroy()
    return root.directory


# pre_handlers function to handle text
def pre_handlers(list_path):

    list_words = []

    for i in range(len(list_path)):
        text = get_text(list_path[i])
        text_cleared = clear_html(text)

        sents = sent_tokenize(text_cleared)
        sents_cleared = [remove_special_character(s) for s in sents]
        text_sents_join = ' '.join(sents_cleared)

        words = word_tokenize(text_sents_join)

        words = [word.lower() for word in words]

        words = [word for word in words if word not in my_stopwords]

        ps = PorterStemmer()
        words = [ps.stem(word) for word in words]

        list_words.append(words)

    return list_words

# convert array text to string
def listToString(list_words):
    string_text = [];
    for i in range(len(list_words)):
        separate = ' '.join(list_words[i])
        string_text.append(separate)

    return string_text


# bag of Words function return matrix of todense
def bagOfWordsTodense(corpus):
    result = CountVectorizer()
    return result.fit_transform(corpus).todense()

# bag of Words function return dictionaries of todense
def bagOfWordVocabulary(corpus):
    result = CountVectorizer()
    result.fit_transform(corpus).todense()
    return result.vocabulary_

# tf-idf function
def tfIdf(corpus):
    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df= 0, stop_words='english')
    tf_idf_matrix = tf.fit_transform(corpus)
    # feature_name = tf.get_feature_names()
    return tf_idf_matrix.todense()

# distance cosin function
def distanceCosin(matrix, list_path):
    list = matrix.tolist()


    for i in range(0, len(list_path) -1 ):
        for j in range(i + 1, len(list_path)):
            result = 1 - spatial.distance.cosine(list[i], list[j])
            print(path_leaf(list_path[i]) + " to " + path_leaf(list_path[j]) + ": "+ str(result))


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def matrixToStringAndNameFile(matrix, list_path):
    string = ""
    stringMaTrixSplit = str(matrix).replace('[','').replace(']','').replace('\n','/').split('/')

    # format content on output file
    for i in range(len(list_path)):
        string += path_leaf(list_path[i]) + ": " + stringMaTrixSplit[i * 2] + stringMaTrixSplit[i *2 + 1] + "\n"

    return string

def matrixToStringOfTFIDF(matrix):
    return str(matrix).replace('[', '').replace(']', '\n -').replace('\n', ' ')

def dictionariesToString(dictionaries):
    string = str(dictionaries)

    return string



# init global variables
list_path = []


# main function
def main():
    print("Choose input folder\n")
    inputFolderPath =  choice_inputDirectory()
    appendListPath(inputFolderPath)
   
    wordsPreHandeler = pre_handlers(list_path)
    corpus =  listToString(wordsPreHandeler)  

    print("Enter number to choose: 1/ Bag of Words\t\t 2/TF-IDF\t\t 3/Distance Cosin\t\t 0/ Exit\n");
    choose = int(input("Choose: "))

    if(choose == 1):
        matrix =  bagOfWordsTodense(corpus)
        stringResultOfMatrix =  matrixToStringAndNameFile(matrix, list_path)
        dictionaries = bagOfWordVocabulary(corpus)
        stringResultOfDictionaries = dictionariesToString(dictionaries)


        result = stringResultOfMatrix + "\n" + stringResultOfDictionaries 

        write_file('BoW.txt', result)
    elif(choose == 2):
        matrix = tfIdf(corpus)
        stringResultOfMatrix = matrixToStringOfTFIDF(matrix)

        write_file('TF-IDF_CosSin.txt', stringResultOfMatrix)
    elif (choose == 3):
        matrix = tfIdf(corpus)

        distanceCosin(matrix, list_path)
    else:
        print("Close program!")
        exit()

main()