#Author: Claudio Moises Valiense de Andrade. Licence: MIT. Objective: Create general purpose library
import timeit  # calcular metrica de tempo
from datetime import datetime  # Datetime for time in file
import nltk
import random
import scipy.stats as stats # Calcular intervalo de confiança
import unidecode # remove accents
import numpy as np
import numpy  # Manipular arrau numpy
from scipy.stats import norm
from pyjarowinkler import distance # similarity string per caracters
from sklearn.metrics import confusion_matrix
from sklearn import svm  # Classifier SVN
import collections # quantifica elementos em listas e conjuntos
import io
import gensim
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score
import json  # Manipulate extension json
import operator # intevalo de confianca teste nao parametrico
from sklearn.metrics import precision_score
import sklearn.metrics
import os  # Variable in system
import re # Regular expression
import ast # String in dict

def load_dict_file(file):
    """Load dict in file. Example: load_dict_file('myfile.json')"""
    try:
        with open(file, 'r', newline='') as csv_reader:
            return json.load(csv_reader)
    except OSError:
        return {}

def reorder_id(x, y, ids_original, ids_new):
    """Reorder id after shuffle"""
    x_ok = []; y_ok = []
    for id_ok in ids_original:
        position_new=0
        for new_id in ids_new:
            if id_ok == new_id:
                x_ok.append(x[position_new])
                y_ok.append(y[position_new])
                break
            position_new+=1
    return x_ok, y_ok

def clean_text2(texts):
    texts_lem = []
    from nltk.stem import WordNetLemmatizer
    #stop_words = set(stopwords.words("english")) 
    lemmatizer = WordNetLemmatizer() 
    for index in range(len(texts)): 
        text = texts[index]
        #text = re.sub(r'[^\w\s]','',text, re.UNICODE)
        #text = text.lower()
        text = word_tokenize(text)
        #text = [lemmatizer.lemmatize(token) for token in text]
        text = [lemmatizer.lemmatize(token, "v") for token in text]
        #text = [word for word in text if not word in stop_words]
        text = " ".join(text)
        texts_lem.append(text)
    return texts_lem


def my_vocab(x_tokens):
    """Create vocabulary"""
    vocab = dict() 
    vocab['unk'] = 0 # token desconhecido
    cont=1
    for tokens in x_tokens:
        for t in tokens:
            if vocab.get(t) == None: 
                vocab[t] = cont
                cont+=1
    return vocab

def my_stop_word(text, language="english"):
    """ Stop word in text. Example: my_stop_word(['the orange in book', 'hello house'], 'english'); Return=['orange book', 'hello house'] """
    #import nltk
    #nltk.download('punkt')
    #stemmer = SnowballStemmer("english") #stemmer  
    for index in range(len(text)):       
        stop_words = set(stopwords.words(language))         
        #stop_words = set(stopwords.words('portuguese')) 
        word_tokens = text[index].split(" ")
        #word_tokens = word_tokenize(text[index]) #melhor resulto                
        
        filtered_sentence = [] 
        for w in word_tokens: 
            if w not in stop_words: 
                filtered_sentence.append(w)
        #filtered_sentence = [stemmer.stem(t) for t in filtered_sentence] #stemmer
        text[index] = " ".join(filtered_sentence)
    return text 

def file_to_corpus(name_file):
    """Transforma as linahs de um arquivo em uma lista. Example: function(my_file.txt) """
    rows = []
    #with open(name_file, 'r') as read:
    #, encoding = "ISO-8859-1"
    with io.open(name_file, newline='\n', errors='ignore') as read: #erro ignore caracteres
        for row in read:
            row = row.replace("\n", "")
            rows.append(row)
    return rows


def ids_train_test_shuffle(ids_file, datas, labels, id_fold):
    """Return data and labels starting of ids file. Example: ids_train_test('ids.txt', 'texts.txt', 'labels.txt', 0); Return=x_train, y_train, x_test, y_test"""
    ids = file_to_corpus(ids_file)
    train_test = str(ids[id_fold]).split(';')
    ids_train = [int(id) for id in train_test[0].strip().split(' ')]
    ids_test = [int(id) for id in train_test[1].strip().split(' ')]
    new_ids_train = ids_train.copy(); new_ids_test = ids_test.copy()
    random.shuffle(new_ids_train); random.shuffle(new_ids_test)
    total = file_to_corpus(datas)
    labels = file_to_corpus(labels)
    x_train = [total[index] for index in new_ids_train]       
    y_train = [int(labels[index]) for index in new_ids_train] 
    x_test = [total[index] for index in new_ids_test]
    y_test = [int(labels[index]) for index in new_ids_test] 
    return x_train, y_train, x_test, y_test, ids_train, ids_test, new_ids_train, new_ids_test

def ids_train_test_shuffle(ids_file, datas, labels, id_fold):
    """Return data and labels starting of ids file. Example: ids_train_test('ids.txt', 'texts.txt', 'labels.txt', 0); Return=x_train, y_train, x_test, y_test"""
    ids = file_to_corpus(ids_file)
    train_test = str(ids[id_fold]).split(';')
    ids_train = [int(id) for id in train_test[0].strip().split(' ')]
    ids_test = [int(id) for id in train_test[1].strip().split(' ')]
    new_ids_train = ids_train.copy(); new_ids_test = ids_test.copy()
    random.shuffle(new_ids_train); random.shuffle(new_ids_test)
    total = file_to_corpus(datas)
    labels = file_to_corpus(labels)
    x_train = [total[index] for index in new_ids_train]       
    y_train = [int(labels[index]) for index in new_ids_train] 
    x_test = [total[index] for index in new_ids_test]
    y_test = [int(labels[index]) for index in new_ids_test] 
    return x_train, y_train, x_test, y_test, ids_train, ids_test, new_ids_train, new_ids_test

def clean_text(text):
    from nltk.stem import WordNetLemmatizer
    stop_words = set(stopwords.words("english")) 
    lemmatizer = WordNetLemmatizer() 
    text = re.sub(r'[^\w\s]','',text, re.UNICODE)
    text = text.lower()
    text = [lemmatizer.lemmatize(token) for token in text.split(" ")]
    text = [lemmatizer.lemmatize(token, "v") for token in text]
    text = [word for word in text if not word in stop_words]
    text = " ".join(text)
    return text

def remove_caracters_especiais_por_espaco(text):
    text =  re.sub("[()!;':?><,.?/+-=-_#$%ˆ&*]", " ", text)
    return re.sub(' +', ' ', text) # remove multiple space

def preprocessor(text):
    """ Preprocessoing data. Example: cv.preprocessor('a155a 45638-000'); Return='a ParsedDigits a Parsed-ZopcodePlusFour'"""        
    replace_patterns = [
    ('<[^>]*>', 'parsedhtml') # remove HTML tags       
    ,(r'(\D)\d\d:\d\d:\d\d(\D)', '\\1 ParsedTime \\2') # text_time_text
    ,(r'(\D)\d\d:\d\d(\D)', '\\1 ParsedTime \\2') # text_time_text
    ,(r'(\D)\d:\d\d:\d\d(\D)', '\\1 ParsedTime \\2') # text_time_text
    ,(r'(\D)\d:\d\d(\D)', '\\1 ParsedTime \\2') # text_time_text
    ,(r'(\D)\d\d\d\-\d\d\d\d(\D)', 'ParsedPhoneNum') # text_phone_text
    ,(r'(\D)\d\d\d\D\d\d\d\-\d\d\d\d(\D)', '\\1 ParsedPhoneNum \\2') # text_phone_text
    ,(r'(\D\D)\d\d\d\D\D\d\d\d\-\d\d\d\d(\D)', '\\1 ParsedPhoneNum \\2') # text_phone_text
    ,(r'(\D)\d\d\d\d\d\-\d\d\d\d(\D)', '\\1 ParsedZipcodePlusFour \\2') # text_zip_text
    ,(r'(\D)\d\d\d\d-\d\d\d\d(\D)', '\\1 ParsedPhoneNum \\2') # text_phone_text  
    ,(r'\d\d:\d\d:\d\d', 'ParsedTime') #time
    ,(r'\d:\d\d:\d\d', 'ParsedTime') #time
    ,(r'\d\d:\d\d', 'ParsedTime') # time
    ,(r'\d\d\d-\d\d\d\d', 'ParsedPhoneNum') # phone US
    ,(r'\d\d\d\d-\d\d\d\d', 'ParsedPhoneNum') # phone brasil
    ,(r'\d\d\d\d\d\-\d\d\d\d', 'ParsedZipcodePlusFour') # zip
    ,(r'\d\d\d\d\d\-\d\d\d', 'ParsedZipcodePlusFour') # zip brasil
    ,(r'(\D)\d+(\D)', '\\1 ParsedDigits \\2') # text_digit_text
    ]    
    compiled_replace_patterns = [(re.compile(p[0]), p[1]) for p in replace_patterns]
    
    # For each pattern, replace it with the appropriate string
    for pattern, replace in compiled_replace_patterns:        
        text = re.sub(pattern, replace, text)
    
    #text = remove_caracters_especiais_por_espaco(text)
    text = remove_accents(text)
    text = text.lower()        
    #text = remove_point_virgula(text)
    
    text = text.split(" ")
    index=0
    for t in text:
        if text[index].__contains__("http://"):
            text[index] = 'parsedhttp'
        elif text[index].__contains__("@"):
            text[index] = 'parsedref'
        index+=1
    return " ".join(text)

def f1(y_test_folds, y_pred_folds, average='macro'):
    """Return f1 score of the various lists. Example: f1([[1,0,1]], [[1,1,1]], 'macro'); Return=0.4"""
    metric=[]
    for index in range(len(y_test_folds)):    
        metric.append(sklearn.metrics.f1_score(y_test_folds[index], y_pred_folds[index], average=average))
    return metric

def save_dict_file(file, dict):
    """Save dict in file. Example: save_dict_file('myfile.json', {'1' : 'hello world'})"""
    with open(file, 'w', newline='') as json_write:
        json.dump(dict, json_write)

def limit_data(x_train, y_train, x_test, y_test, limit, limit_min=0):
    """limit_data(x_train, y_train, x_test, y_test, 30)"""
    return x_train[limit_min:limit], y_train [limit_min:limit], x_test[limit_min:limit], y_test[limit_min:limit]


def stratified_data(x, y):
    """Return train e valid. ([ "ola", "tresasdas", "aeqwwq", "oixzxxz", "olaas", "tresdd", "a", 'b','c'], [1,1,1,1,0,0,0,0,0]) """
    splits =  list(  StratifiedKFold(n_splits=3, shuffle=True, random_state=42).split(x,y) )
    x_train = [ x[index] for index in splits[0][0]]
    y_train = [ y[index] for index in splits[0][0]]
    x_valid = [ x[index] for index in splits[0][1]]
    y_valid = [ y[index] for index in splits[0][1]]
    return x_train, y_train, x_valid, y_valid

def remove_accents(string):
    """ Remove accents string. """
    return unidecode.unidecode(string)

def ic(tamanho, std, confianca, type='normal', lado=2):
    """Calcula o intervalo de confianca"""
    if lado is 1:
        lado = (1 - confianca) # um lado o intervalo fica mais estreito
    else:
        lado = (1 - confianca) /2 
        
    #print(f'Valor de t: {stats.t.ppf(1- (lado), tamanho-1) }')    
    if type is 'normal':
        return stats.norm.ppf(1 - (lado)) * ( std / ( tamanho ** (1/2) ) )
    return stats.t.ppf(1- (lado), tamanho-1) * ( std / ( tamanho ** (1/2) ) ) 

def transform_emb(x_train, x_test):
    """Transform token in embedding"""
    fix_num_token=20
    glove = gensim.models.KeyedVectors.load_word2vec_format('../glove.6B.300d.txt', limit=50000)
    for index_doc in range( len( x_train) ):
        vet_tokens = []
        for t in x_train[index_doc]:
            if len(vet_tokens) == fix_num_token: 
                break
            try:  
                vet_tokens.append( glove[t] )
            except KeyError:
                pass
        while( len(vet_tokens) < fix_num_token ):
            vet_tokens.append( numpy.zeros(300) )
        x_train[index_doc] = vet_tokens
    for index_doc in range( len( x_test) ):
        vet_tokens = []
        for t in x_test[index_doc]:
            if len(vet_tokens) == fix_num_token: break
            try:
                vet_tokens.append( glove[t] )
            except KeyError:
                pass
        while( len(vet_tokens) < fix_num_token ):
            vet_tokens.append( numpy.zeros(300) )
        x_test[index_doc] = vet_tokens
    return x_train, x_test

def remove_point_virgula(text):
    text =  re.sub("[.,]", " ", text)
    return re.sub(' +', ' ', text) # remove multiple space

