import glob
import io
import os
import pdb
import re
import sys
import numpy as np
import pandas as pd

import nltk
from nltk import sent_tokenize
from nltk import word_tokenize
from nltk import pos_tag
from nltk import ne_chunk
from nltk import ngrams
from nltk.corpus import stopwords
nltk.download('stopwords')

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score
from sklearn.naive_bayes import GaussianNB


from textblob import TextBlob

def get_entity(text):
    """Prints the entity inside of the text."""
    #print(text)
    namelength=[]
    space=[]
    lengthgram1=[]
    lengthgram2=[]
    lengthgram3=[]
    features={}
    previous=""
    next=""
    names=re.findall(re.compile(r'\█+\s?\█\s?\█+\D█+'),text)
    #print(names)
    stop_words=set(stopwords.words('english'))
    words=word_tokenize(text)
    fil=[i for i in words if not i.lower() in stop_words]
    fil=[]
    x=[]
    length=0
    score=0
    for i in words:
        if i not in stop_words:
            fil.append(i)
    if len(names)!=0:
        length=len(names[0])
    else:
        length=0


    score=TextBlob(text).sentiment.polarity
    features={'Name_length': length,'stop_words':len(fil),'sentiment score':score}
    #print(features)
    return features
def doextraction(glob_text):
    """Get all the files from the given glob and pass them to the extractor."""
    #print(glob_text)
    total_list_features=[]
    name_list=[]
    for thefile in glob.glob(glob_text):
        #print(thefile)
        with io.open(thefile, 'r', encoding='utf-8') as fyl:
            text = fyl.read()
            #print(text)
    header=['git_name','data_type','redacted_name','sentence']
    t = pd.read_csv('unredactor.tsv', sep='\t',header=None,names=header)
    x=t[t['data_type']=='training']
    #print(x['sentence'])
    for i in x['sentence']:
        entity = get_entity(i)
        total_list_features.append(entity)
    #print(total_list_features)
    name_list=x['redacted_name'].tolist()
    return name_list,total_list_features

def get_entity_validation(text):
    """Prints the entity inside of the text."""
    #print(text)
    namelength=[]
    space=[]
    lengthgram1=[]
    lengthgram2=[]
    lengthgram3=[]
    features={}
    previous=""
    next=""
    names=re.findall(re.compile(r'\█+\s?\█\s?\█+\D█+'),text)
    #print(names)
    stop_words=set(stopwords.words('english'))
    words=word_tokenize(text)
    fil=[i for i in words if not i.lower() in stop_words]
    fil=[]
    x=[]
    length=0
    score=0
    for i in words:
        if i not in stop_words:
            fil.append(i)
    if len(names)!=0:
        length=len(names[0])
    else:
        length=0

    score=TextBlob(text).sentiment.polarity
    features={'Name_length': length,'stop_words':len(fil),'sentiment score':score}
    #print(features)
    return features
def doextraction_validation(glob_text):
    """Get all the files from the given glob and pass them to the extractor."""
    #print(glob_text)
    total_list_features=[]
    name_list=[]
    for thefile in glob.glob(glob_text):
        #print(thefile)
        with io.open(thefile, 'r', encoding='utf-8') as fyl:
            text = fyl.read()
            #print(text)
    header=['git_name','data_type','redacted_name','sentence']
    t = pd.read_csv('unredactor.tsv', sep='\t',header=None,names=header)
    x=t[t['data_type']=='validation']
    #print(x['sentence'])
    for i in x['sentence']:
        entity = get_entity_validation(i)
        total_list_features.append(entity)
    #print(total_list_features)
    name_list=x['redacted_name'].tolist()
    return name_list,total_list_features

def get_entity_test(text):
    """Prints the entity inside of the text."""
    #print(text)
    namelength=[]
    space=[]
    lengthgram1=[]
    lengthgram2=[]
    lengthgram3=[]
    features={}
    previous=""
    next=""
    names=re.findall(re.compile(r'\█+\s?\█\s?\█+\D█+'),text)
    #print(names)
    stop_words=set(stopwords.words('english'))
    words=word_tokenize(text)
    fil=[i for i in words if not i.lower() in stop_words]
    fil=[]
    x=[]
    length=0
    score=0
    for i in words:
        if i not in stop_words:
            fil.append(i)
    if len(names)!=0:
        length=len(names[0])
    else:
        length=0

    score=TextBlob(text).sentiment.polarity
    features={'Name_length': length,'stop_words':len(fil),'sentiment score':score}
    #print(features)
    return features

def doextraction_test(glob_text):
    """Get all the files from the given glob and pass them to the extractor."""
    #print(glob_text)
    total_list_features=[]
    name_list=[]
    for thefile in glob.glob(glob_text):
        #print(thefile)
        with io.open(thefile, 'r', encoding='utf-8') as fyl:
            text = fyl.read()
            #print(text)
    header=['git_name','data_type','redacted_name','sentence']
    t = pd.read_csv('unredactor.tsv', sep='\t',header=None,names=header)
    x=t[t['data_type']=='testing']
    #print(x['sentence'])
    for i in x['sentence']:
        entity = get_entity_test(i)
        total_list_features.append(entity)
    #print(total_list_features)
    name_list=x['redacted_name'].tolist()
    return name_list,total_list_features

if __name__ == '__main__':
    # Usage: python3 entity-extractor.py 'train/pos/*.txt'
    y_train,x_train=doextraction(sys.argv[-1])
    y_validation,x_validation=doextraction_validation(sys.argv[-1])
    #print(y_validation)
    y_test, x_test = doextraction_test(sys.argv[-1])
    vec = DictVectorizer()
    vec_features = vec.fit_transform(x_train).toarray()
    #model=LogisticRegression()
    model= GaussianNB()
    model.fit(vec_features, y_train)
    vec_features_validation=vec.fit_transform(x_validation).toarray()
    vec_features_test = vec.fit_transform(x_test).toarray()
    output=model.predict(vec_features_validation)
    print("validation",output)
    print("Validation Precision", precision_score(y_validation, output, average='macro'))
    print("Validation Recall", recall_score(y_validation, output, average='macro'))
    print("Validation f1 score", f1_score(y_validation, output, average='macro'))
    print("Validation Accuracy score", accuracy_score(y_validation, output))
    output1=model.predict(vec_features_test)
    print("test",output1)
    print("test Precision", precision_score(y_test, output1, average='macro'))
    print("test Recall", recall_score(y_test, output1, average='macro'))
    print("test f1 score", f1_score(y_test, output1, average='macro'))
    print("Test Accuracy score", accuracy_score(y_validation, output))