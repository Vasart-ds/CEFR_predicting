import pandas
import streamlit as st
import pickle
import os
import re

from io import StringIO

import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pymystem3 import Mystem
import spacy

from sklearn.feature_extraction.text import CountVectorizer


# content
st.title('_Jules Winnfield CEFR english predicting_')
welcome_img, welcome_text = st.columns(2)
welcome_img = welcome_img.image('./img/welcome_image.jpg')
welcome_text = welcome_text.markdown('**_Sup fella. Ya know where ya come? Here we talkin about CEFR levels - the system of knowin foreign languages. If ya know me, then u like to watch movies and ya know Im talking in English, dearfrend. Wanna try some? Push da button below, upload subs of ur best movie and enjoy!_**')

movie_name, upload_button, free_space_2 = st.columns(3)
upload_button = upload_button.file_uploader(label='English, dearfrend, can u read it?!', accept_multiple_files = False)

# функции для очистки субтитров
stop_words = stopwords.words('english')
nlp = spacy.load("en_core_web_sm")

del_n = re.compile('\n')                # перенос каретки
del_tags = re.compile('<[^>]*>')        # html-теги
del_brackets = re.compile('\([^)]*\)')  # содержимое круглых скобок
clean_text = re.compile('[^а-яa-z\s]')  # все небуквенные символы кроме пробелов
del_spaces = re.compile('\s{2,}')       # пробелы от 2-ух подряд

def prepare_text(text):
    text = del_n.sub(' ', str(text).lower())
    text = del_tags.sub('', text)
    text = del_brackets.sub('', text)
    res_text = clean_text.sub('', text)
    return del_spaces.sub(' ', res_text)

def del_stopwords(text):
    clean_tokens = tuple(
        map(lambda x: x if x not in stop_words else '', word_tokenize(text))
    )
    res_text = ' '.join(clean_tokens)
    return res_text

def lemmatize(text):    
    lemmatized_text = ''.join(Mystem().lemmatize(text))
    return lemmatized_text.split('|')
  
def clean_subs(sub_list):
    filtered = []
    for word in range(len(sub_list)):
        text = prepare_text(sub_list[word])
        text = del_stopwords(text)
        text = lemmatize(text)
        filtered.append(text)
    return filtered
 
def preprocess(text):
    # удаление символов
    document = re.sub(r'\W', ' ', str(text))
    # удаление одиноко стоящих слов
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
    document = re.sub('[^а-яa-z\s]', ' ', document)
    # приведение к нижнему регистру 
    document = document.lower()
    # токенизация
    #document = nltk.word_tokenize(document,language = "english")
    # лемматизация
    spacy_results = nlp(document)
    document = ' '.join([token.lemma_ for token in spacy_results])
    return document

# реализация программы
if upload_button is not None:
    stringio = StringIO(upload_button.getvalue().decode('iso-8859-1'))
    string_data = stringio.read()
    
    # ВАРИАНТ 1
    #text = prepare_text(string_data)
    #text = del_stopwords(text)
    #text = nltk.word_tokenize(text, language = "english")
    #spacy_results = nlp(text)
    #text = ' '.join([token.lemma_ for token in text])
    #st.write(text)
    
    #cleaned_sub = clean_subs(string_data)
    ##cleaned_sub = cleaned_sub.apply(preprocess)
    #st.write(cleaned_sub)
    
    # ВАРИАНТ 2
    # string_data = re.sub(r'\n', ' ', str(string_data))
    # string_data = re.sub(r'\s+[a-zA-Z]\s+', ' ', string_data)
    # string_data = re.sub(r'[^а-яa-z\s]', ' ', string_data)
    # string_data = string_data.lower()
    # spacy_results = nlp(string_data)
    # string_data = ' '.join([token.lemma_ for token in spacy_results])
    # st.write(string_data)
    
    # ВАРИАНТ 3
    data = preprocess(string_data)
    data = nltk.word_tokenize(data, language = "english")
    
    vectorizer = CountVectorizer(stop_words = stop_words)
    vectorized_sub = vectorizer.fit_transform(data).toarray()
    
    model = pickle.load(open(r'.\catboost_clf.pcl', 'rb'))
    prediction_clf = model.predict(vectorized_sub)
    
    movie_name = movie_name.write(prediction_clf)
