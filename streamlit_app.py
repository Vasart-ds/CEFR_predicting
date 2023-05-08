import pandas
import streamlit as st
import pickle
import os

from io import StringIO

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pymystem3 import Mystem
import spacy


# content
st.title('_Jules Winnfield CEFR english predicting_')
welcome_img, welcome_text = st.columns(2)
welcome_img = welcome_img.image('./img/welcome_image.jpg')
welcome_text = welcome_text.markdown('**_Sup fella. Ya know where ya come? Here we talkin about CEFR levels - the system of knowin foreign languages. If ya know me, then u like to watch movies and ya know Im talking in English, dearfrend. Wanna try some? Push da button below, upload subs of ur best movie and enjoy!_**')

free_space_1, upload_button, free_space_2 = st.columns(3)
upload_button = upload_button.file_uploader(label='English, dearfrend, can u read it?!')

# интерфейс
if upload_button is not None:
  stringio = StringIO(upload_button.getvalue().decode('iso-8859-1'))
  string_data = stringio.read()
  
  stop_words = stopwords.words('english')
  nlp = spacy.load("en_core_web_sm")
  
  def preprocess(text):
    # удаление символов
    document = re.sub(r'\W', ' ', str(text))
    # удаление одиноко стоящих слов
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
    # приведение к нижнему регистру 
    document = document.lower()
    # токенизация
    #document = nltk.word_tokenize(document,language = "english")
    # лемматизация
    spacy_results = nlp(document)
    document = ' '.join([token.lemma_ for token in spacy_results])
    return document
  
 # string_data = string_data.apply(preprocess)
  st.write(string_data)
