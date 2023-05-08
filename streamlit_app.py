import pandas
import streamlit as st
import pickle
import os

from io import StringIO

# content
st.title('_Jules Winnfield CEFR english predicting_')
welcome_img, welcome_text = st.columns(2)
welcome_img = welcome_img.image('./img/welcome_image.jpg')
welcome_text = welcome_text.markdown('**_Sup fella. Ya know where ya come? Here we talkin about CEFR levels - the system of knowin foreign languages. If ya know me, then u like to watch movies and ya know Im talking in English, dearfrend. Wanna try some? Push da button below, upload subs of ur best movie and enjoy!_**')

free_space_1, upload_button, free_space_2 = st.columns(3)
upload_button = upload_button.file_uploader(label='English, dearfrend, can u read it?!')

if upload_button is not None:
  stringio = StringIO(upload_button.getvalue().decode('iso-8859-1'))
  st.write(stringio)
