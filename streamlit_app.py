import pandas
import streamlit as st
import pickle
import os

# content
st.title('_Samuel L. Jackson CEFR english predicting_')
welcome_img, welcome_text = st.columns(2)
welcome_img = welcome_img.image('./img/welcome_image.jpg')
welcome_text = welcome_text.markdown('**Sup fella. Ya know where ya come? Here we talkin about CEFR levels - the system of knowin foreign languages. \n
                                      If ya know me, then u like to watch movies and ya know Im talking in English, dearfrend. \n
                                      Wanna try some? Push da button below, upload subs of ur best movie and enjoy!**'
                                     )
