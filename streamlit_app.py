import pandas
import streamlit as st
import pickle
import os

# content
st.title('_Samuel L. Jackson CEFR english predicting_')
welcome_img, welcome_text = st.columns(2)
welcome_img = welcome_img.image('https://cdn.mos.cms.futurecdn.net/shuXAAqrMyYzUVMEusFTpm-1200-80.jpg')
welcome_text = welcome_text.text('Sup fella. Ya know where ya come? Here we talkin about CEFR levels - ',
                                 'the system of knowin foreign languages. Wanna try some? Push da button below, ',
                                 'load subs of ur best movie and enjoy!')
