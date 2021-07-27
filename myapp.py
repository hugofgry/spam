import streamlit as st
import requests
import pandas as pd
from pages.eda_page import display_page_eda
from pages.preprocessing_page import display_preprocessing
from pages.model_page import display_model

st.set_page_config(page_title='NLP with MultinomialNB')
st.set_option('deprecation.showPyplotGlobalUse', False)
st.title('Spam or ham ?')

##### PREPARATION PAGES ###

# Load data
data = requests.get('http://127.0.0.1:8000/all').json()
data = pd.DataFrame(data)
data = data.rename(columns={1:'label', 2:'text'})

# Add numerical target - 0=ham / 1=spam and length messages
data_final = data.copy()
data_final['spam'] = data_final['label'].map({'ham':1, 'spam':2})
data_final['length'] = data_final['text'].apply(len)

# Separate ham and spam messages
data_ham  = data_final[data_final['spam'] == 1]
data_spam = data_final[data_final['spam'] == 2]

# Navigation menu
st.sidebar.title('Menu de navigation')
nav = st.sidebar.radio('Choisir une section',['EDA', 'Preprocessing', 'Modèle'])
if nav == 'EDA':
    display_page_eda(data, data_final)
elif nav == 'Preprocessing':
    display_preprocessing(data_final, data_ham, data_spam)
else :
    display_model(data_final)



