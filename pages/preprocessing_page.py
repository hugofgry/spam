import streamlit as st
from functions.preprocessing_functions import *
from functions.viz_functions import display_barplot, show_wordcloud

def display_preprocessing(data_final, data_ham, data_spam) :
    st.header('Preprocessing')
    data_load_state = st.text('Loading data...')
    
    # Remove punctuation and stopword
    data_clean = data_final['text'].apply(remove_punctuation_and_stopwords).head()
    st.subheader('Suppression ponctuation et stopwords')
    st.write(data_clean)
    data_ham_clean = data_ham['text'].apply(remove_punctuation_and_stopwords).tolist()
    data_spam_clean = data_spam['text'].apply(remove_punctuation_and_stopwords).tolist()
    
    # Count the ham and spam words without punctuation and stopwords
    df_hamwords_top20 = counter_ham_words(data_ham_clean)
    df_spamwords_top20 = counter_spam_words(data_spam_clean)
    
    # Display wordcloud
    st.subheader('Nuages de mots')
    
    col1, col2= st.beta_columns(2)
    with col1:
        fig = show_wordcloud(data_ham, 'Messages non-spams')
        st.pyplot(fig)
    with col2:
        fig = show_wordcloud(data_spam, 'Messages spams')
        st.pyplot(fig)
    

    # Visualization top 30 with Counter class
    st.subheader('Top 20 des mots les plus utilisés dans les messages spams et non-spams')
    st.write('Utilisation de la classe Counter')
    
    # Viz top 30 hams
    fig = display_barplot(df_hamwords_top20, 'Top 20 des mots les plus utilisés dans les message non-spams')
    st.plotly_chart(fig)
    # Viz top 30 spams
    fig = display_barplot(df_spamwords_top20,'Top 20 des mots les plus utilisés dans les message spams')
    st.plotly_chart(fig)
    
    data_load_state.text('Loading data...DONE!')