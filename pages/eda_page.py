import streamlit as st
from functions.viz_functions import display_hist

def display_page_eda(data, data_final):
    st.header('Exploratory Data Analysis')
    data_load_state = st.text('Loading data...')
    
    # Display strating dataframe
    st.subheader ('Dataframe')
    st.dataframe(data)

    # Statistical description
    st.subheader('Description générale')
    st.write(data.describe())
    st.write(data.groupby('label').describe())
    
    # Display new dataframe with numerical target and messages lenght
    st.subheader('Ajout target numérique et longueur des messages')
    st.dataframe(data_final)
    
    # Visualization distribution and boxplot of final dataframe
    st.subheader('Visualisation')
    fig = display_hist(data_final, 'length', 'label')
    st.plotly_chart(fig)
    
    data_load_state.text('Loading data...DONE!')