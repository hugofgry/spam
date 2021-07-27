import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.utils.multiclass import _check_partial_fit_first_call
import wordcloud
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score


# Display displot with boxplot
def display_hist(data, column_name, color) :
   fig = px.histogram(data, x=column_name, nbins=100, color=color, marginal="box")
   fig.update_layout(
      title="Nombre de messages (hams/spams) en fonction de leur longueurs",
      xaxis_title="Longueur des messages",
      yaxis_title="Nombre de messages",
      legend_title="Type de messages",
      barmode='overlay')
   fig.update_traces(opacity=0.75)
   return fig


# Display wordcloud
def show_wordcloud(data_spam_or_ham, title):
    text = ' '.join(data_spam_or_ham['text'].astype(str).tolist())
    stopwords = set(wordcloud.STOPWORDS)
    fig_wordcloud = wordcloud.WordCloud(stopwords=stopwords,
                                        background_color='white',
                                        colormap='tab10',
                                        width=800,
                                        height=600).generate(text)
    plt.figure(figsize=(10, 7), frameon=True)
    plt.imshow(fig_wordcloud)
    plt.axis('off')
    plt.title(title, fontsize=20)
    plt.show()
   

# Display  barplot
def display_barplot(data, title):
   fig = px.bar(data, x='count', y='word', orientation='h', color='word')
   fig.update_layout(
      title=title,
      xaxis_title="Fréquence",
      yaxis_title="Mots",
      legend_title="Mots")
   return fig


# Display confusion matrix
def display_confusion_matrix(y_true, y_pred):
    mtx = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6,6))
    group_names = ['True Neg','False Pos','False Neg','True Pos']
    group_counts = ['{0:0.0f}'.format(value) for value in mtx.flatten()]
    group_percentages = ['{0:.2%}'.format(value) for value in mtx.flatten()/np.sum(mtx)]
    labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(mtx, annot=labels, fmt='', cmap='viridis', linewidths=.5, cbar=False, ax=ax)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return fig

   