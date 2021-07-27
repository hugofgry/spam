from sklearn.feature_extraction.text import TfidfVectorizer
from functions.preprocessing_functions import remove_punctuation_and_stopwords
from sklearn.preprocessing import MaxAbsScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Pipeline for train the MNB model
def Mnb(X_train, y_train) :
    pipe_MNB_tfidfvec = Pipeline([('tfidf_vec', TfidfVectorizer(analyzer=remove_punctuation_and_stopwords)),
                              ('scaler', MaxAbsScaler()),
                              ('MNB', MultinomialNB()),
                             ])
    model = pipe_MNB_tfidfvec.fit(X_train, y_train)
    return model


