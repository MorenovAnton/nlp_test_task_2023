from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

class TextToTfIdf:
    def __init__(self, text_column: str):
        self.word_vectorizer = TfidfVectorizer()
        self.text_column = text_column

    def fit(self, df: pd.DataFrame):
        corpus = df[self.text_column].tolist()
        self.word_vectorizer.fit(corpus)

    def transform(self, df: pd.DataFrame):
        corpus = df[self.text_column].tolist()
        tfidf_matrix = self.word_vectorizer.transform(corpus)
        return tfidf_matrix.toarray()





