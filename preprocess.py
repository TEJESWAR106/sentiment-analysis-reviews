import re
import nltk
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

nltk.download('stopwords', quiet=True)
STOP_WORDS = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|@\w+|#\w+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = ' '.join(
        w for w in text.split() if w not in STOP_WORDS)
    return text

def run_eda(df):
    # Sentiment distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(x='sentiment', data=df,
                  palette='Set2',
                  order=['positive','neutral','negative'])
    plt.title('Sentiment Distribution')
    plt.savefig('plots/sentiment_distribution.png', dpi=150)
    plt.show()

    # Review length distribution
    df['length'] = df['text'].apply(lambda x: len(x.split()))
    plt.figure(figsize=(8, 4))
    sns.histplot(data=df, x='length', hue='sentiment',
                 bins=30, palette='Set2')
    plt.title('Review Length by Sentiment')
    plt.savefig('plots/review_length.png', dpi=150)
    plt.show()

def preprocess(df, max_words=10000, max_len=50):
    df['clean_text'] = df['text'].apply(clean_text)

    le = LabelEncoder()
    y = le.fit_transform(df['sentiment'])

    tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(df['clean_text'])
    sequences = tokenizer.texts_to_sequences(df['clean_text'])
    X = pad_sequences(sequences, maxlen=max_len, padding='post')

    return X, y, tokenizer, le