import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Load the dataset
df = pd.read_csv("climate_change_indicators.csv")

# Clean the text
def clean_text(text):
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    return text

df['cleaned_text'] = df['text'].apply(clean_text)

# Perform EDA
print(df['category'].value_counts())

# Plot category distribution
sns.countplot(y=df['category'])
plt.title('Category Distribution')
plt.show()


from sklearn.model_selection import train_test_split

# Encode the labels
df['label'] = df['category'].factorize()[0]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(df['cleaned_text'], df['label'], test_size=0.2, random_state=42)


from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Train Word2Vec model
sentences = [text.split() for text in X_train]
w2v_model = Word2Vec(sentences, vector_size=100, window=5, min_count=2, sg=1)

# Generate sentence embeddings using Word2Vec
def get_sentence_embedding(text, model):
    words = text.split()
    embedding = np.mean([model.wv[word] for word in words if word in model.wv], axis=0)
    return embedding if type(embedding) != float else np.zeros(100)

X_train_w2v = np.array([get_sentence_embedding(text, w2v_model) for text in X_train])
X_test_w2v = np.array([get_sentence_embedding(text, w2v_model) for text in X_test])


# Load pre-trained GloVe embeddings
glove_file = "glove.6B.100d.txt"
glove_embeddings = {}
with open(glove_file, 'r', encoding='utf-8') as file:
    for line in file:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype='float32')
        glove_embeddings[word] = vector

# Generate sentence embeddings using GloVe
def get_glove_embedding(text, embeddings):
    words = text.split()
    embedding = np.mean([embeddings[word] for word in words if word in embeddings], axis=0)
    return embedding if type(embedding) != float else np.zeros(100)

X_train_glove = np.array([get_glove_embedding(text, glove_embeddings) for text in X_train])
X_test_glove = np.array([get_glove_embedding(text, glove_embeddings) for text in X_test])
