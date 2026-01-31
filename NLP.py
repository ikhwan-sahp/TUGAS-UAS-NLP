import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


df = pd.read_csv('data/dataset_sentimen.csv', sep=';')
df = df.dropna(subset=['review_text', 'sentiment'])


factory = StemmerFactory()
stemmer = factory.create_stemmer()
indo_stopwords = set(stopwords.words('indonesian'))

def preprocess(text):
   
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)

    tokens = word_tokenize(text)
    
    tokens = [w for w in tokens if w not in indo_stopwords]
    stemmed = [stemmer.stem(w) for w in tokens]
    return " ".join(stemmed)

print("Sedang memproses preprocessing data...")
df['clean_text'] = df['review_text'].apply(preprocess)


df.to_csv('data/dataset_bersih.csv', index=False, sep=';')


df[df['sentiment'] == 0.0].to_csv('data/hasil_negatif.csv', index=False, sep=';')
df[df['sentiment'] == 1.0].to_csv('data/hasil_netral.csv', index=False, sep=';')
df[df['sentiment'] == 2.0].to_csv('data/hasil_positif.csv', index=False, sep=';')

print("Data CSV untuk semua kategori telah dibuat di folder 'data'.")


X_train, X_test, y_train, y_test = train_test_split(df['clean_text'], df['sentiment'], test_size=0.2, random_state=42)
print(f"Jumlah Data Latih: {len(X_train)}")
print(f"Jumlah Data Uji: {len(X_test)}")
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vec, y_train)
y_pred = model.predict(X_test_vec)


print("\n=== EVALUASI MODEL ===")
print(classification_report(y_test, y_pred))
print("CONFUSION MATRIX:")
print(confusion_matrix(y_test, y_pred))


ConfusionMatrixDisplay.from_estimator(model, X_test_vec, y_test, cmap=plt.cm.Blues)
plt.show()















