import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import joblib

# Fayl yo'lini belgilang
file_path = "20000-Utterances-Training-dataset-for-chatbots-virtual-assistant-Bitext-sample.csv"  # CSV faylingizning to'liq yo'lini kiriting

# CSV faylni yuklash
df = pd.read_csv(file_path)

# Mavjud ustunlar
print("Mavjud ustunlar:", df.columns)

# Matn va belgi ustunlarini tanlash
df = df[['utterance', 'intent']]  # To'g'ri ustunlarni belgilang
df.columns = ['text', 'label']  # Ustunlarni nomini o'zgartirish

# Ma'lumotlarni tozalash (agar kerak bo'lsa)
df = df.dropna()  # NaN qiymatlarni olib tashlash

# Belgilar va maqsadni ajratish
X = df['text']
y = df['label']

# Ma'lumotlarni o'quv va test to'plamlariga bo'lish
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Matnni vektorlashtirish
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Modelni yaratish va o'rgatish
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Test ma'lumotlarini baholash
y_pred = model.predict(X_test_vec)
print(classification_report(y_test, y_pred))

# Modelni saqlash
model_filename = "trained_model.pkl"
vectorizer_filename = "vectorizer.pkl"

joblib.dump(model, model_filename)
joblib.dump(vectorizer, vectorizer_filename)

print(f"Model '{model_filename}' nomi bilan saqlandi.")
print(f"Vectorizer '{vectorizer_filename}' nomi bilan saqlandi.")
