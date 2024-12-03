import pickle
from spellchecker import SpellChecker
import streamlit as st

# Pickle faylidan modelni yuklash
with open('vectorizer.pkl', 'rb') as f:
    model = pickle.load(f)

# Imloviy xatoliklarni tuzatish uchun funksiyani yaratamiz
def correct_spelling(text):
    spell = SpellChecker()
    words = text.split()  # So'zlarni bo'lib olish
    corrected_words = [spell.correction(word) for word in words]
    corrected_text = ' '.join(corrected_words)
    return corrected_text

# Streamlit ilovasini yaratish
def main():
    st.title("Inglizcha Imloviy Xatoliklarni To'g'irlash")
    
    # Foydalanuvchidan matn kiritishni so'rash
    text = st.text_area("Matnni kiriting:", "")
    
    if text:
        # Imloviy xatoliklarni tuzatish
        corrected_text = correct_spelling(text)
        
        # Foydalanuvchiga natijani ko'rsatish
        st.subheader("To'g'irlangan Matn:")
        st.write(corrected_text)

if __name__ == "__main__":
    main()
