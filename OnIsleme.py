 #Gerekli kütüphaneleri içe aktar
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import spacy

# NLTK'de stopword'leri yükleyin
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# PorterStemmer gövdeleme için kullanılıyor
stemmer = PorterStemmer()

# SpaCy İngilizce modelini yükleyin (lemmatization için)
nlp = spacy.load("en_core_web_sm")

# İşlenecek örnek metin
text = "Natural language processing techniques can help in text preprocessing, which is essential in machine learning."

# 1. Küçük harfe çevirme
text = text.lower()

# 2. Stopword temizleme
tokens = text.split()
tokens = [word for word in tokens if word not in stop_words]

# 3. Gövdeleme (Stemming)
stemmed_tokens = [stemmer.stem(word) for word in tokens]

# 4. Lemmatiation (Kök bulma)
lemmatized_tokens = [token.lemma_ for token in nlp(" ".join(tokens))]

# Sonuçları yazdırma
print("Orijinal Metin:", text)
print("Stopword'ler Temizlenmiş:", tokens)
print("Gövdeleme Sonucu:", stemmed_tokens)
print("Kök Bulma (Lemmatization) Sonucu:", lemmatized_tokens)
