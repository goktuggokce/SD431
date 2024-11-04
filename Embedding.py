import nltk
nltk.download('punkt')  # Tokenizer için gerekli
from nltk.tokenize import word_tokenize

# Örnek veri
sentences = [
    "I love natural language processing",
    "Word embeddings are important for NLP",
    "Natural language processing is fascinating",
    "Machine learning and deep learning are essential for AI"
]

# Her cümledeki kelimeleri tokenize edelim
tokenized_sentences = [word_tokenize(sentence.lower()) for sentence in sentences]


from gensim.models import Word2Vec

# Word2Vec Modeli oluşturma (CBOW kullanarak)
w2v_model = Word2Vec(sentences=tokenized_sentences, vector_size=50, window=3, min_count=1, sg=0)

# Bir kelimenin embedding vektörünü gösterme
print("Embedding for 'natural':\n", w2v_model.wv['natural'])

# İki kelime arasındaki benzerliği hesaplama
similarity = w2v_model.wv.similarity('natural', 'language')
print("\nSimilarity between 'natural' and 'language':", similarity)

import gensim.downloader as api

# GloVe modelini indirme ve yükleme
glove_model = api.load("glove-wiki-gigaword-50")  # 50 boyutlu GloVe vektörleri

# Bir kelimenin embedding vektörünü gösterme
print("Embedding for 'natural':\n", glove_model['natural'])

# İki kelime arasındaki benzerliği hesaplama
similarity = glove_model.similarity('natural', 'language')
print("\nSimilarity between 'natural' and 'language':", similarity)



from sklearn.feature_extraction.text import TfidfVectorizer

# TF-IDF modelini oluşturma
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform([" ".join(sentence) for sentence in tokenized_sentences])

# TF-IDF vektörlerini gösterme
print("TF-IDF Vectors:\n", tfidf_matrix.toarray())

# Kelime indekslerini gösterme
print("\nVocabulary:\n", tfidf_vectorizer.vocabulary_)



