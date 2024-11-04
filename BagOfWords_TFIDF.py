from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Örnek belgeler
documents = [
    "Natural language processing makes machine learning easier.",
    "Machine learning and natural language processing are interesting fields.",
    "Natural language understanding is a part of natural language processing."
]

# Bag of Words (BOW) Uygulaması
# 1. CountVectorizer'ı tanımla
bow_vectorizer = CountVectorizer()

# 2. Belgeleri vektörize et
bow_matrix = bow_vectorizer.fit_transform(documents)

# 3. Bag of Words sonuçlarını yazdır
print("Bag of Words (BOW) Özellik İsimleri:\n", bow_vectorizer.get_feature_names_out())
print("\nBag of Words (BOW) Matrisi:\n", bow_matrix.toarray())

# TF-IDF Uygulaması
# 1. TfidfVectorizer'ı tanımla
tfidf_vectorizer = TfidfVectorizer()

# 2. Belgeleri vektörize et
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

# 3. TF-IDF sonuçlarını yazdır
print("\nTF-IDF Özellik İsimleri:\n", tfidf_vectorizer.get_feature_names_out())
print("\nTF-IDF Matrisi:\n", tfidf_matrix.toarray())
