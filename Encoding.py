import pandas as pd

# Örnek veri
data = pd.DataFrame({
    'City': ['Istanbul', 'Ankara', 'Izmir', 'Istanbul', 'Ankara', 'Antalya', 'Izmir', 'Bursa', 'Antalya', 'Istanbul'],
    'Category': ['A', 'B', 'A', 'C', 'B', 'C', 'A', 'B', 'C', 'C']
})


# One-Hot Encoding
one_hot_encoded = pd.get_dummies(data, columns=['City', 'Category'])
print("One-Hot Encoded Data:\n", one_hot_encoded)


from sklearn.preprocessing import LabelEncoder

# Label Encoding
data['City_LabelEncoded'] = LabelEncoder().fit_transform(data['City'])
print("Label Encoded Data:\n", data[['City', 'City_LabelEncoded']])



# Rare Encoding - 'Antalya' ve 'Bursa' değerlerini tek kategori altında topluyoruz
threshold = 0.2  # Verilerin %20'sinden az olan kategorileri 'Rare' olarak adlandır
frequency = data['City'].value_counts(normalize=True)  # Kategori frekansları
data['City_RareEncoded'] = data['City'].apply(lambda x: 'Rare' if frequency[x] < threshold else x)

print("Rare Encoded Data:\n", data[['City', 'City_RareEncoded']])
