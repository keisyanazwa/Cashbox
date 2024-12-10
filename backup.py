import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from difflib import SequenceMatcher

# 1. Load dataset
def load_data(file_path):
    data = pd.read_csv(file_path)
    urgency_map = {"Tinggu": 1, "Sedang": 2, "Rendah": 3}
    data["Urgensi"] = data["Urgensi"].replace(urgency_map).astype(int)
    return data

# 2. Preprocess data
def preprocess_data(data):
    le = LabelEncoder()
    data["Kategori_Asli"] = data["Kategori"]  # Simpan kategori asli untuk pencocokan
    data["Kategori"] = le.fit_transform(data["Kategori"])  # Encode kategori
    scaler = MinMaxScaler()
    data["Jumlah_Normalized"] = scaler.fit_transform(data[["Jumlah"]])
    return data, le, scaler

# 3. Build model
def build_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(2,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    return model

# 4. Train model
def train_model(model, X_train, y_train, X_val, y_val, epochs=30, batch_size=32):
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size)
    return model

# 5. Predict urgency scores
def predict_urgency(model, input_data, le, scaler):
    # Proses data untuk prediksi
    input_data["Kategori_Encoded"] = le.transform(input_data["Kategori"])
    input_data["Jumlah_Normalized"] = scaler.transform(input_data[["Jumlah"]])
    features = input_data[["Jumlah_Normalized", "Kategori_Encoded"]]
    predictions = model.predict(features)
    input_data["Skor Urgensi"] = predictions

    # Peta untuk label urgensi
    urgency_reverse_map = {1: "Tinggi", 2: "Sedang", 3: "Rendah"}
    
    # Tentukan label urgensi berdasarkan skor prediksi
    input_data["Label Urgensi"] = pd.Series(predictions.ravel().round().clip(1, 3).astype(int)).map(urgency_reverse_map)

    # Urutkan data berdasarkan skor urgensi
    sorted_data = input_data.sort_values(by="Skor Urgensi", ascending=True).reset_index(drop=True)
    sorted_data["No"] = range(1, len(sorted_data) + 1)  # Beri nomor urut

    # Output kolom sesuai input pengguna
    return sorted_data[["No", "Kategori_User", "Jumlah", "Skor Urgensi", "Label Urgensi"]]


# 6. Handle input categories with Pengeluaran + Kategori
def handle_input_category(input_category, dataset_categories):
    best_match = None
    best_ratio = 0.0
    
    # Pencocokan input pengguna dengan data kombinasi 'Pengeluaran + Kategori Asli'
    for pengeluaran, kategori_asli in zip(dataset_categories['Pengeluaran'], dataset_categories['Kategori_Asli']):
        combined_category = f"{pengeluaran} {kategori_asli}"  # Gabungkan Pengeluaran + Kategori Asli
        ratio = SequenceMatcher(None, input_category.lower(), combined_category.lower()).ratio()  # Case insensitive matching
        
        if ratio > best_ratio:  # Ambil kecocokan terbaik di atas threshold
            best_match = kategori_asli
            best_ratio = ratio

    if best_match:
        print(f"Kategori ditemukan: '{best_match}', Ratio: {best_ratio:.2f}")
        return input_category, best_match  # Kategori asli ditemukan
    else:
        print(f"Kategori tidak ditemukan. Menggunakan input pengguna sebagai 'Lain lain', Ratio: {best_ratio:.2f}")
        return input_category, "Lain lain"  # Kategori backend "Lain lain", tetapi tampilkan input asli pengguna

# Main function
def main():
    file_path = "data/dataset.csv"
    data = load_data(file_path)
    data, le, scaler = preprocess_data(data)
    
    X = data[["Jumlah_Normalized", "Kategori"]]
    y = data["Urgensi"]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = build_model(input_shape=(2,))
    model = train_model(model, X_train, y_train, X_val, y_val, epochs=30)
    
    input_data = []  # Initialize an empty list to store input data
    print("===== Prediksi Urgensi Pengeluaran =====")
    while True:
        kategori = input("Masukkan kategori: ")
        jumlah = input("Masukkan nominal: ")

        try:
            jumlah = float(jumlah)  # Konversi jumlah ke float
            kategori_user, kategori_backend = handle_input_category(kategori, data[['Pengeluaran', 'Kategori_Asli']])
            input_data.append({"Kategori_User": kategori_user, "Kategori": kategori_backend, "Jumlah": jumlah})
        except ValueError:
            print("Jumlah harus berupa angka. Coba lagi.")

        tambah = input("Tambah pengeluaran lain? (y/n): ").lower()
        if tambah != 'y':
            break

    # Create DataFrame from user input
    input_data = pd.DataFrame(input_data)
    
    # Predict urgency
    results = predict_urgency(model, input_data, le, scaler)
    
    # Display results with formatted columns
    print("===== Hasil Prediksi =====")
    print(results.rename(columns={"Kategori_Asli": "Kategori"}).to_string(index=False))

if __name__ == "__main__":
    main()

