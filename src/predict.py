import tensorflow as tf
import pandas as pd

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
