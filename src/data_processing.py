import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

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
