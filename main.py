import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import pandas as pd
import tensorflow as tf
from src.data_processing import load_data, preprocess_data
from src.handler import handle_input_category
from src.predict import predict_urgency
from sklearn.model_selection import train_test_split

# Function to load the pre-trained model
def load_model():
    model_path = 'models/predict_model.h5'
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully!")
        return model
    else:
        print("Model not found. Please train the model first.")
        exit()

# Main function
def main():
    # Load the pre-trained model
    model = load_model()
    
    # Load and preprocess the data+
    file_path = "data/dataset.csv"
    data = load_data(file_path)
    data, le, scaler = preprocess_data(data)
    
    input_data = []  # Initialize an empty list to store input data
    print("===== Prediksi Urgensi Pengeluaran =====")
    while True:
        kategori = input("Masukkan kategori: ")
        jumlah = input("Masukkan nominal: ")

        try:
            jumlah = float(jumlah)  # Convert the amount to float
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
