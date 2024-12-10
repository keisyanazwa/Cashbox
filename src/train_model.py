import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from data_processing import load_data, preprocess_data

# 3. Build and train model
def build_and_train_model(file_path, epochs=30, batch_size=32):
    # Load and preprocess data
    data = load_data(file_path)
    data, le, scaler = preprocess_data(data)
    
    # Prepare training and validation data
    X = data[["Jumlah_Normalized", "Kategori"]]
    y = data["Urgensi"]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Build model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(2,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])
    model.summary()
    
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.MeanAbsoluteError()]
    )
    
    # Train model
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size)
    
    # Plot training history
    plot_training_history(history)
    
    # Save and convert the model
    save_model(model)
    
    return model

# Function to plot training history
def plot_training_history(history):
    plt.figure(figsize=(12, 6))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Over Epochs')
    plt.legend()
    
    # Plot MAE
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mean_absolute_error'], label='Training MAE')
    plt.plot(history.history['val_mean_absolute_error'], label='Validation MAE')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Absolute Error')
    plt.title('MAE Over Epochs')
    plt.legend()
    
    # Show the plot
    plt.tight_layout()
    plt.show()

# Function to save the model
def save_model(model):
    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)
    
    # Save the model in .h5 format
    model.save(os.path.join(model_dir, 'predict_model.h5'))
    
    # Convert the model to TensorFlow Lite format
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    # Save the converted model as .tflite
    with open(os.path.join(model_dir, 'predict_model.tflite'), 'wb') as f:
        f.write(tflite_model)
    print("Model saved and converted successfully!")

# Main function to train and save model
def main():
    file_path = "data/dataset.csv"
    model = build_and_train_model(file_path, epochs=30)
    print("Model training and saving completed!")

if __name__ == "__main__":
    main()
