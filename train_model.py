import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import os
import warnings

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def create_and_train_model():
    # --- Data Loading and Preparation ---
    data = pd.read_csv("ngpf_data.csv")
    data.rename(columns={'Day': 'date', 'Price in Dollars per Million Btu': 'gas_price'}, inplace=True)
    data['date'] = pd.to_datetime(data['date'], format="%d/%m/%Y")
    data = data.sort_values(by='date').set_index('date')
    data = data.fillna(method='pad')

    # --- Data Modeling ---
    train = data['1997-01-07': '2020-01-06']
    
    # --- LSTM Model Training ---
    slot = 15
    x_train, y_train = [], []
    for i in range(slot, len(train)):
        x_train.append(train.iloc[i-slot:i, 0])
        y_train.append(train.iloc[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    lstm_model = tf.keras.Sequential()
    lstm_model.add(tf.keras.layers.LSTM(units=50, input_shape=(slot, 1), return_sequences=True, activation='relu'))
    lstm_model.add(tf.keras.layers.LSTM(units=50, activation='relu', return_sequences=True))
    lstm_model.add(tf.keras.layers.LSTM(units=50, return_sequences=True))
    lstm_model.add(tf.keras.layers.LSTM(units=50, return_sequences=False))
    lstm_model.add(tf.keras.layers.Dense(units=1))
    lstm_model.compile(loss='mean_squared_error', optimizer='adam')

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=7)
    print("Starting LSTM model training...")
    lstm_model.fit(x_train, y_train, epochs=100, batch_size=64, verbose=1, shuffle=False, callbacks=[early_stopping])
    print("LSTM model training complete.")
    
    # Save the trained model
    lstm_model.save("lstm_model.h5")
    print("Model saved as lstm_model.h5")

if __name__ == '__main__':
    create_and_train_model()