import numpy as np

from src.data_preprocessing import (
    load_data,
    scale_close_price
)

from src.feature_engineering import create_sequences

from src.train_rnn import build_rnn
from src.train_lstm import build_lstm

from sklearn.model_selection import train_test_split

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint
)

import joblib

df = load_data("data/TSLA.csv")

scaled_data, scaler = scale_close_price(df)

joblib.dump(scaler, "models/scaler.pkl")

X, y = create_sequences(
    scaled_data,
    time_step=60
)

X = X.reshape(
    X.shape[0],
    X.shape[1],
    1
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=False
)

# RNN

rnn = build_rnn()

rnn.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
)

rnn.save("models/rnn_model.keras")

# LSTM

lstm = build_lstm()

callbacks = [
    EarlyStopping(
        patience=5,
        restore_best_weights=True
    ),
    ModelCheckpoint(
        "models/best_lstm.keras",
        save_best_only=True
    )
]

lstm.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    callbacks=callbacks
)

print("Training Complete")