from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout


def build_lstm():

    model = Sequential()

    model.add(
        LSTM(
            64,
            return_sequences=True,
            input_shape=(60,1)
        )
    )

    model.add(Dropout(0.2))

    model.add(
        LSTM(
            64
        )
    )

    model.add(Dropout(0.2))

    model.add(Dense(1))

    model.compile(
        optimizer="adam",
        loss="mse"
    )

    return model