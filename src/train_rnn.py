from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout


def build_rnn():

    model = Sequential()

    model.add(
        SimpleRNN(
            units=50,
            input_shape=(60,1)
        )
    )

    model.add(Dropout(0.2))

    model.add(Dense(1))

    model.compile(
        optimizer="adam",
        loss="mse"
    )

    return model