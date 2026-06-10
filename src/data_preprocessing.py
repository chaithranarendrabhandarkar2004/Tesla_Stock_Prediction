import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_data(path):

    df = pd.read_csv(path)

    df["Date"] = pd.to_datetime(df["Date"])

    df.set_index("Date", inplace=True)

    print(df.info())

    print(df.isnull().sum())

    df = df.ffill()

    return df


def scale_close_price(df):

    scaler = MinMaxScaler()

    scaled = scaler.fit_transform(df[["Close"]])

    return scaled, scaler