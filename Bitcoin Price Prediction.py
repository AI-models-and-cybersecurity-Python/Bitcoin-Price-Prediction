import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df["value"] = [float(x.replace(",", ".")) for x in df["value"].values]
    df["date"] = pd.to_datetime(df["date"], dayfirst=True)
    return df

def split_and_scale_data(df):
    train, test = train_test_split(df, test_size=0.2, shuffle=False)
    sc = MinMaxScaler()  
    X_train = train.loc[:, train.columns[1:]]
    X_train = sc.fit_transform(X_train)
    train[train.columns[1:]] = X_train
    X_test = test.loc[:, test.columns[1:]]
    X_test = sc.transform(X_test)
    test[test.columns[1:]] = X_test
    return train, test, sc

def create_time_series_generators(train, test):
    train_x = train.loc[:, train.columns[1:]].values
    test_x = test.loc[:, test.columns[1:]].values 
    train_gen = TimeseriesGenerator(train_x, train_x, length=7, batch_size=4)
    test_gen = TimeseriesGenerator(test_x, test_x, length=7, batch_size=4)
    return train_gen, test_gen, test_x

def build_model():
    model = Sequential()
    model.add(Flatten(input_shape=(7,)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse", metrics=["mse"])
    return model

def evaluate_model(model, test_gen, test_x, sc):
    y_pred = model.predict(test_gen)
    y_pred = y_pred.reshape(-1, y_pred.shape[-1])
    test_x_reshaped = test_x.reshape(-1, test_x.shape[-1])
    pred = sc.inverse_transform(y_pred)
    true = sc.inverse_transform(test_x_reshaped)
    r2 = r2_score(y_true=true[7:], y_pred=pred)
    return r2, pred, true

def plot_predictions(pred, true):
    figure = plt.figure()
    plt.title("Bitcoin value prediction")
    plt.xlabel("Number of days")
    plt.ylabel("Value")
    plt.plot(pred, label="Predicted")
    plt.plot(true[7:], label="True")
    plt.legend()
    return figure

def main():
    df = load_and_preprocess_data("bitcoin.csv")
    train, test, sc = split_and_scale_data(df)
    train_gen, test_gen, test_x = create_time_series_generators(train, test)
    
    model = build_model()
    model.fit(train_gen, epochs=100)
    
    r2, pred, true = evaluate_model(model, test_gen, test_x, sc)
    print(f"R2 Score: {r2}")
    
    fig = plot_predictions(pred, true)
    plt.show()

if __name__ == "__main__":
    main()
=======
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df["value"] = [float(x.replace(",", ".")) for x in df["value"].values]
    df["date"] = pd.to_datetime(df["date"], dayfirst=True)
    return df

def split_and_scale_data(df):
    train, test = train_test_split(df, test_size=0.2, shuffle=False)
    sc = MinMaxScaler()  
    X_train = train.loc[:, train.columns[1:]]
    X_train = sc.fit_transform(X_train)
    train[train.columns[1:]] = X_train
    X_test = test.loc[:, test.columns[1:]]
    X_test = sc.transform(X_test)
    test[test.columns[1:]] = X_test
    return train, test, sc

def create_time_series_generators(train, test):
    train_x = train.loc[:, train.columns[1:]].values
    test_x = test.loc[:, test.columns[1:]].values 
    train_gen = TimeseriesGenerator(train_x, train_x, length=7, batch_size=4)
    test_gen = TimeseriesGenerator(test_x, test_x, length=7, batch_size=4)
    return train_gen, test_gen, test_x

def build_model():
    model = Sequential()
    model.add(Flatten(input_shape=(7,)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse", metrics=["mse"])
    return model

def evaluate_model(model, test_gen, test_x, sc):
    y_pred = model.predict(test_gen)
    y_pred = y_pred.reshape(-1, y_pred.shape[-1])
    test_x_reshaped = test_x.reshape(-1, test_x.shape[-1])
    pred = sc.inverse_transform(y_pred)
    true = sc.inverse_transform(test_x_reshaped)
    r2 = r2_score(y_true=true[7:], y_pred=pred)
    return r2, pred, true

def plot_predictions(pred, true):
    figure = plt.figure()
    plt.title("Bitcoin value prediction")
    plt.xlabel("Number of days")
    plt.ylabel("Value")
    plt.plot(pred, label="Predicted")
    plt.plot(true[7:], label="True")
    plt.legend()
    return figure

def main():
    df = load_and_preprocess_data("bitcoin.csv")
    train, test, sc = split_and_scale_data(df)
    train_gen, test_gen, test_x = create_time_series_generators(train, test)
    
    model = build_model()
    model.fit(train_gen, epochs=100)
    
    r2, pred, true = evaluate_model(model, test_gen, test_x, sc)
    print(f"R2 Score: {r2}")
    
    fig = plot_predictions(pred, true)
    plt.show()

if __name__ == "__main__":
    main()
