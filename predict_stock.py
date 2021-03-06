import pandas as pd
import numpy as np
from sklearn import preprocessing
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import utils
import os, sys
from stock_rnn_model import StockRnnModel
import datetime

if len(sys.argv) < 6:
    print("Usage:")
    print("  python predict_stock.py ${merged_csv_path} [m2o|m2m] [LSTM|GRU] ${past_day} ${future_day}")
    print("Ex.")
    print("  # python predict_stock.py ./2330_new.csv m2o LSTM 20 1")
    print("  # python predict_stock.py ./2330_new.csv m2m GRU 20 5")
    sys.exit(0)

csv_raw_file = sys.argv[1]
input_output_type = sys.argv[2] # "m2o" or "m2m"
model_type = sys.argv[3] # "LSTM" or "GRU"
past_day = int(sys.argv[4])
future_day = int(sys.argv[5])

def info(message):
    print("%s --- %s" % (datetime.datetime.now(), message))

def precheck():
    if past_day <= 0 or future_day <= 0:
        info("Please check past_day or future_day parameters.")
        info("They can not be zero or negative integer.")
        sys.exit(1)
    if input_output_type == "m2o" and future_day != 1:
        info("Please assign future_day as 1 for m2o.")
        sys.exit(1)
    if input_output_type == "m2m" and future_day == 1:
        info("Please assign future_day as the number greater than 1 for m2m.")
        sys.exit(1)

def normalize_dataframe(df):
    new_df = df.drop(["date"], axis=1)
    min_max_scaler = preprocessing.MinMaxScaler()
    for column in new_df:
        new_df[column] = min_max_scaler.fit_transform(new_df[column].values.reshape(-1, 1))
    return new_df

def denormalize_dataframe(raw_df, target_df):
    min_max_scaler = preprocessing.MinMaxScaler()
    min_max_scaler.fit_transform(raw_df)
    reversed_target = min_max_scaler.inverse_transform(target_df)
    return pd.DataFrame({'close': reversed_target[:, 0]})

def plot_stock(date_series, raw_close, predicted_close):
    date_list_index = date_series.index.tolist()
    date_list = date_series.values.tolist()
    part_date_list_index = date_list_index[::int(len(date_list_index)/10)]
    part_date_list = date_list[::int(len(date_list)/10)]
    part_date_list_index.pop()
    part_date_list.pop()
    part_date_list_index.append(date_list_index[-1])
    part_date_list.append(date_list[-1])
    plt.clf()
    plt.title('Close price comparison')
    plt.xlabel('Date')
    plt.ylabel('Close')
    plt.xticks(part_date_list_index, part_date_list, rotation=30)
    plt.plot(raw_close['close'], color='red', label='close')
    plt.plot(predicted_close['close'], color='blue', label='close')
    plt.legend(['raw_close', 'predicted_close'], loc='upper left')
    plt.savefig(model_type + "_" + input_output_type + "_train_price_comparison.png")
    plt.show()

def model_score(model, X_train, y_train, X_test, y_test):
    training_score = model.evaluate(X_train, y_train, verbose=0) * 100
    validation_score = model.evaluate(X_test, y_test, verbose=0) * 100
    return training_score,validation_score

def plot_loss(history_dict):
    plt.plot(history_dict['loss'])
    plt.plot(history_dict['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(model_type + "_" + input_output_type + '_lose.png')
    #plt.show()

def build_train(train_norm_df, pastDay, futureDay):
    X_train, Y_train = [], []
    for i in range(train_norm_df.shape[0] - futureDay - pastDay + 1):
        X_train.append(np.array(train_norm_df.iloc[i:i + pastDay]))
        Y_train.append(np.array(train_norm_df.iloc[i + pastDay:i + pastDay + futureDay]["close"]))
    return np.array(X_train), np.array(Y_train)

def shuffle(X, Y):
    np.random.seed(10)
    randomList = np.arange(X.shape[0])
    np.random.shuffle(randomList)
    return X[randomList], Y[randomList]

def split_data(X, Y, rate):
    X_train = X[int(X.shape[0]*rate):]
    Y_train = Y[int(Y.shape[0]*rate):]
    X_val = X[:int(X.shape[0]*rate)]
    Y_val = Y[:int(Y.shape[0]*rate)]
    return X_train, Y_train, X_val, Y_val

def get_change_error(close_df, predicted_df):
    total_change = 0
    for index in range(0, len(close_df)):
        pc = predicted_df.iloc[index]['close']
        c = close_df.iloc[index]['close']
        total_change = total_change + (abs(pc - c) / c)
    return total_change * 100 / len(close_df)

if __name__ == "__main__":
    precheck()
    info("Start to read csv to dataframe.")
    raw_df = utils.read_csv_to_df(csv_raw_file)
    close_df = raw_df['close'].to_frame().reset_index(drop=True)
    #print(raw_df)
    info("Start to normalize dataset.")
    train_norm_df = normalize_dataframe(raw_df)
    #print(train_norm_df)

    #info("Create normalize dataset for cache.")
    #utils.df_to_csv(train_norm_df, csv_norm_file)
    #train_norm_df = utils.read_csv_to_df(csv_norm_file)

    info("Start to build training data.")
    X_train, Y_train = build_train(train_norm_df, past_day, future_day)
    info("Feature shape: %s" % str(X_train.shape))
    info("Ground truth shape: %s" % str(Y_train.shape))
    final_test_for_all = X_train
    first_half_close_df = train_norm_df['close'].iloc[0:past_day].to_frame()
    info("=================================")

    info("Start to shuffle and split data for training and validation.")
    X_train, Y_train = shuffle(X_train, Y_train)
    X_train, Y_train, X_val, Y_val = split_data(X_train, Y_train, 0.2)
    info("Training feature shape: %s" % str(X_train.shape))
    info("Training ground truth shape: %s" % str(Y_train.shape))
    info("Validation feature shape: %s" % str(X_val.shape))
    info("Validation ground truth shape: %s" % str(Y_val.shape))
    info("=================================")


    stock_rnn_model = StockRnnModel(input_output_type, model_type, X_train.shape, Y_train.shape)
    model = stock_rnn_model.model
    callback = EarlyStopping(monitor="loss", patience=10, verbose=1, mode="auto")
    training_history = model.fit(X_train, Y_train, epochs=1000, batch_size=128,
                                 validation_data=(X_val, Y_val), callbacks=[callback])

    model.save(model_type + "_stock_" + input_output_type + "_inference.h5")
    training_history_dict = training_history.history
    utils.dict_to_json(training_history_dict, "./" + model_type + "_" + input_output_type + "_training_history")
    plot_loss(training_history_dict)
    training_score, validation_score = model_score(model, X_train, Y_train, X_val, Y_val)
    info("=================================")
    info("Train MSE: %.5f%%" % training_score)
    info("Validation MSE: %.5f%%" % validation_score)
    info("=================================")

    predicted_close = model.predict(final_test_for_all)
    #print(predicted_close)
    #print(predicted_close.shape)
    predicted_close = stock_rnn_model.get_avereage_predicted_close(predicted_close, "mean")
    #print(predicted_close)
    #print(predicted_close.shape)
    predicted_close_df = pd.DataFrame({'close': predicted_close[:, 0]})
    predicted_norm_df = pd.concat([first_half_close_df, predicted_close_df], ignore_index=True)
    predicted_df = denormalize_dataframe(close_df, predicted_norm_df)
    info("Predicted close for whole dataset:")
    print(predicted_df)
    info("Change percentage error: %s%%" % get_change_error(close_df, predicted_df))
    plot_stock(raw_df['date'], close_df, predicted_df)
