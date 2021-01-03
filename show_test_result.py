import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import utils
from stock_rnn_model import StockRnnModel
import os, sys
import datetime

if len(sys.argv) < 7:
    print("Usage:")
    print("  python show_test_result.py ${merged_csv_path} [m2o|m2m] [LSTM|GRU] ${past_day} ${future_day} ${show_interval}")
    print("Ex.")
    print("  # python show_test_result.py ./2330_new.csv m2o LSTM 20 1 60")
    print("  # python show_test_result.py ./2330_new.csv m2m GRU 20 5 60")
    sys.exit(1)

csv_raw_file = sys.argv[1]
input_output_type = sys.argv[2] # "m2o" or "m2m"
model_type = sys.argv[3] # "LSTM" or "GRU"
past_day = int(sys.argv[4])
future_day = int(sys.argv[5])
show_interval = "all" if sys.argv[6] == "all" else int(sys.argv[6]) # "all" or any integer

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
        new_df[column] = min_max_scaler.fit_transform(new_df[column].values.reshape(-1,1))
    return new_df

def denormalize_dataframe(raw_df, target_df):
    min_max_scaler = preprocessing.MinMaxScaler()
    min_max_scaler.fit_transform(raw_df)
    reversed_target = min_max_scaler.inverse_transform(target_df)  # + raw_df['close'].mean()
    return pd.DataFrame({'close': reversed_target[:, 0]})

def plot_stock(predicted_close):
    date_list_index = predicted_close['date'].index.tolist()
    date_list = predicted_close['date'].values.tolist()
    part_date_list_index = date_list_index[::int(len(date_list_index)/10)]
    part_date_list = date_list[::int(len(date_list)/10)]
    if show_interval != "all" or input_output_type == "m2m":
        part_date_list_index.append(date_list_index[-1])
        part_date_list.append(date_list[-1])
    plt.clf()
    plt.title('Close price comparison')
    plt.xlabel('Date')
    plt.ylabel('Close')
    plt.xticks(part_date_list_index, part_date_list, rotation=30)
    plt.plot(predicted_close['raw_close'], color='red', label='close')
    plt.plot(predicted_close['close'], color='blue', label='close')
    plt.legend(['raw_close', 'predicted_close'], loc='upper left')
    plt.savefig(model_type + "_" + input_output_type + "_" + str(len(predicted_close)) + "_test_price_comparison.png")
    plt.show()

def model_score(model, X_test, y_test):
    testing_score = model.evaluate(X_test, y_test, verbose=0) * 100
    return testing_score

def build_train_test(train_norm_df, pastDay, futureDay):
    X_train, Y_train, X_test = [], [], []
    for i in range(train_norm_df.shape[0] - pastDay + 1):
        if i < (train_norm_df.shape[0] - futureDay - pastDay + 1):
            X_train.append(np.array(train_norm_df.iloc[i:i + pastDay]))
            Y_train.append(np.array(train_norm_df.iloc[i + pastDay:i + pastDay + futureDay]["close"]))
        X_test.append(np.array(train_norm_df.iloc[i:i + pastDay]))
    return np.array(X_train), np.array(Y_train), np.array(X_test)

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

def get_prediction_date(future_day):
    predicted_date_list = []
    for index in range(1, future_day+1):
        predicted_date_list.append("Prediction " + str(index))
    return predicted_date_list

def get_nan_price(future_day):
    nan_price_list = []
    for index in range(1, future_day+1):
        nan_price_list.append(np.nan)
    return nan_price_list

if __name__ == "__main__":
    precheck()
    info("Start to read csv to dataframe.")
    raw_df = utils.read_csv_to_df(csv_raw_file)
    close_df = raw_df['close'].to_frame().reset_index(drop=True)
    #print(raw_df)
    info("Start to normalize dataset.")
    test_norm_df = normalize_dataframe(raw_df)
    print(test_norm_df)

    #csv_norm_file = "./norm_2330_debug.csv"
    #info("Create normalize dataset for cache.")
    #utils.df_to_csv(test_norm_df, csv_norm_file)
    #test_norm_df = utils.read_csv_to_df(csv_norm_file)

    info("Start to build testing data.")
    X_train, Y_train, X_test = build_train_test(test_norm_df, past_day, future_day)
    info("Feature shape: %s" % str(X_train.shape))
    info("Ground truth shape: %s" % str(Y_train.shape))
    final_test_for_all = X_test
    first_half_close_df = test_norm_df['close'].iloc[0:past_day].to_frame()
    info("=================================")

    stock_rnn_model = StockRnnModel(input_output_type, model_type, X_train.shape, Y_train.shape)
    model = stock_rnn_model.model
    model.load_weights(model_type + "_stock_" + input_output_type + "_inference.h5")

    testing_score = model_score(model, X_train, Y_train)
    info("=================================")
    info("Testing MSE: %.5f%%" % testing_score)
    info("=================================")

    predicted_close = model.predict(final_test_for_all)
    print(predicted_close.shape)
    predicted_close = stock_rnn_model.get_avereage_predicted_close(predicted_close, "mean")
    predicted_close_df = pd.DataFrame({'close': predicted_close[:, 0]})

    predicted_norm_df = pd.concat([first_half_close_df, predicted_close_df], ignore_index=True)
    predicted_df = denormalize_dataframe(close_df, predicted_norm_df)
    info("Predicted close for whole dataset:")
    print(predicted_df)
    info("Change percentage error: %s%%" % get_change_error(close_df, predicted_df))
    close_list = close_df['close'].values.tolist()
    close_list = close_list + get_nan_price(future_day)
    predicted_df['raw_close'] = close_list
    date_list = raw_df['date'].values.tolist()
    date_list = date_list + get_prediction_date(future_day)
    predicted_df['date'] = date_list
    utils.df_to_csv(predicted_df, "./diff.csv")
    if show_interval != "all":
        close_df = close_df.tail(show_interval)
        predicted_df = predicted_df.tail(show_interval)
    info("Change percentage error: %s%%" % get_change_error(close_df, predicted_df))
    plot_stock(predicted_df)

