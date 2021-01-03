from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU
import numpy as np
import statistics

class StockRnnModel:
    def __init__(self, input_output_type, model_type, input_shape, output_shape):
        self.input_output_type = input_output_type
        if self.input_output_type == "m2o":
            self.model = self.build_many_to_one_model(model_type, input_shape)
        elif self.input_output_type == "m2m":
            self.model = self.build_many_to_many_model(model_type, input_shape, output_shape)

    def build_many_to_one_model(self, model_type, input_shape):
        model = Sequential()
        if model_type == "GRU":
            model.add(GRU(input_shape[1], input_length=input_shape[1], input_dim=input_shape[2]))
        else:
            model.add(LSTM(input_shape[1], input_length=input_shape[1], input_dim=input_shape[2]))
        model.add(Dense(1))
        model.compile(loss="mse", optimizer="adam")
        model.summary()
        return model

    def build_many_to_many_model(self, model_type, input_shape, output_shape):
        model = Sequential()
        if model_type == "GRU":
            model.add(GRU(units=input_shape[1], input_shape=(input_shape[1], input_shape[2])))
        else:
            model.add(LSTM(units=input_shape[1], input_shape=(input_shape[1], input_shape[2])))
        model.add(Dense(output_shape[1]))
        model.compile(loss="mse", optimizer="adam")
        model.summary()
        return model

    def get_avereage_predicted_close(self, predicted_close, method):
        if self.input_output_type == "m2m":
            new_predicted_close = []
            predicted_close_dict = {}
            predicted_index = 0
            for row in predicted_close:
                inner_index = predicted_index
                for column in row:
                    if not str(inner_index) in predicted_close_dict:
                        predicted_close_dict[str(inner_index)] = [column]
                    else:
                        predicted_close_dict[str(inner_index)].append(column)
                    inner_index = inner_index + 1
                predicted_index = predicted_index + 1
            for key, value in predicted_close_dict.items():
                if method == "mean":
                    new_predicted_close.append([statistics.mean(value)])
                elif method == "median":
                    new_predicted_close.append([statistics.median(value)])
            return np.array(new_predicted_close)
        return predicted_close
