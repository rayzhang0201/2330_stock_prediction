import pandas as pd
import json

def read_csv_to_df(file):
    train = pd.read_csv(file)
    return train

def dict_to_json(dict, file):
    json.dump(dict, open(file, 'w'))

def df_to_csv(df, file):
    df.to_csv(file, index=False, header=True)

