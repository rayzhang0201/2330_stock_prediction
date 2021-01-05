import utils
import re
import requests
import datetime
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import os, sys

if len(sys.argv) < 3:
    print("Usage:")
    print("  python ./get_latest_stock_info.py ${download_csv_path} ${merged_csv_path}")
    sys.exit(0)

default_csv = "./2330_20201112_20201204.csv"
download_csv = sys.argv[1]
merged_csv = sys.argv[2]

def info(message):
    print("%s --- %s" % (datetime.datetime.now(), message))

def rename_dataframe(df):
    return df.rename(columns={'Date': 'date', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close',
                              'Change': 'change', 'Change%': 'change_percentage', 'Volume(\'000 shares)': 'volume'})

def formalize_data(data):
    new_data = re.sub(",", "", data)
    new_data = "".join(new_data.split())
    return new_data

def get_year_month_data(year, month, url):
    pageRequest = requests.get("https://stock.wearn.com/" + url + ".asp?Year=" + str(year) + "&month=" + str(month) +
                               "&kind=2330")
    soup = BeautifulSoup(pageRequest.content, 'html.parser')
    soup.encoding = 'utf-8'
    return soup

def get_margin_data(soup, year, month):
    margin_trading = []
    short_selling = []
    table = soup.find('table')
    table_rows = table.find_all('tr')
    for tr in table_rows:
        if tr.has_attr('class'):
            td = tr.find_all('td')
            row = [tr.text for tr in td]
            mod_year = formalize_data(row[0]).split("/", 1)[0]
            mod_month = formalize_data(row[0]).split("/", 1)[1]
            if not (mod_year == year and month in mod_month):
                continue
            margin_trading.append(formalize_data(row[1]))
            short_selling.append(formalize_data(row[3]))
    return list(reversed(margin_trading)), list(reversed(short_selling))

def get_institutional_investor(soup, year, month):
    da = []
    it = []
    d = []
    fi = []
    table = soup.find('table')
    table_rows = table.find_all('tr')
    for tr in table_rows:
        if tr.has_attr('class'):
            td = tr.find_all('td')
            row = [tr.text for tr in td]
            mod_year = formalize_data(row[0]).split("/", 1)[0]
            mod_month = formalize_data(row[0]).split("/", 1)[1]
            if not(mod_year == year and month in mod_month):
                continue
            mod_date = str(int(formalize_data(row[0]).split("/", 1)[0]) + 1911) + "/" + formalize_data(row[0]).split("/", 1)[1]
            da.append(mod_date)
            it.append(formalize_data(row[1]))
            d.append(formalize_data(row[2]))
            fi.append(formalize_data(row[3]))
    return list(reversed(da)), list(reversed(it)), list(reversed(d)), list(reversed(fi))

def get_detail_info_df(rename_df):
    year_month_pair_list = []
    for index, row in rename_df.iterrows():
        year = row['date'].split("/")[0]
        mod_year = str(int(year) - 1911)
        month = row['date'].split("/")[1]
        year_month_pair = mod_year + "/" + month
        if not year_month_pair in year_month_pair_list:
            year_month_pair_list.append(year_month_pair)

    date = []
    investment_trust = []
    dealer = []
    foreign_invenstor = []
    margin_trading = []
    short_selling = []
    for year_month_pair in year_month_pair_list:
        year = year_month_pair.split("/")[0]
        month = year_month_pair.split("/")[1]
        soup = get_year_month_data(year, month, "netbuy")
        tmp_date, tmp_investment_trust, tmp_dealer, tmp_foreign_invenstor = get_institutional_investor(soup, year, month)
        date = date + tmp_date
        investment_trust = investment_trust + tmp_investment_trust
        dealer = dealer + tmp_dealer
        foreign_invenstor = foreign_invenstor + tmp_foreign_invenstor
        #print(tmp_date, tmp_investment_trust, tmp_dealer)
        soup = get_year_month_data(year, month, "acredit")
        tmp_margin_trading, tmp_short_selling = get_margin_data(soup, year, month)
        margin_trading = margin_trading + tmp_margin_trading
        short_selling = short_selling + tmp_short_selling
        #print(tmp_margin_trading, tmp_short_selling)

    detail_info_dict = {
        'date': date,
        'short_selling': short_selling,
        'margin_trading': margin_trading,
        'investment_trust': investment_trust,
        'dealer': dealer,
        'foreign_invenstor': foreign_invenstor
    }
    detail_info_df = pd.DataFrame.from_dict(detail_info_dict)
    return detail_info_df

def merge_rename_df_and_detail_info_df(default_last_date, rename_df, detail_info_df):
    target_rename_df_index = rename_df[rename_df['date'].str.match(default_last_date)].index.values.astype(int)[0] + 1
    target_detail_info_df_index = detail_info_df[detail_info_df['date'].str.match(default_last_date)].index.values.astype(int)[0] + 1
    rename_df = rename_df.iloc[target_rename_df_index:].reset_index(drop=True)
    detail_info_df = detail_info_df.iloc[target_detail_info_df_index:].reset_index(drop=True)
    detail_info_df = detail_info_df.drop(['date'], axis=1)
    recent_df = pd.concat([rename_df, detail_info_df], axis=1)
    return recent_df

def calculate_ma_and_bband(default_df, recent_df):
    recent_ma = []
    recent_bband_up = []
    recent_bband_down = []
    default_last_20_close = default_df.tail(20)['close'].values.tolist()
    default_last_20_ma = default_df.tail(20)['ma'].values.tolist()
    for index, row in recent_df.iterrows():
        tmp_ma = (default_last_20_ma[-1] * 20 - default_last_20_close.pop(0) + float(row['close'])) / 20
        default_last_20_ma.pop(0)
        recent_ma.append(tmp_ma)
        default_last_20_ma.append(tmp_ma)
        default_last_20_close.append(float(row['close']))
        sd_total_np = np.array(default_last_20_close)
        s_sd = np.std(sd_total_np, ddof=0)
        recent_bband_up.append(tmp_ma + 2 * s_sd)
        recent_bband_down.append(tmp_ma - 2 * s_sd)
    recent_df['ma'] = recent_ma
    recent_df['bband_up'] = recent_bband_up
    recent_df['bband_down'] = recent_bband_down
    return recent_df

def rename_date(df):
    date_list = []
    for index in range(len(df)):
        date_list.append(re.sub("/0", "/", df.iloc[index]['date']))
    df['date'] = date_list
    return df

if __name__ == '__main__':
    info("Load default 2330 csv.")
    default_df = utils.read_csv_to_df(default_csv)
    #print(default_df)

    info("Load downloaded 2330 csv.")
    new_df = utils.read_csv_to_df(download_csv)
    rename_df = rename_dataframe(new_df).iloc[::-1].reset_index(drop=True)
    rename_df = rename_date(rename_df)
    #print(rename_df)

    info("Try to get detail information from other websites.")
    detail_info_df = get_detail_info_df(rename_df)
    detail_info_df = rename_date(detail_info_df)
    #print(detail_info_df)

    info("Try to merge downloaded csv and detail information.")
    default_last_date = default_df.tail(1).iloc[0]['date']
    recent_df = merge_rename_df_and_detail_info_df(default_last_date, rename_df, detail_info_df)
    #print(recent_df)
    recent_df = calculate_ma_and_bband(default_df, recent_df)
    info("Create new csv successfully.")

    info("Try to merge default csv and new csv.")
    final_df = pd.concat([default_df, recent_df], ignore_index=True)
    final_df = final_df.dropna()
    utils.df_to_csv(final_df, merged_csv)
    print(final_df)
    info("Check the file: %s" % merged_csv)
    info("Create lastest csv successfully.")



