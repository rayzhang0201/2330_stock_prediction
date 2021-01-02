# 2330_stock_prediction
# This is final project for NCTU 2020/09 Deep Learning lesson.

## 1. To run the program, please install the following packages first.

* python 3.7
* Keras 2.4.3
* beautifulsoup4 4.9.3
* matplotlib 3.3.3
* numpy 1.19.4
* pandas 1.1.5
* requests 2.25.1
* scikit-learn 0.24.0
* tensorflow 2.4.0

## 2. Download the latest dataset for extended training or testing

> If you want to make the model learn from less dataset, you can skip this step.
> 
> Please modify the content of file "2330\_20201112\_20201204.csv" and continue.

2330 TSMC Price：<https://invest.cnyes.com/twstock/TWS/2330/history>

Click the link and choose the date interval from 2020/12/04 to the date you want.

After that, download the csv and run "get\_latest\_stock\_info.py".

It will merge the default dataset "2330\_20201112\_20201204.csv" and the file you download.

Also, please assign merged csv name with the parameter. 

`# python get_latest_stock_info.py ${download_csv_path} ${merged_csv_path}`


## 3. Let's start to train the RNN model

Usage:

` # python predict_stock.py ${merged_csv_path} [m2o|m2m] [LSTM|GRU] ${past_day} ${future_day} `
> If you skip the step 2, please use "2330\_20201112\_20201204.csv" as ${merged\_csv\_path} parameter.


Ex.

Train a LSTM model and input 20 past day close prices to predict 1 future day price.

`# python predict_stock.py ./2330_new.csv m2o LSTM 20 1 `

Train a GRU model and input 20 past day close prices to predict 5 future day prices.

***Don't worry, we will use "mean" strategy to make multiple close prices become 1 close price.***

`# python predict_stock.py ./2330_new.csv m2m GRU 20 5`

> This action will generate a **training weight** with past day and future day based on LSTM or GRU model.
> 
> You can use m2o (many to one) or (many to many) to get the prediction result 
> 
> **Please make sure your environment is GUI compatible system.**
> 
> **It will generate a training price image result in the end.**

### 4. Let's start to test the RNN model

Usage:

` # python show_test_result.py ${merged_csv_path} [m2o|m2m] [LSTM|GRU] ${past_day} ${future_day} ${show_interval}`
> You can use the dataset you want to test as ${merged\_csv\_path} parameter.
>
> Using "all" for ${show\_interval} will show all prediction.
>
> Using positive ineger for ${show\_interval} will only show the tail of result.

Ex.

Test a LSTM model and input 20 past day close prices to predict 1 future day price from **last training weight**. And only show the last 60 days result.

`# python show_test_result.py ./2330_new.csv m2o LSTM 20 1 60`

Test a GRU model and input 20 past day close prices to predict 5 future day prices from **last training weight**. And only show the last 60 days result.

`# python show_test_result.py ./2330_new.csv m2m GRU 20 5 60`

### 5. Enjoy the inference result
