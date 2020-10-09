# RNN_Trend_Analysis_Price_Prediction
Trend analysis and price prediction for Stocks and Crypto Currencies.

This repository has 4 folders :
1. Classification
2. Regression
3. training_data
4. plots

**training_data** folder has per minute data of BTC,ETH,BCH and LTC. This is the data that i found from https://pythonprogramming.net/cryptocurrency-recurrent-neural-network-deep-learning-python-tensorflow-keras/ 

Other data from the code is fetched from Alpha Vantage API which provides that data for both stock and crypto currencies.


**plots** folder has all the plots images.


# Classification
We handle this problem in two different ways - classification & regression

For the **classification** problem we predict 3 minutes into the future that weather the price of the currency will go up or not.

We do this by training the model with the big 4 of the crypto currrency market with their closing price and volume.

We tried it with just closing price and also with single currency of which we are analyzing.

# Take a look at the Accuracy and Loss Plots:
![](https://github.com/bharatdhyani13/RNN_Trend_Analysis_Price_Prediction/blob/main/plots/class_BTC_Loss_Acc.png)
![](https://github.com/bharatdhyani13/RNN_Trend_Analysis_Price_Prediction/blob/main/plots/class_BCH_Loss_Acc.png)
![](https://github.com/bharatdhyani13/RNN_Trend_Analysis_Price_Prediction/blob/main/plots/class_LTC_Loss_Acc.png)
![](https://github.com/bharatdhyani13/RNN_Trend_Analysis_Price_Prediction/blob/main/plots/class_ETH_Loss_Acc.png)

Now this can be done with daily price also predicting the price of the currency in the following few days.

# Regression
For the **regression** problem we take a windiw of 40 days or 60 days to predict the price for the next day using LSTM model.

We predict the prices for the validation set and plot them with the actual prices and analyze the trend of our predicted data and original data.

Moreover in addition to this we added a funtionality to predict the future prices, so we can analyze the probable trend predicted by the model which is 2 months in future from current date.

# Crypto(BTC)
![](https://github.com/bharatdhyani13/RNN_Trend_Analysis_Price_Prediction/blob/main/plots/trend_analysis_price_loss(0.0055).png) 
![](https://github.com/bharatdhyani13/RNN_Trend_Analysis_Price_Prediction/blob/main/plots/trend_analysis_price_volume_loss(0.005).png) 
![](https://github.com/bharatdhyani13/RNN_Trend_Analysis_Price_Prediction/blob/main/plots/experiment_window_size.png) 

**1.(Close Price as feature)** 
**2.(Close Price, and Volume Traded as feature)** 
**3.(Experimenting with hyper-parameters)**

# Crypto Future(BTC)
![](https://github.com/bharatdhyani13/RNN_Trend_Analysis_Price_Prediction/blob/main/plots/trend_analysis_price_future_prediction.png)


# Stocks(MSFT)
![](https://github.com/bharatdhyani13/RNN_Trend_Analysis_Price_Prediction/blob/main/plots/stock_close_vol_permin(volume%20feature%20doesn't%20do%20much%20difference%20in%20stocks).png)
![]
![](https://github.com/bharatdhyani13/RNN_Trend_Analysis_Price_Prediction/blob/main/plots/stocks_close_perday.png)


# Stocks Future(MSFT)
![](https://github.com/bharatdhyani13/RNN_Trend_Analysis_Price_Prediction/blob/main/plots/stocks_close_permin.png)
![](https://github.com/bharatdhyani13/RNN_Trend_Analysis_Price_Prediction/blob/main/plots/stock_close_vol_perday(volume%20feature%20doesn't%20do%20much%20difference%20in%20stocks).png)

Stocks trend for future follows is quite more reliable when we work on daily data instead of per minutes data.
The model understands the features more precisely for each data stocks data.

# RNN Trend Ananlysis Folder
![](https://github.com/bharatdhyani13/RNN_Trend_Analysis_Price_Prediction/blob/main/plots/btc_trend_analysis_old.png)
![](https://github.com/bharatdhyani13/RNN_Trend_Analysis_Price_Prediction/blob/main/plots/btc_trend_analysis_old_future_pred.png)

This is a directory from an old project that i worked on. I enhanced the code and added a few more functionalities.

# Training
We use Bidirectional LSTM which are an extension of traditional LSTMs that can improve model performance on sequence classification problems. In problems where all timesteps of the input sequence are available, Bidirectional LSTMs train two instead of one LSTMs on the input sequence. We fetch the data from Alpha Vantage API.

# Sources

https://pythonprogramming.net/cryptocurrency-recurrent-neural-network-deep-learning-python-tensorflow-keras/ 
