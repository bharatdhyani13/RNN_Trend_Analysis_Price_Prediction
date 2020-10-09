# RNN_Trend_Analysis_Price_Prediction
Trend analysis and price prediction for Stocks and Crypto Currencies.

This repository has 4 folders :
1. Classification
2. Regression
3. training_data
4. plots

**training_data** folder has per minute data of BTC,ETH,BCH and LTC.
**plots** folder has per minute data of BTC,ETH,BCH and LTC.


We handle this problem in two different ways - classification & regression

For the **classification** problem we predict 3 minutes into the future that weather the price of the currency will go up or not.
We do this by training the model with the big 4 of the crypto currrency market with their closing price and volume.
We tried it with just closing price and also with single currency of which we are analyzing.



# Take a look at the Accuracy and Loss Plots:
![](https://github.com/bharatdhyani13/RNN_Trend_Analysis_Price_Prediction/plots/class_BTH_Loss_Acc.PNG)
![](https://github.com/bharatdhyani13/RNN_Trend_Analysis_Price_Prediction/plots/class_BCH_Loss_Acc.PNG)
![](https://github.com/bharatdhyani13/RNN_Trend_Analysis_Price_Prediction/plots/class_LTC_Loss_Acc.PNG)
![](https://github.com/bharatdhyani13/RNN_Trend_Analysis_Price_Prediction/plots/class_ETH_Loss_Acc.PNG)


# How does GAN Works?
Basic mathematical foundation for GAN is to find the PDF(probability distribution function) of the dataset **D** and get its parameters.

Now genarate random samples from this which will be our **D'**. 
Now if D and D' have the same distribution, their values should be similar.

We try to **maximize Log Loss** at the Discriminative part of GAN, as in order to dicriminate between the two data,
we assign prediction for D = 0 and D' = 1.
This technique is used to distinguish between distributions. Greater the loss of this model, higher is the similarity.

# Training
We use DCGAN ([read it here](https://medium.com/@jonathan_hui/gan-dcgan-deep-convolutional-generative-adversarial-networks-df855c438f)). Training is done on 62,608 images.
Images are loaded into memory batch wis

# Sources
https://medium.com/@jonathan_hui/gan-dcgan-deep-convolutional-generative-adversarial-networks-df855c438f
https://medium.com/@nikitasharma_43692/my-mangagan-building-my-first-generative-adversarial-network-2ec1920257e3?sk=0eef45a3ef8d8b13f23f620abe48ef07
