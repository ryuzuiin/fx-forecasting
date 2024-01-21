# Fx Forecasting Model From Zero

This note shares some of my experiences in time series analysis and modeling, providing a comprehensive guide from handling missing values, feature analysis, to training and cross-validation processes. This project does not delve into model hyperparameter tuning and focuses primarily on feature analysis and engineering. It can serve as a baseline for future predictive modeling projects related to time-series events.

Vesrion.1 Baseline

## 1. Data Description

Daily USDJPY OHLC data, downloaded from Yahoo Finance. The time span covers from October 30, 1996, to January 12, 2024.

## 2. Problem Analysis and Roadmap

For this dataset, I have framed the problem as predicting the USDJPY price three days ahead.

In other words, this is a time series forecasting problem, and all time series forecasting problems are essentially autoregressive (AR) problems. In this case, we are trying to predict the future price of USDJPY based on the OHLC (Open, High, Low, Close) data. This can be viewed as a regression problem with autoregressive properties.

In a typical regression problem, we have an equation like Y = aX1 + bX2, where Y is the dependent variable, and X1 and X2 are independent variables. The goal is to find the optimal values of 'a' and 'b' to make the predicted value 'Y' as close as possible to the real value 'Y'. However, autoregressive models are different. In autoregression, both the dependent variable (Y) and the independent variables (X1, X2) are the same variable, which is Y itself. In other words, Y(t) = aY(t-1) + bY(t-2), where Y(t-1) is the value of Y at time t-1, and Y(t-2) is the value of Y at time t-2. This means that the current value of Y depends on past values of Y. Hence, both the independent and dependent variables are the same, which is known as autoregression.

Based on the OHLC data we have(we have X1,X2,...), the training data mainly consists of past data, and we want to predict future data. In other words, the current value of Y depends on past values of Y, making it an autoregressive problem.

Therefore, I will approach the data with both a regression problem perspective and an autoregressive problem perspective, taking into account the characteristics of financial data. I will conduct Exploratory Data Analysis (EDA) and feature engineering from these different angles.

## 3. Application and Metrics
### 3.1 Application
To make a signal, help traders to make strategies based on it.
### 3.2 Metrics

## 4. Exploratory Data Analysis
### 4.1 Regression model perspective
#### 4.1.1 Pearsons Correlation


## 5. Feature engineering

## 6. Models
### 6.1 ARIMA
employing ARIMA model as an analysis tool to indicate that the target has no trend and no season. Since the time intervals of validation and test set are not constant, ARIMA model can not be applied to do the out 0ff sample forcast. In this context, it violates the initial assumption of the ARIMA model, which requires all data samples to be collected at the same time intervals.

