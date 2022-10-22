# Data manipulation
# ==============================================================================
import pandas as pd
import numpy as np
import datetime
from cryptocmd import CmcScraper

# Plots
# ==============================================================================
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
plt.style.use('ggplot')

# Bitcoin colors
# ==============================================================================
palette_btc = {'orange': '#f7931a',
               'white' : '#ffffff',
               'gray'  : '#4d4d4d',
               'blue'  : '#0d579b',
               'green' : '#329239'
              }

# Modelling and Forecasting
# ==============================================================================
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRFClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import StackingClassifier
from lightgbm import LGBMClassifier
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix
import pickle
from scipy.stats import gmean

# Data download
# ==============================================================================
# Scraper is initialized, symbol, start and end of download are included
scraper = CmcScraper('BTC', '01-06-2017', '31-03-2022')

# Transform collected data into a dataframe
BTC_ori = scraper.get_dataframe()
BTC_ori.sort_values(by='Date', ascending=True, inplace=True)

pd.set_option('display.max_columns', None)
pd.reset_option('display.max_columns')

# Data preparation
# ==============================================================================
BTC_ori = BTC_ori.reset_index()

#del BTC_ori['index']
del BTC_ori['Volume']
del BTC_ori['index']

BTC_ori['price_change'] = BTC_ori['Close']
BTC_ori['high_open'] = BTC_ori['Close']
BTC_ori['low_open'] = BTC_ori['Close']

for i in range(1, len(BTC_ori)):
    BTC_ori['price_change'][i] = (BTC_ori['Close'][i] - BTC_ori['Close'][i-1]) / BTC_ori['Close'][i-1]
    BTC_ori['high_open'][i] = (BTC_ori['High'][i] - BTC_ori['Open'][i]) / BTC_ori['Open'][i]
    BTC_ori['low_open'][i] = (BTC_ori['Low'][i] - BTC_ori['Open'][i]) / BTC_ori['Open'][i]
BTC_ori['price_change'][0] = 0

# Momentum
BTC_ori['mom'] = np.where(BTC_ori['price_change'] > 0, 1, 0)
BTC_ori['mom3'] = BTC_ori.mom.rolling(3).mean()
BTC_ori['mom5'] = BTC_ori.mom.rolling(5).mean()
BTC_ori['mom10'] = BTC_ori.mom.rolling(10).mean()

# Moving Average
window_size = (30, 5)
for i in window_size:
    j = 0
    globals()['moving_average%s' % i] = [None] * (i - 1)
    while j < len(BTC_ori['Close']) - i + 1:
        # Store elements from i to i + window in list to get the current window
        window = BTC_ori['Close'][j:j+i]
        # calculate the average of current window
        window_average = int(sum(window) / i)
        # Store the average of current window in moving average list
        globals()['moving_average%s' % i].append(window_average)
        # shift window to right by one position
        j += 1

BTC_ori['ma30'] = moving_average30
BTC_ori['ma5'] = moving_average5
BTC = BTC_ori[BTC_ori['Date'] >= '2018-01-01'].reset_index(drop=True)

train = BTC[BTC['Date'] < '2021-01-01']
test = BTC[BTC['Date'] >= '2021-01-01']
# 7:3

# Hurst for MA30
def get_hurst_exponent_30(time_series, max_lag = 20):
    lags = range(2, max_lag)
    # variances of the lagged differences
    tau = []
    for lag in lags:
        t_s_1 = time_series[lag:]
        t_s_2 = time_series[:-lag]
        cc = np.std(np.subtract(t_s_1, t_s_2))
        tau.append(cc)
    # calculate the slope of the log plot -> the Hurst Exponent
    reg = np.polyfit(np.log(lags), np.log(tau), 1)

    hurst = reg[0] #*2

    return hurst

hurst30 = []
for i in range(len(BTC)):
    hurst30.append(get_hurst_exponent_30(moving_average30[-len(BTC)-30:][i:i+30]))


train_set = {'Hurst': hurst30[:len(BTC[BTC['Date'] < '2021-01-01'])], 'mom3': train['mom3'], 'mom5': train['mom5'], 'mom10': train['mom10'], 'ma30': train['ma30'], 'ma5': train['ma5'], 'price_change': train['price_change'], 'high_open': train['high_open'], 'low_open': train['low_open'], 'OpenPrice': train['Open'], 'ClosePrice': train['Close'], 'target': train['mom']}
train_set = pd.DataFrame(train_set)
test_set = {'Hurst': hurst30[len(BTC[BTC['Date'] < '2021-01-01']):], 'mom3': test['mom3'], 'mom5': test['mom5'], 'mom10': test['mom10'], 'ma30': test['ma30'], 'ma5': test['ma5'], 'price_change': test['price_change'], 'high_open': test['high_open'], 'low_open': test['low_open'], 'OpenPrice': test['Open'], 'ClosePrice': test['Close'], 'target': test['mom']}
test_set = pd.DataFrame(test_set)

train_set_1 = train_set[["Hurst","mom3","mom5", 'mom10', 'ma30', 'ma5', 'price_change', 'high_open', 'low_open']].shift(1)
train_data = train_set_1.values
min_max_scaler = preprocessing.MinMaxScaler()
train_data_scaled = min_max_scaler.fit_transform(train_data)
train_set_1 = pd.DataFrame(train_data_scaled)
train_set_1["target"] = train_set["target"]
train_set_1 = train_set_1.dropna()
train_set_1.set_axis(["Hurst","mom3","mom5", 'mom10', 'ma30', 'ma5', 'price_change', 'high_open', 'low_open', 'target'], axis=1, inplace=True)

test_set_1 = test_set[["Hurst","mom3","mom5", 'mom10', 'ma30', 'ma5', 'price_change', 'high_open', 'low_open']].shift(1)
test_data = test_set_1.values
min_max_scaler = preprocessing.MinMaxScaler()
test_data_scaled = min_max_scaler.fit_transform(test_data)
test_set_1 = pd.DataFrame(test_data_scaled)
test_set_1["target"] = test_set["target"].reset_index(drop=True)
test_set_1 = test_set_1.dropna()
test_set_1.set_axis(["Hurst","mom3","mom5", 'mom10', 'ma30', 'ma5', 'price_change', 'high_open', 'low_open', 'target'], axis=1, inplace=True)

def combs(a):
    if len(a) == 0:
        return [[]]
    cs = []
    for c in combs(a[1:]):
        cs += [c, c+[a[0]]]
    return cs
features = ["Hurst", "mom3", "mom5", 'mom10', 'ma30', 'ma5', 'price_change', 'high_open', 'low_open']
combinations = combs(features)

x_train = train_set_1[['mom5', 'mom10', 'price_change']]
y_train = train_set_1['target']
x_test = test_set_1[['mom5', 'mom10', 'price_change']]
y_test = test_set_1['target']
model = ('LGBM', LGBMClassifier())

model[1].fit(x_train, y_train)
pred = model[1].predict(x_test)
accuracy = model[1].score(x_test, y_test)
print(model[0], 'accuracy:', accuracy)
print('Confusion Matrix:\n', confusion_matrix(pred, y_test))

# save the model
LGBM_model = 'LGBM_model.pkl'
pickle.dump(model[1], open(LGBM_model, 'wb'))

revList = []
returns = []
acc_rev = 0
maxdrawdown = 0
for i in range(1, len(test_set)):
    if pred[i-1] == 1:
        rev = test_set['ClosePrice'].iloc[i] - test_set['OpenPrice'].iloc[i]
        daily_ret = rev / test_set['OpenPrice'].iloc[i]
        acc_rev += rev
        revList.append(rev)
    if pred[i-1] == 0:
        rev = test_set['OpenPrice'].iloc[i] - test_set['ClosePrice'].iloc[i]
        daily_ret = rev / test_set['ClosePrice'].iloc[i]
        acc_rev += rev
        revList.append(rev)
    if rev < maxdrawdown:
        maxdrawdown = rev
    daily_ret += 1
    returns.append(daily_ret)

ret_mean = gmean(returns)
ret_mean -= 1
pct_change = test_set['ClosePrice'][1:].pct_change()
risk = pct_change.std()
sharpe = ret_mean / risk * (365 ** 0.5)
print('Sharpe Ratio:', sharpe)
print('Max drawdown:', maxdrawdown)

# TenFold Rolling Validation
Date = BTC['Date'][1:]
TenFold = BTC[['mom5', 'mom10', 'price_change']].shift(1)
data = TenFold.values
min_max_scaler = preprocessing.MinMaxScaler()
data_scaled = min_max_scaler.fit_transform(data)
TenFold = pd.DataFrame(data_scaled)
TenFold["target"] = BTC["mom"]
TenFold = TenFold.dropna()
TenFold.set_axis(["mom5", "mom10", 'price_change', 'target'], axis=1, inplace=True)

X_test = TenFold[['mom5', 'mom10', 'price_change']]
Y_test = TenFold['target']

split = 0.7
model = ('LGBM', LGBMClassifier())

loop = 0
acc = []
for i in range(len(TenFold) - len(TenFold)//10 + 1):
    model[1].fit(X_test.iloc[i : i + round(len(TenFold)//10 * split)], Y_test.iloc[i : i + round(len(TenFold)//10 * split)])
    pred = model[1].predict(X_test.iloc[i + round(len(TenFold)//10 * split) : i + len(TenFold)//10])
    accuracy = model[1].score(X_test.iloc[i + round(len(TenFold)//10 * split) : i + len(TenFold)//10], Y_test.iloc[i + round(len(TenFold)//10 * split) : i + len(TenFold)//10])
    acc.append(accuracy)
    loop += 1

# create date format
Date = []
for i in range(loop):
    Date.append("{} ~ {}".format(BTC['Date'].iloc[i].strftime('%Y-%m-%d'), BTC['Date'].iloc[i+len(TenFold)//10].strftime('%Y-%m-%d')))

fin_res = {}
fin_res['Period'] = Date
fin_res['Accuracy'] = acc
fin_res = pd.DataFrame(fin_res)
print(fin_res)

quantile = fin_res.quantile([.0, .25, .5, .75, 1.], axis = 0)
print(quantile)

print(fin_res[fin_res['Accuracy'] == max(acc)])
print(fin_res[fin_res['Accuracy'] == min(acc)])