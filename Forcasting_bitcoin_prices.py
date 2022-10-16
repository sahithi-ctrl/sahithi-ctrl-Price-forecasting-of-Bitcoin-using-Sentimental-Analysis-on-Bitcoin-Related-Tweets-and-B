#!/usr/bin/env python
# coding: utf-8

# # Quick Introduction
# 
# In this project, we aim to predict the price of bitcoin using sentiment analysis and finalcial features to support the sentiment.
# 
# We will compute the error (Mean absolute error of two models we developed to predict the prices)
# 
# The two models are:
# 1. Random Forrest Regression
# 2. Linear Regression
# 
# The sentiment was detected using Vader, used to detect polarity

# In[2]:


#import libraries to import the data
import pandas as pd
import numpy as np
import re
get_ipython().system('pip install clean-text')
get_ipython().system('pip install requirements.txt')



# In[104]:


# Since Elon only started tweeting about crypto from 2019, lets only include tweets from 2019,2020,2021,2022

elon_tweets_2019 = pd.read_csv('2019.csv')
elon_tweets_2020 = pd.read_csv('2020.csv')
elon_tweets_2021 = pd.read_csv('2021.csv')
elon_tweets_2022 = pd.read_csv('2022.csv')

# we would also be forcasting in bitcoin tweets from 2021 on general, to make our results more generalizable. we chose 2021
# because the volume of bitcoin tweets in 2021 has skyrocketed.

bitcoin_tweets_2021 = pd.read_csv('Bitcoin_tweets.csv')


# In[105]:


# combining all tweets from elon musk 2019-2022
elon_tweets = pd.concat([elon_tweets_2019, elon_tweets_2020, elon_tweets_2021,elon_tweets_2022], axis = 0)
elon_tweets = elon_tweets.reset_index(drop=True)


# In[4]:



elon_tweets


# In[5]:


elon_tweets.describe()


# In[106]:


# we only need the tweet and the data of the tweet for analysis

elon_tweets = elon_tweets[['date','tweet']]
elon_tweets


# In[107]:


pd.options.mode.chained_assignment = None
# let's remove the timestamp from the date field and only include the date
elon_tweets.date = elon_tweets.date.str[:10] #tweets.date.str[:10]


# In[108]:


def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)        
    return input_txt
def clean_tweets(tweets):
    #remove twitter Return handles (RT @xxx:)
    tweets = np.vectorize(remove_pattern)(tweets, "RT @[\w]*:") 
    
    #remove twitter handles (@xxx)
    tweets = np.vectorize(remove_pattern)(tweets, "@[\w]*")
    
    #remove URL links (httpxxx)
    tweets = np.vectorize(remove_pattern)(tweets, "https?://[A-Za-z0-9./]*")
    
    #remove special characters, numbers, punctuations (except for #)
    tweets = np.core.defchararray.replace(tweets, "[^a-zA-Z0-9]", " ")
    tweets = np.core.defchararray.replace(tweets, "#", " ")
    tweets = np.core.defchararray.replace(tweets, "$", " ")
    tweets = np.core.defchararray.replace(tweets, "[^a-zA-Z]", " ")
   
    
    return tweets


# In[109]:


elon_tweets['tweet'] = clean_tweets(elon_tweets['tweet'])


# In[110]:


elon_tweets.reset_index(inplace=True, drop=True)


# In[111]:


elon_tweets['tweet'].head(50)


# In[112]:


elon_tweets['tweet'] = elon_tweets['tweet'].str.lower()
elon_tweets['tweet'] = elon_tweets['tweet'].str.lstrip()





# In[113]:


# after manually browsing the internet for crytocurrency hashtags and keywords, we found a comprehensive list of words listed below
crpyto_keywords = ["#Bitcoin","#BTC","#Cryptocurrency","#Crypto",'Bitcoin','BTC',"#Ethereum","Ethereum", "DOGE","dogecoin",'doge','shibainu','shiba','floki','snl',
                   'SNL', 'HOLD','hold','HODL','hodl','BabyDoge','babydoge']
                   


# In[114]:


# running the dataset against the above keywords to filter only bitcoin related tweets
elon_crypto = pd.DataFrame(columns = ['date', 'tweet'])

for i in range(len(elon_tweets)):
    cur = elon_tweets.loc[i].tweet
    cur = cur.lower().split()
    for j in cur:
            if j in crpyto_keywords:
                elon_crypto = elon_crypto.append({'date' : elon_tweets.loc[i].date, 'tweet' : elon_tweets.loc[i].tweet}, 
                ignore_index = True)

elon_crypto['date'] = pd.to_datetime(elon_crypto['date'], format='%Y-%m-%d')


            


# In[115]:


# after cleaning the tweets and using regex to retrieve only elon musk's tweets, we filtered our list from 24172 to 135 tweets
elon_crypto.shape
# importing libraries for sentiment analysis
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()


# In[116]:


scores_elon = []
# Declare variables for scores
compound_list1 = []
positive_list1 = []
negative_list1 = []
neutral_list1 = []
for i in range(elon_crypto['tweet'].shape[0]):
#print(analyser.polarity_scores(sentiments_pd['text'][i]))
    compound = analyzer.polarity_scores(elon_crypto['tweet'][i])["compound"]
    pos = analyzer.polarity_scores(elon_crypto['tweet'][i])["pos"]
    neu = analyzer.polarity_scores(elon_crypto['tweet'][i])["neu"]
    neg = analyzer.polarity_scores(elon_crypto['tweet'][i])["neg"]
    
    scores_elon.append({"Compound": compound,
                       "Positive": pos,
                       "Negative": neg,
                       "Neutral": neu
                  })


# In[117]:


sentiments_score_elon = pd.DataFrame.from_dict(scores_elon)
elon_crypto = elon_crypto.join(sentiments_score_elon)


# In[118]:


elon_crypto = elon_crypto[elon_crypto['date']>='2019-01-01']


# In[119]:


elon_crypto.reset_index(drop=True,inplace=True)


# In[126]:


elon_crypto

elon_bit_doge = elon_crypto.merge(bitusd,on='date', how='left')


# In[ ]:


elon_bit_doge.isnull().sum()


# In[ ]:


# let's perform similar operations on bitcoin_2021 dataframe, from cleaning the data to performing sentiment analysis

bitcoin_tweets_2021.shape


# In[ ]:


bitcoin_tweets_2021.columns


# In[ ]:


bitcoin_tweets_2021 = bitcoin_tweets_2021[['user_name','date','text','hashtags','user_verified','user_followers'
                                          ,'user_friends','user_favourites']]


# In[ ]:


bitcoin_tweets_2021 = bitcoin_tweets_2021[bitcoin_tweets_2021['user_verified'] ==True]
bitcoin_tweets_2021 = bitcoin_tweets_2021.reset_index(drop=True)


# In[ ]:


print(bitcoin_tweets_2021['user_followers'].median())
print(bitcoin_tweets_2021['user_followers'].mean())
print(bitcoin_tweets_2021['user_followers'].mode())


# In[ ]:


print(bitcoin_tweets_2021['user_favourites'].median())
print(bitcoin_tweets_2021['user_favourites'].mean())
print(bitcoin_tweets_2021['user_favourites'].mode())


# In[ ]:


bitcoin_tweets_2021_new = bitcoin_tweets_2021.loc[(bitcoin_tweets_2021['user_followers']>=77581) &
                                                  (bitcoin_tweets_2021['user_favourites']>=6166)]
bitcoin_tweets_2021_new


# In[ ]:


bitcoin_tweets_2021_new.reset_index(drop=True)


# In[ ]:


#cleaning the tweets
def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)        
    return input_txt
def clean_tweets(tweets):
    #remove twitter Return handles (RT @xxx:)
    tweets = np.vectorize(remove_pattern)(tweets, "RT @[\w]*:") 
    
    #remove twitter handles (@xxx)
    tweets = np.vectorize(remove_pattern)(tweets, "@[\w]*")
    
    #remove URL links (httpxxx)
    tweets = np.vectorize(remove_pattern)(tweets, "https?://[A-Za-z0-9./]*")
    
    #remove special characters, numbers, punctuations (except for #)
    tweets = np.core.defchararray.replace(tweets, "[^a-zA-Z0-9]", " ")
    tweets = np.core.defchararray.replace(tweets, "#", " ")
    tweets = np.core.defchararray.replace(tweets, "$", " ")
    tweets = np.core.defchararray.replace(tweets, "[^a-zA-Z]", " ")
    
    
    
    
    return tweets


# In[ ]:


bitcoin_tweets_2021_new['text'] = clean_tweets(bitcoin_tweets_2021_new['text'])


# In[ ]:


bitcoin_tweets_2021_new.reset_index(inplace=True, drop=True)


# In[ ]:


bitcoin_tweets_2021_new['text'] = bitcoin_tweets_2021_new['text'].str.lower()
bitcoin_tweets_2021_new['text'] = bitcoin_tweets_2021_new['text'].str.lstrip()


# In[ ]:


bitcoin_tweets_2021_new.shape[0]


# In[ ]:


scores = []
# Declare variables for scores
compound_list = []
positive_list = []
negative_list = []
neutral_list = []
for i in range(bitcoin_tweets_2021_new['text'].shape[0]):
#print(analyser.polarity_scores(sentiments_pd['text'][i]))
    compound = analyzer.polarity_scores(bitcoin_tweets_2021_new['text'][i])["compound"]
    pos = analyzer.polarity_scores(bitcoin_tweets_2021_new['text'][i])["pos"]
    neu = analyzer.polarity_scores(bitcoin_tweets_2021_new['text'][i])["neu"]
    neg = analyzer.polarity_scores(bitcoin_tweets_2021_new['text'][i])["neg"]
    
    scores.append({"Compound": compound,
                       "Positive": pos,
                       "Negative": neg,
                       "Neutral": neu
                  })


# In[ ]:


sentiments_score = pd.DataFrame.from_dict(scores)
bitcoin_tweets_2021_new = bitcoin_tweets_2021_new.join(sentiments_score)
bitcoin_tweets_2021_new.head()


# In[ ]:


bitcoin_tweets_2021_new.shape[0]


# In[ ]:


bitcoin_tweets_2021_new.date = bitcoin_tweets_2021_new.date.str[:10] #tweets.date.str[:10]


# In[ ]:


bitcoin_tweets_2021_new


# In[123]:


bitusd = pd.read_csv('/Users/arjun/Downloads/BTCUSD.csv')


# In[124]:


bitusd.rename(columns={'Date':'date'}, inplace=True)


# In[125]:


bitusd['date'] = pd.to_datetime(bitusd['date'])


# In[ ]:


# combining bitcoin historic price data to our tweets dataset
bitcoin_tweets_prices = bitcoin_tweets_2021_new.merge(bitusd, on='date', how='left')


# In[ ]:


bitcoin_tweets_2021_new['date'] = pd.to_datetime(bitcoin_tweets_2021_new['date'])


# In[ ]:


bitcoin_tweets_2021_new


# In[ ]:


bitcoin_tweets_prices.shape


# In[ ]:


elon_bit_doge.isnull().sum()


# In[ ]:


bitcoin_tweets_prices.isnull().sum()


# In[ ]:


elon_bit_doge.columns


# In[ ]:


bitcoin_tweets_prices.columns


# In[127]:


elon_bit_doge_final = elon_bit_doge[['date', 'tweet', 'Compound', 'Open',
        'Close',  'Volume']]


# In[ ]:


bitcoin_tweets_2021_prices_final = bitcoin_tweets_prices[[ 'date', 'text',
       'user_followers', 'user_favourites', 'Compound',
        'Open', 'Close',
        'Volume']]


# In[ ]:


elon_bit_doge_final.shape


# In[ ]:


bitcoin_tweets_2021_prices_final.shape


# In[ ]:


elon_bit_doge_final.isnull().sum()


# In[ ]:


bitcoin_tweets_2021_prices_final.isnull().sum()


# In[ ]:


elon_bit_doge_final.columns


# In[ ]:


bitcoin_tweets_2021_prices_final.columns


# In[ ]:


elon_bit_doge_final.to_csv('elon_bit_doge_final.csv')


# In[ ]:


bitcoin_tweets_2021_prices_final.to_csv('bitcoin_tweets_2021_prices_final.csv')


# In[60]:


import numpy as dragon
import pylab as p
import matplotlib.pyplot as plot
from collections import Counter
import re

#importing packages for the prediction of time-series data
import statsmodels.api as sm
import statsmodels.tsa.api as smt
import statsmodels.formula.api as smf

from sklearn.metrics import mean_squared_error


# In[ ]:


from statsmodels.tsa.stattools import adfuller

def test_stationarity(x):


    #Determing rolling statistics
    rolmean = x.rolling(window=22,center=False).mean()

    rolstd = x.rolling(window=12,center=False).std()
    
    #Plot rolling statistics:
    orig = plot.plot(x, color='blue',label='Original')
    mean = plot.plot(rolmean, color='red', label='Rolling Mean')
    std = plot.plot(rolstd, color='black', label = 'Rolling Std')
    plot.legend(loc='best')
    plot.title('Rolling Mean & Standard Deviation')
    plot.show(block=False)
    
    #Perform Dickey Fuller test    
    result=adfuller(x)
    print('ADF Stastistic: %f'%result[0])
    print('p-value: %f'%result[1])
    pvalue=result[1]
    for key,value in result[4].items():
        if result[0]>value:
            print("The graph is non stationery")
            break
        else:
            print("The graph is stationery")
            break;
    print('Critical values:')
    for key,value in result[4].items():
        print('\t%s: %.3f ' % (key, value))
        
     


# In[ ]:


test_stationarity(bitcoin_tweets_2021_prices_final['Close'])


# In[ ]:


close = dragon.log(bitcoin_tweets_2021_prices_final['Close'])
plot.plot(close,color="green")
plot.show()

test_stationarity(close)

close


# In[ ]:


close_diff = close - close.shift()
plot.plot(close_diff)
plot.show()


# In[ ]:


close_diff.dropna(inplace=True)
test_stationarity(close_diff)


# In[ ]:


close_diff


# In[3]:


bitcoin_tweets_2021_prices_final = pd.read_csv('bitcoin_tweets_2021_prices_final.csv')


# In[ ]:


bitcoin_tweets_2021_prices_final['Close']


# In[4]:


import numpy as np
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import cross_val_score, GridSearchCV, cross_val_predict
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler


# In[5]:


features = ['user_followers', 'user_favourites', 'Compound', 'Open', 'Volume']


# In[6]:


y = bitcoin_tweets_2021_prices_final['Close']
x = bitcoin_tweets_2021_prices_final[features]


# In[7]:


train_X, val_X, train_y, val_y = train_test_split(x, y, random_state=1)


# In[8]:


## since the dataset is very large spliting the dataset using the default 75:25 offered by random forrest model is ideal
from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(random_state=1,n_estimators=10)


# In[9]:


rf_model.fit(train_X, train_y)


# In[10]:


rf_pred = rf_model.predict(val_X)


# In[11]:


print(rf_model.predict(x))
bitcoin_tweets_2021_prices_final['Close']


# In[12]:


from sklearn.metrics import mean_absolute_error
rf_val_mae = mean_absolute_error(val_y,rf_pred)
rf_val_mae


# In[ ]:


# let's analyze the features user_followers and user_favourites to determine the correlation with price


# In[56]:


plt.scatter(bitcoin_tweets_2021_prices_final['Close'],bitcoin_tweets_2021_prices_final['user_followers'])
plt.xlabel('Bitcoin Price')
plt.ylabel('User Followers')
plt.savefig('price_followers_relation.png')
plt.show()


# In[57]:


plt.scatter(bitcoin_tweets_2021_prices_final['Close'],bitcoin_tweets_2021_prices_final['user_favourites'])
plt.xlabel('Bitcoin Price')
plt.ylabel('User Favourites')
plt.savefig('price_favourites_relation.png')
plt.show()


# On the basis of the plots above, there does not seem to be much correlation between user_favourutes,user_followers 
# and Closing price of bitcoin, lets exclude these features and test our model again

# In[16]:


features_updated = ['Compound', 'Open', 'Volume']

x_upd = bitcoin_tweets_2021_prices_final[features_updated]


# In[17]:


train_X_upd, val_X_upd, train_y_upd, val_y_upd = train_test_split(x_upd, y, random_state=1)


# In[18]:


rf_model_upd = RandomForestRegressor(random_state=1,n_estimators=10)


# In[20]:


rf_model_upd.fit(train_X_upd, train_y_upd)


# In[21]:


rf_pred_upd = rf_model_upd.predict(val_X_upd)


# In[22]:


print(rf_model_upd.predict(x_upd))
bitcoin_tweets_2021_prices_final['Close']


# In[25]:


rf_val_mae_upd = mean_absolute_error(val_y_upd,rf_pred_upd)

print(rf_val_mae_upd)


# In[128]:


from sklearn.metrics import r2_score
r2_rf_bitcoin = r2_score(val_y_upd,rf_pred_upd)
print(r2_rf_bitcoin)


# Our error has reduced from 16.02597987046904 to 10.48463847893668 after remove user_followers and user_favourite features, these features as seen in the plots above do not effect the price movements. And the r2 score is 0.9996616189541517 which is a very good range for a regression model

# In[65]:


plt.figure(figsize=(6,6))
plt.scatter(val_y_upd, rf_pred_upd, c='crimson')
plt.yscale('linear')
plt.xscale('linear')
plt.xlabel('True Values')
plt.ylabel('Predictions')

p1 = max(max(rf_pred_upd), max(val_y_upd))
p2 = min(min(rf_pred_upd), min(val_y_upd))
plt.plot([p1, p2], [p1, p2], 'b-')
plt.xlabel('True Values', fontsize=10)
plt.ylabel('Predictions', fontsize=10)
plt.axis('equal')
plt.savefig('true_predict_bitcoin_correlation.png')
plt.show()


# In[58]:


# lets perform the prediction for elon musks tweets


# In[28]:


elon_bit_doge_final = pd.read_csv('elon_bit_doge_final.csv')


# In[29]:


elon_bit_doge_final.columns


# In[30]:


features_elon = ['Compound','Open', 'Volume']


# In[31]:


y_elon = elon_bit_doge_final['Close']
x_elon = elon_bit_doge_final[features_elon]


# In[32]:


train_X_elon, val_X_elon, train_y_elon, val_y_elon = train_test_split(x_elon, y_elon, random_state=1)


# In[33]:


rf_model_elon = RandomForestRegressor(random_state=1, n_estimators=90, max_depth=6)


# In[34]:


rf_model_elon.fit(train_X_elon, train_y_elon)


# In[35]:


rf_pred_elon = rf_model_elon.predict(val_X_elon)


# In[36]:


print(rf_model_elon.predict(x_elon))
elon_bit_doge_final['Close']


# In[129]:


rf_val_mae_elon = mean_absolute_error(val_y_elon,rf_pred_elon)
print(rf_val_mae_elon)
r2_rf_elon = r2_score(val_y_elon,rf_pred_elon)
print(r2_rf_elon)


# After changing the number of estimator trees to 90 and max_depth to 6, the MAE dropped down to 741.6235132323729 and an R2 score of 0.9945283139354377, which is very good for a regression model. Especially with such limited data (132 datapoints)

# In[66]:


plt.figure(figsize=(6,6))
plt.scatter(val_y_elon, rf_pred_elon, c='crimson')
plt.yscale('linear')
plt.xscale('linear')
plt.xlabel('True Values')
plt.ylabel('Predictions')

p1 = max(max(rf_pred_elon), max(val_y_elon))
p2 = min(min(rf_pred_elon), min(val_y_elon))
plt.plot([p1, p2], [p1, p2], 'b-')
plt.xlabel('True Values', fontsize=10)
plt.ylabel('Predictions', fontsize=10)
plt.axis('equal')
plt.savefig('true_predict_elon_bitcoin_correlation.png')
plt.show()


# the mae for elon musks tweets is not the greatest, partly because we considered day by day close data for bitcoin most of the tweets by musk effect intraday(within a day) trading. 

# In[42]:


def rfr_model(X, y):
# Perform Grid-Search
    gsc = GridSearchCV(
        estimator=RandomForestRegressor(),
        param_grid={
            'max_depth': range(3,7),
            'n_estimators': (10, 50, 100, 1000),
        },
        cv=5, scoring='neg_mean_squared_error', verbose=0,                         n_jobs=-1)
    
    grid_result = gsc.fit(X, y)
    best_params = grid_result.best_params_
    
    rfr = RandomForestRegressor(max_depth=best_params["max_depth"], n_estimators=best_params["n_estimators"],                               random_state=False, verbose=False)
# Perform K-Fold CV
    scores = cross_val_score(rfr, X, y, cv=10, scoring='neg_mean_absolute_error')

    return scores


# In[67]:


elon_kfolded = pd.read_csv("elon_bit_doge_final.csv")
elon_fet = ['Compound','Open','Volume']
elon_price = elon_kfolded['Close']

fet = elon_kfolded[elon_fet]


# In[68]:


rfr = rfr_model(fet,elon_price)

print(rfr)


# Even after using kfold cross validation, the mean absolute error only reduced by 100 to 634. The relative loss on accuracy on elon musks tweets compared to the whole of bitcoin community is because of the fact that the dataset of the tweets pertinet to elon musk is 130, while the dataset for bitcoin tweets in general is 4300.  
# 
# From this point and below, Linear Regression model is being trained and implemented on the same datasets
# 

# In[69]:


import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')


# In[70]:


df = pd.read_csv('bitcoin_tweets_2021_prices_final.csv')
df.head()


# In[71]:


df.describe()


# In[47]:


## lets use linearregression to perform the same prediction
from sklearn.linear_model import LinearRegression


# In[72]:


df.isnull().sum()


# In[73]:


df.info()


# In[74]:


df.corr()


# In[75]:


plt.figure(figsize=(30,30))
sns.heatmap(df.corr(), annot=True , cmap=plt.cm.Accent_r,annot_kws={'fontsize':15})
plt.show()


# In[76]:


def correlation(data, threshold):
  corr = data.corr()['Close'].sort_values(ascending=False)[1:]
  abs_corr = abs(corr)
  relevent_features = abs_corr[abs_corr>threshold]
  return relevent_features


# In[77]:


corr_features = correlation(df,0.81)


# In[78]:


corr_features


# In[79]:


sns.jointplot(df['Open'] , df['Close'] , color='red')


# In[80]:


df1 = df[corr_features.index]


# In[81]:


df1


# In[82]:


X = df1
y = df['Close']


# In[83]:


X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2 , random_state=51)


# In[84]:


X_train.shape , X_test.shape , y_train.shape , y_test.shape


# In[85]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()


# In[86]:


X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[87]:


lr = LinearRegression()


# In[88]:


lr.fit(X_train , y_train)


# In[89]:


pred = lr.predict(X_test)


# In[90]:


pred[0]


# In[91]:


y_test.iloc[0]


# In[92]:


from sklearn.metrics import mean_absolute_error , mean_squared_error
import math


# In[93]:


print(f'Mean Absolute Error = {mean_absolute_error(y_test,pred)}')


# In[94]:


df_elon = pd.read_csv('elon_bit_doge_final.csv')
df_elon.head()


# In[95]:


df_elon.describe()


# In[96]:


df_elon.isnull().sum()


# In[97]:


df_elon.corr()


# In[98]:


plt.figure(figsize=(30,30))
sns.heatmap(df_elon.corr(), annot=True , cmap=plt.cm.Accent_r,annot_kws={'fontsize':15})
plt.show()


# In[99]:


def correlation(data, threshold):
  corr = data.corr()['Close'].sort_values(ascending=False)[1:]
  abs_corr = abs(corr)
  relevent_features = abs_corr[abs_corr>threshold]
  return relevent_features


# In[100]:


corr_features = correlation(df_elon,0.81)


# In[101]:


corr_features


# In[102]:


sns.jointplot(df_elon['Open'] , df_elon['Close'] , color='red')


# In[103]:


df1 = df_elon[corr_features.index]


# In[104]:


df1


# In[105]:


X = df1
y = df_elon['Close']


# In[124]:


X_train , X_test , y_train , y_test_elon = train_test_split(X,y,test_size=0.2 , random_state=51)


# In[107]:


X_train.shape , X_test.shape , y_train.shape , y_test.shape


# In[108]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()


# In[109]:


X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[110]:


lr = LinearRegression()


# In[111]:


lr.fit(X_train , y_train)


# In[123]:


pred_elon = lr.predict(X_test)


# In[113]:


pred[0]


# In[114]:


y_test.iloc[0]


# In[115]:


from sklearn.metrics import mean_absolute_error , mean_squared_error
import math


# In[116]:


print(f'Mean Absolute Error = {mean_absolute_error(y_test,pred_elon)}')


# In[131]:


r2_lr_elon = r2_score(pred_elon,y_test )
r2_lr_elon


# The MAE score using linear regression model is  is 1375.2581805400412 for bitcoin set and 1511.5784879089535 for elon musk dataset and R2 score is 0.9824045484548013.
