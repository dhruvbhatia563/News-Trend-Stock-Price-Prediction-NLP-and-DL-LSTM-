#!/usr/bin/env python
# coding: utf-8

# # News - Oriented Stock Price Trend Prediction - (NLP & Stacked LSTM)

# ### Problem Statement:
# 
#  - The prediction of stock prices in the market has always been an important task. However, due to the market volatility, it is difficult to make the correct prediction solely based on historical stock data. Therefore in this notebook, with the analysis of daily news’ impact on stock markets, will identify some key features that could be useful in stock price prediction and propose a deep learning model to capture the dynamics of stock price trend with rich news textual information.
#  
# ### Dataset:
# 
#  - The dataset working on is a combination of Reddit news and the Dow Jones Industrial Average (DJIA) stock price from 2008 to 2016. 
#      - The news dataset contains the top 25 news on Reddit of each day from 2008 to 2016.
#      - The DJIA contains the core stock market information for each trading day such as Open, Close, and Volume. The label of the dataset represents whether the stock price is increase (labeled as 1) or decrease (labeled as 0) on that day. The total number of days in the dataset is 1989.

# # Importing Libraries and Packages

# In[1]:


#pip install -q wordcloud
#pip install gensim
#!pip install vaderSentiment

# download nltk if not installed previously
# nltk.download('vader_lexicon')

#pip install --upgrade tensorflow
#pip install Keras
#pip install tensorflow==2.1.0


# In[3]:


# importing libraries
import warnings
warnings.filterwarnings('ignore')

import math
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from keras.layers import Dense, LSTM, Dropout, Dense, Activation

import nltk
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *

import wordcloud

import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger') 
nltk.download('vader_lexicon')

from sklearn import preprocessing, metrics
from sklearn.preprocessing import MinMaxScaler


# # Getting the Dataset

# In[4]:


stock = pd.read_csv("upload_DJIA_table.csv")
stock.head()


# In[5]:


Reddit = pd.read_csv("RedditNews.csv")
Reddit.head()


# In[6]:


Top_Headlines = pd.read_csv("Combined_News_DJIA.csv")
Top_Headlines.head()


# In[7]:


data=Top_Headlines.iloc[:,2:27]
data.replace("[^a-zA-Z]"," ",regex=True, inplace=True)

# Renaming column names for ease of access
list1= [i for i in range(25)]
new_Index=[str(i) for i in list1]
data.columns= new_Index
data.head(5)


# In[8]:


# Convertng headlines to lower case
for index in new_Index:
    data[index]=data[index].str.lower()
data.head()


# In[9]:


' '.join(str(x) for x in data.iloc[1,0:25])


# In[10]:


headlines = []
for row in range(0,len(data.index)):
    headlines.append(' '.join(str(x) for x in data.iloc[row,0:25]))


# In[11]:


headlines[0]


# In[12]:


Top_Headlines["headlines"] = ""


# In[13]:


Top_Headlines.head(1)


# In[14]:


for i in range(len(headlines)):
    Top_Headlines["headlines"][i] = headlines[i]
    
Top_Headlines.head()


# In[15]:


Top_Headlines=Top_Headlines.drop(['Top1','Top2','Top3','Top4','Top5','Top6','Top7','Top8','Top9','Top10','Top11','Top12','Top13', 
             'Top14','Top15','Top16','Top17','Top18','Top19','Top20','Top21','Top22','Top23','Top24','Top25'],axis=1)


# In[16]:


mergedDf = stock.merge(Top_Headlines)
mergedDf.head()


# ### Feature Engineering

# In[17]:


# coverting the datatype of column 'Date' from type object to type 'datetime'
mergedDf['Date'] = pd.to_datetime(mergedDf['Date']).dt.normalize()


# In[18]:


#DROP - Adj Close column as it is similer with Close Column
#DROP - Label Column as on what basis label is provided is not mentioned and this can effect further in analysis
mergedDf=mergedDf.drop(['Adj Close'],axis=1)
mergedDf=mergedDf.drop(['Label'],axis=1)


# filtering the important columns required
mergedDf = mergedDf.filter(['Date', 'Close', 'Open', 'High', 'Low', 'Volume','headlines'])


# In[19]:


# setting column 'Date' as the index column
mergedDf.set_index('Date', inplace= True)

# sorting the data according to the index i.e 'Date'
mergedDf = mergedDf.sort_index(ascending=True, axis=0)

mergedDf


# In[20]:


# dropping duplicates
mergedDf = mergedDf.drop_duplicates()


# # Calculating Sentiment Polarity and Subjectivity
# 
#  - The subjectivity shows how subjective or objective a statement is.
# 
#  - The polarity shows how positive/negative the statement is, a value equal to 1 means the statement is positive, a value equal to 0 means the statement is neutral and a value of -1 means the statement is negative.

# In[21]:


from textblob import TextBlob     # for performing NLP Functions i.e detection of Polarity and Subjectivity

polarity=[]     #list that contains polarity of tweets
subjectivity=[]    ##list that contains subjectivity of tweets

for i in mergedDf.headlines.values:
    try:
        analysis = TextBlob(i) # [i] records to the first data in dataset
        polarity.append(analysis.sentiment.polarity)
        subjectivity.append(analysis.sentiment.subjectivity)
        
    except:
        polarity.append(0)
        subjectivity.append(0)
        

        
# adding sentiment polarity and subjectivity column to dataframe

mergedDf['polarity'] = polarity
mergedDf['subjectivity'] = subjectivity
mergedDf.head()


# # Calculating Sentiment Scores

# In[22]:


# adding empty sentiment columns to data for later calculation
mergedDf['compound'] = ''
mergedDf['negative'] = ''
mergedDf['neutral'] = ''
mergedDf['positive'] = ''
mergedDf.head()


# In[23]:


# importing requires libraries to analyze the sentiments
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import unicodedata

# instantiating the Sentiment Analyzer
sid = SentimentIntensityAnalyzer()

# calculating sentiment scores
mergedDf['compound'] = mergedDf['headlines'].apply(lambda x: sid.polarity_scores(x)['compound'])
mergedDf['negative'] = mergedDf['headlines'].apply(lambda x: sid.polarity_scores(x)['neg'])
mergedDf['neutral'] = mergedDf['headlines'].apply(lambda x: sid.polarity_scores(x)['neu'])
mergedDf['positive'] = mergedDf['headlines'].apply(lambda x: sid.polarity_scores(x)['pos'])

# displaying the stock data
mergedDf.head()


# # Word Cloud

# In[24]:


# word cloud
from wordcloud import WordCloud, STOPWORDS
comment_words = ''
stopwords = set(STOPWORDS)
  
# iterate through the csv file
for val in mergedDf.headlines:
      
    # typecaste each val to string
    val = str(val)
  
    # split the value
    tokens = val.split()
      
    # Converts each token into lowercase
    for i in range(len(tokens)):
        tokens[i] = tokens[i].lower()
      
    comment_words += " ".join(tokens)+" "

wordcloud = WordCloud(width = 800, height = 800,
                background_color ='black',
                stopwords = stopwords,
                min_font_size = 10).generate(comment_words)
  
# plot the WordCloud image                       
plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)
  
plt.show()


# # To find sentiments of news headlines

# In[25]:


# create a function get the sentiment text
def getSentiment(score):
    if score < 0:
        return "negative"
    elif score == 0:
        return "neutral"
    else:
        return "positive"


# In[26]:


# create a column to store the news sentiment
mergedDf['news_sentiment'] = mergedDf['polarity'].apply(getSentiment)
mergedDf.head()


# In[27]:


# create a function get the sentiment text
def getSentiment(score):
    if score < 0:
        return 0 #negative
    elif score == 0:
        return 1  #neutral
    else:
        return 2 #positive


# In[28]:


# create a column to store the news sentiment
mergedDf['news_sentiment_flag'] = mergedDf['polarity'].apply(getSentiment)
mergedDf.head()


# In[29]:


mergedDf.news_sentiment_flag.value_counts(normalize=True)*100


# In[30]:


# print percentage on top of each bar
def displaypercentage(ax,feature):
    
    total = len(mergedDf[feature])
    for p in ax.patches:
        percentage = '{:.2f}%'.format(100 * p.get_height()/total)
        x = p.get_x() + p.get_width()/2
        y = p.get_height()
        ax.annotate(percentage, (x, y),ha='center',va='bottom',fontsize=12,color='blue')


# In[31]:


# check the 'Converted' column
pl=sns.catplot(data=mergedDf,x='news_sentiment_flag',kind="count")
plt.title('Sentiment V/s. Count', fontsize = 14)
ax = pl.facet_axis(0,0)
displaypercentage(ax,'news_sentiment_flag')
plt.show()


# ### Negative Sentimets Count = 30.22%
# ### Neutral Sentimets Count = 0%
# ### Positive Sentimets Count = 69.78%

# In[32]:


# scatter plot to show the subjectivity and the polarity
plt.figure(figsize=(14,10))

for i in range(mergedDf.shape[0]):
    plt.scatter(mergedDf["polarity"].iloc[[i]].values[0], mergedDf["subjectivity"].iloc[[i]].values[0], color="Purple")

plt.title("Sentiment Analysis Scatter Plot")
plt.xlabel('polarity')
plt.ylabel('subjectivity')
plt.show()


# # EDA for Stock Data

# In[33]:


# setting figure size
plt.figure(figsize=(16,10))

# plotting close price
mergedDf['Close'].plot()

# setting plot title, x and y labels
plt.title("Close Price")
plt.xlabel('Date')
plt.ylabel('Close Price ($)')


# In[34]:


# calculating 7 day rolling mean
mergedDf.rolling(7).mean().head(20)


# In[35]:


# setting figure size
plt.figure(figsize=(16,10))

# plotting the close price and a 30-day rolling mean of close price
mergedDf['Close'].plot()
mergedDf.rolling(window=30).mean()['Close'].plot()


# In[36]:


mergedDf.head(3)


# In[37]:


#Create a list of columns to keep in the completed data set and show the data.

keep_columns = ['Close','Open','polarity','subjectivity','compound','negative','neutral','positive']
df = mergedDf[keep_columns]
df.head()


# In[38]:


df.shape


# In[39]:


#visualize the closing price history

plt.figure(figsize=(16,8))
plt.title('Close Price History')
plt.plot(df['Close'])
plt.xlabel('Date')
plt.ylabel('Close Price in USD ($)')
plt.show()


# # Data Preparation for Modelling

# In[40]:


#convert dataframe to numpy array

dataset = df.values

#Get the number of rows to train LSTM model
train_data_len = math.ceil(len(dataset)*.8)

train_data_len


# In[41]:


#Scale the data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

scaled_data


# In[42]:


scaled_data[0]


# In[43]:


#Create the training dataset
#Create the sclaed training dataset

train_data = scaled_data[0:train_data_len, :]

#Split the data into X_train and y_train dataset
x_train=[]
y_train=[]

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0:])
    y_train.append(train_data[i,0:])
    
    if i<=61:
        print(x_train)
        print(y_train)
        print()


# In[44]:


# Note: y_train -> contains 61st value which model needs to predict on the basis of past 60 days values stored in x_train


# In[45]:


#Convert x_train and y_train dataset into numpy arrays so that can train for LSTM Model

x_train, y_train = np.array(x_train), np.array(y_train)


# In[46]:


#Reshape the data
x_train.shape #2D array


# In[47]:


#LSTM required 3D
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 8))
x_train.shape


# In[48]:


#Build LSTM Model

# setting the model architecture
model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(x_train.shape[1],8)))
model.add(LSTM(50,return_sequences=False))
model.add(Dense(25))
model.add(Dense(8))

# printing the model summary
model.summary()


# In[49]:


#Compile Model
model.compile(optimizer='adam', loss='mean_squared_error')


# In[50]:


#Train Model
model.fit(x_train,y_train,batch_size=1,epochs=1)


# In[51]:


#Create the testing dataset
#Create new array containing scaled values from index 1532 to 1989
test_data=scaled_data[train_data_len-60: ,:]
#Create datasets --> x_test and y_test
x_test=[]
y_test=dataset[train_data_len: ,:]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0:8])


# In[52]:


x_test[0]


# In[53]:


#Convert data to a numpy array
x_test=np.array(x_test)
x_test[0]


# In[54]:


#Reshape the data
x_test=np.reshape(x_test, (x_test.shape[0],x_test.shape[1],8))
x_test


# In[55]:


#Get the Models Predicted Price Values
predictions = model.predict(x_test)


# In[56]:


predictions[0]


# In[57]:


predictions.shape


# In[58]:


predictions = scaler.inverse_transform(predictions) #unscaling the values


# In[59]:


#Get root mean squared error (RMSE)
rmse=np.sqrt(np.mean(((predictions- y_test)**2)))
rmse


# In[60]:


predictions[0]


# In[61]:


predictions[0][0]


# In[62]:


predictions[0][1]


# In[63]:


len(predictions)


# In[64]:


train = df[:train_data_len]
train


# In[65]:


valid = df[train_data_len:]
valid


# In[66]:


p_list=[] #prediction list
for i in range(len(predictions)):
    p_list.append(predictions[i][0])
    
p_list 


# In[67]:


valid['Predicted Close Price'] = 0
for i in range(len(valid)):
    valid['Predicted Close Price'][i] = p_list[i]


# In[68]:


valid.head(3)


# In[69]:


#Visulaize the data
plt.figure(figsize=(16,8))
plt.title('Model')
plt.ylabel('Close Price')
plt.plot(train[['Close']])
plt.plot(valid[['Close','Predicted Close Price']])
plt.legend(['Train','Valid','Predictions'], loc='lower right')
plt.show()


# In[70]:


#Visulaize the data
plt.figure(figsize=(16,8))
plt.title('Model')
plt.ylabel('Close Price')
plt.plot(valid[['Close','Predicted Close Price']])
plt.legend(['Valid','Predictions'], loc='lower right')
plt.show()


# In[71]:


valid[['Close','Predicted Close Price']]


# In[72]:


valid['Price Up/Down'] = 0
for i in range(len(valid)):
    if (valid['Close'][i] > valid['Predicted Close Price'][i]):
        valid['Price Up/Down'][i] = 0 #price down
    else:
        valid['Price Up/Down'][i] = 1 #price up


# In[73]:


valid[['Close','Predicted Close Price','Price Up/Down']]


# In[74]:


valid['Price Up/Down'].value_counts(normalize=True)


# In[75]:


# print percentage on top of each bar
def displaypercentage(ax,feature):
    
    total = len(valid[feature])
    for p in ax.patches:
        percentage = '{:.2f}%'.format(100 * p.get_height()/total)
        x = p.get_x() + p.get_width()/2
        y = p.get_height()
        ax.annotate(percentage, (x, y),ha='center',va='bottom',fontsize=12,color='blue')


# In[76]:


# check the 'Converted' column
pl=sns.catplot(data=valid,x='Price Up/Down',kind="count")
plt.title('Percentage of Predicted Close Price in terms of Down[0] and Up[1]', fontsize = 14)
ax = pl.facet_axis(0,0)
displaypercentage(ax,'Price Up/Down')
plt.show()


# In[91]:


from tensorflow.keras import Model


# In[92]:


from tensorflow.keras.models import model_from_json


# In[94]:


from tensorflow.keras.models import load_model


# In[105]:


model.save('stock_news_model.h5')


# In[106]:


#new_model = load_model('stock_news_model.h5')


# In[107]:


#new_model.summary()


# In[108]:


#new_model.get_weights()


# In[113]:


#new_model.optimizer


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




