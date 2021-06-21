#!/usr/bin/env python
# coding: utf-8

# In[87]:


import math
import numpy as np
import pandas as pd
import re
import os
import sys
import random
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from sklearn import preprocessing, metrics
from sklearn.preprocessing import MinMaxScaler


# In[88]:


from flask import Flask, request, jsonify, render_template, url_for
from flask_cors import cross_origin
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model,load_model
from tensorflow.keras import Sequential
from keras.layers import Dense, LSTM, Dropout, Dense, Activation


# In[89]:


app = Flask(__name__) #to create flask App


# In[90]:


model = load_model('stock_news_model.h5')


# In[91]:


#to go to route/main directory of the hirearchy of web application direct
@app.route('/')
@cross_origin()
def home():
    return render_template('index.html')


# In[92]:


@app.route('/predict',methods=['GET','POST'])
@cross_origin()
def predict():
    
    if request.method == 'POST':
        user = request.form["Enter Headline"]
        open_price = float(request.form["Open Price"])
        close_price = float(request.form["Close Price"])
        
        data = {'o_t':  [user]}
        user_df = pd.DataFrame (data, columns = ['o_t'])
        
        user_df['headline'] = user_df['o_t']


        # for performing NLP Functions i.e detection of Polarity and Subjectivity

        polarity=[]     #list that contains polarity of tweets
        subjectivity=[]    ##list that contains subjectivity of tweets

        for i in user_df.headline.values:
            try:
                analysis = TextBlob(i) # [i] records to the first data in dataset
                polarity.append(analysis.sentiment.polarity)
                subjectivity.append(analysis.sentiment.subjectivity)
            except:
                polarity.append(0)
                subjectivity.append(0)

        # adding sentiment polarity and subjectivity column to dataframe

        user_df['polarity'] = polarity
        user_df['subjectivity'] = subjectivity


        #Create a function to get the sentiment scores (using Sentiment Intensity Analyzer)
        def getSIA(text):
            sia = SentimentIntensityAnalyzer()
            sentiment = sia.polarity_scores(text)
            return sentiment

        #Get the sentiment scores 
        compound = []
        neg = []
        neu = []
        pos = []
        SIA = 0
        for i in range(0, len(user_df['headline'])):
            SIA = getSIA(user_df['headline'][i])
            compound.append(SIA['compound'])
            neg.append(SIA['neg'])
            neu.append(SIA['neu'])
            pos.append(SIA['pos'])

        #Store the sentiment scores in the data frame
        user_df['Compound'] =compound
        user_df['Negative'] =neg
        user_df['Neutral'] =neu
        user_df['Positive'] = pos
        

        a=[]
        a.append(close_price)
        a.append(open_price)
        a.append(user_df['polarity'][0])
        a.append(user_df['subjectivity'][0])
        a.append(user_df.Compound[0])
        a.append(user_df.Negative[0])
        a.append(user_df.Neutral[0])
        a.append(user_df.Positive[0])
        
        list_new = [[17900.09961,17910.01953,-0.075678,0.430321,-0.9980,0.183,0.787,0.030],[a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7]]]
        new_df = pd.DataFrame(list_new,columns=[['Close','Open','polarity','subjectivity','compound','negative','neutral','positive']])
        
        x = new_df.values
        s = MinMaxScaler(feature_range=(0,1))
        s_d = s.fit_transform(x)
        t_d_len = math.ceil(len(x)*.8)
        t_d = s_d[0:t_d_len, :]

        #Split the data into X_train and y_train dataset
        x_t=[]
        y_t=[]

        for i in range(1, len(t_d)):
            x_t.append(t_d[i-1:i, 0:])
            y_t.append(t_d[i,0:])
            
        x_t, y_t = np.array(x_t), np.array(y_t)
        x_t = np.reshape(x_t, (x_t.shape[0], x_t.shape[1], 8))
        
        # printing the model summary
        model.summary()
        
        # compiling the model
        model.compile(loss='mse' , optimizer='adam')

        # fitting the model using the training dataset
        model.fit(x_t,y_t,batch_size=1,epochs=1)

        p = model.predict(x_t)
        p = s.inverse_transform(p) #unscaling the values

        if (p[0][0] > close_price):
            output = 1
        else:
            output = 0
        
        if user_df['polarity'][0] > 0:   # Positive Sentiment
            if output == 1:
                prediction_text1='Entered Headline is = {}'.format(user)
                prediction_text2='News Sentiment is "POSITIVE" as polarity = {}'.format(user_df['polarity'][0])
                prediction_text3='Price Up as value predicted is = {}'.format(output)
                prediction_text4='Close Price is ={}'.format(close_price)
                prediction_text5='Predicted Close Price is={}'.format(p[0][0])
                combine = prediction_text1 + ' || ' + prediction_text2 + ' || ' + prediction_text3 + ' || ' + prediction_text4 + ' || ' + prediction_text5
                return render_template('index.html', prediction_text = combine)
            else:
                prediction_text1='Entered Headline is = {}'.format(user)
                prediction_text2='News Sentiment is "POSITIVE" as polarity = {}'.format(user_df['polarity'][0])
                prediction_text3='Price Down as value predicted is = {}'.format(output)
                prediction_text4='Close Price is ={}'.format(close_price)
                prediction_text5='Predicted Close Price is={}'.format(p[0][0])
                combine = prediction_text1 + ' || ' + prediction_text2 + ' || ' + prediction_text3 + ' || ' + prediction_text4 + ' || ' + prediction_text5
                return render_template('index.html', prediction_text = combine)
            
        elif user_df['polarity'][0] < 0:  # Negative Sentiment
            if output == 1:
                prediction_text1='Entered Headline is = {}'.format(user)
                prediction_text2='News Sentiment is "NEGATIVE" as polarity = {}'.format(user_df['polarity'][0])
                prediction_text3='Price Up as value predicted is = {}'.format(output)
                prediction_text4='Close Price is ={}'.format(close_price)
                prediction_text5='Predicted Close Price is={}'.format(p[0][0])
                combine = prediction_text1 + ' || ' + prediction_text2 + ' || ' + prediction_text3 + ' || ' + prediction_text4 + ' || ' + prediction_text5
                return render_template('index.html', prediction_text = combine)
            else:
                prediction_text1='Entered Headline is = {}'.format(user)
                prediction_text2='News Sentiment is "NEGATIVE" as polarity = {}'.format(user_df['polarity'][0])
                prediction_text3='Price Down as value predicted is = {}'.format(output)
                prediction_text4='Close Price is ={}'.format(close_price)
                prediction_text5='Predicted Close Price is={}'.format(p[0][0])
                combine = prediction_text1 + ' || ' + prediction_text2 + ' || ' + prediction_text3 + ' || ' + prediction_text4 + ' || ' + prediction_text5
                return render_template('index.html', prediction_text = combine)

        else:    # Neutral Sentiment
            if output == 1:
                prediction_text1='Entered Headline is = {}'.format(user)
                prediction_text2='News Sentiment is "NEUTRAL" as polarity = {}'.format(user_df['polarity'][0])
                prediction_text3='Price Up as value predicted is = {}'.format(output)
                prediction_text4='Close Price is ={}'.format(close_price)
                prediction_text5='Predicted Close Price is={}'.format(p[0][0])
                combine = prediction_text1 + ' || ' + prediction_text2 + ' || ' + prediction_text3 + ' || ' + prediction_text4 + ' || ' + prediction_text5
                return render_template('index.html', prediction_text = combine)            
            else:
                prediction_text1='Entered Headline is = {}'.format(user)
                prediction_text2='News Sentiment is "NEUTRAL" as polarity = {}'.format(user_df['polarity'][0])
                prediction_text3='Price Down as value predicted is = {}'.format(output)
                prediction_text4='Close Price is ={}'.format(close_price)
                prediction_text5='Predicted Close Price is={}'.format(p[0][0])
                combine = prediction_text1 + ' || ' + prediction_text2 + ' || ' + prediction_text3 + ' || ' + prediction_text4 + ' || ' + prediction_text5
                return render_template('index.html', prediction_text = combine)        
       
        #if output == 1:
            #return render_template('index.html', prediction_text='Price Up as value predicted is = {}'.format(output))
        #else:
            #return render_template('index.html', prediction_text='Price Down as value predicted is = {}'.format(output))
    
    return render_template(index.html)


# In[93]:


if __name__ == "__main__":
    app.run(debug=False)


# In[1]:


#pip install gunicorn


# In[1]:


#pip freeze > requirements.text


# In[2]:


#echo web: run this thing >Procfile


# In[ ]:




