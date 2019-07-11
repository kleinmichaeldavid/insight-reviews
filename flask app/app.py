
#from flaskexample import app
from flask import Flask, request, render_template

import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.externals import joblib

import os
import sys

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

import myfunctions as myfunc


app = Flask(__name__)


@app.route('/', methods=["GET"])
def index():

	## initialize the predictions to a number out of the range of the plot
    return render_template('index.html',
        predict_uniq = -2,
        predict_humt = -2,
        predict_qual = -2,
        review_text = '')


@app.route('/',methods=['POST'])
def submit_review():

	## import the lemmatizer and load the model
    from nltk.stem import WordNetLemmatizer
    model_nb = joblib.load(open('model_nb_200sellers.pkl','rb'))

    def topic_sentiment_model(review, lemmatizer):

        ## define the word lists to be used to label the topics
        uniqueness_pool = ['unique', 'creative', 'special',
        'inventive', 'innovative', 'handmade',
        'original', 'handcrafted', 'creativity', 'crafted']

        humantouch_pool = ['helpful', 'communication', 'response',
        'respond', 'contact', 'friendly',
        'attentive', 'accommodating', 'courteous',
        'polite', 'respect', 'pleasant',
        'service','kind','personal','seller',
        'personable','interaction','reply','answer',
        'honest','rude','unhelpful','unresponsive',
        'personalised','personalized']

        quality_pool = ['quality', 'condition', 'described',
        'detailed', 'craftmanship', 'craftsmanship',
        'detail', 'workmanship', 'material','broke','break',
        'fell','apart','cheap']

        ## lemmatize the word lists (since the input will be lemmatized)
        uniqueness_pool = [lemmatizer.lemmatize(w) for w in uniqueness_pool]
        humantouch_pool = [lemmatizer.lemmatize(w) for w in humantouch_pool]
        quality_pool = [lemmatizer.lemmatize(w) for w in quality_pool]

        ## divide the input into sentences and check each for the topics
        sentences = myfunc.sentence_splitter([review], split_on = "[\.!,]+[ ]*")
        uniqueness = [any([w in uniqueness_pool for w in s.split()]) for s in sentences]
        humantouch = [any([w in humantouch_pool for w in s.split()]) for s in sentences]
        quality = [any([w in quality_pool for w in s.split()]) for s in sentences]

        ## convert the inputted text to a dataframe (for preprocessing for sentiment)
        app_input_df = pd.DataFrame({'review_text' : [review]})

        ## preprocess the input for sentiment analysis
        data_for_sentiment = myfunc.preprocessing(app_input_df,
                                split_sentences = True,
                                replace_numbers = True,
                                tokenize = False,
                                lemmatize = True,
                                remove_stops = False,
                                split_on = "[\.!,]+[ ]*",
                                embed = False)
                                
        ## predict sentiment for each sentiment
        sentiments = model_nb.predict_proba(data_for_sentiment['clean_text'])[:,1] * 2 - 1

        ## compute sentiment score for each topic
        ## equal to the average sentiment on those sentences that are about the topic
        uniqueness_score = np.sum(np.multiply(sentiments,uniqueness))/np.max([1,np.sum(uniqueness)])
        humantouch_score = np.sum(np.multiply(sentiments,humantouch))/np.max([1,np.sum(humantouch)])
        quality_score = np.sum(np.multiply(sentiments,quality))/np.max([1,np.sum(quality)])

        ## return the sentiment scores
        return np.round(uniqueness_score,2),np.round(humantouch_score,2), np.round(quality_score,2)

    if request.method == 'POST':

        if request.form['submit_button'] == "Analyze Your Review":
        	## run analysis of input unless the input is empty
            review = request.form['review_input']
            if review != '':
            	uniqueness_score, humantouch_score, quality_score = topic_sentiment_model(review, WordNetLemmatizer())
            	print(uniqueness_score, humantouch_score, quality_score)
            else:
            	uniqueness_score, humantouch_score, quality_score = -2,-2,-2
        elif request.form['submit_button'] == "Positive":
        	## run analysis of positive example review
            review = "I have had the pleasure of buying several pieces from him, and each is as unique as they are high quality. Love dealing with him."
            uniqueness_score, humantouch_score, quality_score = topic_sentiment_model(review, WordNetLemmatizer())
        elif request.form['submit_button'] == "Negative":
        	## run analysis of negative example review
            review = "This seller pathological scammer and liar: 1. Sent to me defective earrings,one with broken lock,dirty,scratched, with different design-swirls on the top are different on each earring 2. Seller lied about shipping- see screenshots - said earrings were shipped, and they are not, liar shipped them only after my message Seller refusing admit her fault, arguing and lying, shame, i recommend avoid this shop"
            uniqueness_score, humantouch_score, quality_score = topic_sentiment_model(review, WordNetLemmatizer())
        elif request.form['submit_button'] == "Unique":
        	## run analysis of unique example review
            review = "I love this unique bracelet. I am happy as always with my purchases from this shop!"
            uniqueness_score, humantouch_score, quality_score = topic_sentiment_model(review, WordNetLemmatizer())
        elif request.form['submit_button'] == "Human":
        	## run analysis of human touch example review
            review = "She really cares and goes out of her way to accommodate all of your needs! I received item very quickly and everything is so beautiful!!! THANK YOU!!! I cant wait to see my daughter's face Christmas morning!"
            uniqueness_score, humantouch_score, quality_score = topic_sentiment_model(review, WordNetLemmatizer())
        elif request.form['submit_button'] == "Quality":
        	## run analysis of quality example review
            review = "I ordered these bracelets for wedding gifts for my future mother-in-law and sisters-in-law. I understand that the materials aren't the greatest, but I haven't even been able to give them the bracelets and they are already changing color. Definitely upset. Anyone know how to polish this material in order it not to look tarnished ?"
            uniqueness_score, humantouch_score, quality_score = topic_sentiment_model(review, WordNetLemmatizer())

    return render_template('index.html',
        predict_uniq = uniqueness_score,
        predict_humt = humantouch_score,
        predict_qual = quality_score,
        review_text = review)


if __name__ == '__main__':
    app.run(host="0.0.0.0")