
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

    return render_template('index.html',
        predict_uniq = -2,
        predict_humt = -2,
        predict_qual = -2,
        review_text = '')


@app.route('/',methods=['POST'])
def submit_review():
    ### also ,this should load when the page is opened, not when a review is submitted
    ### save as session?
    # can do multiple models and put if statement here
    from nltk.stem import WordNetLemmatizer
    #W2Vmodel = joblib.load(open('W2Vmodel.pkl','rb'))
    model_nb = joblib.load(open('model_nb_200sellers.pkl','rb'))

    def topic_sentiment_model(review, lemmatizer):
        ### function to determine the topic of each sentence in a review.

        # stopwords_mike = ['bracelets','necklace','earrings',
        #     'bracelet','ring','daughter','mother',
        #     'anklet','cindy','gift','gorgeous',
        #     'item','product','color','colors',
        #     'lovely','perfect','cute','beautiful','great',
        #     'super','order','wonderful','work','awesome',
        #     'loved','lt','pretty','little','love','absolutely',
        #     'bunch','wear','friend','piece','amazing','happy',
        #     'fantastic','xx','summer','excellent','million',
        #     'michael','lisa','good','looks','highly','recommend',
        #     'price','leather','future','pleased','high','beads',
        #     'you','nice','size','looking',
        #     'design','buy','definitely','fast','quickly',
        #     'products','product','promptly','quickly','quick',
        #     'numbertoken','like','ordered','fit','purchase','bought',
        #     'christmas','beautifully','perfectly','purchased','husband',
        #     'ordering','adorable','way','small','better','wrist',
        #     'jewelry','wearing','day','wanted', 'easy', 'new','lot',
        #     'best','going', 'custom','sure', 'simple','right','person',
        #     'boyfriend','took', 'charm', 'sent', 'photo', 'week','use',
        #     'birthday','son','extra','bit','favorite', 'clasp','present',
        #     'nose','bag','sister','chain','mom', 'year', 'stone','wedding',
        #     'thought','delicate','feel','gave', 'different','dainty',
        #     'box','big','second','blanket','sweet','comfortable',
        #     'thing','extremely','baby','silver','want']

        uniqueness_pool = ['unique', 'creative', 'special',
            'inventive', 'innovative', 'handmade',
            'original', 'handcrafted',
            'creativity', 'crafted','made']

        humantouch_pool = ['helpful', 'communication', 'response',
            'respond', 'contact', 'friendly',
            'attentive', 'accommodating', 'accommodate',
            'courteous',
            'polite', 'respect', 'pleasant',
            'service','kind','personal',
            'personable','interaction','reply','answer',
            'honest','rude','unhelpful','unresponsive',
            'personalised','personalized','you','deal','dealing',
            'liar','thank']

        quality_pool = ['quality', 'condition', 'described',
            'detailed', 'craftmanship', 'craftsmanship',
            'detail', 'workmanship', 'material','broke','break',
            'fell','apart','defective','broken']

        # custom_stopwords = myfunc.create_stopwords(stopwords_mike)


        app_input_df = pd.DataFrame({'review_text' : [review]})
        
        ## process the input for the topic classification
        # input_data_for_topic = myfunc.preprocessing(app_input_df,
        #     split_sentences = True,
        #     replace_numbers = True,
        #     tokenize = False,
        #     lemmatize = True,
        #     remove_stops = False,
        #     embed = False,
        #     split_on = "[\.!,]+[ ]*",
        #     stopwords = custom_stopwords)

        ## lemmatize the word lists
        uniqueness_pool = [lemmatizer.lemmatize(w) for w in uniqueness_pool]
        humantouch_pool = [lemmatizer.lemmatize(w) for w in humantouch_pool]
        quality_pool = [lemmatizer.lemmatize(w) for w in quality_pool]
        
        ## check for the brand words within the review sentences
        # sentences = input_data_for_topic['clean_text']
        # isin_unique = [np.any([w in uniqueness_pool for w in s.split()]) for s in sentences]
        # isin_humantouch = [np.any([w in humantouch_pool for w in s.split()]) for s in sentences]
        # isin_quality = [np.any([w in quality_pool for w in s.split()]) for s in sentences]

        ## process the input for sentiment analysis
        data_for_sentiment = myfunc.preprocessing(app_input_df,
                                split_sentences = True,
                                replace_numbers = True,
                                tokenize = False,
                                lemmatize = True,
                                remove_stops = False,
                                split_on = "[\.!,]+[ ]*",
                                embed = False)
                                
        sentiments = model_nb.predict_proba(data_for_sentiment['clean_text'])[:,1] * 2 - 1

        # ## compute sentiment score for each topic
        # ## equal to the sentiment for on-topic sentences, or 0 otherwise
        # uniqueness_score = np.sum(np.multiply(sentiments,isin_unique))
        # humantouch_score = np.sum(np.multiply(sentiments,isin_humantouch))
        # quality_score = np.sum(np.multiply(sentiments,isin_quality))


        sentences = myfunc.sentence_splitter([review], split_on = "[\.!,]+[ ]*")
        uniqueness = [any([w in uniqueness_pool for w in s.split()]) for s in sentences]
        humantouch = [any([w in humantouch_pool for w in s.split()]) for s in sentences]
        quality = [any([w in quality_pool for w in s.split()]) for s in sentences]

        ## compute sentiment score for each topic
        ## equal to the average sentiment on those sentences that are about the topic
        uniqueness_score = np.sum(np.multiply(sentiments,uniqueness))/np.max([1,np.sum(uniqueness)])
        humantouch_score = np.sum(np.multiply(sentiments,humantouch))/np.max([1,np.sum(humantouch)])
        quality_score = np.sum(np.multiply(sentiments,quality))/np.max([1,np.sum(quality)])


        # data_topics = pd.DataFrame({'sentence' : sentences,
        #                             'uniqueness' : uniqueness,
        #                             'humantouch' : humantouch,
        #                             'quality' : quality})

        # uniqueness_text = data_topics.loc[data_topics['uniqueness'], 'sentence'].str.cat(sep='. ')
        # humantouch_text = data_topics.loc[data_topics['humantouch'], 'sentence'].str.cat(sep='. ')
        # quality_text = data_topics.loc[data_topics['quality'], 'sentence'].str.cat(sep='. ')

        return np.round(uniqueness_score,2),np.round(humantouch_score,2), np.round(quality_score,2)

    if request.method == 'POST':
        # review = request.form['review_input']
        # uniqueness_score, humantouch_score, quality_score = topic_sentiment_model(review)
        # print(request.form)
        if request.form['submit_button'] == "Analyze Your Review":
            review = request.form['review_input']
            if review != '':
            	uniqueness_score, humantouch_score, quality_score = topic_sentiment_model(review, WordNetLemmatizer())
            	print(uniqueness_score, humantouch_score, quality_score)
            else:
            	uniqueness_score, humantouch_score, quality_score = -2,-2,-2
        elif request.form['submit_button'] == "Positive":
            review = "I have had the pleasure of buying several pieces from him, and each is as unique as they are high quality. Love dealing with him."
            uniqueness_score, humantouch_score, quality_score = topic_sentiment_model(review, WordNetLemmatizer())
        elif request.form['submit_button'] == "Negative":
            review = "This seller pathological scammer and liar: 1. Sent to me defective earrings,one with broken lock,dirty,scratched, with different design-swirls on the top are different on each earring 2. Seller lied about shipping- see screenshots - said earrings were shipped, and they are not, liar shipped them only after my message Seller refusing admit her fault, arguing and lying, shame, i recommend avoid this shop"
            uniqueness_score, humantouch_score, quality_score = topic_sentiment_model(review, WordNetLemmatizer())
        elif request.form['submit_button'] == "Unique":
            review = "I love this unique bracelet. I am happy as always with my purchases from this shop!"
            uniqueness_score, humantouch_score, quality_score = topic_sentiment_model(review, WordNetLemmatizer())
        elif request.form['submit_button'] == "Human":
            review = "She really cares and goes out of her way to accommodate all of your needs! I received item very quickly and everything is so beautiful!!! THANK YOU!!! I cant wait to see my daughter's face Christmas morning!"
            uniqueness_score, humantouch_score, quality_score = topic_sentiment_model(review, WordNetLemmatizer())
        elif request.form['submit_button'] == "Quality":
            review = "I ordered these bracelets for wedding gifts for my future mother-in-law and sisters-in-law. I understand that the materials aren't the greatest, but I haven't even been able to give them the bracelets and they are already changing color. Definitely upset. Anyone know how to polish this material in order it not to look tarnished ?"
            uniqueness_score, humantouch_score, quality_score = topic_sentiment_model(review, WordNetLemmatizer())

    # session['unique_value'] = uniqueness_score
    # session['humantouch_value'] = humantouch_score
    # session['quality_value'] = quality_score

    return render_template('index.html',
        predict_uniq = uniqueness_score,
        predict_humt = humantouch_score,
        predict_qual = quality_score,
        review_text = review)


if __name__ == '__main__':
    app.run(host="0.0.0.0")