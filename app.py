
from flaskexample import app
from flask import request, render_template

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

@app.route('/', methods=["GET"])
def index():

    return render_template('index.html',
        predict_uniq = -2,
        predict_humt = 0.5,
        predict_qual = -2,
        review_text = '')


# @app.route("/", methods=["POST"])
# def submit_review():
#     review_text = request.form["review_input"]
#     print('doo doo doo')
#     return render_template('index.html',review_text = review_text)


@app.route('/',methods=['POST'])
def submit_review():
    ### also ,this should load when the page is opened, not when a review is submitted
    ### save as session?
    # can do multiple models and put if statement here
    W2Vmodel = joblib.load(open('W2Vmodel.pkl','rb'))
    model_nb = joblib.load(open('model_nb_200sellers.pkl','rb'))

    def topic_sentiment_model(review):
        ### function to determine the topic of each sentence in a review.

        stopwords_mike = ['bracelets','necklace','earrings',
            'bracelet','ring','daughter','mother',
            'anklet','cindy','gift','gorgeous',
            'item','product','color','colors',
            'lovely','perfect','cute','beautiful','great',
            'super','order','wonderful','work','awesome',
            'loved','lt','pretty','little','love','absolutely',
            'bunch','wear','friend','piece','amazing','happy',
            'fantastic','xx','summer','excellent','million',
            'michael','lisa','good','looks','highly','recommend',
            'price','leather','future','pleased','high','beads',
            'you','nice','size','looking',
            'design','buy','definitely','fast','quickly',
            'products','product','promptly','quickly','quick',
            'numbertoken','like','ordered','fit','purchase','bought',
            'christmas','beautifully','perfectly','purchased','husband',
            'ordering','adorable','way','small','better','wrist',
            'jewelry','wearing','day','wanted', 'easy', 'new','lot',
            'best','going', 'custom','sure', 'simple','right','person',
            'boyfriend','took', 'charm', 'sent', 'photo', 'week','use',
            'birthday','son','extra','bit','favorite', 'clasp','present',
            'nose','bag','sister','chain','mom', 'year', 'stone','wedding',
            'thought','delicate','feel','gave', 'different','dainty',
            'box','big','second','blanket','sweet','comfortable',
            'thing','extremely','baby','silver','want']

        uniqueness_pool = ['unique', 'creative', 'special',
            'inventive', 'innovative', 'handmade',
            'original', 'handcrafted',
            'creativity', 'crafted','made']

        humantouch_pool = ['helpful', 'communication', 'response',
            'respond', 'contact', 'friendly',
            'attentive', 'accommodating', 'courteous',
            'polite', 'respect', 'pleasant',
            'service','kind','personal',
            'personable','interaction','reply','answer',
            'honest','rude','unhelpful','unresponsive',
            'personalised','personalized']

        quality_pool = ['quality', 'condition', 'described',
            'detailed', 'craftmanship', 'craftsmanship',
            'detail', 'workmanship', 'material','broke','break',
            'fell','apart']

        custom_stopwords = myfunc.create_stopwords(stopwords_mike)


        app_input_df = pd.DataFrame({'review_text' : [review]})
        
        ## process the input for the topic classification
        input_data_for_topic = myfunc.preprocessing(app_input_df,
            split_sentences = True,
            replace_numbers = True,
            tokenize = True,
            lemmatize = True,
            remove_stops = True,
            embed = False,
            split_on = "[\.!,]+[ ]*",
            stopwords = custom_stopwords)

        sentences = input_data_for_topic['clean_text']
        uniqueness_sims = [myfunc.similarity_test(r, uniqueness_pool, W2Vmodel.wv) for r in sentences]
        humantouch_sims = [myfunc.similarity_test(r, humantouch_pool, W2Vmodel.wv) for r in sentences]
        quality_sims = [myfunc.similarity_test(r, quality_pool, W2Vmodel.wv) for r in sentences]

        uniqueness_conts = [myfunc.topic_contribution(sim) for sim in uniqueness_sims]
        humantouch_conts = [myfunc.topic_contribution(sim) for sim in humantouch_sims]
        quality_conts = [myfunc.topic_contribution(sim) for sim in quality_sims]

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

        ## compute sentiment score for each topic
        uniqueness_score = np.sum(np.multiply(sentiments,uniqueness_conts))
        humantouch_score = np.sum(np.multiply(sentiments,humantouch_conts))
        quality_score = np.sum(np.multiply(sentiments,quality_conts))

        sentences = review.strip().split('.')
        uniqueness = [any([w in uniqueness_pool for w in s.split()]) for s in sentences]
        humantouch = [any([w in humantouch_pool for w in s.split()]) for s in sentences]
        quality = [any([w in quality_pool for w in s.split()]) for s in sentences]

        data_topics = pd.DataFrame({'sentence' : sentences,
                                    'uniqueness' : uniqueness,
                                    'humantouch' : humantouch,
                                    'quality' : quality})

        uniqueness_text = data_topics.loc[data_topics['uniqueness'], 'sentence'].str.cat(sep='. ')
        humantouch_text = data_topics.loc[data_topics['humantouch'], 'sentence'].str.cat(sep='. ')
        quality_text = data_topics.loc[data_topics['quality'], 'sentence'].str.cat(sep='. ')

        return np.round(uniqueness_score,2),np.round(humantouch_score,2), np.round(quality_score,2)

    if request.method == 'POST':
     	review = request.form['review_input']
     	uniqueness_score, humantouch_score, quality_score = topic_sentiment_model(review)

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