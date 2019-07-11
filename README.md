# BrandGuard
### Know The Companies You Keep
---

This is the repository for my project for Insight Data Science, which was completed from start (defining a problem) to finish (deploying a web app) over the span of a few weeks during my time as a Data Science Fellow at Insight. The web app can be accessed at [mikeklein.ca](mikeklein.ca). Presentation slides can be found [here](https://github.com/kleinmichaeldavid/insight-reviews/blob/master/Michael_Klein_Demo.pdf). 

### Background and Problem

Branding is an extremely important component of a company’s value. Many top companies have brands that are valued in the tens of [billions of dollars](https://www.usatoday.com/story/money/business/2018/10/09/most-valuable-brands-apple-google-amazon/38061893/). Although a brand does not make money directly, maintaining a strong brand adds value by bringing in new customers and increasing the loyalty of current customers whose values are aligned with the brand. For this project, I’ve put myself in the shoes of Etsy, which is an online marketplace that allows anyone to sign up for an online shop through which they can sell handmade goods. Etsy works hard to maintain their branding, highlighting values such as Uniqueness, High Quality, and Human Touch. Unfortunately, the fact that anyone can sell through Etsy makes it difficult to curate the content in their marketplace, resulting in many shops and products on their website that do not live up to their branding. It is easy to find examples of people who have gone to Etsy because of their branding and come away disappointed, leaving negative comments online. The challenge I took on for this project was to build a model that could determine how well each of the 2+ million shops on Etsy align with their brand values, so that Etsy can highlight and promote only those shops that are best aligned.

### My Approach

To determine how well each online shop aligns with Etsy’s values of Uniqueness, High Quality, and Human Touch, I made use of the customer reviews available on the website, which often discuss topics relevant to Etsy’s values. For the purposes of this project, I defined a shop to be in line with Etsy’s values if, when the reviews for that shop discuss topics relevant to each value, they tend to use positive rather than negative (or neutral) language. With this in mind, I divided the problem into two parts: (1) identifying the reviews that are relevant to each value, and (2) determining the sentiment of those reviews. 

### Part 1: Which reviews discuss Uniqueness, Quality, and/or Human Touch?

The main challenges for this problem were (a) the problem was unsupervised – that is, I did not have topic labels for the customer reviews in my data set, and (b) a single review could not only cover multiple topics, but also have different sentiment for those separate topics. To account for this possibility, I divided each review into individual sentences with the idea that each sentence could be labelled with its own topic and then undergo sentiment analysis separately. Although this is not a perfect solution, a scan of the reviews suggests that when they discuss multiple topics, they do tend to do so in separate sentences. 

Because our projects at Insight need to be completed within only a couple weeks, it is very important to create a minimum viable product as quickly as possible. With this in mind, my first approach was simply to come up with a representative list of words for each value to and consider a sentence to be relevant to that value if it contained a word from its list. I generated the lists using the following procedure: (a) finding key words from Etsy’s website, particularly their ‘About Us’ page, (b) adding synonyms and antonyms of words already included, (c) checking the reviews in my data set for words that had been missed by the previous steps, and (d) filtering out words based on performance on a validation set (outlined below). The goal of this task was not necessarily to correctly label every single sentence but rather to find a set of sentences that would be representative of all the relevant sentences for a particular shop. I therefore erred on the side of reducing false positives, even if it came at the expense of missing some sentences that used atypical language to discuss a relevant topic – I wanted to make sure that the sentences that I labelled as relevant were not ‘contaminated’ by sentences that were not actually relevant.

Here is the list of words I used to represent each value:

> Uniqueness: unique, creative, special, inventive, innovative, handmade, original, handcrafted, creativity,  crafted

> Quality: quality, condition, described, detailed, craftmanship, craftsmanship, detail, workmanship, material,  broke, break, fell, apart, cheap

> Human Touch: helpful, communication, response, respond, contact, friendly, attentive, accommodating, courteous,  polite, respect, pleasant, service, kind, personal, seller, personable, interaction, reply, answer, honest, rude,  unhelpful, unresponsive, personalised, personalized

To test the performance of this approach, I labelled the topics of 600 sentences that I had set aside earlier to use as a validation set. The confusion matrix for each value is shown below. As can be seen, although only a fraction of the truly relevant sentences are identified (ranging from 30 to 41%), a very small proportion of irrelevant sentences are misidentified (1 to 3%). 

![alt text](https://github.com/kleinmichaeldavid/insight-reviews/blob/master/images_for_readme/uniqueness_confusion.png) ![alt text](https://github.com/kleinmichaeldavid/insight-reviews/blob/master/images_for_readme/quality_confusion.png) ![alt text](https://github.com/kleinmichaeldavid/insight-reviews/blob/master/images_for_readme/human_touch_confusion.png)

To account for class imbalance (most sentences are not relevant to any of the topics), I also measured precision, which is the proportion of sentences that were predicted to be relevant that actually were relevant. Precision scores for Uniqueness, Quality, and Human Touch were, respectively, 0.73, 0.83, and 0.64, indicating that most of the reviews identified as being relevant actually were, although a fairly large proportion were not. 

### Part 2: What is the sentiment of the relevant reviews?

To determine the sentiment of each review, I made use of the fact that although they were not strictly labelled as positive or negative, the reviews were all accompanied by a rating out of 5 stars. I used this rating as a proxy for sentiment, treating reviews accompanied by a rating of 4 or 5 stars as positive, and those accompanied by a rating of 1-3 stars as negative. I selected a Naive Bayes Classifier as an initial model for this task for several reasons, including their simplicity (e.g. few hyperparameters to choose), interpretability, and speed on both training and prediction. The pipeline included preprocessing (removing non-english reviews, replacing numbers, and lemmatization), TF-IDF vectorization (I included both single words and bigrams in the vectorization so that the model could account for basic interactions such as negation), random oversampling of negative sentences (over 95% of reviews were positive), and finally, training. Note that although the model was used to predict the sentiment of individual sentences, it was trained using full reviews since the star ratings corresponded to the full reviews but not necessarily to each sentence within the reviews. Despite its simplicity, the model performed quite well, achieving over 90% accuracy at detecting both positive and negative sentiment in validation set reviews.

![alt text](https://github.com/kleinmichaeldavid/insight-reviews/blob/master/images_for_readme/sentiment_confusion.png)

However, due to the extreme class imbalance, precision was quite poor, at only 0.35 (taking the ‘positive’ class to be negative sentiment – that is, only 35% of reviews identified as negative actually were). The recall (of all negative reviews, how many were identified) was quite good at 0.91.
	
I compared the performance of this model to the performance of VADER, which is a rule-based sentiment analysis tool that uses a sentiment lexicon to determine sentiment. The Naive Bayes model clearly outperformed the VADER baseline.

![alt text](https://github.com/kleinmichaeldavid/insight-reviews/blob/master/images_for_readme/sentiment_roc.png)

To further validate the Naive Bayes model, I determined the most positive and most negative features by finding those features that, if present, were most predictive of a review being either positive or negative. The features returned by this analysis make sense, with tokens such as “overpriced”, “still waiting” and “rude” appearing among the most negative, and “love love”, “perfect thanks”, and “great seller” appearing among the most positive.

![alt text](https://github.com/kleinmichaeldavid/insight-reviews/blob/master/images_for_readme/most_negative_features.png) ![alt text](https://github.com/kleinmichaeldavid/insight-reviews/blob/master/images_for_readme/most_positive_features.png)

Given more time, two avenues I would like to explore to improve the model would be to (a) collect enough data to be able to train the sentiment analysis model using only reviews that are relevant to Etsy’s brand values, instead of all reviews in my data set, and (b) try higher complexity models such as gradient-boosted decision trees or neural nets to maximize performance enough to overcome the extreme class imbalance. 

### How would this look in practice?

There are many ways that this model could be implemented in practice to help to promote those sellers that best align with Etsy’s branding. One basic approach might be to simply use the model as a filter on top of the methods Etsy currently uses to determine where to display products. For example, if a user conducted a search or clicked on a product category, the model could filter shops that do not align with Etsy’s branding from the first page. I analyzed the shops appearing in the ‘Bracelets’ category using my model, finding that many of the shops that appeared on the very first page did not align well with Etsy’s branding, having negative sentiment on Uniqueness, Quality, and/or Human Touch (see boxplots below). By pushing these shops off the first page and replacing them with shops from the second page, potential customers could have a much improved experience.

![alt text](https://github.com/kleinmichaeldavid/insight-reviews/blob/master/images_for_readme/bracelets_first_page.png)

Similarly, there is no trend in average sentiment per page across the first 100 pages of the Bracelet category (see line plots below), suggesting that the shops that best align with Etsy’s values are not being highlighted relative to those that do not align well. By rearranging the order of shops within this category, potential customers could go through 50 or 60 pages of items before encountering a shop that doesn’t align well with Etsy’s brand values. 

![alt text](https://github.com/kleinmichaeldavid/insight-reviews/blob/master/images_for_readme/bracelets_uniqueness.png) ![alt text](https://github.com/kleinmichaeldavid/insight-reviews/blob/master/images_for_readme/bracelets_quality.png) ![alt text](https://github.com/kleinmichaeldavid/insight-reviews/blob/master/images_for_readme/bracelets_human_touch.png)

