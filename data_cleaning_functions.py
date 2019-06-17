from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

def basic_cleaning(review_df, sentiment_cutoff = 4,
                   text_col = 'review_text', rating_col = 'rating',
                  keep_cols = ['review_text','y']):

    ## This function performs basic cleaning on my raw dataset to
    ## convert it to a form that can be processed by other functions.
    
    ## drop rows that don't contain any review text
    review_df = review_df[~review_df[text_col].isnull()].reset_index(drop=True)
    
    ## drop rows with improper rating
    review_df = review_df[review_df[rating_col].isin([5,4,3,2,1,'5','4','3','2','1'])]
    
    ## convert ratings to numeric (in case some are str)
    review_df[rating_col]= review_df[rating_col].astype(int) 
    
    ## add y column for sentiment classification. ratings > x are positive, <= are negative
    review_df['y'] = (review_df[rating_col] > sentiment_cutoff)
    
    ## remove unneccessary columns
    review_df = review_df.loc[:,keep_cols]
    
    return review_df

def sentence_splitter(reviews, split_on, cat = None):

    ## Takes df with 1 review per row and outputs df with 1 sentence per row.
    ## Keeps track of which sentence came from where using the cat variable.

    split_sentences = [[s for s in re.split(split_on,rev) if s != ""] for rev in reviews]
    count_sentences = [len(s) for s in split_sentences]
    sentences = [item.strip() for sublist in split_sentences for item in sublist]
    if cat is None:
        cat = np.arange(len(reviews))
    index = cat.repeat(count_sentences)
    return sentences, index

def remove_accented_chars(text):

    ## remove non-ascii characters
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text

def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):

    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), 
                                      flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)\
                                if contraction_mapping.get(match)\
                                else contraction_mapping.get(match.lower())                       
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction

    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text

def number_replacer(sentence):
    return re.sub('[A-Za-z]*[0-9]+[A-Za-z]*', 'numbertoken', sentence)

def tokenize_text(sentence):
    sentence = sentence.split(' ')
    return sentence

def alphanumerize(sentence, remove_pattern):
    removed = re.sub('<br>', '', sentence)
    removed = re.sub('&amp', '', removed) ## since the word 'amp' keeps showing up
    removed = re.sub('[Aa]\+', 'amazing', removed) ## A+ --> amazing
    removed = re.sub('thank you', 'thanks', removed) ## maybe remove this
    removed = re.sub(remove_pattern, '', removed)
    return removed

def lemmatize_text(text):
    return ' '.join([lemmatizer.lemmatize(w) for w in text.split(' ')])

def remove_stopwords(sentence, stopwords):
    return ' '.join([word for word in sentence.split(' ') if word not in stopwords])

def get_w2v_features(w2v_model, sentence):
    
    index2word_set = set(w2v_model.wv.vocab.keys()) # create vocab set
    embedding = np.zeros(w2v_model.vector_size, dtype="float32") # init vector
    
    ## there shouldn't be empty sentences, but include this just in case
    if len(sentence) > 0:
        nwords = 0 # word counter
        for word in sentence:
            if word in index2word_set:
                # if the word is in the set, add it and increment counter
                embedding = np.add(embedding, w2v_model[word])
                nwords += 1.
        if nwords > 0: # in case no words were aded
            embedding = np.divide(embedding, nwords)
        
    return embedding

def preprocessing(review_df,
                  split_sentences = False,
                  replace_numbers = True,
                  tokenize = True,
                  lemmatize = True,
                  remove_stops = True,
                  embed = True,
                  output_embeddings = True,
                  split_on = "[\.!]+[ ]*",
                  stopwords = '',
                  remove_pattern = '[^0-9A-Za-z ]+',
                  W2Vmodel = '',
                  cat = None,
                  old_col='review_text',
                  new_col='clean_text'):
    
    ## other options...
    # replace rare words with a certain token
    
    ## split into sentences (optional)
    if split_sentences:
        print('splitting into sentences...')
        sentences, indices = sentence_splitter(review_df[old_col],split_on,cat)
        review_df = pd.DataFrame({old_col : sentences,
                                 'index' : indices})
    
    ## remove accents
    print('removing accents...')
    review_df[new_col] = list(map(remove_accented_chars, review_df[old_col]))
    
    ## convert all text to lowercase
    print('converting to lowercase...')
    review_df[new_col] = review_df[new_col].str.lower()
    
    ## expand contractions
    print('expanding contractions...')
    review_df[new_col] = list(map(expand_contractions,review_df[new_col]))
    
    ## replace numbers (optional)
    if replace_numbers:
        print('replacing numbers...')
        review_df[new_col] = list(map(number_replacer,review_df[new_col]))
    
    ## clear non-alphanumerics
    print('alphanumerizing...')
    review_df[new_col] = [alphanumerize(rev, remove_pattern) for rev in review_df[new_col]]
    
    ## lemmatize (optional)
    if lemmatize:
        print('lemmatizing...')
        review_df[new_col] = list(map(lemmatize_text,review_df[new_col]))
        
    ## remove stop words (optional)
    if remove_stops:
        print('removing stop words...')
        if stopwords == '':
            stopwords = nltk.corpus.stopwords.words('english')
            stopwords.remove('not')
        if lemmatize:
            stopwords = [lemmatizer.lemmatize(w) for w in stopwords]
        review_df[new_col] = [remove_stopwords(rev, stopwords) for rev in review_df[new_col]]

    ## tokenize (optional)
    if tokenize:
        print('tokenizing...')
        review_df[new_col] = list(map(tokenize_text,review_df[new_col]))
    
    ## embed (optional)
    if embed:
        print('embedding words...')
        review_df['embeddings'] = list(map(lambda sentence:get_w2v_features(W2Vmodel, sentence),
                                    review_df[new_col]))
        
        if output_embeddings:
            print('converting to embedded form...')
            review_df = review_df['embeddings'].apply(pd.Series)
    
    return review_df

def create_stopwords(my_stopwords):
    stopwords_sklearn = list(sw_sklearn.ENGLISH_STOP_WORDS)
    stopwords_nltk = list(sw_nltk.words('english'))
    stopwords_gensim = list(sw_gensim)

    custom_stopwords = set(stopwords_sklearn +
                           stopwords_nltk +
                           stopwords_gensim +
                           my_stopwords)
    
    return custom_stopwords

def similarity_test(rev, pool, vecs):
    similarity = 0
    for rev_word in rev:
        if rev_word in vecs:
            for pool_word in pool:
                if pool_word in vecs:
                    word_sim = vecs.similarity(rev_word,pool_word)
                    similarity = np.max([word_sim,similarity])
    return similarity

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax