
# coding: utf-8

# # Assignment 3
# 
# In this assignment you will explore text message data and create models to predict if a message is spam or not. 

# In[1]:


import pandas as pd
import numpy as np

spam_data = pd.read_csv('spam.csv')

spam_data['target'] = np.where(spam_data['target']=='spam',1,0)
spam_data.head(10)


# In[2]:


from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(spam_data['text'], 
                                                    spam_data['target'], 
                                                    random_state=0)


# ### Question 1
# What percentage of the documents in `spam_data` are spam?
# 
# *This function should return a float, the percent value (i.e. $ratio * 100$).*

# In[3]:


def answer_one():
    
    answer = len(spam_data[spam_data['target'] == 1])/len(spam_data) *100
    return answer


# In[4]:


answer_one()


# ### Question 2
# 
# Fit the training data `X_train` using a Count Vectorizer with default parameters.
# 
# What is the longest token in the vocabulary?
# 
# *This function should return a string.*

# In[5]:


from sklearn.feature_extraction.text import CountVectorizer

def answer_two():
    
    from sklearn.feature_extraction.text import CountVectorizer
    vect = CountVectorizer().fit(X_train)
    answer = max(vect.get_feature_names(), key = len)
    
    return answer


# In[6]:


answer_two()


# ### Question 3
# 
# Fit and transform the training data `X_train` using a Count Vectorizer with default parameters.
# 
# Next, fit a fit a multinomial Naive Bayes classifier model with smoothing `alpha=0.1`. Find the area under the curve (AUC) score using the transformed test data.
# 
# *This function should return the AUC score as a float.*

# In[7]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score

def answer_three():
    vect = CountVectorizer()
    X_train_vectorized = vect.fit_transform(X_train)
    clfrNB = MultinomialNB(alpha = 0.1)
    clfrNB.fit(X_train_vectorized, y_train)
    
    X_test_vectorized = vect.transform(X_test) # NOT fit_transform since it's fitted with train data before already
    y_pred = clfrNB.predict(X_test_vectorized) 
    answer = roc_auc_score(y_test, y_pred)
    
    return answer


# In[8]:


answer_three()


# ### Question 4
# 
# Fit and transform the training data `X_train` using a Tfidf Vectorizer with default parameters.
# 
# What 20 features have the smallest tf-idf and what 20 have the largest tf-idf?
# 
# Put these features in a two series where each series is sorted by tf-idf value and then alphabetically by feature name. The index of the series should be the feature name, and the data should be the tf-idf.
# 
# The series of 20 features with smallest tf-idfs should be sorted smallest tfidf first, the list of 20 features with largest tf-idfs should be sorted largest first. 
# 
# *This function should return a tuple of two series
# `(smallest tf-idfs series, largest tf-idfs series)`.*

# In[112]:


# For each feature compute first the maximum tf-idf value across all documents in X_train. 
# What 20 features have the smallest maximum tf-idf value and what 20 features have the largest maximum tf-idf value?

from sklearn.feature_extraction.text import TfidfVectorizer

def answer_four():
    
    tfidf = TfidfVectorizer().fit(X_train)
    feature = np.array(tfidf.get_feature_names())
    maxi = np.amax(tfidf.transform(X_train).toarray(), axis=0)
    
    df = pd.DataFrame({'feature': feature, 'maxi': maxi}).sort_values(by = ['maxi', 'feature'])
    answer1 = df.iloc[0:20].values
    df = pd.DataFrame({'feature': feature, 'maxi': maxi}).sort_values(by = ['maxi', 'feature'], ascending=[False, True])
    answer2 = df.iloc[0:20].values
    
    return  answer1, answer2


# In[113]:


answer_four()


# ### Question 5
# 
# Fit and transform the training data `X_train` using a Tfidf Vectorizer ignoring terms that have a document frequency strictly lower than **3**.
# 
# Then fit a multinomial Naive Bayes classifier model with smoothing `alpha=0.1` and compute the area under the curve (AUC) score using the transformed test data.
# 
# *This function should return the AUC score as a float.*

# In[32]:


def answer_five():
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    tfidf = TfidfVectorizer(min_df=3).fit(X_train)
    X_train_vectorized = tfidf.transform(X_train)
    clfrNB = MultinomialNB(alpha = 0.1)
    clfrNB.fit(X_train_vectorized, y_train)
    
    X_test_vectorized = tfidf.transform(X_test) # NOT fit_transform since it's fitted with train data before already
    y_pred = clfrNB.predict(X_test_vectorized) 
    answer = roc_auc_score(y_test, y_pred)
    
    return answer


# In[33]:


answer_five()


# ### Question 6
# 
# What is the average length of documents (number of characters) for not spam and spam documents?
# 
# *This function should return a tuple (average length not spam, average length spam).*

# In[34]:


def answer_six():
    
    df = spam_data
    df['length'] = [len(w) for w in df['text']]
    
    return (np.mean(df[df['target']==0]['length']), np.mean(df[df['target']==1]['length']))


# In[35]:


answer_six()


# <br>
# <br>
# The following function has been provided to help you combine new features into the training data:

# In[36]:


def add_feature(X, feature_to_add):
    """
    Returns sparse feature matrix with added feature.
    feature_to_add can also be a list of features.
    """
    from scipy.sparse import csr_matrix, hstack
    return hstack([X, csr_matrix(feature_to_add).T], 'csr')


# ### Question 7
# 
# Fit and transform the training data X_train using a Tfidf Vectorizer ignoring terms that have a document frequency strictly lower than **5**.
# 
# Using this document-term matrix and an additional feature, **the length of document (number of characters)**, fit a Support Vector Classification model with regularization `C=10000`. Then compute the area under the curve (AUC) score using the transformed test data.
# 
# *This function should return the AUC score as a float.*

# In[37]:


from sklearn.svm import SVC

def answer_seven():
    
    df = pd.DataFrame(X_train)
    df['length'] = [len(w) for w in X_train]
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    tfidf = TfidfVectorizer(min_df=5).fit(df.text)
    X_train_vectorized = tfidf.transform(df.text)
    X_train_vectorized = add_feature(X_train_vectorized, df.length)

    clfrSVC = SVC(C = 10000)
    clfrSVC.fit(X_train_vectorized, y_train)
    
    df = pd.DataFrame(X_test)
    df['length'] = [len(w) for w in X_test]
    
    X_test_vectorized = tfidf.transform(df.text)
    X_test_vectorized = add_feature(X_test_vectorized, df.length)
    y_pred = clfrSVC.predict(X_test_vectorized) 
    answer = roc_auc_score(y_test, y_pred)
    
    return answer


# In[38]:


answer_seven()


# ### Question 8
# 
# What is the average number of digits per document for not spam and spam documents?
# 
# *This function should return a tuple (average # digits not spam, average # digits spam).*

# In[39]:


def answer_eight():
    df = spam_data
    df['digits'] = df['text'].str.findall('\d')
    df['length'] = [len(w) for w in df['digits']]
    return (np.mean(df[df['target']==0]['length']), np.mean(df[df['target']==1]['length']))


# In[40]:


answer_eight()


# ### Question 9
# 
# Fit and transform the training data `X_train` using a Tfidf Vectorizer ignoring terms that have a document frequency strictly lower than **5** and using **word n-grams from n=1 to n=3** (unigrams, bigrams, and trigrams).
# 
# Using this document-term matrix and the following additional features:
# * the length of document (number of characters)
# * **number of digits per document**
# 
# fit a Logistic Regression model with regularization `C=100`. Then compute the area under the curve (AUC) score using the transformed test data.
# 
# *This function should return the AUC score as a float.*

# In[41]:


from sklearn.linear_model import LogisticRegression

def answer_nine():
    df = pd.DataFrame(X_train)
    df['length'] = [len(w) for w in X_train]
    df['digit'] = df['text'].str.findall('\d')
    df['digits'] = [len(w) for w in df['digit']]
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    tfidf = TfidfVectorizer(min_df=5, ngram_range=(1,3)).fit(df.text)
    X_train_vectorized = tfidf.transform(df.text)
    X_train_vectorized = add_feature(X_train_vectorized, df.length)
    X_train_vectorized = add_feature(X_train_vectorized, df.digits)

    clfrLR = LogisticRegression(C = 100)
    clfrLR.fit(X_train_vectorized, y_train)
    
    df = pd.DataFrame(X_test)
    df['length'] = [len(w) for w in X_test]
    df['digit'] = df['text'].str.findall('\d')
    df['digits'] = [len(w) for w in df['digit']]
    
    X_test_vectorized = tfidf.transform(df.text)
    X_test_vectorized = add_feature(X_test_vectorized, df.length)
    X_test_vectorized = add_feature(X_test_vectorized, df.digits)
    y_pred = clfrLR.predict(X_test_vectorized) 
    answer = roc_auc_score(y_test, y_pred)

    return answer


# In[42]:


answer_nine()


# ### Question 10
# 
# What is the average number of non-word characters (anything other than a letter, digit or underscore) per document for not spam and spam documents?
# 
# *Hint: Use `\w` and `\W` character classes*
# 
# *This function should return a tuple (average # non-word characters not spam, average # non-word characters spam).*

# In[43]:


def answer_ten():
    
    df = spam_data
    df['nonword'] = df['text'].str.findall('\W')
    df['length'] = [len(w) for w in df['nonword']]
    return (np.mean(df[df['target']==0]['length']), np.mean(df[df['target']==1]['length']))


# In[44]:


answer_ten()


# ### Question 11
# 
# Fit and transform the training data X_train using a Count Vectorizer ignoring terms that have a document frequency strictly lower than **5** and using **character n-grams from n=2 to n=5.**
# 
# To tell Count Vectorizer to use character n-grams pass in `analyzer='char_wb'` which creates character n-grams only from text inside word boundaries. This should make the model more robust to spelling mistakes.
# 
# Using this document-term matrix and the following additional features:
# * the length of document (number of characters)
# * number of digits per document
# * **number of non-word characters (anything other than a letter, digit or underscore.)**
# 
# fit a Logistic Regression model with regularization C=100. Then compute the area under the curve (AUC) score using the transformed test data.
# 
# Also **find the 10 smallest and 10 largest coefficients from the model** and return them along with the AUC score in a tuple.
# 
# The list of 10 smallest coefficients should be sorted smallest first, the list of 10 largest coefficients should be sorted largest first.
# 
# The three features that were added to the document term matrix should have the following names should they appear in the list of coefficients:
# ['length_of_doc', 'digit_count', 'non_word_char_count']
# 
# *This function should return a tuple `(AUC score as a float, smallest coefs list, largest coefs list)`.*

# In[45]:


def answer_eleven():
    
    df = pd.DataFrame(X_train)
    df['length'] = [len(w) for w in X_train]
    df['digit'] = df['text'].str.findall('\d')
    df['nonword'] = df['text'].str.findall('\W')
    df['digits'] = [len(w) for w in df['digit']]
    df['nonwords'] = [len(w) for w in df['nonword']]
    
    
    vect = CountVectorizer(min_df=5, ngram_range=(2,5), analyzer='char_wb').fit(df.text)
    X_train_vectorized = vect.transform(df.text)
    X_train_vectorized = add_feature(X_train_vectorized, df.length)
    X_train_vectorized = add_feature(X_train_vectorized, df.digits)
    X_train_vectorized = add_feature(X_train_vectorized, df.nonwords)

    clfrLR = LogisticRegression(C = 100)
    clfrLR.fit(X_train_vectorized, y_train)
    
    small = np.sort(clfrLR.coef_)[0][:10]
    large = np.sort(clfrLR.coef_)[0][-10:]
    
    df = pd.DataFrame(X_test)
    df['length'] = [len(w) for w in X_test]
    df['digit'] = df['text'].str.findall('\d')
    df['nonword'] = df['text'].str.findall('\W')
    df['digits'] = [len(w) for w in df['digit']]
    df['nonwords'] = [len(w) for w in df['nonword']]
    
    X_test_vectorized = vect.transform(df.text)
    X_test_vectorized = add_feature(X_test_vectorized, df.length)
    X_test_vectorized = add_feature(X_test_vectorized, df.digits)
    X_test_vectorized = add_feature(X_test_vectorized, df.nonwords)
    
    y_pred = clfrLR.predict(X_test_vectorized) 
    answer = roc_auc_score(y_test, y_pred)


    return answer, small, large


# In[46]:


answer_eleven()


# In[ ]:




