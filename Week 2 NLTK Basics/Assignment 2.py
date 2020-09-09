
# # Assignment 2 - Introduction to NLTK
# 
# In part 1 of this assignment you will use nltk to explore the Herman Melville novel Moby Dick. Then in part 2 you will create a spelling recommender function that uses nltk to find words similar to the misspelling. 

# ## Part 1 - Analyzing Moby Dick


import nltk
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# If you would like to work with the raw text you can use 'moby_raw'
with open('moby.txt', 'r') as f:
    moby_raw = f.read()
    
# If you would like to work with the novel in nltk.Text format you can use 'text1'
moby_tokens = nltk.word_tokenize(moby_raw)
text1 = nltk.Text(moby_tokens)


# ### Example 1
# 
# How many tokens (words and punctuation symbols) are in text1?
# 
# *This function should return an integer.*



def example_one():
    
    return len(nltk.word_tokenize(moby_raw)) # or alternatively len(text1)

example_one()


# ### Example 2
# 
# How many unique tokens (unique words and punctuation) does text1 have?
# 
# *This function should return an integer.*



def example_two():
    
    return len(set(nltk.word_tokenize(moby_raw))) # or alternatively len(set(text1))

example_two()


# ### Example 3
# 
# After lemmatizing the verbs, how many unique tokens does text1 have?
# 
# *This function should return an integer.*



from nltk.stem import WordNetLemmatizer
def example_three():

    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(w,'v') for w in text1]
    return len(set(lemmatized))

example_three()


# ### Question 1
# 
# What is the lexical diversity of the given text input? (i.e. ratio of unique tokens to the total number of tokens)
# 
# *This function should return a float.*



def answer_one():
    
    #return (len(set(nltk.word_tokenize(moby_raw)))/len(nltk.word_tokenize(moby_raw)))
    return (len(set(text1))/len(text1))

answer_one()


# ### Question 2
# 
# What percentage of tokens is 'whale'or 'Whale'?
# 
# *This function should return a float.*



def answer_two():
    
    from nltk import FreqDist
    dist = FreqDist(text1)
    whale = dist['whale'] + dist['Whale']
    return (whale*100)/len(text1)

answer_two()


# ### Question 3
# 
# What are the 20 most frequently occurring (unique) tokens in the text? What is their frequency?
# 
# *This function should return a list of 20 tuples where each tuple is of the form `(token, frequency)`. The list should be sorted in descending order of frequency.*



def answer_three():
    
    from nltk import FreqDist
    dist = FreqDist(text1)
    return dist.most_common(20)

answer_three()


# ### Question 4
# 
# What tokens have a length of greater than 5 and frequency of more than 150?
# 
# *This function should return an alphabetically sorted list of the tokens that match the above constraints. To sort your list, use `sorted()`*



def answer_four():
    
    from nltk import FreqDist
    dist = FreqDist(text1)
    vocab = dist.keys()
    freqwords = sorted([w for w in vocab if len(w)>5 and dist[w]>150])
    return freqwords

answer_four()


# ### Question 5
# 
# Find the longest word in text1 and that word's length.
# 
# *This function should return a tuple `(longest_word, length)`.*



def answer_five():
    
    from nltk import FreqDist
    dist = FreqDist(text1)
    vocab = dist.keys()
    return (sorted(vocab, key=len)[-1], len(sorted(vocab, key=len)[-1]))

answer_five()


# ### Question 6
# 
# What unique words have a frequency of more than 2000? What is their frequency?
# 
# "Hint:  you may want to use `isalpha()` to check if the token is a word and not punctuation."
# 
# *This function should return a list of tuples of the form `(frequency, word)` sorted in descending order of frequency.*



def answer_six():
    
    from nltk import FreqDist
    dist = FreqDist(text1)
    vocab = dist.keys()
    freqwords = sorted([(dist[w], w) for w in vocab if w.isalpha() and dist[w]>2000], reverse = True)
    
    return freqwords

answer_six()


# ### Question 7
# 
# What is the average number of tokens per sentence?
# 
# *This function should return a float.*



def answer_seven():
    
    sent = nltk.sent_tokenize(moby_raw)
    word = nltk.word_tokenize(moby_raw)
    
    return len(word)/len(sent)

answer_seven()


# ### Question 8
# 
# What are the 5 most frequent parts of speech in this text? What is their frequency?
# 
# *This function should return a list of tuples of the form `(part_of_speech, frequency)` sorted in descending order of frequency.*



def answer_eight():
    
    from nltk import FreqDist
    postag = nltk.pos_tag(text1)
    pos_counts = nltk.FreqDist(tag for (word, tag) in postag)
    return pos_counts.most_common(5)

answer_eight()


# ## Part 2 - Spelling Recommender
# 
# For this part of the assignment you will create three different spelling recommenders, that each take a list of misspelled words and recommends a correctly spelled word for every word in the list.
# 
# For every misspelled word, the recommender should find find the word in `correct_spellings` that has the shortest distance*, and starts with the same letter as the misspelled word, and return that word as a recommendation.
# 
# *Each of the three different recommenders will use a different distance measure (outlined below).
# 
# Each of the recommenders should provide recommendations for the three default words provided: `['cormulent', 'incendenece', 'validrate']`.



from nltk.corpus import words
correct_spellings = words.words()


# ### Question 9
# 
# For this recommender, your function should provide recommendations for the three default words provided above using the following distance metric:
# 
# **[Jaccard distance](https://en.wikipedia.org/wiki/Jaccard_index) on the trigrams of the two words.**
# 
# *This function should return a list of length three:
# `['cormulent_reccomendation', 'incendenece_reccomendation', 'validrate_reccomendation']`.*



def answer_nine(entries=['cormulent', 'incendenece', 'validrate']):
    distance = []
    for i in range(0,len(entries)):
        # get first letter of the word
        c = [w for w in correct_spellings if w[0]== entries[i][0]]
        # calculate the distance of the word with entry and link both together
        distance.append([(nltk.jaccard_distance(set(nltk.ngrams(entries[i], n=3)), set(nltk.ngrams(a, n=3))), a) for a in c])

    # sort them to ascending order so shortest distance is on top.
    # extract the word only
    output = [sorted(distance[0])[0][1], sorted(distance[1])[0][1], sorted(distance[2])[0][1]]
    
    return output
    
answer_nine()


# ### Question 10
# 
# For this recommender, your function should provide recommendations for the three default words provided above using the following distance metric:
# 
# **[Jaccard distance](https://en.wikipedia.org/wiki/Jaccard_index) on the 4-grams of the two words.**
# 
# *This function should return a list of length three:
# `['cormulent_reccomendation', 'incendenece_reccomendation', 'validrate_reccomendation']`.*



def answer_ten(entries=['cormulent', 'incendenece', 'validrate']):
    
    distance = []
    for i in range(0,len(entries)):
        # get first letter of the word
        c = [w for w in correct_spellings if w[0]== entries[i][0]]
        # calculate the distance of the word with entry and link both together
        distance.append([(nltk.jaccard_distance(set(nltk.ngrams(entries[i], n=4)), set(nltk.ngrams(a, n=4))), a) for a in c])

    # sort them to ascending order so shortest distance is on top.
    # extract the word only
    output = [sorted(distance[0])[0][1], sorted(distance[1])[0][1], sorted(distance[2])[0][1]]
    
    return output
    
answer_ten()


# ### Question 11
# 
# For this recommender, your function should provide recommendations for the three default words provided above using the following distance metric:
# 
# **[Edit distance on the two words with transpositions.](https://en.wikipedia.org/wiki/Damerau%E2%80%93Levenshtein_distance)**
# 
# *This function should return a list of length three:
# `['cormulent_reccomendation', 'incendenece_reccomendation', 'validrate_reccomendation']`.*



def answer_eleven(entries=['cormulent', 'incendenece', 'validrate']):
    
    distance = []
    for i in range(0,len(entries)):
        # get first letter of the word
        c = [w for w in correct_spellings if w[0]== entries[i][0]]
        # calculate the distance of the word with entry and link both together
        distance.append([((nltk.edit_distance(entries[i], a)), a) for a in c])

    # sort them to ascending order so shortest distance is on top.
    # extract the word only
    output = [sorted(distance[0])[0][1], sorted(distance[1])[0][1], sorted(distance[2])[0][1]]
    
    return output
    
answer_eleven()
