
# coding: utf-8

# # Assignment 1
# 
# In this assignment, you'll be working with messy medical data and using regex to extract relevant infromation from the data. 
# 
# Each line of the `dates.txt` file corresponds to a medical note. Each note has a date that needs to be extracted, but each date is encoded in one of many formats.
# 
# The goal of this assignment is to correctly identify all of the different date variants encoded in this dataset and to properly normalize and sort the dates. 
# 
# Here is a list of some of the variants you might encounter in this dataset:
# * 04/20/2009; 04/20/09; 4/20/09; 4/3/09
# * Mar-20-2009; Mar 20, 2009; March 20, 2009;  Mar. 20, 2009; Mar 20 2009;
# * 20 Mar 2009; 20 March 2009; 20 Mar. 2009; 20 March, 2009
# * Mar 20th, 2009; Mar 21st, 2009; Mar 22nd, 2009
# * Feb 2009; Sep 2009; Oct 2010
# * 6/2008; 12/2009
# * 2009; 2010
# 
# Once you have extracted these date patterns from the text, the next step is to sort them in ascending chronological order accoring to the following rules:
# * Assume all dates in xx/xx/xx format are mm/dd/yy
# * Assume all dates where year is encoded in only two digits are years from the 1900's (e.g. 1/5/89 is January 5th, 1989)
# * If the day is missing (e.g. 9/2009), assume it is the first day of the month (e.g. September 1, 2009).
# * If the month is missing (e.g. 2010), assume it is the first of January of that year (e.g. January 1, 2010).
# * Watch out for potential typos as this is a raw, real-life derived dataset.
# 
# With these rules in mind, find the correct date in each note and return a pandas Series in chronological order of the original Series' indices.
# 
# For example if the original series was this:
# 
#     0    1999
#     1    2010
#     2    1978
#     3    2015
#     4    1985
# 
# Your function should return this:
# 
#     0    2
#     1    4
#     2    0
#     3    1
#     4    3
# 
# Your score will be calculated using [Kendall's tau](https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient), a correlation measure for ordinal data.
# 
# *This function should return a Series of length 500 and dtype int.*

# In[1]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
doc = []
with open('dates.txt') as file:
    for line in file:
        doc.append(line)

df = pd.DataFrame(doc, columns=['text'])
df.head()


# In[2]:


def date_sorter():
    global df
    # 1
    df['extract'] = df['text'].str.findall(r'(?P<month>\d{1,2})[-/](?P<day>\d{1,2})[-/](?P<year>\d{2,4})')
    df['mm'] = df['extract'].str[0].str[0]
    df['dd'] = df['extract'].str[0].str[1]
    df['yyyy'] = df['extract'].str[0].str[2]

    # 2
    df.loc[df['extract'].astype(str)=='[]', 'extract'] = df['text'].str.findall(r'(?P<day>\d{1,2}) (?P<month>(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*) (?P<year>\d{2,4})')
    df.loc[df['mm'].isnull(), 'mm'] = df['extract'].str[0].str[1]
    df.loc[df['dd'].isnull(), 'dd'] = df['extract'].str[0].str[0]
    df.loc[df['yyyy'].isnull(), 'yyyy'] = df['extract'].str[0].str[2]

    # 3
    df.loc[df['extract'].astype(str)=='[]', 'extract'] = df['text'].str.findall(r'(?P<month>(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*) (?P<day>\d{1,2}) (?P<year>\d{2,4})')
    df.loc[df['extract'].astype(str)=='[]', 'extract'] = df['text'].str.findall(r'(?P<month>(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*) (?P<day>\d{1,2}), (?P<year>\d{2,4})')
    df.loc[df['extract'].astype(str)=='[]', 'extract'] = df['text'].str.findall(r'(?P<month>(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*). (?P<day>\d{1,2}), (?P<year>\d{2,4})')
    df.loc[df['mm'].isnull(), 'mm'] = df['extract'].str[0].str[0]
    df.loc[df['dd'].isnull(), 'dd'] = df['extract'].str[0].str[1]
    df.loc[df['yyyy'].isnull(), 'yyyy'] = df['extract'].str[0].str[2]

    # 4
    df.loc[df['extract'].astype(str)=='[]', 'extract'] = df['text'].str.findall(r'(?P<month>(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*) (?P<year>\d{2,4})')
    df.loc[df['extract'].astype(str)=='[]', 'extract'] = df['text'].str.findall(r'(?P<month>(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*), (?P<year>\d{2,4})')
    df.loc[df['extract'].astype(str)=='[]', 'extract'] = df['text'].str.findall(r'(?P<month>\d{1,2})/(?P<year>\d{2,4})')
    df.loc[df['mm'].isnull(), 'mm'] = df['extract'].str[0].str[0]
    df.loc[df['dd'].isnull(), 'dd'] = '01'
    df.loc[df['yyyy'].isnull(), 'yyyy'] = df['extract'].str[0].str[1]

    # 5
    df.loc[df['extract'].astype(str)=='[]', 'extract'] = df['text'].str.findall(r'(?P<year>\d{4})')
    df.loc[df['yyyy'].isnull(), 'yyyy'] = df['extract'].str[0]
    df.loc[df['mm'].isnull(), 'mm'] = '01'
    df.loc[df['dd'].isnull(), 'dd'] = '01'   

    # replace
    df['mm'] = df['mm'].replace({'Jan':'01', 'January':'01', 'Janaury':'01', '1':'01',
                                 'Feb':'02', 'February':'02', '2':'02',
                                 'Mar':'03', 'March':'03', '3':'03',
                                 'Apr':'04', 'April':'04', '4':'04',
                                 'May': '05', '5':'05',
                                 'Jun':'06', 'June':'06', '6':'06',
                                 'Jul':'07', 'July':'07', '7':'07',
                                 'Aug':'08', 'August':'08', '8':'08',
                                 'Sep':'09', 'September':'09', '9':'09',
                                 'Oct':'10', 'October':'10', 
                                 'Nov':'11', 'November':'11', 
                                 'Dec':'12', 'December':'12', 'Decemeber':'12'})
    

    df['length'] = [len(w) for w in df['dd']]
    df.loc[df['length'] == 1, 'dd'] = '0' + df['dd']
    df['length'] = [len(w) for w in df['yyyy']]
    df.loc[df['length'] == 2, 'yyyy'] = '19' + df['yyyy']
    df['date'] = df['yyyy'] + '-' + df['mm'] + '-' + df['dd']
    df['date'] = pd.to_datetime(df['date'])
    
    return pd.Series(np.argsort(df['date']), dtype="int32")


# In[5]:


date_sorter()


# In[ ]:




