#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as md
import seaborn as sns
import datetime as dt
import time
import datetime
from textblob import TextBlob
import nltk
from nltk import ngrams
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt')
import operator
import re


# In[2]:


# Function to get the n-grams

def getNGrams(df, col, n): # n is the 'n' in 'n-grams'
    ndiction = {}
    textbody = df[col]
    for text in textbody: # take each text/entry in df[col] and tokenize it
        #text = re.sub(r"\S*https?:\S*", "", text)
        text = re.sub(r'http\S+', '', text)
        text = text.encode('ascii', 'ignore').decode() # removing the unicode characters like the emojis
        text = text.replace('-', ' ')
        text = text.lower()
        candidates = word_tokenize(text)
        tokens = []
        stop_words = stopwords.words('english')
        stop_words.extend(['www', 'nlm', 'ncbi', 'x200b', 'pubmed', 'amp',
                           'pjpg', 'com', 'r', 'stopsmoking', 'azvz9tuefvotx9tr2zmhawmpgduxus',
                           'yqjqftiisfbn91zmhwjcxvcfncgylnlkcwbvd'])
        for tk in candidates:
            if tk not in stop_words and tk.isalnum():
                tokens.append(tk)
        n_grams = ngrams(tokens, n)
        for grams in n_grams:
            joined_grams = '-'.join(grams)
            if joined_grams in ndiction:
                ndiction[joined_grams] += 1
            else:
                ndiction[joined_grams] = 1
    sorted_grams_lst = sorted(ndiction.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_grams_lst


# In[3]:


# This Function will take in a column, add a new column of sentiments to it

def getSentimentPolarity(df, col): 
    df['sentiment'] = df[col].apply(lambda x: TextBlob(x).sentiment.polarity)
    return df


# In[4]:


# reading in the csv files from local source
# This is stopdrinking data

df_S = pd.read_csv('/Users/nayza/Downloads/stopdrinking_allsubmissions.csv') 
# not .tsv file so no separator specified
dfS = df_S.drop_duplicates(subset=['selftext'])
    
df_C = pd.read_csv('/Users/nayza/Downloads/stopdrinking_allcomments.csv')
dfC = df_C.drop_duplicates(subset=['body'])


# In[5]:


# splitting the comments and submissions files into one before covid and one during covid

beforedf_C =dfC.loc[df_C['created_utc'] < datetime.datetime(2020, 3, 10, 0, 0).timestamp()]
afterdf_C =dfC.loc[df_C['created_utc'] > datetime.datetime(2020, 3, 11, 0, 0).timestamp()]
    
beforedf_S =dfS.loc[df_S['created_utc'] < datetime.datetime(2020, 3, 10, 0, 0).timestamp()]
afterdf_S =dfS.loc[df_S['created_utc'] > datetime.datetime(2020, 3, 11, 0, 0).timestamp()]


# In[6]:


dfS.head()


# In[7]:


temp_dfS = dfS[['author', 'selftext', 'subreddit', 'date_normal']]
temp_dfS.head()


# In[ ]:


# DON'T run the cells below yet !!!!!!!


# In[8]:


df_S = getSentimentPolarity(dfS, 'selftext') # calling on getSentimentPolarity() function to get the sent polar values
df_S.head()


# In[9]:


df_C = getSentimentPolarity(dfC, 'body')
df_C.head()


# In[10]:


# plot the sentiment polarity vs. date (year-month)
# For this, we'd need to group the sentiments by date (year-month) and
# get the average sentiment of each month

count_C = [1]*len(df_C)
df_C['count'] = pd.Series(count_C).values # so that we can get the count of comments in each year-month category

unique_ym_C = set(df_C['year-month']) # this is just to compare

# Comments
df_C_sentGroup = df_C.groupby('year-month') # group data by year-month's
dates = [] 
means = []
freq = []
for group, subdf in df_C_sentGroup:
    dates.append(group)
    means.append(round(subdf['sentiment'].mean(), 3))
    freq.append(subdf['count'].sum())
    
# creating a dataframe with only Date, Average Sent. Polarity of each month
# and count of Comments
df_C_sentMeans = pd.DataFrame({'Year-Month': dates, 'Sentiment Polarity of Comments': means, 'Comment Count': freq})
df_C_sentMeans['Year-Month'] = pd.to_datetime(df_C_sentMeans['Year-Month'],
                                         format='%Y-%m').dt.strftime('%Y-%m')

sns.set(font_scale=1.5)
fig = plt.figure(figsize=(20,5))
ax1 = fig.add_subplot(111)
plt.xticks(rotation=60)
ax2 = ax1.twinx()
sns.barplot(x='Year-Month', y='Comment Count', data=df_C_sentMeans, ax=ax1)
sns.lineplot(x='Year-Month', y='Sentiment Polarity of Comments', data=df_C_sentMeans, ax=ax2, legend='auto', color='r')
plt.legend(labels=['Sentiment Polarity'])
plt.axvline(x='2020-03', linestyle='--', color='b', linewidth=2)
plt.show()


# In[11]:


import pymannkendall as mk

data_mk_test_C = df_C_sentMeans['Sentiment Polarity of Comments']

mk.original_test(data_mk_test_C[18:])
#print(df_C_sentMeans)


# In[12]:



count_S = [1]*len(df_S)
df_S['count'] = pd.Series(count_S).values

unique_ym_S = set(df_S['year-month']) # this is just to compare

# Submissions
dates_S = []
means_S = []
freq_S = []

df_S_sentGroup = df_S.groupby('year-month')
for group, subdf in df_S_sentGroup:
    dates_S.append(group)
    means_S.append(round(subdf['sentiment'].mean(), 3))
    freq_S.append(subdf['count'].sum())


# creating a dataframe with only Date, Average Sent. Polarity of each month
# and count of Submissions

df_S_sentMeans = pd.DataFrame({'Year-Month': dates_S, 'Sentiment Polarity of Submissions': means_S, 'Submission Count': freq_S})
df_S_sentMeans['Year-Month'] = pd.to_datetime(df_S_sentMeans['Year-Month'],
                                         format='%Y-%m').dt.strftime('%Y-%m')

sns.set(font_scale=1.5)
fig = plt.figure(figsize=(20,5))
ax1 = fig.add_subplot(111)
plt.xticks(rotation=60)
ax2 = ax1.twinx()
#ax2.set(ylim=(0, 0.20)) # new addition - might delete it later
sns.barplot(x='Year-Month', y='Submission Count', data=df_S_sentMeans, ax=ax1)
sns.lineplot(x='Year-Month', y='Sentiment Polarity of Submissions', data=df_S_sentMeans, ax=ax2, color='purple')
# adding a lineplot for Moving Average in the following line of code
rolling_avg = df_S_sentMeans['Sentiment Polarity of Submissions'].rolling(window=7).mean()
df_S_sentMeans['Rolling Average'] = rolling_avg.values
sns.lineplot(x='Year-Month', y='Rolling Average', data=df_S_sentMeans, ax=ax2, color='green')

plt.legend(labels=['Sentiment Polarity', 'Rolling Average of Sent. Polarity'])
plt.axvline(x='2020-03', linestyle='--', color='b', linewidth=2)
plt.show()


# In[13]:


# will do a trend line test using pymannkendall 

data_mk_test = df_S_sentMeans['Sentiment Polarity of Submissions'].tolist()

mk.original_test(data_mk_test[25:]) # trend test shows that there is 'no trend' after covid started although
                                    # trend test shows 'decreasing' for the entire period


# In[14]:


# Now, we will extract from four groups of data: submissions BEFORE COVID, submissions DURING COVID, comments
# BEFORE COVID and comments DURING COVID (we had split the date into four separate dataframes earlier)

# First tri-grams for COMMENTS BEFORE COVID
tri_before_C = getNGrams(beforedf_C, 'body', 3) # the output is a list
tri_before_C[:15]


# In[15]:


# Quad-grams for COMMENTS BEFORE COVID
quad_before_C = getNGrams(beforedf_C, 'body', 4)
quad_before_C[:15]


# In[16]:


# Tri-grams for COMMENTS DURING COVID
tri_during_C = getNGrams(afterdf_C, 'body', 3)
tri_during_C[:15]


# In[17]:


# Quad-grams for COMMENTS DURING COVID
quad_during_C = getNGrams(afterdf_C, 'body', 4)
quad_during_C[:15]


# In[18]:


# Now, we repeat the same steps for the submissions
# First tri-grams for COMMENTS BEFORE COVID
tri_before_S = getNGrams(beforedf_S, 'selftext', 3) # the output is a list
tri_before_S[:15]


# In[19]:


quad_before_S = getNGrams(beforedf_S, 'selftext', 4)
quad_before_S[:15]


# In[20]:


tri_during_S = getNGrams(afterdf_S, 'selftext', 3) # the output is a list
tri_during_S[:15]


# In[21]:


quad_during_S = getNGrams(afterdf_S, 'selftext', 4)
quad_during_S[:15]


# In[8]:


# Now will do the LIWC (Linguistic Inquiry and Word Count) analysis so we need to import the LIWC files

# Will work on the stopdrinking data first and then go back to stopsmoking
path = '/Users/nayza/Downloads/'
dfS_liwc = pd.read_csv(path+'stopdrinking_submissions_LIWC.csv', low_memory=False)
dfC_liwc = pd.read_csv(path+'stopdrinking_comments_LIWC.csv', low_memory=False)


# In[10]:


# Here we would need to drop the unnecessary columns which have the NaN values
pd.set_option('display.max_columns', None)
dfS_liwc.head()


# In[11]:


dfS = dfS_liwc.dropna(subset=['selftext'])


# In[12]:


dfC_liwc.head()


# In[13]:


dfC = dfC_liwc.dropna(subset=['G'])


# In[14]:


dfS_liwc.columns.tolist()


# In[15]:


# Now, we have dropped the rows with missing entries.
# Next, we have to split the two data sets into BEFORE and DURING COVID based on 'created_utc' column.
# BUT the column names are somehow different


# In[16]:


print(type(dfS['created_utc'].tolist()[0]))
print(type(dfC['AC'].tolist()[0])) # Somehow the 'created_utc' column in comments data set got renamed to 'AC'


# In[17]:


# debugging: there is at least one row in created_utc that can't be converted to int from str

odd_vals = []
for utc in dfS['created_utc'].tolist():
    try:
        temp = int(utc)
    except ValueError:
        odd_vals.append(utc)
        
odd_vals


# In[18]:


# Odd values in created_utc have been identified. Now will go ahead and drop those rows
odd_vals_set = set(odd_vals)

for utc in dfS['created_utc'].tolist():
    if utc in odd_vals_set:
        dfS = dfS.drop(dfS.index[dfS['created_utc'] == utc])


# In[19]:


# the 'created_ufc' column of dfS ended up being a 'str' type somehow so will need to convert back to int

dfS['created_utc'] = dfS['created_utc'].apply(lambda x: int(x))
print(type(dfS['created_utc'].tolist()[0]))


# In[20]:


# Split both data sets into BEFORE and AFTER COVID
b_dfC =dfC.loc[dfC['AC'] < datetime.datetime(2020, 3, 10, 0, 0).timestamp()]
a_dfC =dfC.loc[dfC['AC'] > datetime.datetime(2020, 3, 11, 0, 0).timestamp()]

b_dfS =dfS.loc[dfS['created_utc'] < datetime.datetime(2020, 3, 10, 0, 0).timestamp()]
a_dfS =dfS.loc[dfS['created_utc'] > datetime.datetime(2020, 3, 11, 0, 0).timestamp()]


# In[21]:


len(a_dfS)


# In[22]:


# Now that we have split the data into before and during COVID, let's drop the columns that 
# are unnecessary for the t-test. 
# First let's drop all the columns to the left of 'WC'.

# For b_dfC and a_dfC
index = b_dfC.columns.tolist().index('WC')
cols_to_keep = b_dfC.columns.tolist()[index:]
b_dfC2 = b_dfC[cols_to_keep]
a_dfC2 = a_dfC[cols_to_keep]


# In[23]:


# For b_dfS and a_dfS
index_start = b_dfS.columns.tolist().index('WC')
index_end = a_dfS.columns.tolist().index('OtherP')
cols_to_keep2 = b_dfS.columns.tolist()[index_start:index_end+1]
b_dfS2 = b_dfS[cols_to_keep2]
a_dfS2 = a_dfS[cols_to_keep2]
b_dfS2.head()


# In[24]:


cols_to_keep == cols_to_keep2


# In[25]:


# Finally we can do the t-test between BEFORE data sets and DURING/AFTER data sets.

from scipy.stats import ttest_ind

# First, let's do it for Comments data set (i.e. comparing BEFORE and AFTER for Comments)
cols_with_small_p_C = set()
for col in cols_to_keep:
    vals1 = b_dfC2[col].tolist()
    vals2 = a_dfC2[col].tolist()
    p_val = round(ttest_ind(vals1, vals2)[1], 4)
    if p_val < 0.05:
        cols_with_small_p_C.add(col)
        print("BEFORE COVID: ")
        print(b_dfC2[col].describe())
        print("DURING COVID: ")
        print(a_dfC2[col].describe())
        print(p_val)
        print('-------------------------------------')
        print()
        


# In[40]:


# Latest addition: Instead of just outputting 'BEFORE COVID...' and 'AFTER COVID...' as in the last cell, we can do
# output the results into a dataframe and ultimately to be a csv file.

# Will need to do for loop over all the columns with p<0.05 and store the count, mean, std and p-value of 
# those columns.

means = []
stds = []
p_vals = []

for col in list(cols_with_small_p_C):
    means.append((round(b_dfC2[col].mean(), 4), round(a_dfC2[col].mean(), 4)))
    stds.append((round(b_dfC2[col].std(), 4), round(a_dfC2[col].std(), 4)))
    vals1 = b_dfC2[col].tolist()
    vals2 = a_dfC2[col].tolist()
    p_val = ttest_ind(vals1, vals2)[1]
    p_vals.append(p_val)
    
dict_res_C = {'Column': list(cols_with_small_p_C), 'Mean B&D COVID': means, 'STD B&D COVID': stds,
             'P-value': p_vals}
df_res_C = pd.DataFrame(dict_res_C)
                


# In[44]:


#df_res_C.to_csv(path+'ttest_res_C.csv')


# In[45]:


# Before repeating the same thing for Submissions data set, we need to convert some columns to int/float
# since they are currently str type

# Some NaN values remaining so removing them first.
b_dfS2 = b_dfS2.dropna()
a_dfS2 = a_dfS2.dropna()


# In[46]:


for col in cols_to_keep:
    if type(b_dfS2[col].tolist()[0]) == str:
        b_dfS2[col] = b_dfS2[col].apply(lambda x: float(x))
        a_dfS2[col] = a_dfS2[col].apply(lambda x: float(x))


# In[47]:


# Now, let's do it for Submissions data set (i.e. comparing BEFORE and AFTER for Submissions)
cols_with_small_p_S = set()
for col in cols_to_keep:
    vals1 = b_dfS2[col].tolist()
    vals2 = a_dfS2[col].tolist()
    p_val = round(ttest_ind(vals1, vals2)[1], 4)
    if p_val < 0.05:
        cols_with_small_p_S.add(col)
        print("BEFORE COVID: ")
        print(b_dfS2[col].describe())
        print("DURING COVID: ")
        print(a_dfS2[col].describe())
        print(p_val)
        print('-------------------------------------')
        print()


# In[51]:


# Repeating the same for SUBMISSION data with getting the results outputed to a csv file.
del means
del stds
del p_vals

means = []
stds = []
p_vals = []

for col in list(cols_with_small_p_S):
    means.append((round(b_dfS2[col].mean(), 4), round(a_dfS2[col].mean(), 4)))
    stds.append((round(b_dfS2[col].std(), 4), round(a_dfS2[col].std(), 4)))
    vals1 = b_dfS2[col].tolist()
    vals2 = a_dfS2[col].tolist()
    p_val = ttest_ind(vals1, vals2)[1]
    p_vals.append(p_val)
    
dict_res_S = {'Column': list(cols_with_small_p_S), 'Mean B&D COVID': means, 'STD B&D COVID': stds,
             'P-value': p_vals}
df_res_S = pd.DataFrame(dict_res_S)

#df_res_S.to_csv(path+'ttest_res_S.csv')


# In[52]:


df_res_S.head()


# In[91]:


# Repeating the exact same steps in getting the t-test results for stopsmoking data

dfC = pd.read_csv(path+'stopsmoking_allcomments_LIWC.csv')
dfS = pd.read_csv(path+'stopsmoking_allsubmissions_LIWC.csv')


# In[100]:


dfC['body'].head()


# In[96]:


dfC.drop(columns=['body'], inplace=True)


# In[98]:


dfC.rename(columns={'body.1': 'body'}, inplace=True)


# In[101]:


# Splitting the stopsmoking data by March 11, 2020 data

bdfC =dfC.loc[dfC['created_utc'] < datetime.datetime(2020, 3, 10, 0, 0).timestamp()]
adfC =dfC.loc[dfC['created_utc'] > datetime.datetime(2020, 3, 11, 0, 0).timestamp()]

bdfS =dfS.loc[dfS['created_utc'] < datetime.datetime(2020, 3, 10, 0, 0).timestamp()]
adfS =dfS.loc[dfS['created_utc'] > datetime.datetime(2020, 3, 11, 0, 0).timestamp()]


# In[102]:


b_dfC = bdfC[cols_to_keep]
a_dfC = adfC[cols_to_keep]

b_dfS = bdfS[cols_to_keep]
a_dfS = adfS[cols_to_keep]

b_dfC.head()


# In[112]:


# Literally repeating the same code from above. Now for Comments
del means
del stds
del p_vals

means = []
stds = []
p_vals = []
cols = []

for col in b_dfC.columns.tolist():
    vals1 = b_dfC[col].tolist()
    vals2 = a_dfC[col].tolist()
    p_val = ttest_ind(vals1, vals2)[1]
    if p_val < 0.05:
        cols.append(col)
        p_vals.append(p_val)
        means.append((round(b_dfC[col].mean(), 4), round(a_dfC[col].mean(), 4)))
        stds.append((round(b_dfC[col].std(), 4), round(a_dfC[col].std(), 4)))
    
    
dict_res_C = {'Column': cols, 'Mean B&D COVID': means, 'STD B&D COVID': stds,
             'P-value': p_vals}
df_res_C = pd.DataFrame(dict_res_C)


# In[117]:


# df_res_C.to_csv(path+'ttest_res_C_stopsmoke.csv')


# In[119]:


# Now doing the same for Submissions

del means
del stds
del p_vals
del cols

means = []
stds = []
p_vals = []
cols = []

for col in b_dfS.columns.tolist():
    vals1 = b_dfS[col].tolist()
    vals2 = a_dfS[col].tolist()
    p_val = ttest_ind(vals1, vals2)[1]
    if p_val < 0.05:
        cols.append(col)
        p_vals.append(p_val)
        means.append((round(b_dfS[col].mean(), 4), round(a_dfS[col].mean(), 4)))
        stds.append((round(b_dfS[col].std(), 4), round(a_dfS[col].std(), 4)))
    
    
dict_res_S = {'Column': cols, 'Mean B&D COVID': means, 'STD B&D COVID': stds,
             'P-value': p_vals}
df_res_S = pd.DataFrame(dict_res_S)


# In[120]:


# df_res_S.to_csv(path+'ttest_res_S_stopsmoke.csv')


# In[122]:


len(df_res_S)


# In[ ]:




