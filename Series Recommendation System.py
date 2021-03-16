#!/usr/bin/env python
# coding: utf-8

# In[80]:


import pandas as pd
import numpy as np
series = pd.read_csv("imdb_top_1000.csv")


# In[81]:


series.head()


# In[82]:


print("Series Dataframe:",series.shape)


# In[83]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfv = TfidfVectorizer(min_df=1,  max_features=None,
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 1),
            stop_words = None)


# In[84]:


# Fitting the TF-IDF on the 'overview' text
tfv_matrix = tfv.fit_transform(series['Overview'])
print(tfv_matrix)
print(tfv_matrix.shape)


# In[85]:


series['Overview']


# In[86]:


from sklearn.metrics.pairwise import sigmoid_kernel

# Compute the sigmoid kernel
sig = sigmoid_kernel(tfv_matrix, tfv_matrix)
print(sig[0])


# In[87]:


# Reverse mapping of indices and movie titles
indices = pd.Series(series.index, index=series['Series_Title']).drop_duplicates()
print(indices)
print(indices['The Godfather'])
print(sig[999])
print(list(enumerate(sig[indices['The Godfather']])))
print(sorted(list(enumerate(sig[indices['The Godfather']])), key=lambda x: x[1], reverse=True))


# In[88]:


def give_recomendations(title, sig=sig):
    # Get the index corresponding to original_title
    idx = indices[title]

    # Get the pairwsie similarity scores
    sig_scores = list(enumerate(sig[idx]))

    # Sort the movies
    sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)

    # Scores of the 10 most similar movies
    sig_scores = sig_scores[1:11]

    # Movie indices
    series_indices = [i[0] for i in sig_scores]

    # Top 10 most similar movies
    return series['Series_Title'].iloc[series_indices]


# In[89]:


print(give_recomendations('The Dark Knight'))


# In[ ]:




