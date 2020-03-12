
# coding: utf-8

# In[1]:

get_ipython().system('pip install lifetimes')


# In[3]:

import pandas as pd 
df = pd.read_excel('OnlineRetailDataset.xlsx')


# In[1]:

import pandas as pd


# In[4]:

from matplotlib import pyplot as plt
df.head()


# In[5]:

from lifetimes.utils import summary_data_from_transaction_data


# In[37]:

summary = summary_data_from_transaction_data(df, 'CustomerID', 'InvoiceDate', observation_period_end='2011-12-30')

print(summary.head())


# In[43]:

from lifetimes import BetaGeoFitter

# similar API to scikit-learn and lifelines.
bgf = BetaGeoFitter(penalizer_coef=28)
bgf.fit(summary['frequency'], summary['recency'], summary['T'])
print(bgf)


# In[44]:

from lifetimes.plotting import plot_frequency_recency_matrix

plot_frequency_recency_matrix(bgf)
plt.show()


# In[45]:

from lifetimes.plotting import plot_probability_alive_matrix

plot_probability_alive_matrix(bgf)
plt.show()


# In[48]:

t = 5
summary['predicted_purchases'] = bgf.conditional_expected_number_of_purchases_up_to_time(t, summary['frequency'], summary['recency'], summary['T'])
summary.sort_values(by='predicted_purchases')


# In[ ]:




# In[ ]:



