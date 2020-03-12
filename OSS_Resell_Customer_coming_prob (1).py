
# coding: utf-8

# In[1]:

get_ipython().system('pip install lifetimes')


# In[3]:

##This is the online retail dataset file which is available online

import pandas as pd 
df = pd.read_excel('OnlineRetailDataset.xlsx')


# In[4]:

from matplotlib import pyplot as plt

##The attributes which are our interest are the Stockcode which is the unique number of the product, Invoice date saying when the product was bought and Customer ID which is the unique ID of the customer. 

df.head()


# In[5]:

from lifetimes.utils import summary_data_from_transaction_data

#This transforms a DataFrame of transaction data of the form to a DataFrame of the form:  customer_id, frequency, recency, T [, monetary_value]


# In[37]:

summary = summary_data_from_transaction_data(df, 'CustomerID', 'InvoiceDate', observation_period_end='2011-12-30')
##Observation period end should be the end date of the last transaction data 

print(summary.head())


# In[43]:

from lifetimes import BetaGeoFitter

# similar API to scikit-learn and lifelines.
# change the penalizer coefficient according to the length of the dataset and max value of frequency
bgf = BetaGeoFitter(penalizer_coef=28)
bgf.fit(summary['frequency'], summary['recency'], summary['T'])
print(bgf)


# In[44]:

#The best customers are those in right bottom and the cold customers are those right top customers. 

from lifetimes.plotting import plot_frequency_recency_matrix

plot_frequency_recency_matrix(bgf)
plt.show()


# In[45]:

from lifetimes.plotting import plot_probability_alive_matrix

plot_probability_alive_matrix(bgf)
plt.show()


# In[48]:

t = 5 #this says the number of days for which the prediction is done
summary['predicted_purchases'] = bgf.conditional_expected_number_of_purchases_up_to_time(t, summary['frequency'], summary['recency'], summary['T'])
summary.sort_values(by='predicted_purchases')


# In[ ]:




# In[ ]:



