#!/usr/bin/env python
# coding: utf-8

# ## Business Problem:
# 
# The objective is to understand the factors leading to customer churn in a bank, identify key patterns, and provide actionable insights to reduce churn. By analyzing customer demographics, account activity, and transaction behavior, the bank aims to predict which customers are likely to leave and take preemptive measures to improve retention. This analysis will help the bank develop strategies to better engage customers, enhance loyalty, and ultimately minimize losses due to churn.

# ### Library Import and Data Loading

# In[1]:


#Importing necessary libraries for data manipulation, visualization, and warnings.

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
import warnings
warnings.simplefilter('ignore')


# In[2]:


#Loading the churn dataset from the specified path.

df=pd.read_csv("C:\\Users\\Masooma\\Downloads\\Churn_Modelling.csv")


# ### Data Understanding

# In[3]:


df.head()   #Displaying the first few rows of the dataset.


# In[4]:


df.info()   #Displaying the information about the dataset (data types, non-null values).


# In[5]:


#Note that there are no null values.
#row no, customer id and surname seem unnecessary rn


# In[6]:


#Specify categorical, discrete count, continuous  variables.

cat=['Geography','Gender']
disc_count=['CreditScore','Tenure','NumOfProducts','HasCrCard','IsActiveMember','Exited']
cont=['Age','Balance','EstimatedSalary']


# ### EDA

# In[7]:


df[cont].describe()  #Performing descriptive statistics on continuous variables.


# In[8]:


df[cat].value_counts()   #Displaying the value counts of categorical variables.


# In[9]:


df[disc_count].describe()   #Perform descriptive statistics on discrete count variables.


# In[10]:


df['Exited'].value_counts() 


# In[11]:


print(f"percentage of customers exited is {2037/10000*100}")
print(f"percentage of customers retained is {100-2037/10000*100}")


# In[12]:


for var in cont:
    print(f"corr b/w 'Exited' and {var} is {df['Exited'].corr(df[var])}")


# In[13]:


# cont variables age, salary and balance have low corr with output var 'Exited'


# In[14]:


for var in disc_count:
    print(f"corr b/w 'Exited' and {var} is {df['Exited'].corr(df[var])}")


# In[15]:


# all disc count vars are inversely related to exited col


# In[240]:


#Creating a dataframe with numeric variables for further analysis.

numeric=pd.DataFrame(df,columns=['CreditScore','Age','Tenure','Balance','NumOfProducts','EstimatedSalary','Exited'])


# In[242]:





# ### Univariate Analysis

# ### CreditScore

# In[23]:


df['CreditScore'].nunique()


# In[54]:


s=df['CreditScore'].value_counts().sort_index(ascending=True)  #Counting the values of 'CreditScore' and displaying in ascending order.



# In[61]:


s.iloc[368:460]


# In[101]:


#  Creating a categorical variable for 'CreditScore' based on defined bins.

cred_cat=pd.cut(df['CreditScore'],bins=[350,450,550,650,750,851],
                labels=['very low','low','medium','fair','high'],right=False)
cred_cat


# In[100]:


cred_cat.value_counts() #Displaying the value counts for 'CreditScore' categories.


# In[77]:


sns.displot(cred_cat)  #Visualizing the distribution of 'CreditScore' categories using a displot.


# In[103]:


#Calculating the number of exits in each 'CreditScore' category.

cred_exit=df[df['Exited']==1].groupby(cred_cat).size().reset_index(name='Exits') 
cred_exit


# In[107]:


sns.barplot(cred_exit,x='CreditScore',y='Exits')  #Visualize the exits by 'CreditScore' category using a barplot.


# In[108]:


#highest exits are among people with medium to fair credit score ie lying b/w 550-750


# ### Geography

# In[78]:


sns.displot(data=df,x='Geography',hue='Gender')   #Visualizing the distribution of 'Geography' with 'Gender' as hue.


# In[109]:


#Group by 'Gender', 'Geography', and 'IsActiveMember' to find exit counts.
v=df.groupby(['Gender','Geography','IsActiveMember'])['Exited'].sum().unstack()  
v


# In[ ]:


#MoSt of the exits are among females of France followed by Germany


# In[ ]:


u=df[df[.groupby(['Gender','Geography','IsActiveMember'])['Exited'].sum().unstack()
u


# In[110]:


v.plot(kind='bar',stacked=True,color='brbg')


# In[111]:


counts_df = df.groupby(['Geography', 'Gender']).size().reset_index(name='Counts')
counts_df


# ### Gender 

# In[131]:


df['Gender'].value_counts()


# In[135]:


p=(df.groupby('Gender')['Exited'].mean()*(100)).reset_index(name='Exits%')
p


# In[174]:


fig,axs=plt.subplots(1,2,figsize=(12,6))  #Visualizing the distribution of gender by count and by exit percentage.
sns.countplot(df,x=df['Gender'],ax=axs[0])
axs[0].set_title('Distribution of Gender by Count')

sns.barplot(p,x='Gender',y='Exits%',ax=axs[1])
axs[1].set_title('Distribution of Gender by %Exits')


# In[ ]:


# as compared to males more females exited


# In[118]:


total_exits=df['Exited'].sum()
total_exits


# ### Age

# In[141]:


df['Age'].unique()


# In[142]:


df['Age'].describe()


# In[166]:


df['Age'].value_counts()


# In[220]:


df['age_cat']=pd.cut(df['Age'],bins=[18,30,42,54,93],labels=['young','mature','adult','old'],include_lowest=True)
df['age_cat']    #Creating a categorical variable for 'Age' based on defined bins.


# In[ ]:


age_cat
mature    5138
adult     2012
young     1968
old        882


# In[221]:


df['age_cat'].value_counts()


# In[336]:


zero_bal_coun=df[df['Balance'] == 0].groupby('age_cat').size() 
zero_bal_coun   #Calculating the number of customers with zero balance in each age category.


# In[ ]:


#people in Age grp 'mature'(30-42 yrs) have more chance of having '0' balance account


# In[223]:


t=(df.groupby('age_cat')['Exited'].mean()*(100)).reset_index(name='Exits%')
t


# In[225]:


#Visualize the distribution of age and exits by percentage.

fig,axs=plt.subplots(1,2,figsize=(12, 6))
sns.countplot(data=df,x=df['age_cat'],ax=axs[0])
sns.barplot(data=t,x='age_cat',y='Exits%',ax=axs[1])
axs[0].set_title('Frequency Plot of Age')
axs[1].set_title('Frequency Distribution of Age by %Exits')


plt.show()


# In[ ]:





# #most Exits are from the adult age grp 42-54yrs    
# **18-30 -- young  
#   30-42 -- adult   
#   42-54 -- mature    
#   54-93 -- old**

# In[ ]:


#age_cat.dtype


# In[157]:


df['Age'].value_counts()


# ### Tenure

# In[192]:


df['Tenure'].value_counts()


# In[201]:


g=df.groupby('Tenure')['Exited'].sum().reset_index(name='Exits')  
g   #Group by 'Tenure' to calculate the number of exits.


# In[227]:


#Calculating the mean tenure for each age group.

s=df.groupby('age_cat')['Tenure'].mean().reset_index(name='Mean Tenure')
s


# In[229]:


#Visualizing the distribution of tenure and the mean tenure by age group.

fig,axs=plt.subplots(1,2,figsize=(12,6))
sns.barplot(g,x='Tenure',y='Exits',ax=axs[0])
axs[0].set_label('Distribution of Tenure with no. of Exits')

sns.barplot(s,x='age_cat',y='Mean Tenure',ax=axs[1])
axs[1].set_label('Mean Tenure wrt Age-group')


# ### Balance

# In[230]:


df['Balance'].describe()


# In[243]:


df['Balance'].value_counts().sort_index(ascending=True)


# In[233]:


df['Balance'].nunique()


# In[234]:


sns.histplot(df['Balance'],kde=True)   #Visualize the distribution of 'Balance' using a histogram.


# In[325]:


d=df[df['Balance']==0]   #Filtering data for customers with zero balance.
d


# In[329]:


#Calculating the number of zero balance accounts by geography.

zero_balance_counts = df[df['Balance'] == 0].groupby('Geography').size()
zero_balance_counts


# In[333]:


# Calculating the number of zero balance accounts by credit card ownership.


zero_bal_counts = df[df['Balance'] == 0].groupby('HasCrCard').size()
zero_bal_counts


# In[ ]:


#Germany has no of account with '0' balance
# Around 67% accounts with '0' balance are from  France
#People with a credit card have a higher chance to have'0' balance.


# In[246]:


balance_cat=pd.cut(df['Balance'],bins=[0,50000,100000,150000,200000,250900],labels=['low','medium','fair','high','very high'],include_lowest=True)
balance_cat  #creating a categorical variable for 'Balance' based on defined bins.


# In[257]:


df.groupby('age_cat')['Balance'].mean()


# In[ ]:


#there is not much difference in mean Balance amount of different age-grps


# ### NumOfProducts

# In[259]:


df['NumOfProducts'].unique()


# In[260]:


df['NumOfProducts'].value_counts()


# In[262]:


df.groupby('age_cat')['NumOfProducts'].mean()


# In[273]:


#Visualize the exits by 'NumOfProducts'.

sns.countplot(df[df['Exited']==1],x='NumOfProducts')
plt.ylabel('No. of Exits')


# In[278]:


#Most of the exits are of the people that have 1 product


# In[277]:


df.groupby('NumOfProducts')['Exited'].sum()   #total number of exits by 'NumOfProducts'.


# In[ ]:


#no.of products is invariant among age_cat


# ### HasCrCard

# In[265]:


df['HasCrCard'].unique()


# In[279]:


df.groupby('HasCrCard')['Exited'].sum()  # number of exits by 'HasCrCard'.


# In[280]:


#More exits are among the people who hold a credit card


# In[ ]:





# ### IsActiveMember

# In[281]:


df.groupby('IsActiveMember')['Exited'].sum()   #number of exits by 'IsActiveMember'.


# In[282]:


df.groupby('age_cat')['IsActiveMember'].sum()    # number of active members by age category.


# In[283]:


#Mature(30-42 yrs) people are most active


# ### EstimatedSalary

# In[284]:


df['EstimatedSalary'].nunique()


# In[ ]:


df['EstimatedSalary'].value_counts()


# In[295]:


df['EstimatedSalary'].describe()


# In[301]:


#Creating a categorical variable for 'EstimatedSalary' based on defined bins.

Sal_cat=pd.cut(df['EstimatedSalary'],bins=[10,50000,100200,149400,199995],labels=['low','medium','fair','high'],include_lowest=True)


# In[302]:


Sal_cat


# In[305]:


Sal_cat.value_counts()  


# In[306]:


#frequency distn  is uniform over different salary grps


# In[313]:


o=df.groupby(Sal_cat)['HasCrCard'].sum()  #Calculating the number of credit card holders in each salary category.
o


# In[304]:


sns.countplot(x=Sal_cat)  #Visualizing the distribution of salary categories using a countplot.


# In[314]:


df.groupby('age_cat')['EstimatedSalary'].mean()   #Calculating the mean salary for each age group.


# In[317]:


df.groupby(Sal_cat)['Exited'].mean()*100  #Calculating the percentage of exits in each salary category.


# In[ ]:


#% of exits is uniform(around 20%) for all levels of salary
#Average salary is also almost the same (around 100000 $) for each age grp

