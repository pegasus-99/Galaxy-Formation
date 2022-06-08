#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv("dark_matter.csv")


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


df.corr()


# In[7]:


df_copy = df


# In[8]:


fig = plt.figure(figsize = (30, 20))
data_plotting = df.corr(method= 'pearson')
sns.heatmap(data_plotting, cmap='Reds', linecolor='black', linewidths= 2 )
plt.show()


# In[9]:


df = df.drop(df.index[6815])


# In[10]:


df = df.drop(df.index[8474])


# In[11]:


df.info()


# In[12]:


df_copy = df


# In[13]:


for column in df.loc[:,~df.columns.isin(['dispersion'])]:
    
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(121)
    ax.set_xlabel("".format(column))
    ax.set_ylabel("frequency")
    ax.set_title("Checking normality of {}".format(column))
    df[column].plot(kind='hist')


# ### As we can see mass, kinetic energy, Group_M_Crit200, Group_M_Crit500, Group_M_Mean200, Group_M_TopHat200, Group_R_Crit200, Group_R_Crit500, Group_R_Mean200, Group_R_TopHat200 need to be normalized

# In[14]:


for column in df.loc[:,~df.columns.isin(['dispersion'])]:
    
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(121)
    ax.set_ylabel("frequency")
    ax.set_title("Boxplot - {}".format(column))
    df.boxplot(column=column)


# In[15]:


x = df.loc[:, ~df.columns.isin(['dispersion'])] #Creating dependent variables dataframe
y = df.loc[:, df.columns.isin(['dispersion'])]  #Creating predictor variable dataframe


# In[16]:


x.head()


# In[17]:


y.head()


# In[18]:


x_robust = x


# There are a lot of outliers, scaling will be required

# #### Using Robust Scaler to scale this data due to the large number of outliers

# In[19]:


from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
x_robust[x_robust.columns] = scaler.fit_transform(x_robust[x_robust.columns])


# In[20]:


x_robust.head()


# In[29]:


for column in x_robust:
    
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(121)
    ax.set_ylabel("frequency")
    ax.set_title("Boxplot - {}".format(column))
    x_robust.boxplot(column=column)


# In[34]:


for column in x_robust:
    
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(121)
    ax.set_xlabel("".format(column))
    ax.set_ylabel("frequency")
    ax.set_title("Checking normality of {}".format(column))
    x_robust[column].plot(kind='hist')


# ## Visualizations

# In[35]:


from scipy.stats import norm
sns.distplot(y['dispersion'], fit=norm)
plt.show()


# In[36]:


from scipy.stats import norm
for column in x_robust:
    sns.distplot(x_robust[column], fit=norm)
    plt.show()


# In[40]:


for column in x_robust:
    # generate line of best fit
    m, b = np.polyfit(x_robust[column], y, 1)
    best_fit = m*x_robust[column] + b
    
    # plotting
    plt.subplots(figsize=(8, 5))
    plt.scatter(x_robust[column], y, color="blue")
    plt.plot(x_robust, best_fit, color="red")
    plt.title("The impact of feature at {} on Dispersion".format(column))
    plt.xlabel("Feature at {}".format(column))
    plt.ylabel("Dispersion")
    plt.show()


# ## Train test split

# In[41]:


import statsmodels.api as sm
import statsmodels.formula.api as smf
import sklearn.linear_model as lm
from sklearn import datasets, linear_model, model_selection
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import mean_squared_error, r2_score


# In[43]:


X=x_robust
Y=y

X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size = 0.20, random_state = 5)


# In[44]:


X_train.shape, X_test.shape


# In[46]:


Y.head()


# In[47]:


X.head()


# #### Multiple Linear Regression

# In[45]:


#Model statistics
#Must add constant for y-intercept
model = sm.OLS(Y_train, sm.add_constant(X_train)).fit()
Y_pred = model.predict(sm.add_constant(X_test))
print_model = model.summary()
print(print_model)


# In[48]:


from scipy import stats
sns.distplot(model.resid, fit=stats.norm);


# In[49]:


#Normal Q-Q Plot

plt.rc('figure', figsize=(10,10))
plt.style.use('ggplot')

probplot = sm.ProbPlot(model.get_influence().resid_studentized_internal, fit=True)
fig = probplot.qqplot(line='45', marker='o', color='black')
plt.title('Normal Q-Q', fontsize=20)
plt.show()


# In[50]:


#Residuals vs Fitted

# Plotting the residuals of y and pred_y
sns.residplot(Y_test,Y_pred, lowess=True, 
                          scatter_kws={'facecolors':'none', 'edgecolors':'black'}, 
                          line_kws={'color': 'blue', 'lw': 1, 'alpha': 0.8})
plt.title('Residuals vs Fitted', fontsize=20)
plt.xlabel('Fitted Values', fontsize=15)
plt.ylabel('Residuals', fontsize=15)


# In[51]:


#Scale-Location Plot

sns.regplot(model.fittedvalues, 
           np.sqrt(np.abs(model.get_influence().resid_studentized_internal)), 
            scatter=True, 
            ci=False, 
            lowess=True,
            line_kws={'color': 'blue', 'lw': 1, 'alpha': 0.8},
          scatter_kws={'facecolors':'none', 'edgecolors':'black'})

plt.title('Scale-Location', fontsize=20)
plt.xlabel('Fitted Values', fontsize=15)
plt.ylabel('$\sqrt{|Standardized Residuals|}$', fontsize=15)


# In[52]:


#Residuals vs Leverage

from numpy import sqrt

def one_line(x):
    return sqrt((1 * len(model.params) * (1 - x)) / x)

def point_five_line(x):
    return sqrt((0.5 * len(model.params) * (1 - x)) / x)
    
    
    
def show_cooks_distance_lines(tx,inc,color,label):
    plt.plot(inc,tx(inc), label=label,color=color, ls='--')
    
        

sns.regplot(model.get_influence().hat_matrix_diag, 
           model.get_influence().resid_studentized_internal, 
            scatter=True, 
            ci=False, 
            lowess=True,
            line_kws={'color': 'blue', 'lw': 1, 'alpha': 0.8},
          scatter_kws={'facecolors':'none', 'edgecolors':'black'})

show_cooks_distance_lines(one_line,
                        np.linspace(.01,.14,100),
                          'red',
                          'Cooks Distance (D=1)' )

show_cooks_distance_lines(point_five_line,
                          np.linspace(.01,.14,100),
                          'black',
                          'Cooks Distance (D=0.5)')

plt.title('Residuals vs Leverage', fontsize=20)
plt.xlabel('Leverage', fontsize=15)
plt.ylabel('Standardized Residuals', fontsize=15)
plt.legend()


# In[54]:


Y.head()


# In[57]:


Y_max = Y.max()
Y_min = Y.min()

ax = sns.scatterplot(model.fittedvalues, Y)
ax.set(ylim=(Y_min, Y_max))
ax.set(xlim=(Y_min, Y_max))
ax.set_xlabel("Predicted value of Dispersion")
ax.set_ylabel("Observed value of Dispersion")

X_ref = Y_ref = np.linspace(Y_min, Y_max, 100)
plt.plot(X_ref, Y_ref, color='red', linewidth=1)
plt.show()


# In[56]:


print(Y_min)


# In[61]:


print(model.fittedvalues)

