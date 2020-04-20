#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# timeit


# **Course Case: Apprentice Chef   
# Case Challenge Part II**
# 
# Student Name : Thanh Tam Luu  
# Cohort       : FMSBA2

# <h1>Import Packages

# In[ ]:


# Importing libraries
import pandas as pd # data science essentials
import matplotlib.pyplot as plt # essential graphical output
import seaborn as sns # enhanced graphical output
from sklearn.model_selection import train_test_split # train/test split
from sklearn.metrics import roc_auc_score            # auc score
from sklearn.tree import DecisionTreeClassifier      # classification trees
from sklearn.metrics import confusion_matrix         # confusion matrix
from sklearn.metrics import classification_report    # classification report
from sklearn.model_selection import GridSearchCV     # hyperparameter tuning
from sklearn.metrics import make_scorer              # customizable scorer

# Setting pandas print options
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# <h1>Load Data

# In[ ]:


# Specifying file name
file = 'Apprentice_Chef_Dataset.xlsx'


# Reading the file into Python
original_df = pd.read_excel(file)


# <h1>Feature Engineering

# In[ ]:


# Calculating the average price per order 
price_per_order = original_df['REVENUE']/original_df['TOTAL_MEALS_ORDERED']

price_per_order.mean()

# Creating a new column
original_df['PRICE_PER_ORDER'] = price_per_order

# Checking for the result
original_df.head()

# Working with additional information from the case study -- I want to distinguish
# customers that include wine in their meals and people that order multiple meals 
# in one order and analyze this segment. Perhaps they are more interested in the
# promotion because they like wine or they want to enjoy wine with other people.

# 10 - 23: Meal Only 
# 24 - 28: Meal with Water
# 28 - 48: Meal with Wine 
# 48 +: More meals in one order (multiple meals)

# Calculating the average meal with wine
#print((28+48)/2) 

# Output: 38 --> prices per order above this average will be considered Meal with Wine or Multiple Meals


# In[ ]:


# Setting outlier thresholds
REVENUE_hi                     = 5000  
TOTAL_MEALS_ORDERED_lo         = 25
TOTAL_MEALS_ORDERED_hi         = 215   
UNIQUE_MEALS_PURCH_hi          = 9  
CONTACTS_W_CUSTOMER_SERVICE_lo = 3 
CONTACTS_W_CUSTOMER_SERVICE_hi = 12           
AVG_TIME_PER_SITE_VISIT_hi     = 220                  
CANCELLATIONS_BEFORE_NOON_hi   = 5         
CANCELLATIONS_AFTER_NOON_hi    = 2              
MOBILE_LOGINS_lo               = 5  
MOBILE_LOGINS_hi               = 6  
PC_LOGINS_lo                   = 1
PC_LOGINS_hi                   = 2 
WEEKLY_PLAN_hi                 = 14
EARLY_DELIVERIES_hi            = 4       
LATE_DELIVERIES_hi             = 7       
AVG_PREP_VID_TIME_lo           = 60 
AVG_PREP_VID_TIME_hi           = 280     
LARGEST_ORDER_SIZE_lo          = 2
LARGEST_ORDER_SIZE_hi          = 8   
MASTER_CLASSES_ATTENDED_hi     = 2   
MEDIAN_MEAL_RATING_lo          = 2   
MEDIAN_MEAL_RATING_hi          = 4  
AVG_CLICKS_PER_VISIT_lo        = 8
AVG_CLICKS_PER_VISIT_hi        = 18      
TOTAL_PHOTOS_VIEWED_lo         = 1
TOTAL_PHOTOS_VIEWED_hi         = 300
PRICE_PER_ORDER_hi              = 38

# Developing features (columns) for outliers

# REVENUE
original_df['out_REVENUE'] = 0
condition_hi = original_df.loc[0:,'out_REVENUE'][original_df['REVENUE'] > REVENUE_hi]

original_df['out_REVENUE'].replace(to_replace = condition_hi,
                                   value      = 1,
                                   inplace    = True)

# TOTAL_MEALS_ORDERED
original_df['out_TOTAL_MEALS_ORDERED'] = 0
condition_hi = original_df.loc[0:,'out_TOTAL_MEALS_ORDERED'][original_df['TOTAL_MEALS_ORDERED'] > TOTAL_MEALS_ORDERED_hi]
condition_lo = original_df.loc[0:,'out_TOTAL_MEALS_ORDERED'][original_df['TOTAL_MEALS_ORDERED'] < TOTAL_MEALS_ORDERED_lo]

original_df['out_TOTAL_MEALS_ORDERED'].replace(to_replace = condition_hi,
                                               value      = 1,
                                               inplace    = True)

original_df['out_TOTAL_MEALS_ORDERED'].replace(to_replace = condition_lo,
                                               value      = 1,
                                               inplace    = True)

# UNIQUE_MEALS_PURCH
original_df['out_UNIQUE_MEALS_PURCH'] = 0
condition_hi = original_df.loc[0:,'out_UNIQUE_MEALS_PURCH'][original_df['UNIQUE_MEALS_PURCH'] > UNIQUE_MEALS_PURCH_hi]

original_df['out_UNIQUE_MEALS_PURCH'].replace(to_replace = condition_hi,
                                               value      = 1,
                                               inplace    = True)

# CONTACTS_W_CUSTOMER_SERVICE
original_df['out_CONTACTS_W_CUSTOMER_SERVICE'] = 0
condition_hi = original_df.loc[0:,'out_CONTACTS_W_CUSTOMER_SERVICE'][original_df['CONTACTS_W_CUSTOMER_SERVICE'] > CONTACTS_W_CUSTOMER_SERVICE_hi]
condition_lo = original_df.loc[0:,'out_CONTACTS_W_CUSTOMER_SERVICE'][original_df['CONTACTS_W_CUSTOMER_SERVICE'] < CONTACTS_W_CUSTOMER_SERVICE_lo]

original_df['out_CONTACTS_W_CUSTOMER_SERVICE'].replace(to_replace = condition_hi,
                                               value      = 1,
                                               inplace    = True)

original_df['out_CONTACTS_W_CUSTOMER_SERVICE'].replace(to_replace = condition_lo,
                                               value      = 1,
                                               inplace    = True)

# AVG_TIME_PER_SITE_VISIT
original_df['out_AVG_TIME_PER_SITE_VISIT'] = 0
condition_hi = original_df.loc[0:,'out_AVG_TIME_PER_SITE_VISIT'][original_df['AVG_TIME_PER_SITE_VISIT'] > AVG_TIME_PER_SITE_VISIT_hi]

original_df['out_AVG_TIME_PER_SITE_VISIT'].replace(to_replace = condition_hi,
                                                   value      = 1,
                                                   inplace    = True)

# CANCELLATIONS_BEFORE_NOON
original_df['out_CANCELLATIONS_BEFORE_NOON'] = 0
condition_hi = original_df.loc[0:,'out_CANCELLATIONS_BEFORE_NOON'][original_df['CANCELLATIONS_BEFORE_NOON'] > CANCELLATIONS_BEFORE_NOON_hi]

original_df['out_CANCELLATIONS_BEFORE_NOON'].replace(to_replace = condition_hi,
                                                     value      = 1,
                                                     inplace    = True)

# CANCELLATIONS_AFTER_NOON
original_df['out_CANCELLATIONS_AFTER_NOON'] = 0
condition_hi = original_df.loc[0:,'out_CANCELLATIONS_AFTER_NOON'][original_df['CANCELLATIONS_AFTER_NOON'] > CANCELLATIONS_AFTER_NOON_hi]

original_df['out_CANCELLATIONS_AFTER_NOON'].replace(to_replace = condition_hi,
                                                    value      = 1,
                                                    inplace    = True)

# MOBILE_LOGINS
original_df['out_MOBILE_LOGINS'] = 0
condition_hi = original_df.loc[0:,'out_MOBILE_LOGINS'][original_df['MOBILE_LOGINS'] > MOBILE_LOGINS_hi]
condition_lo = original_df.loc[0:,'out_MOBILE_LOGINS'][original_df['MOBILE_LOGINS'] < MOBILE_LOGINS_lo]

original_df['out_MOBILE_LOGINS'].replace(to_replace = condition_hi,
                                         value      = 1,
                                         inplace    = True)

original_df['out_MOBILE_LOGINS'].replace(to_replace = condition_lo,
                                         value      = 1,
                                         inplace    = True)

# PC_LOGINS
original_df['out_PC_LOGINS'] = 0
condition_hi = original_df.loc[0:,'out_PC_LOGINS'][original_df['PC_LOGINS'] > PC_LOGINS_hi]
condition_lo = original_df.loc[0:,'out_PC_LOGINS'][original_df['PC_LOGINS'] < PC_LOGINS_lo]

original_df['out_PC_LOGINS'].replace(to_replace = condition_hi,
                                     value      = 1,
                                     inplace    = True)

original_df['out_PC_LOGINS'].replace(to_replace = condition_lo,
                                     value      = 1,
                                     inplace    = True)

# WEEKLY_PLAN
original_df['out_WEEKLY_PLAN'] = 0
condition_hi = original_df.loc[0:,'out_WEEKLY_PLAN'][original_df['WEEKLY_PLAN'] > WEEKLY_PLAN_hi]

original_df['out_WEEKLY_PLAN'].replace(to_replace = condition_hi,
                                       value      = 1,
                                       inplace    = True)

# EARLY_DELIVERIES 
original_df['out_EARLY_DELIVERIES'] = 0
condition_hi = original_df.loc[0:,'out_EARLY_DELIVERIES'][original_df['EARLY_DELIVERIES'] > EARLY_DELIVERIES_hi]

original_df['out_EARLY_DELIVERIES'].replace(to_replace = condition_hi,
                                            value      = 1,
                                            inplace    = True)

# LATE_DELIVERIES
original_df['out_LATE_DELIVERIES'] = 0
condition_hi = original_df.loc[0:,'out_LATE_DELIVERIES'][original_df['LATE_DELIVERIES'] > LATE_DELIVERIES_hi]

original_df['out_LATE_DELIVERIES'].replace(to_replace = condition_hi,
                                           value      = 1,
                                           inplace    = True)

# AVG_PREP_VID_TIME
original_df['out_AVG_PREP_VID_TIME'] = 0
condition_hi = original_df.loc[0:,'out_AVG_PREP_VID_TIME'][original_df['AVG_PREP_VID_TIME'] > AVG_PREP_VID_TIME_hi]
condition_lo = original_df.loc[0:,'out_AVG_PREP_VID_TIME'][original_df['AVG_PREP_VID_TIME'] < AVG_PREP_VID_TIME_lo]

original_df['out_AVG_PREP_VID_TIME'].replace(to_replace = condition_hi,
                                             value      = 1,
                                             inplace    = True)

original_df['out_AVG_PREP_VID_TIME'].replace(to_replace = condition_lo,
                                             value      = 1,
                                             inplace    = True)

# LARGEST_ORDER_SIZE
original_df['out_LARGEST_ORDER_SIZE'] = 0
condition_hi = original_df.loc[0:,'out_LARGEST_ORDER_SIZE'][original_df['LARGEST_ORDER_SIZE'] > LARGEST_ORDER_SIZE_hi]
condition_lo = original_df.loc[0:,'out_LARGEST_ORDER_SIZE'][original_df['LARGEST_ORDER_SIZE'] < LARGEST_ORDER_SIZE_lo]

original_df['out_LARGEST_ORDER_SIZE'].replace(to_replace = condition_hi,
                                              value      = 1,
                                              inplace    = True)

original_df['out_LARGEST_ORDER_SIZE'].replace(to_replace = condition_lo,
                                              value      = 1,
                                              inplace    = True)

# MASTER_CLASSES_ATTENDED
original_df['out_MASTER_CLASSES_ATTENDED'] = 0
condition_hi = original_df.loc[0:,'out_MASTER_CLASSES_ATTENDED'][original_df['MASTER_CLASSES_ATTENDED'] > MASTER_CLASSES_ATTENDED_hi]

original_df['out_MASTER_CLASSES_ATTENDED'].replace(to_replace = condition_hi,
                                                   value      = 1,
                                                   inplace    = True)

# MEDIAN_MEAL_RATING
original_df['out_MEDIAN_MEAL_RATING'] = 0
condition_hi = original_df.loc[0:,'out_MEDIAN_MEAL_RATING'][original_df['MEDIAN_MEAL_RATING'] > MEDIAN_MEAL_RATING_hi]
condition_lo = original_df.loc[0:,'out_MEDIAN_MEAL_RATING'][original_df['MEDIAN_MEAL_RATING'] < MEDIAN_MEAL_RATING_lo]

original_df['out_MEDIAN_MEAL_RATING'].replace(to_replace = condition_hi,
                                              value      = 1,
                                              inplace    = True)

original_df['out_MEDIAN_MEAL_RATING'].replace(to_replace = condition_lo,
                                              value      = 1,
                                              inplace    = True)

# AVG_CLICKS_PER_VISIT
original_df['out_AVG_CLICKS_PER_VISIT'] = 0
condition_hi = original_df.loc[0:,'out_AVG_CLICKS_PER_VISIT'][original_df['AVG_CLICKS_PER_VISIT'] > AVG_CLICKS_PER_VISIT_hi]
condition_lo = original_df.loc[0:,'out_AVG_CLICKS_PER_VISIT'][original_df['AVG_CLICKS_PER_VISIT'] < AVG_CLICKS_PER_VISIT_lo]

original_df['out_AVG_CLICKS_PER_VISIT'].replace(to_replace = condition_hi,
                                                value      = 1,
                                                inplace    = True)

original_df['out_AVG_CLICKS_PER_VISIT'].replace(to_replace = condition_lo,
                                                value      = 1,
                                                inplace    = True)

# TOTAL_PHOTOS_VIEWED
original_df['out_TOTAL_PHOTOS_VIEWED'] = 0
condition_hi = original_df.loc[0:,'out_TOTAL_PHOTOS_VIEWED'][original_df['TOTAL_PHOTOS_VIEWED'] > TOTAL_PHOTOS_VIEWED_hi]
condition_lo = original_df.loc[0:,'out_TOTAL_PHOTOS_VIEWED'][original_df['TOTAL_PHOTOS_VIEWED'] < TOTAL_PHOTOS_VIEWED_lo]

original_df['out_TOTAL_PHOTOS_VIEWED'].replace(to_replace = condition_hi,
                                               value      = 1,
                                               inplace    = True)

original_df['out_TOTAL_PHOTOS_VIEWED'].replace(to_replace = condition_lo,
                                               value      = 1,
                                               inplace    = True)

# PRICE_PER_ORDER
original_df['out_PRICE_PER_ORDER'] = 0
condition_hi = original_df.loc[0:,'out_PRICE_PER_ORDER'][original_df['PRICE_PER_ORDER'] > PRICE_PER_ORDER_hi]

original_df['out_PRICE_PER_ORDER'].replace(to_replace = condition_hi,
                                               value      = 1,
                                               inplace    = True)


# In[ ]:


# Working with Email Addresses

# Step 1: Splitting personal emails 

# Placeholder list
placeholder_lst = []  

# Looping over each email address
for index, col in original_df.iterrows(): 
    
    # Splitting email domain at '@'
    split_email = original_df.loc[index, 'EMAIL'].split(sep = '@') 
    
    # Appending placeholder_lst with the results
    placeholder_lst.append(split_email)
    

# Converting placeholder_lst into a DataFrame 
email_df = pd.DataFrame(placeholder_lst)


# Displaying the results
#email_df


# In[ ]:


# Step 2: Concatenating with original DataFrame

# Renaming column to concatenate
email_df.columns = ['NAME' , 'EMAIL_DOMAIN']


# Concatenating personal_email_domain with friends DataFrame 
original_df = pd.concat([original_df, email_df.loc[:, 'EMAIL_DOMAIN']], 
                   axis = 1)


# Printing value counts of personal_email_domain
#original_df.loc[: ,'EMAIL_DOMAIN'].value_counts()


# In[ ]:


# Email domain types
professional_email_domains = ['@mmm.com', '@amex.com', '@apple.com', 
                              '@boeing.com', '@caterpillar.com', '@chevron.com',
                              '@cisco.com', '@cocacola.com', '@disney.com',
                              '@dupont.com', '@exxon.com', '@ge.org',
                              '@goldmansacs.com', '@homedepot.com', '@ibm.com',
                              '@intel.com', '@jnj.com', '@jpmorgan.com',
                              '@mcdonalds.com', '@merck.com', '@microsoft.com',
                              '@nike.com', '@pfizer.com', '@pg.com',
                              '@travelers.com', '@unitedtech.com', '@unitedhealth.com',
                              '@verizon.com', '@visa.com', '@walmart.com']
personal_email_domains     = ['@gmail.com', '@yahoo.com', '@protonmail.com']
junk_email_domains         = ['@me.com', '@aol.com', '@hotmail.com',
                              '@live.com', '@msn.com', '@passport.com']


# Placeholder list
placeholder_lst = []  # good practice, overwriting the one above, everything in the workspace takes up place, we are renaming this - saves place on computer


# Looping to group observations by domain type
for domain in original_df['EMAIL_DOMAIN']:
        if '@' + domain in professional_email_domains: # has to be an exact match, that's why '@'
            placeholder_lst.append('professional')
            
        elif '@' + domain in personal_email_domains:
            placeholder_lst.append('personal')
            
        elif '@' + domain in junk_email_domains:
            placeholder_lst.append('junk')
            
        else:
            print('Unknown')


# Concatenating with original DataFrame
original_df['DOMAIN_GROUP'] = pd.Series(placeholder_lst)


# Checking results
#original_df['DOMAIN_GROUP'].value_counts()


# In[ ]:


# One hot encoding categorical variables
one_hot_DOMAIN_GROUP      = pd.get_dummies(original_df['DOMAIN_GROUP'])

# Dropping categorical variables after they've been encoded
original_df = original_df.drop('DOMAIN_GROUP', axis = 1)

# Joining codings together
original_df = original_df.join([one_hot_DOMAIN_GROUP])

# Saving new columns
new_columns = original_df.columns


# <h1>Train/Test Split

# In[ ]:


# Declaring explanatory variables
original_df_data = original_df.drop('CROSS_SELL_SUCCESS', axis = 1)

# Declaring response variable
original_df_target = original_df.loc[ : , 'CROSS_SELL_SUCCESS']

# Train-test split with the full model
X_train, X_test, y_train, y_test = train_test_split(
            original_df_data,
            original_df_target,
            test_size = 0.25,
            random_state = 222,
            stratify = original_df_target)


# Merging training data for statsmodels
original_df_train = pd.concat([X_train, y_train], axis = 1)


# In[ ]:


# Explanatory set of variables

x_variables = ['MOBILE_NUMBER', 'CANCELLATIONS_BEFORE_NOON',
               'TASTES_AND_PREFERENCES', 'PC_LOGINS',
               'FOLLOWED_RECOMMENDATIONS_PCT', 'out_PRICE_PER_ORDER',
               'personal', 'professional', 'REFRIGERATED_LOCKER',
               'CANCELLATIONS_AFTER_NOON', 'MOBILE_LOGINS', 'AVG_PREP_VID_TIME']


# In[ ]:


# Train/test split with the fit model
original_df_data   =  original_df.loc[ : , x_variables]
original_df_target =  original_df.loc[ : , 'CROSS_SELL_SUCCESS']


# Train-test split 
X_train, X_test, y_train, y_test = train_test_split(
            original_df_data,
            original_df_target,
            random_state = 222,
            test_size    = 0.25,
            stratify     = original_df_target)


# <h1>Final Model

# In[ ]:


################################################################################
# Tuned Decision Tree Classifier using GridSearchCV      
################################################################################

# declaring a hyperparameter space
depth_space          = pd.np.arange(1, 10, 1)
samples_leaf_space   = pd.np.arange(1,10,1)
criterion_space      = ['gini']


# creating a hyperparameter grid
param_grid = {'criterion' : criterion_space,
              'max_depth' : depth_space,
              'min_samples_leaf' : samples_leaf_space
              }


# INSTANTIATING the model object without hyperparameters
tree_tuned = DecisionTreeClassifier(random_state = 222)


# GridSearchCV object (due to cross-validation)
tree_tuned_cv = GridSearchCV(estimator  = tree_tuned,
                           param_grid = param_grid,
                           cv         = 3,
                           scoring    = make_scorer(roc_auc_score,
                                                    needs_threshold = False))


# FITTING to the FULL DATASET
tree_tuned_cv.fit(original_df_data, original_df_target)


# printing the optimal parameters and best score
print("Tuned Parameters  :", tree_tuned_cv.best_params_)
print("Tuned CV AUC      :", tree_tuned_cv.best_score_.round(4))


# In[ ]:


# Calling for the best estimator
tree_tuned_cv.best_estimator_


# In[ ]:


# building a model based on hyperparameter tuning results

# INSTANTIATING a logistic regression model with tuned values
tree_tuned = tree_tuned_cv.best_estimator_


# FIT step is not needed


# PREDICTING based on the testing set
tree_tuned_pred = tree_tuned.predict(X_test)


# In[ ]:


################################################################################
# Confusion Matrix       
################################################################################

# Creating a confusion matrix
print(confusion_matrix(y_true = y_test,
                       y_pred = tree_tuned_pred))


# In[ ]:


# Calculating correct and incorrect predictions

correct_predictions = 121 + 291
incorrect_predictions = 40 + 35
accuracy = (121 + 291) / 487
misclassification_rate = (40 + 35) / 487

print("Correct predictions:", correct_predictions)
print("Incorrect predictions:", incorrect_predictions)
print("Accuracy:", accuracy)
print("Misclassification Rate:", misclassification_rate)  

# Printing the classification report
print(classification_report(y_test, tree_tuned_pred))


# **Findings:**  
# 
# The result is telling us that we have 412 (85%) correct predictions and 75 (15%) incorrect predictions.   
# 
# The classifier has an Accuracy of 85% -- how often the classifier is correct.  
# The classifier has a True Positive Rate/Sensitivity/Recall of 88% -- when it's actually yes, how often it predicts yes.  
# The classifier has a True Negative Rate/Specificity of 78% -- when it's actually no, how often it predicts no.  
# The classifier has a Precision of 89% -- when it predicts yes, how often it is correct.  

# <h1>Final Model Score

# In[ ]:


# SCORING the results
print('Training ACCURACY:', tree_tuned.score(X_train, y_train).round(4))
print('Testing  ACCURACY:', tree_tuned.score(X_test, y_test).round(4))
print('AUC Score        :', roc_auc_score(y_true  = y_test,
                                          y_score = tree_tuned_pred).round(4))

# Saving scoring data for future use
train_score = tree_tuned.score(X_train, y_train).round(4)
test_score = tree_tuned.score(X_test, y_test).round(4)
auc_score = roc_auc_score(y_true  = y_test,
                          y_score = tree_tuned_pred).round(4)

