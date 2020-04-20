{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Course Case: Apprentice Chef   \n",
    "Case Challenge Part II**\n",
    "\n",
    "Student Name : Thanh Tam Luu  \n",
    "Cohort       : FMSBA2\n",
    "\n",
    "**Business Problem**: Apprentice Chef, Inc. has launched a cross-selling promotion \"Halfway There\" for subscribers in order to diversify the   revenue stream and create a competitive advantage in the fierce market. The task is to analyze the data, develop insights and build a model to predict the likelihood of responding the cross-sell offer. "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h1>Packages Import"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/Users/luuttami/anaconda3/lib/python3.7/site-packages/sklearn/externals/six.py:31: DeprecationWarning: The module is deprecated in version 0.21 and will be removed in version 0.23 since we've dropped support for Python 2.7. Please rely on the official version of six (https://pypi.org/project/six/).\n",
      "  \"(https://pypi.org/project/six/).\", DeprecationWarning)\n"
     ]
    }
   ],
   "source": [
    "# Importing libraries\n",
    "import pandas as pd # data science essentials\n",
    "import matplotlib.pyplot as plt # essential graphical output\n",
    "import seaborn as sns # enhanced graphical output\n",
    "import statsmodels.formula.api as smf # regression modeling\n",
    "from sklearn.model_selection import train_test_split # train/test split\n",
    "from sklearn.linear_model import LogisticRegression  # logistic regression\n",
    "from sklearn.metrics import confusion_matrix         # confusion matrix\n",
    "from sklearn.metrics import roc_auc_score            # auc score\n",
    "from sklearn.neighbors import KNeighborsClassifier   # KNN for classification\n",
    "from sklearn.neighbors import KNeighborsRegressor    # KNN for regression\n",
    "from sklearn.preprocessing import StandardScaler     # standard scaler\n",
    "from sklearn.tree import DecisionTreeClassifier      # classification trees\n",
    "from sklearn.tree import export_graphviz             # exports graphics\n",
    "from sklearn.externals.six import StringIO           # saves objects in memory\n",
    "from sklearn.metrics import classification_report    # classification report\n",
    "from IPython.display import Image                    # displays on frontend\n",
    "import pydotplus                                     # interprets dot objects\n",
    "\n",
    "# Setting pandas print options\n",
    "pd.set_option('display.max_rows', 500)\n",
    "pd.set_option('display.max_columns', 500)\n",
    "pd.set_option('display.width', 1000)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h1>Loading Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Specifying file name\n",
    "file = 'Apprentice_Chef_Dataset.xlsx'\n",
    "\n",
    "\n",
    "# Reading the file into Python\n",
    "original_df = pd.read_excel(file)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h1>Fundamental Dataset Exploration"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Displaying all column names\n",
    "#print(original_df.columns)\n",
    "\n",
    "# Displaying the first rows of the DataFrame\n",
    "#original_df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 1946 entries, 0 to 1945\n",
      "Data columns (total 29 columns):\n",
      "REVENUE                         1946 non-null float64\n",
      "CROSS_SELL_SUCCESS              1946 non-null int64\n",
      "NAME                            1946 non-null object\n",
      "EMAIL                           1946 non-null object\n",
      "FIRST_NAME                      1946 non-null object\n",
      "FAMILY_NAME                     1899 non-null object\n",
      "TOTAL_MEALS_ORDERED             1946 non-null int64\n",
      "UNIQUE_MEALS_PURCH              1946 non-null int64\n",
      "CONTACTS_W_CUSTOMER_SERVICE     1946 non-null int64\n",
      "PRODUCT_CATEGORIES_VIEWED       1946 non-null int64\n",
      "AVG_TIME_PER_SITE_VISIT         1946 non-null float64\n",
      "MOBILE_NUMBER                   1946 non-null int64\n",
      "CANCELLATIONS_BEFORE_NOON       1946 non-null int64\n",
      "CANCELLATIONS_AFTER_NOON        1946 non-null int64\n",
      "TASTES_AND_PREFERENCES          1946 non-null int64\n",
      "MOBILE_LOGINS                   1946 non-null int64\n",
      "PC_LOGINS                       1946 non-null int64\n",
      "WEEKLY_PLAN                     1946 non-null int64\n",
      "EARLY_DELIVERIES                1946 non-null int64\n",
      "LATE_DELIVERIES                 1946 non-null int64\n",
      "PACKAGE_LOCKER                  1946 non-null int64\n",
      "REFRIGERATED_LOCKER             1946 non-null int64\n",
      "FOLLOWED_RECOMMENDATIONS_PCT    1946 non-null int64\n",
      "AVG_PREP_VID_TIME               1946 non-null float64\n",
      "LARGEST_ORDER_SIZE              1946 non-null int64\n",
      "MASTER_CLASSES_ATTENDED         1946 non-null int64\n",
      "MEDIAN_MEAL_RATING              1946 non-null int64\n",
      "AVG_CLICKS_PER_VISIT            1946 non-null int64\n",
      "TOTAL_PHOTOS_VIEWED             1946 non-null int64\n",
      "dtypes: float64(3), int64(22), object(4)\n",
      "memory usage: 441.0+ KB\n"
     ]
    }
   ],
   "source": [
    "# Getting a concise summary\n",
    "original_df.info()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Findings**:  \n",
    "\n",
    "We have a lot of missing values in the FAMILY_NAME variable."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>REVENUE</th>\n",
       "      <th>CROSS_SELL_SUCCESS</th>\n",
       "      <th>TOTAL_MEALS_ORDERED</th>\n",
       "      <th>UNIQUE_MEALS_PURCH</th>\n",
       "      <th>CONTACTS_W_CUSTOMER_SERVICE</th>\n",
       "      <th>PRODUCT_CATEGORIES_VIEWED</th>\n",
       "      <th>AVG_TIME_PER_SITE_VISIT</th>\n",
       "      <th>MOBILE_NUMBER</th>\n",
       "      <th>CANCELLATIONS_BEFORE_NOON</th>\n",
       "      <th>CANCELLATIONS_AFTER_NOON</th>\n",
       "      <th>TASTES_AND_PREFERENCES</th>\n",
       "      <th>MOBILE_LOGINS</th>\n",
       "      <th>PC_LOGINS</th>\n",
       "      <th>WEEKLY_PLAN</th>\n",
       "      <th>EARLY_DELIVERIES</th>\n",
       "      <th>LATE_DELIVERIES</th>\n",
       "      <th>PACKAGE_LOCKER</th>\n",
       "      <th>REFRIGERATED_LOCKER</th>\n",
       "      <th>FOLLOWED_RECOMMENDATIONS_PCT</th>\n",
       "      <th>AVG_PREP_VID_TIME</th>\n",
       "      <th>LARGEST_ORDER_SIZE</th>\n",
       "      <th>MASTER_CLASSES_ATTENDED</th>\n",
       "      <th>MEDIAN_MEAL_RATING</th>\n",
       "      <th>AVG_CLICKS_PER_VISIT</th>\n",
       "      <th>TOTAL_PHOTOS_VIEWED</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <td>0.2</td>\n",
       "      <td>1285.00</td>\n",
       "      <td>0.0</td>\n",
       "      <td>35.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>5.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>61.58</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>5.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>10.0</td>\n",
       "      <td>108.6</td>\n",
       "      <td>3.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>12.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>0.4</td>\n",
       "      <td>1558.00</td>\n",
       "      <td>1.0</td>\n",
       "      <td>51.0</td>\n",
       "      <td>4.0</td>\n",
       "      <td>6.0</td>\n",
       "      <td>4.0</td>\n",
       "      <td>86.51</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>5.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>5.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>20.0</td>\n",
       "      <td>134.4</td>\n",
       "      <td>4.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>13.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>0.6</td>\n",
       "      <td>1910.00</td>\n",
       "      <td>1.0</td>\n",
       "      <td>71.0</td>\n",
       "      <td>6.0</td>\n",
       "      <td>7.0</td>\n",
       "      <td>6.0</td>\n",
       "      <td>102.62</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>6.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>10.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>40.0</td>\n",
       "      <td>156.7</td>\n",
       "      <td>5.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>14.0</td>\n",
       "      <td>28.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>0.8</td>\n",
       "      <td>2895.00</td>\n",
       "      <td>1.0</td>\n",
       "      <td>106.0</td>\n",
       "      <td>7.0</td>\n",
       "      <td>9.0</td>\n",
       "      <td>9.0</td>\n",
       "      <td>123.80</td>\n",
       "      <td>1.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>6.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>16.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>5.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>70.0</td>\n",
       "      <td>183.8</td>\n",
       "      <td>6.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>16.0</td>\n",
       "      <td>210.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>1.0</td>\n",
       "      <td>8793.75</td>\n",
       "      <td>1.0</td>\n",
       "      <td>493.0</td>\n",
       "      <td>19.0</td>\n",
       "      <td>18.0</td>\n",
       "      <td>10.0</td>\n",
       "      <td>1645.60</td>\n",
       "      <td>1.0</td>\n",
       "      <td>13.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>7.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>52.0</td>\n",
       "      <td>9.0</td>\n",
       "      <td>19.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>90.0</td>\n",
       "      <td>564.2</td>\n",
       "      <td>11.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>5.0</td>\n",
       "      <td>19.0</td>\n",
       "      <td>1600.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "     REVENUE  CROSS_SELL_SUCCESS  TOTAL_MEALS_ORDERED  UNIQUE_MEALS_PURCH  CONTACTS_W_CUSTOMER_SERVICE  PRODUCT_CATEGORIES_VIEWED  AVG_TIME_PER_SITE_VISIT  MOBILE_NUMBER  CANCELLATIONS_BEFORE_NOON  CANCELLATIONS_AFTER_NOON  TASTES_AND_PREFERENCES  MOBILE_LOGINS  PC_LOGINS  WEEKLY_PLAN  EARLY_DELIVERIES  LATE_DELIVERIES  PACKAGE_LOCKER  REFRIGERATED_LOCKER  FOLLOWED_RECOMMENDATIONS_PCT  AVG_PREP_VID_TIME  LARGEST_ORDER_SIZE  MASTER_CLASSES_ATTENDED  MEDIAN_MEAL_RATING  AVG_CLICKS_PER_VISIT  TOTAL_PHOTOS_VIEWED\n",
       "0.2  1285.00                 0.0                 35.0                 3.0                          5.0                        2.0                    61.58            1.0                        0.0                       0.0                     0.0            5.0        1.0          0.0               0.0              1.0             0.0                  0.0                          10.0              108.6                 3.0                      0.0                 2.0                  12.0                  0.0\n",
       "0.4  1558.00                 1.0                 51.0                 4.0                          6.0                        4.0                    86.51            1.0                        1.0                       0.0                     1.0            5.0        1.0          5.0               0.0              2.0             0.0                  0.0                          20.0              134.4                 4.0                      0.0                 3.0                  13.0                  0.0\n",
       "0.6  1910.00                 1.0                 71.0                 6.0                          7.0                        6.0                   102.62            1.0                        1.0                       0.0                     1.0            6.0        2.0         10.0               1.0              3.0             0.0                  0.0                          40.0              156.7                 5.0                      1.0                 3.0                  14.0                 28.0\n",
       "0.8  2895.00                 1.0                106.0                 7.0                          9.0                        9.0                   123.80            1.0                        2.0                       0.0                     1.0            6.0        2.0         16.0               3.0              5.0             1.0                  0.0                          70.0              183.8                 6.0                      1.0                 3.0                  16.0                210.0\n",
       "1.0  8793.75                 1.0                493.0                19.0                         18.0                       10.0                  1645.60            1.0                       13.0                       3.0                     1.0            7.0        3.0         52.0               9.0             19.0             1.0                  1.0                          90.0              564.2                11.0                      3.0                 5.0                  19.0               1600.0"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Running descriptive statistics\n",
    "original_df.describe().round(2)\n",
    "\n",
    "original_df.loc[:, :].quantile([0.20,\n",
    "                                0.40,\n",
    "                                0.60,\n",
    "                                0.80,\n",
    "                                1.00])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>REVENUE</th>\n",
       "      <th>CROSS_SELL_SUCCESS</th>\n",
       "      <th>NAME</th>\n",
       "      <th>EMAIL</th>\n",
       "      <th>FIRST_NAME</th>\n",
       "      <th>FAMILY_NAME</th>\n",
       "      <th>TOTAL_MEALS_ORDERED</th>\n",
       "      <th>UNIQUE_MEALS_PURCH</th>\n",
       "      <th>CONTACTS_W_CUSTOMER_SERVICE</th>\n",
       "      <th>PRODUCT_CATEGORIES_VIEWED</th>\n",
       "      <th>AVG_TIME_PER_SITE_VISIT</th>\n",
       "      <th>MOBILE_NUMBER</th>\n",
       "      <th>CANCELLATIONS_BEFORE_NOON</th>\n",
       "      <th>CANCELLATIONS_AFTER_NOON</th>\n",
       "      <th>TASTES_AND_PREFERENCES</th>\n",
       "      <th>MOBILE_LOGINS</th>\n",
       "      <th>PC_LOGINS</th>\n",
       "      <th>WEEKLY_PLAN</th>\n",
       "      <th>EARLY_DELIVERIES</th>\n",
       "      <th>LATE_DELIVERIES</th>\n",
       "      <th>PACKAGE_LOCKER</th>\n",
       "      <th>REFRIGERATED_LOCKER</th>\n",
       "      <th>FOLLOWED_RECOMMENDATIONS_PCT</th>\n",
       "      <th>AVG_PREP_VID_TIME</th>\n",
       "      <th>LARGEST_ORDER_SIZE</th>\n",
       "      <th>MASTER_CLASSES_ATTENDED</th>\n",
       "      <th>MEDIAN_MEAL_RATING</th>\n",
       "      <th>AVG_CLICKS_PER_VISIT</th>\n",
       "      <th>TOTAL_PHOTOS_VIEWED</th>\n",
       "      <th>PRICE_PER_ORDER</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <td>0</td>\n",
       "      <td>1880.0</td>\n",
       "      <td>1</td>\n",
       "      <td>Addam Osgrey</td>\n",
       "      <td>addam.osgrey@passport.com</td>\n",
       "      <td>Addam</td>\n",
       "      <td>Osgrey</td>\n",
       "      <td>118</td>\n",
       "      <td>4</td>\n",
       "      <td>7</td>\n",
       "      <td>5</td>\n",
       "      <td>86.00</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>6</td>\n",
       "      <td>2</td>\n",
       "      <td>8</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>90</td>\n",
       "      <td>165.8</td>\n",
       "      <td>6</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>11</td>\n",
       "      <td>0</td>\n",
       "      <td>15.932203</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>1</td>\n",
       "      <td>1495.0</td>\n",
       "      <td>1</td>\n",
       "      <td>Aegon Blackfyre</td>\n",
       "      <td>aegon.blackfyre@jnj.com</td>\n",
       "      <td>Aegon</td>\n",
       "      <td>Blackfyre</td>\n",
       "      <td>44</td>\n",
       "      <td>3</td>\n",
       "      <td>6</td>\n",
       "      <td>3</td>\n",
       "      <td>125.60</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>6</td>\n",
       "      <td>1</td>\n",
       "      <td>8</td>\n",
       "      <td>0</td>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>90</td>\n",
       "      <td>150.5</td>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>13</td>\n",
       "      <td>90</td>\n",
       "      <td>33.977273</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>2</td>\n",
       "      <td>2572.5</td>\n",
       "      <td>1</td>\n",
       "      <td>Aegon Frey (son of Aenys)</td>\n",
       "      <td>aegon.frey.(son.of.aenys)@gmail.com</td>\n",
       "      <td>Aegon</td>\n",
       "      <td>Frey</td>\n",
       "      <td>38</td>\n",
       "      <td>1</td>\n",
       "      <td>5</td>\n",
       "      <td>3</td>\n",
       "      <td>58.00</td>\n",
       "      <td>1</td>\n",
       "      <td>5</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>6</td>\n",
       "      <td>2</td>\n",
       "      <td>14</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>80</td>\n",
       "      <td>99.6</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>12</td>\n",
       "      <td>0</td>\n",
       "      <td>67.697368</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>3</td>\n",
       "      <td>1647.0</td>\n",
       "      <td>1</td>\n",
       "      <td>Aegon Targaryen (son of Rhaegar)</td>\n",
       "      <td>aegon.targaryen.(son.of.rhaegar)@ibm.com</td>\n",
       "      <td>Aegon</td>\n",
       "      <td>Targaryen</td>\n",
       "      <td>76</td>\n",
       "      <td>3</td>\n",
       "      <td>8</td>\n",
       "      <td>10</td>\n",
       "      <td>45.51</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>6</td>\n",
       "      <td>1</td>\n",
       "      <td>11</td>\n",
       "      <td>5</td>\n",
       "      <td>4</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>90</td>\n",
       "      <td>125.0</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>13</td>\n",
       "      <td>0</td>\n",
       "      <td>21.671053</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>4</td>\n",
       "      <td>1923.0</td>\n",
       "      <td>1</td>\n",
       "      <td>Aegon V Targaryen</td>\n",
       "      <td>aegon.v.targaryen@goldmansacs.com</td>\n",
       "      <td>Aegon</td>\n",
       "      <td>V Targaryen</td>\n",
       "      <td>65</td>\n",
       "      <td>3</td>\n",
       "      <td>6</td>\n",
       "      <td>9</td>\n",
       "      <td>106.00</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>6</td>\n",
       "      <td>2</td>\n",
       "      <td>12</td>\n",
       "      <td>0</td>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>90</td>\n",
       "      <td>135.3</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>13</td>\n",
       "      <td>253</td>\n",
       "      <td>29.584615</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   REVENUE  CROSS_SELL_SUCCESS                              NAME                                     EMAIL FIRST_NAME  FAMILY_NAME  TOTAL_MEALS_ORDERED  UNIQUE_MEALS_PURCH  CONTACTS_W_CUSTOMER_SERVICE  PRODUCT_CATEGORIES_VIEWED  AVG_TIME_PER_SITE_VISIT  MOBILE_NUMBER  CANCELLATIONS_BEFORE_NOON  CANCELLATIONS_AFTER_NOON  TASTES_AND_PREFERENCES  MOBILE_LOGINS  PC_LOGINS  WEEKLY_PLAN  EARLY_DELIVERIES  LATE_DELIVERIES  PACKAGE_LOCKER  REFRIGERATED_LOCKER  FOLLOWED_RECOMMENDATIONS_PCT  AVG_PREP_VID_TIME  LARGEST_ORDER_SIZE  MASTER_CLASSES_ATTENDED  MEDIAN_MEAL_RATING  AVG_CLICKS_PER_VISIT  TOTAL_PHOTOS_VIEWED  PRICE_PER_ORDER\n",
       "0   1880.0                   1                      Addam Osgrey                 addam.osgrey@passport.com      Addam       Osgrey                  118                   4                            7                          5                    86.00              1                          2                         1                       0              6          2            8                 0                2               1                    0                            90              165.8                   6                        1                   3                    11                    0        15.932203\n",
       "1   1495.0                   1                   Aegon Blackfyre                   aegon.blackfyre@jnj.com      Aegon    Blackfyre                   44                   3                            6                          3                   125.60              1                          0                         1                       0              6          1            8                 0                4               1                    0                            90              150.5                   4                        1                   3                    13                   90        33.977273\n",
       "2   2572.5                   1         Aegon Frey (son of Aenys)       aegon.frey.(son.of.aenys)@gmail.com      Aegon        Frey                    38                   1                            5                          3                    58.00              1                          5                         0                       1              6          2           14                 0                0               0                    0                            80               99.6                   3                        0                   3                    12                    0        67.697368\n",
       "3   1647.0                   1  Aegon Targaryen (son of Rhaegar)  aegon.targaryen.(son.of.rhaegar)@ibm.com      Aegon   Targaryen                    76                   3                            8                         10                    45.51              0                          3                         0                       1              6          1           11                 5                4               0                    0                            90              125.0                   3                        0                   3                    13                    0        21.671053\n",
       "4   1923.0                   1                 Aegon V Targaryen         aegon.v.targaryen@goldmansacs.com      Aegon  V Targaryen                   65                   3                            6                          9                   106.00              1                          1                         1                       0              6          2           12                 0                4               1                    0                            90              135.3                   3                        1                   3                    13                  253        29.584615"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Calculating the average price per order \n",
    "price_per_order = original_df['REVENUE']/original_df['TOTAL_MEALS_ORDERED']\n",
    "\n",
    "price_per_order.mean()\n",
    "\n",
    "# Creating a new column\n",
    "original_df['PRICE_PER_ORDER'] = price_per_order\n",
    "\n",
    "# Checking for the result\n",
    "original_df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Findings:** \n",
    "\n",
    "We have big gaps in the 80% - 100% quantile which need further investigation.  \n",
    "Avg REVENUE = 2107.29  \n",
    "Avg TOTAL_MEALS_ORDERED = 74.63  \n",
    "Avg PRICE_PER_MEAL = 36.5  \n",
    "Avg CROSS_SELL_SUCCESS --> We have more people who subscribed for the promotion in the dataset."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h2>Missing Value Analysis and Imputation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "REVENUE                          0\n",
       "CROSS_SELL_SUCCESS               0\n",
       "NAME                             0\n",
       "EMAIL                            0\n",
       "FIRST_NAME                       0\n",
       "FAMILY_NAME                     47\n",
       "TOTAL_MEALS_ORDERED              0\n",
       "UNIQUE_MEALS_PURCH               0\n",
       "CONTACTS_W_CUSTOMER_SERVICE      0\n",
       "PRODUCT_CATEGORIES_VIEWED        0\n",
       "AVG_TIME_PER_SITE_VISIT          0\n",
       "MOBILE_NUMBER                    0\n",
       "CANCELLATIONS_BEFORE_NOON        0\n",
       "CANCELLATIONS_AFTER_NOON         0\n",
       "TASTES_AND_PREFERENCES           0\n",
       "MOBILE_LOGINS                    0\n",
       "PC_LOGINS                        0\n",
       "WEEKLY_PLAN                      0\n",
       "EARLY_DELIVERIES                 0\n",
       "LATE_DELIVERIES                  0\n",
       "PACKAGE_LOCKER                   0\n",
       "REFRIGERATED_LOCKER              0\n",
       "FOLLOWED_RECOMMENDATIONS_PCT     0\n",
       "AVG_PREP_VID_TIME                0\n",
       "LARGEST_ORDER_SIZE               0\n",
       "MASTER_CLASSES_ATTENDED          0\n",
       "MEDIAN_MEAL_RATING               0\n",
       "AVG_CLICKS_PER_VISIT             0\n",
       "TOTAL_PHOTOS_VIEWED              0\n",
       "PRICE_PER_ORDER                  0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Checking for missing values\n",
    "\n",
    "original_df.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Imputing missing values\n",
    "original_df['FAMILY_NAME'] = original_df['FAMILY_NAME'].fillna('Unknown')\n",
    "\n",
    "\n",
    "# Checking for the results\n",
    "original_df['FAMILY_NAME'].isnull().sum()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h2>Data Types"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**CONTINUOUS OR INTERVAL**  \n",
    "REVENUE                  \n",
    "AVG_TIME_PER_SITE_VISIT  \n",
    "FOLLOWED_RECOMMENDATIONS_PCT  \n",
    "AVG_PREP_VID_TIME   \n",
    "MEDIAN_MEAL_RATING  \n",
    "AVG_CLICKS_PER_VISIT   \n",
    "PRICE_PER_MEAL  \n",
    "\n",
    "**BINARY**  \n",
    "CROSS_SELL_SUCCESS  \n",
    "MOBILE_NUMBER  \n",
    "TASTES_AND_PREFERENCES  \n",
    "PACKAGE_LOCKER  \n",
    "REFRIGERATED_LOCKER  \n",
    "\n",
    "**COUNT**  \n",
    "TOTAL_MEALS_ORDERED  \n",
    "UNIQUE_MEALS_PURCH  \n",
    "CONTACTS_W_CUSTOMER_SERVICE  \n",
    "PRODUCT_CATEGORIES_VIEWED  \n",
    "CANCELLATIONS_BEFORE_NOON  \n",
    "CANCELLATIONS_AFTER_NOON  \n",
    "MOBILE_LOGINS   \n",
    "PC_LOGINS   \n",
    "WEEKLY_PLAN  \n",
    "EARLY_DELIVERIES  \n",
    "LATE_DELIVERIES  \n",
    "LARGEST_ORDER_SIZE   \n",
    "MASTER_CLASSES_ATTENDED  \n",
    "TOTAL_PHOTOS_VIEWED  \n",
    "\n",
    "**CATEGORICAL**  \n",
    "\n",
    "\n",
    "**DISCRETE**      \n",
    "NAME                \n",
    "EMAIL                 \n",
    "FIRST_NAME               \n",
    "FAMILY_NAME               "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h2>Response Variable Exploration"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1    1321\n",
       "0     625\n",
       "Name: CROSS_SELL_SUCCESS, dtype: int64"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Exploring the response variable\n",
    "original_df['CROSS_SELL_SUCCESS'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Visualizing the count of the response variable\n",
    "sns.countplot(x = 'CROSS_SELL_SUCCESS', data = original_df, palette = 'hls')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Percentage of no subscription is 32.11716341212744\n",
      "Percentage of subscription 67.88283658787256\n"
     ]
    }
   ],
   "source": [
    "# Counting the percentage of customers who subsribed and who did not\n",
    "count_no_sub = len(original_df[original_df['CROSS_SELL_SUCCESS'] == 0])\n",
    "count_sub = len(original_df[original_df['CROSS_SELL_SUCCESS'] == 1])\n",
    "pct_of_no_sub = count_no_sub/(count_no_sub + count_sub)\n",
    "print(\"Percentage of no subscription is\", pct_of_no_sub * 100)\n",
    "pct_of_sub = count_sub/(count_no_sub + count_sub)\n",
    "print(\"Percentage of subscription\", pct_of_sub * 100)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Findings:**    \n",
    "\n",
    "Our classes are imbalanced, and the ratio of no-subscription to subscription instances is 32:68.   \n",
    "Before we go ahead to balance the classes, let’s do some more exploration.  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>REVENUE</th>\n",
       "      <th>TOTAL_MEALS_ORDERED</th>\n",
       "      <th>UNIQUE_MEALS_PURCH</th>\n",
       "      <th>CONTACTS_W_CUSTOMER_SERVICE</th>\n",
       "      <th>PRODUCT_CATEGORIES_VIEWED</th>\n",
       "      <th>AVG_TIME_PER_SITE_VISIT</th>\n",
       "      <th>MOBILE_NUMBER</th>\n",
       "      <th>CANCELLATIONS_BEFORE_NOON</th>\n",
       "      <th>CANCELLATIONS_AFTER_NOON</th>\n",
       "      <th>TASTES_AND_PREFERENCES</th>\n",
       "      <th>MOBILE_LOGINS</th>\n",
       "      <th>PC_LOGINS</th>\n",
       "      <th>WEEKLY_PLAN</th>\n",
       "      <th>EARLY_DELIVERIES</th>\n",
       "      <th>LATE_DELIVERIES</th>\n",
       "      <th>PACKAGE_LOCKER</th>\n",
       "      <th>REFRIGERATED_LOCKER</th>\n",
       "      <th>FOLLOWED_RECOMMENDATIONS_PCT</th>\n",
       "      <th>AVG_PREP_VID_TIME</th>\n",
       "      <th>LARGEST_ORDER_SIZE</th>\n",
       "      <th>MASTER_CLASSES_ATTENDED</th>\n",
       "      <th>MEDIAN_MEAL_RATING</th>\n",
       "      <th>AVG_CLICKS_PER_VISIT</th>\n",
       "      <th>TOTAL_PHOTOS_VIEWED</th>\n",
       "      <th>PRICE_PER_ORDER</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>CROSS_SELL_SUCCESS</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <td>0</td>\n",
       "      <td>2099.780800</td>\n",
       "      <td>74.113600</td>\n",
       "      <td>4.900800</td>\n",
       "      <td>6.862400</td>\n",
       "      <td>5.363200</td>\n",
       "      <td>98.597344</td>\n",
       "      <td>0.828800</td>\n",
       "      <td>1.036800</td>\n",
       "      <td>0.200000</td>\n",
       "      <td>0.660800</td>\n",
       "      <td>5.478400</td>\n",
       "      <td>1.515200</td>\n",
       "      <td>11.435200</td>\n",
       "      <td>1.43520</td>\n",
       "      <td>2.944000</td>\n",
       "      <td>0.324800</td>\n",
       "      <td>0.081600</td>\n",
       "      <td>17.472000</td>\n",
       "      <td>148.24848</td>\n",
       "      <td>4.387200</td>\n",
       "      <td>0.569600</td>\n",
       "      <td>2.760000</td>\n",
       "      <td>13.62720</td>\n",
       "      <td>103.756800</td>\n",
       "      <td>37.190232</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>1</td>\n",
       "      <td>2110.846707</td>\n",
       "      <td>74.880394</td>\n",
       "      <td>4.906889</td>\n",
       "      <td>7.040878</td>\n",
       "      <td>5.393641</td>\n",
       "      <td>100.081234</td>\n",
       "      <td>0.900833</td>\n",
       "      <td>1.579107</td>\n",
       "      <td>0.149886</td>\n",
       "      <td>0.739591</td>\n",
       "      <td>5.533687</td>\n",
       "      <td>1.457986</td>\n",
       "      <td>11.280091</td>\n",
       "      <td>1.51022</td>\n",
       "      <td>2.983346</td>\n",
       "      <td>0.369417</td>\n",
       "      <td>0.127933</td>\n",
       "      <td>43.898562</td>\n",
       "      <td>151.64860</td>\n",
       "      <td>4.460257</td>\n",
       "      <td>0.620742</td>\n",
       "      <td>2.811506</td>\n",
       "      <td>13.45193</td>\n",
       "      <td>107.700227</td>\n",
       "      <td>36.172095</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                        REVENUE  TOTAL_MEALS_ORDERED  UNIQUE_MEALS_PURCH  CONTACTS_W_CUSTOMER_SERVICE  PRODUCT_CATEGORIES_VIEWED  AVG_TIME_PER_SITE_VISIT  MOBILE_NUMBER  CANCELLATIONS_BEFORE_NOON  CANCELLATIONS_AFTER_NOON  TASTES_AND_PREFERENCES  MOBILE_LOGINS  PC_LOGINS  WEEKLY_PLAN  EARLY_DELIVERIES  LATE_DELIVERIES  PACKAGE_LOCKER  REFRIGERATED_LOCKER  FOLLOWED_RECOMMENDATIONS_PCT  AVG_PREP_VID_TIME  LARGEST_ORDER_SIZE  MASTER_CLASSES_ATTENDED  MEDIAN_MEAL_RATING  AVG_CLICKS_PER_VISIT  TOTAL_PHOTOS_VIEWED  PRICE_PER_ORDER\n",
       "CROSS_SELL_SUCCESS                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                \n",
       "0                   2099.780800            74.113600            4.900800                     6.862400                   5.363200                98.597344       0.828800                   1.036800                  0.200000                0.660800       5.478400   1.515200    11.435200           1.43520         2.944000        0.324800             0.081600                     17.472000          148.24848            4.387200                 0.569600            2.760000              13.62720           103.756800        37.190232\n",
       "1                   2110.846707            74.880394            4.906889                     7.040878                   5.393641               100.081234       0.900833                   1.579107                  0.149886                0.739591       5.533687   1.457986    11.280091           1.51022         2.983346        0.369417             0.127933                     43.898562          151.64860            4.460257                 0.620742            2.811506              13.45193           107.700227        36.172095"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Exploring the average proportions\n",
    "original_df.groupby('CROSS_SELL_SUCCESS').mean()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Findings:**    \n",
    "\n",
    "The average revenue and number of total meals ordered is slightly higher for the customers who subscribed to the promotion. However, the average price per order is slightly lower for the customers who did not subscribe.\n",
    "The average frequency of contacts with the customer service is higher in case of the customers who subscribed to the promotion.  \n",
    "The average time spent on the site of the customers who subscribed is higher than that of customers who didn't.  \n",
    "Customers who subscribed received on average more early deliveries.  \n",
    "Customers whose buildings have refrigerated locker subscribed on average more to the promotion.  \n",
    "Customers who subscribed for the promotion on average significantly followed the recommendations that were generated for them.  "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h2>Outlier Analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "\"\\n# Creating visual EDA (histograms)\\n\\nfig, ax = plt.subplots(figsize = (10, 8))\\nplt.subplot(2, 2, 1)\\nsns.distplot(original_df['REVENUE'],\\n             rug   = True,\\n             bins  = 'fd',\\n             color = 'g')\\nplt.xlabel('REVENUE')\\n\\n########################\\n\\nplt.subplot(2, 2, 2)\\nsns.distplot(original_df['CROSS_SELL_SUCCESS'],\\n             kde = False,\\n             bins  = 'fd',\\n             color = 'y')\\nplt.xlabel('CROSS_SELL_SUCCESS')\\n\\n########################\\n\\nplt.subplot(2, 2, 3)\\nsns.distplot(original_df['TOTAL_MEALS_ORDERED'],\\n             rug   = True,\\n             bins  = 'fd',\\n             color = 'orange')\\nplt.xlabel('TOTAL_MEALS_ORDERED')\\n\\n########################\\nplt.subplot(2, 2, 4)\\nsns.distplot(original_df['UNIQUE_MEALS_PURCH'],\\n             bins  = 'fd',\\n             color = 'r')\\nplt.xlabel('UNIQUE_MEALS_PURCH')\\nplt.tight_layout()\\nplt.savefig('Apprentice Chef Data Histograms 1 of 6.png')\\nplt.show()\\n\\n########################\\n########################\\n\\nfig, ax = plt.subplots(figsize = (10, 8))\\nplt.subplot(2, 2, 1)\\nsns.distplot(original_df['CONTACTS_W_CUSTOMER_SERVICE'],\\n             bins  = 'fd',\\n             color = 'g')\\nplt.xlabel('CONTACTS_W_CUSTOMER_SERVICE')\\n\\n########################\\n\\nplt.subplot(2, 2, 2)\\nsns.distplot(original_df['PRODUCT_CATEGORIES_VIEWED'],\\n             bins  = 'fd',\\n             color = 'y')\\nplt.xlabel('PRODUCT_CATEGORIES_VIEWED')\\n\\n########################\\n\\nplt.subplot(2, 2, 3)\\nsns.distplot(original_df['AVG_TIME_PER_SITE_VISIT'],\\n             bins  = 'fd',\\n             color = 'orange')\\nplt.xlabel('AVG_TIME_PER_SITE_VISIT')\\n\\n########################\\n\\nplt.subplot(2, 2, 4)\\nsns.distplot(original_df['MOBILE_NUMBER'],\\n             kde = False,\\n             bins  = 'fd',\\n             color = 'r')\\nplt.xlabel('MOBILE_NUMBER')\\nplt.tight_layout()\\nplt.savefig('Apprentice Chef Data Histograms 2 of 6.png')\\nplt.show()\\n\\n########################\\n########################\\n\\nfig, ax = plt.subplots(figsize = (10, 8))\\nplt.subplot(2, 2, 1)\\nsns.distplot(original_df['CANCELLATIONS_BEFORE_NOON'],\\n             rug   = True,\\n             bins  = 'fd',\\n             color = 'y')\\nplt.xlabel('CANCELLATIONS_BEFORE_NOON')\\n\\n########################\\n\\nplt.subplot(2, 2, 2)\\nsns.distplot(original_df['CANCELLATIONS_AFTER_NOON'],\\n             rug   = True,\\n             bins  = 'fd',\\n             color = 'orange')\\nplt.xlabel('CANCELLATIONS_AFTER_NOON')\\n\\n########################\\n\\nplt.subplot(2, 2, 3)\\nsns.distplot(original_df['TASTES_AND_PREFERENCES'],\\n             bins  = 'fd',\\n             color = 'r')\\nplt.xlabel('TASTES_AND_PREFERENCES')\\n\\n########################\\n\\nplt.subplot(2, 2, 4)\\nsns.distplot(original_df['MOBILE_LOGINS'],\\n             bins  = 'fd',\\n             color = 'g')\\nplt.xlabel('MOBILE_LOGINS')\\nplt.tight_layout()\\nplt.savefig('Apprentice Chef Data Histograms 3 of 6.png')\\nplt.show()\\n\\n########################\\n########################\\n\\nfig, ax = plt.subplots(figsize = (10, 8))\\nplt.subplot(2, 2, 1)\\nsns.distplot(original_df['PC_LOGINS'],\\n             bins  = 'fd',\\n             color = 'y')\\nplt.xlabel('PC_LOGINS')\\n\\n########################\\n\\nplt.subplot(2, 2, 2)\\nsns.distplot(original_df['WEEKLY_PLAN'],\\n             bins = 10,\\n             color = 'orange')\\nplt.xlabel('WEEKLY_PLAN')\\n\\n########################\\n\\nplt.subplot(2, 2, 3)\\nsns.distplot(original_df['EARLY_DELIVERIES'],\\n             bins = 'fd',\\n             color = 'r')\\nplt.xlabel('EARLY_DELIVERIES')\\n\\n########################\\n\\nplt.subplot(2, 2, 4)\\nsns.distplot(original_df['LATE_DELIVERIES'],\\n             bins  = 'fd',\\n             color = 'g')\\nplt.xlabel('LATE_DELIVERIES')\\nplt.tight_layout()\\nplt.savefig('Apprentice Chef Data Histograms 4 of 6.png')\\nplt.show()\\n\\n########################\\n########################\\n\\nfig, ax = plt.subplots(figsize = (10, 8))\\nplt.subplot(2, 2, 1)\\nsns.distplot(original_df['PACKAGE_LOCKER'],\\n             kde = False,\\n             bins  = 'fd',\\n             color = 'r')\\nplt.xlabel('PACKAGE_LOCKER')\\n\\n########################\\n\\nplt.subplot(2, 2, 2)\\nsns.distplot(original_df['REFRIGERATED_LOCKER'],\\n             bins  = 'fd',\\n             color = 'r')\\nplt.xlabel('REFRIGERATED_LOCKER')\\n\\n########################\\n\\nplt.subplot(2, 2, 3)\\nsns.distplot(original_df['FOLLOWED_RECOMMENDATIONS_PCT'],\\n             bins  = 'fd',\\n             color = 'r')\\nplt.xlabel('FOLLOWED_RECOMMENDATIONS_PCT')\\n\\n########################\\n\\nplt.subplot(2, 2, 4)\\nsns.distplot(original_df['AVG_PREP_VID_TIME'],\\n             bins  = 'fd',\\n             color = 'r')\\nplt.xlabel('AVG_PREP_VID_TIME')\\nplt.tight_layout()\\nplt.savefig('Apprentice Chef Data Histograms 5 of 6.png')\\nplt.show()\\n\\n########################\\n########################\\n\\nfig, ax = plt.subplots(figsize = (10, 8))\\nplt.subplot(2, 2, 1)\\nsns.distplot(original_df['LARGEST_ORDER_SIZE'],\\n             bins  = 'fd',\\n             color = 'r')\\nplt.xlabel('LARGEST_ORDER_SIZE')\\n\\n########################\\n\\nplt.subplot(2, 2, 2)\\nsns.distplot(original_df['MASTER_CLASSES_ATTENDED'],\\n             bins  = 'fd',\\n             color = 'r')\\nplt.xlabel('MASTER_CLASSES_ATTENDED')\\n\\n########################\\n\\nplt.subplot(2, 2, 3)\\nsns.distplot(original_df['MEDIAN_MEAL_RATING'],\\n             bins  = 'fd',\\n             color = 'r')\\nplt.xlabel('MEDIAN_MEAL_RATING')\\n\\n########################\\n\\nplt.subplot(2, 2, 4)\\nsns.distplot(original_df['AVG_CLICKS_PER_VISIT'],\\n             bins  = 'fd',\\n             color = 'r')\\nplt.xlabel('AVG_CLICKS_PER_VISIT')\\nplt.tight_layout()\\nplt.savefig('Apprentice Chef Data Histograms 5 of 6.png')\\nplt.show()\\n\\n########################\\n########################\\n\\nfig, ax = plt.subplots(figsize = (10, 8))\\nplt.subplot(2, 2, 1)\\nsns.distplot(original_df['TOTAL_PHOTOS_VIEWED'],\\n             bins  = 'fd',\\n             color = 'r')\\nplt.xlabel('TOTAL_PHOTOS_VIEWED')\\n\\nfig, ax = plt.subplots(figsize = (10, 8))\\nplt.subplot(2, 2, 1)\\nsns.distplot(original_df['PRICE_PER_ORDER'],\\n             bins  = 'fd',\\n             color = 'r')\\nplt.xlabel('PRICE_PER_ORDER')\\n\\n\""
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "\"\"\"\n",
    "# Creating visual EDA (histograms)\n",
    "\n",
    "fig, ax = plt.subplots(figsize = (10, 8))\n",
    "plt.subplot(2, 2, 1)\n",
    "sns.distplot(original_df['REVENUE'],\n",
    "             rug   = True,\n",
    "             bins  = 'fd',\n",
    "             color = 'g')\n",
    "plt.xlabel('REVENUE')\n",
    "\n",
    "########################\n",
    "\n",
    "plt.subplot(2, 2, 2)\n",
    "sns.distplot(original_df['CROSS_SELL_SUCCESS'],\n",
    "             kde = False,\n",
    "             bins  = 'fd',\n",
    "             color = 'y')\n",
    "plt.xlabel('CROSS_SELL_SUCCESS')\n",
    "\n",
    "########################\n",
    "\n",
    "plt.subplot(2, 2, 3)\n",
    "sns.distplot(original_df['TOTAL_MEALS_ORDERED'],\n",
    "             rug   = True,\n",
    "             bins  = 'fd',\n",
    "             color = 'orange')\n",
    "plt.xlabel('TOTAL_MEALS_ORDERED')\n",
    "\n",
    "########################\n",
    "plt.subplot(2, 2, 4)\n",
    "sns.distplot(original_df['UNIQUE_MEALS_PURCH'],\n",
    "             bins  = 'fd',\n",
    "             color = 'r')\n",
    "plt.xlabel('UNIQUE_MEALS_PURCH')\n",
    "plt.tight_layout()\n",
    "plt.savefig('Apprentice Chef Data Histograms 1 of 6.png')\n",
    "plt.show()\n",
    "\n",
    "########################\n",
    "########################\n",
    "\n",
    "fig, ax = plt.subplots(figsize = (10, 8))\n",
    "plt.subplot(2, 2, 1)\n",
    "sns.distplot(original_df['CONTACTS_W_CUSTOMER_SERVICE'],\n",
    "             bins  = 'fd',\n",
    "             color = 'g')\n",
    "plt.xlabel('CONTACTS_W_CUSTOMER_SERVICE')\n",
    "\n",
    "########################\n",
    "\n",
    "plt.subplot(2, 2, 2)\n",
    "sns.distplot(original_df['PRODUCT_CATEGORIES_VIEWED'],\n",
    "             bins  = 'fd',\n",
    "             color = 'y')\n",
    "plt.xlabel('PRODUCT_CATEGORIES_VIEWED')\n",
    "\n",
    "########################\n",
    "\n",
    "plt.subplot(2, 2, 3)\n",
    "sns.distplot(original_df['AVG_TIME_PER_SITE_VISIT'],\n",
    "             bins  = 'fd',\n",
    "             color = 'orange')\n",
    "plt.xlabel('AVG_TIME_PER_SITE_VISIT')\n",
    "\n",
    "########################\n",
    "\n",
    "plt.subplot(2, 2, 4)\n",
    "sns.distplot(original_df['MOBILE_NUMBER'],\n",
    "             kde = False,\n",
    "             bins  = 'fd',\n",
    "             color = 'r')\n",
    "plt.xlabel('MOBILE_NUMBER')\n",
    "plt.tight_layout()\n",
    "plt.savefig('Apprentice Chef Data Histograms 2 of 6.png')\n",
    "plt.show()\n",
    "\n",
    "########################\n",
    "########################\n",
    "\n",
    "fig, ax = plt.subplots(figsize = (10, 8))\n",
    "plt.subplot(2, 2, 1)\n",
    "sns.distplot(original_df['CANCELLATIONS_BEFORE_NOON'],\n",
    "             rug   = True,\n",
    "             bins  = 'fd',\n",
    "             color = 'y')\n",
    "plt.xlabel('CANCELLATIONS_BEFORE_NOON')\n",
    "\n",
    "########################\n",
    "\n",
    "plt.subplot(2, 2, 2)\n",
    "sns.distplot(original_df['CANCELLATIONS_AFTER_NOON'],\n",
    "             rug   = True,\n",
    "             bins  = 'fd',\n",
    "             color = 'orange')\n",
    "plt.xlabel('CANCELLATIONS_AFTER_NOON')\n",
    "\n",
    "########################\n",
    "\n",
    "plt.subplot(2, 2, 3)\n",
    "sns.distplot(original_df['TASTES_AND_PREFERENCES'],\n",
    "             bins  = 'fd',\n",
    "             color = 'r')\n",
    "plt.xlabel('TASTES_AND_PREFERENCES')\n",
    "\n",
    "########################\n",
    "\n",
    "plt.subplot(2, 2, 4)\n",
    "sns.distplot(original_df['MOBILE_LOGINS'],\n",
    "             bins  = 'fd',\n",
    "             color = 'g')\n",
    "plt.xlabel('MOBILE_LOGINS')\n",
    "plt.tight_layout()\n",
    "plt.savefig('Apprentice Chef Data Histograms 3 of 6.png')\n",
    "plt.show()\n",
    "\n",
    "########################\n",
    "########################\n",
    "\n",
    "fig, ax = plt.subplots(figsize = (10, 8))\n",
    "plt.subplot(2, 2, 1)\n",
    "sns.distplot(original_df['PC_LOGINS'],\n",
    "             bins  = 'fd',\n",
    "             color = 'y')\n",
    "plt.xlabel('PC_LOGINS')\n",
    "\n",
    "########################\n",
    "\n",
    "plt.subplot(2, 2, 2)\n",
    "sns.distplot(original_df['WEEKLY_PLAN'],\n",
    "             bins = 10,\n",
    "             color = 'orange')\n",
    "plt.xlabel('WEEKLY_PLAN')\n",
    "\n",
    "########################\n",
    "\n",
    "plt.subplot(2, 2, 3)\n",
    "sns.distplot(original_df['EARLY_DELIVERIES'],\n",
    "             bins = 'fd',\n",
    "             color = 'r')\n",
    "plt.xlabel('EARLY_DELIVERIES')\n",
    "\n",
    "########################\n",
    "\n",
    "plt.subplot(2, 2, 4)\n",
    "sns.distplot(original_df['LATE_DELIVERIES'],\n",
    "             bins  = 'fd',\n",
    "             color = 'g')\n",
    "plt.xlabel('LATE_DELIVERIES')\n",
    "plt.tight_layout()\n",
    "plt.savefig('Apprentice Chef Data Histograms 4 of 6.png')\n",
    "plt.show()\n",
    "\n",
    "########################\n",
    "########################\n",
    "\n",
    "fig, ax = plt.subplots(figsize = (10, 8))\n",
    "plt.subplot(2, 2, 1)\n",
    "sns.distplot(original_df['PACKAGE_LOCKER'],\n",
    "             kde = False,\n",
    "             bins  = 'fd',\n",
    "             color = 'r')\n",
    "plt.xlabel('PACKAGE_LOCKER')\n",
    "\n",
    "########################\n",
    "\n",
    "plt.subplot(2, 2, 2)\n",
    "sns.distplot(original_df['REFRIGERATED_LOCKER'],\n",
    "             bins  = 'fd',\n",
    "             color = 'r')\n",
    "plt.xlabel('REFRIGERATED_LOCKER')\n",
    "\n",
    "########################\n",
    "\n",
    "plt.subplot(2, 2, 3)\n",
    "sns.distplot(original_df['FOLLOWED_RECOMMENDATIONS_PCT'],\n",
    "             bins  = 'fd',\n",
    "             color = 'r')\n",
    "plt.xlabel('FOLLOWED_RECOMMENDATIONS_PCT')\n",
    "\n",
    "########################\n",
    "\n",
    "plt.subplot(2, 2, 4)\n",
    "sns.distplot(original_df['AVG_PREP_VID_TIME'],\n",
    "             bins  = 'fd',\n",
    "             color = 'r')\n",
    "plt.xlabel('AVG_PREP_VID_TIME')\n",
    "plt.tight_layout()\n",
    "plt.savefig('Apprentice Chef Data Histograms 5 of 6.png')\n",
    "plt.show()\n",
    "\n",
    "########################\n",
    "########################\n",
    "\n",
    "fig, ax = plt.subplots(figsize = (10, 8))\n",
    "plt.subplot(2, 2, 1)\n",
    "sns.distplot(original_df['LARGEST_ORDER_SIZE'],\n",
    "             bins  = 'fd',\n",
    "             color = 'r')\n",
    "plt.xlabel('LARGEST_ORDER_SIZE')\n",
    "\n",
    "########################\n",
    "\n",
    "plt.subplot(2, 2, 2)\n",
    "sns.distplot(original_df['MASTER_CLASSES_ATTENDED'],\n",
    "             bins  = 'fd',\n",
    "             color = 'r')\n",
    "plt.xlabel('MASTER_CLASSES_ATTENDED')\n",
    "\n",
    "########################\n",
    "\n",
    "plt.subplot(2, 2, 3)\n",
    "sns.distplot(original_df['MEDIAN_MEAL_RATING'],\n",
    "             bins  = 'fd',\n",
    "             color = 'r')\n",
    "plt.xlabel('MEDIAN_MEAL_RATING')\n",
    "\n",
    "########################\n",
    "\n",
    "plt.subplot(2, 2, 4)\n",
    "sns.distplot(original_df['AVG_CLICKS_PER_VISIT'],\n",
    "             bins  = 'fd',\n",
    "             color = 'r')\n",
    "plt.xlabel('AVG_CLICKS_PER_VISIT')\n",
    "plt.tight_layout()\n",
    "plt.savefig('Apprentice Chef Data Histograms 5 of 6.png')\n",
    "plt.show()\n",
    "\n",
    "########################\n",
    "########################\n",
    "\n",
    "fig, ax = plt.subplots(figsize = (10, 8))\n",
    "plt.subplot(2, 2, 1)\n",
    "sns.distplot(original_df['TOTAL_PHOTOS_VIEWED'],\n",
    "             bins  = 'fd',\n",
    "             color = 'r')\n",
    "plt.xlabel('TOTAL_PHOTOS_VIEWED')\n",
    "\n",
    "fig, ax = plt.subplots(figsize = (10, 8))\n",
    "plt.subplot(2, 2, 1)\n",
    "sns.distplot(original_df['PRICE_PER_ORDER'],\n",
    "             bins  = 'fd',\n",
    "             color = 'r')\n",
    "plt.xlabel('PRICE_PER_ORDER')\n",
    "\n",
    "\"\"\""
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h1>Feature Engineering"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'\\nWorking with additional information from the case study -- I want to distinguish\\ncustomers that include wine in their meals and people that order multiple meals \\nin one order and analyze this segment. Perhaps they are more interested in the\\npromotion because they like wine or they want to enjoy wine with other people.\\n\\n10 - 23: Meal Only \\n24 - 28: Meal with Water\\n28 - 48: Meal with Wine \\n48 +: More meals in one order (multiple meals)\\n\\n'"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "\"\"\"\n",
    "Working with additional information from the case study -- I want to distinguish\n",
    "customers that include wine in their meals and people that order multiple meals \n",
    "in one order and analyze this segment. Perhaps they are more interested in the\n",
    "promotion because they like wine or they want to enjoy wine with other people.\n",
    "\n",
    "10 - 23: Meal Only \n",
    "24 - 28: Meal with Water\n",
    "28 - 48: Meal with Wine \n",
    "48 +: More meals in one order (multiple meals)\n",
    "\n",
    "\"\"\"\n",
    "# Calculating the average meal with wine\n",
    "#print((28+48)/2) \n",
    "\n",
    "# Output: 38 --> prices per order above this average will be considered Meal with Wine or Multiple Meals"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h2>Outlier Thresholds"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Setting outlier thresholds\n",
    "REVENUE_hi                     = 5000  \n",
    "TOTAL_MEALS_ORDERED_lo         = 25\n",
    "TOTAL_MEALS_ORDERED_hi         = 215   \n",
    "UNIQUE_MEALS_PURCH_hi          = 9  \n",
    "CONTACTS_W_CUSTOMER_SERVICE_lo = 3 \n",
    "CONTACTS_W_CUSTOMER_SERVICE_hi = 12           \n",
    "AVG_TIME_PER_SITE_VISIT_hi     = 220                  \n",
    "CANCELLATIONS_BEFORE_NOON_hi   = 5         \n",
    "CANCELLATIONS_AFTER_NOON_hi    = 2              \n",
    "MOBILE_LOGINS_lo               = 5  \n",
    "MOBILE_LOGINS_hi               = 6  \n",
    "PC_LOGINS_lo                   = 1\n",
    "PC_LOGINS_hi                   = 2 \n",
    "WEEKLY_PLAN_hi                 = 14\n",
    "EARLY_DELIVERIES_hi            = 4       \n",
    "LATE_DELIVERIES_hi             = 7       \n",
    "AVG_PREP_VID_TIME_lo           = 60 \n",
    "AVG_PREP_VID_TIME_hi           = 280     \n",
    "LARGEST_ORDER_SIZE_lo          = 2\n",
    "LARGEST_ORDER_SIZE_hi          = 8   \n",
    "MASTER_CLASSES_ATTENDED_hi     = 2   \n",
    "MEDIAN_MEAL_RATING_lo          = 2   \n",
    "MEDIAN_MEAL_RATING_hi          = 4  \n",
    "AVG_CLICKS_PER_VISIT_lo        = 8\n",
    "AVG_CLICKS_PER_VISIT_hi        = 18      \n",
    "TOTAL_PHOTOS_VIEWED_lo         = 1\n",
    "TOTAL_PHOTOS_VIEWED_hi         = 300\n",
    "PRICE_PER_ORDER_hi              = 38\n",
    "\n",
    "# Developing features (columns) for outliers\n",
    "\n",
    "# REVENUE\n",
    "original_df['out_REVENUE'] = 0\n",
    "condition_hi = original_df.loc[0:,'out_REVENUE'][original_df['REVENUE'] > REVENUE_hi]\n",
    "\n",
    "original_df['out_REVENUE'].replace(to_replace = condition_hi,\n",
    "                                   value      = 1,\n",
    "                                   inplace    = True)\n",
    "\n",
    "# TOTAL_MEALS_ORDERED\n",
    "original_df['out_TOTAL_MEALS_ORDERED'] = 0\n",
    "condition_hi = original_df.loc[0:,'out_TOTAL_MEALS_ORDERED'][original_df['TOTAL_MEALS_ORDERED'] > TOTAL_MEALS_ORDERED_hi]\n",
    "condition_lo = original_df.loc[0:,'out_TOTAL_MEALS_ORDERED'][original_df['TOTAL_MEALS_ORDERED'] < TOTAL_MEALS_ORDERED_lo]\n",
    "\n",
    "original_df['out_TOTAL_MEALS_ORDERED'].replace(to_replace = condition_hi,\n",
    "                                               value      = 1,\n",
    "                                               inplace    = True)\n",
    "\n",
    "original_df['out_TOTAL_MEALS_ORDERED'].replace(to_replace = condition_lo,\n",
    "                                               value      = 1,\n",
    "                                               inplace    = True)\n",
    "\n",
    "# UNIQUE_MEALS_PURCH\n",
    "original_df['out_UNIQUE_MEALS_PURCH'] = 0\n",
    "condition_hi = original_df.loc[0:,'out_UNIQUE_MEALS_PURCH'][original_df['UNIQUE_MEALS_PURCH'] > UNIQUE_MEALS_PURCH_hi]\n",
    "\n",
    "original_df['out_UNIQUE_MEALS_PURCH'].replace(to_replace = condition_hi,\n",
    "                                               value      = 1,\n",
    "                                               inplace    = True)\n",
    "\n",
    "# CONTACTS_W_CUSTOMER_SERVICE\n",
    "original_df['out_CONTACTS_W_CUSTOMER_SERVICE'] = 0\n",
    "condition_hi = original_df.loc[0:,'out_CONTACTS_W_CUSTOMER_SERVICE'][original_df['CONTACTS_W_CUSTOMER_SERVICE'] > CONTACTS_W_CUSTOMER_SERVICE_hi]\n",
    "condition_lo = original_df.loc[0:,'out_CONTACTS_W_CUSTOMER_SERVICE'][original_df['CONTACTS_W_CUSTOMER_SERVICE'] < CONTACTS_W_CUSTOMER_SERVICE_lo]\n",
    "\n",
    "original_df['out_CONTACTS_W_CUSTOMER_SERVICE'].replace(to_replace = condition_hi,\n",
    "                                               value      = 1,\n",
    "                                               inplace    = True)\n",
    "\n",
    "original_df['out_CONTACTS_W_CUSTOMER_SERVICE'].replace(to_replace = condition_lo,\n",
    "                                               value      = 1,\n",
    "                                               inplace    = True)\n",
    "\n",
    "# AVG_TIME_PER_SITE_VISIT\n",
    "original_df['out_AVG_TIME_PER_SITE_VISIT'] = 0\n",
    "condition_hi = original_df.loc[0:,'out_AVG_TIME_PER_SITE_VISIT'][original_df['AVG_TIME_PER_SITE_VISIT'] > AVG_TIME_PER_SITE_VISIT_hi]\n",
    "\n",
    "original_df['out_AVG_TIME_PER_SITE_VISIT'].replace(to_replace = condition_hi,\n",
    "                                                   value      = 1,\n",
    "                                                   inplace    = True)\n",
    "\n",
    "# CANCELLATIONS_BEFORE_NOON\n",
    "original_df['out_CANCELLATIONS_BEFORE_NOON'] = 0\n",
    "condition_hi = original_df.loc[0:,'out_CANCELLATIONS_BEFORE_NOON'][original_df['CANCELLATIONS_BEFORE_NOON'] > CANCELLATIONS_BEFORE_NOON_hi]\n",
    "\n",
    "original_df['out_CANCELLATIONS_BEFORE_NOON'].replace(to_replace = condition_hi,\n",
    "                                                     value      = 1,\n",
    "                                                     inplace    = True)\n",
    "\n",
    "# CANCELLATIONS_AFTER_NOON\n",
    "original_df['out_CANCELLATIONS_AFTER_NOON'] = 0\n",
    "condition_hi = original_df.loc[0:,'out_CANCELLATIONS_AFTER_NOON'][original_df['CANCELLATIONS_AFTER_NOON'] > CANCELLATIONS_AFTER_NOON_hi]\n",
    "\n",
    "original_df['out_CANCELLATIONS_AFTER_NOON'].replace(to_replace = condition_hi,\n",
    "                                                    value      = 1,\n",
    "                                                    inplace    = True)\n",
    "\n",
    "# MOBILE_LOGINS\n",
    "original_df['out_MOBILE_LOGINS'] = 0\n",
    "condition_hi = original_df.loc[0:,'out_MOBILE_LOGINS'][original_df['MOBILE_LOGINS'] > MOBILE_LOGINS_hi]\n",
    "condition_lo = original_df.loc[0:,'out_MOBILE_LOGINS'][original_df['MOBILE_LOGINS'] < MOBILE_LOGINS_lo]\n",
    "\n",
    "original_df['out_MOBILE_LOGINS'].replace(to_replace = condition_hi,\n",
    "                                         value      = 1,\n",
    "                                         inplace    = True)\n",
    "\n",
    "original_df['out_MOBILE_LOGINS'].replace(to_replace = condition_lo,\n",
    "                                         value      = 1,\n",
    "                                         inplace    = True)\n",
    "\n",
    "# PC_LOGINS\n",
    "original_df['out_PC_LOGINS'] = 0\n",
    "condition_hi = original_df.loc[0:,'out_PC_LOGINS'][original_df['PC_LOGINS'] > PC_LOGINS_hi]\n",
    "condition_lo = original_df.loc[0:,'out_PC_LOGINS'][original_df['PC_LOGINS'] < PC_LOGINS_lo]\n",
    "\n",
    "original_df['out_PC_LOGINS'].replace(to_replace = condition_hi,\n",
    "                                     value      = 1,\n",
    "                                     inplace    = True)\n",
    "\n",
    "original_df['out_PC_LOGINS'].replace(to_replace = condition_lo,\n",
    "                                     value      = 1,\n",
    "                                     inplace    = True)\n",
    "\n",
    "# WEEKLY_PLAN\n",
    "original_df['out_WEEKLY_PLAN'] = 0\n",
    "condition_hi = original_df.loc[0:,'out_WEEKLY_PLAN'][original_df['WEEKLY_PLAN'] > WEEKLY_PLAN_hi]\n",
    "\n",
    "original_df['out_WEEKLY_PLAN'].replace(to_replace = condition_hi,\n",
    "                                       value      = 1,\n",
    "                                       inplace    = True)\n",
    "\n",
    "# EARLY_DELIVERIES \n",
    "original_df['out_EARLY_DELIVERIES'] = 0\n",
    "condition_hi = original_df.loc[0:,'out_EARLY_DELIVERIES'][original_df['EARLY_DELIVERIES'] > EARLY_DELIVERIES_hi]\n",
    "\n",
    "original_df['out_EARLY_DELIVERIES'].replace(to_replace = condition_hi,\n",
    "                                            value      = 1,\n",
    "                                            inplace    = True)\n",
    "\n",
    "# LATE_DELIVERIES\n",
    "original_df['out_LATE_DELIVERIES'] = 0\n",
    "condition_hi = original_df.loc[0:,'out_LATE_DELIVERIES'][original_df['LATE_DELIVERIES'] > LATE_DELIVERIES_hi]\n",
    "\n",
    "original_df['out_LATE_DELIVERIES'].replace(to_replace = condition_hi,\n",
    "                                           value      = 1,\n",
    "                                           inplace    = True)\n",
    "\n",
    "# AVG_PREP_VID_TIME\n",
    "original_df['out_AVG_PREP_VID_TIME'] = 0\n",
    "condition_hi = original_df.loc[0:,'out_AVG_PREP_VID_TIME'][original_df['AVG_PREP_VID_TIME'] > AVG_PREP_VID_TIME_hi]\n",
    "condition_lo = original_df.loc[0:,'out_AVG_PREP_VID_TIME'][original_df['AVG_PREP_VID_TIME'] < AVG_PREP_VID_TIME_lo]\n",
    "\n",
    "original_df['out_AVG_PREP_VID_TIME'].replace(to_replace = condition_hi,\n",
    "                                             value      = 1,\n",
    "                                             inplace    = True)\n",
    "\n",
    "original_df['out_AVG_PREP_VID_TIME'].replace(to_replace = condition_lo,\n",
    "                                             value      = 1,\n",
    "                                             inplace    = True)\n",
    "\n",
    "# LARGEST_ORDER_SIZE\n",
    "original_df['out_LARGEST_ORDER_SIZE'] = 0\n",
    "condition_hi = original_df.loc[0:,'out_LARGEST_ORDER_SIZE'][original_df['LARGEST_ORDER_SIZE'] > LARGEST_ORDER_SIZE_hi]\n",
    "condition_lo = original_df.loc[0:,'out_LARGEST_ORDER_SIZE'][original_df['LARGEST_ORDER_SIZE'] < LARGEST_ORDER_SIZE_lo]\n",
    "\n",
    "original_df['out_LARGEST_ORDER_SIZE'].replace(to_replace = condition_hi,\n",
    "                                              value      = 1,\n",
    "                                              inplace    = True)\n",
    "\n",
    "original_df['out_LARGEST_ORDER_SIZE'].replace(to_replace = condition_lo,\n",
    "                                              value      = 1,\n",
    "                                              inplace    = True)\n",
    "\n",
    "# MASTER_CLASSES_ATTENDED\n",
    "original_df['out_MASTER_CLASSES_ATTENDED'] = 0\n",
    "condition_hi = original_df.loc[0:,'out_MASTER_CLASSES_ATTENDED'][original_df['MASTER_CLASSES_ATTENDED'] > MASTER_CLASSES_ATTENDED_hi]\n",
    "\n",
    "original_df['out_MASTER_CLASSES_ATTENDED'].replace(to_replace = condition_hi,\n",
    "                                                   value      = 1,\n",
    "                                                   inplace    = True)\n",
    "\n",
    "# MEDIAN_MEAL_RATING\n",
    "original_df['out_MEDIAN_MEAL_RATING'] = 0\n",
    "condition_hi = original_df.loc[0:,'out_MEDIAN_MEAL_RATING'][original_df['MEDIAN_MEAL_RATING'] > MEDIAN_MEAL_RATING_hi]\n",
    "condition_lo = original_df.loc[0:,'out_MEDIAN_MEAL_RATING'][original_df['MEDIAN_MEAL_RATING'] < MEDIAN_MEAL_RATING_lo]\n",
    "\n",
    "original_df['out_MEDIAN_MEAL_RATING'].replace(to_replace = condition_hi,\n",
    "                                              value      = 1,\n",
    "                                              inplace    = True)\n",
    "\n",
    "original_df['out_MEDIAN_MEAL_RATING'].replace(to_replace = condition_lo,\n",
    "                                              value      = 1,\n",
    "                                              inplace    = True)\n",
    "\n",
    "# AVG_CLICKS_PER_VISIT\n",
    "original_df['out_AVG_CLICKS_PER_VISIT'] = 0\n",
    "condition_hi = original_df.loc[0:,'out_AVG_CLICKS_PER_VISIT'][original_df['AVG_CLICKS_PER_VISIT'] > AVG_CLICKS_PER_VISIT_hi]\n",
    "condition_lo = original_df.loc[0:,'out_AVG_CLICKS_PER_VISIT'][original_df['AVG_CLICKS_PER_VISIT'] < AVG_CLICKS_PER_VISIT_lo]\n",
    "\n",
    "original_df['out_AVG_CLICKS_PER_VISIT'].replace(to_replace = condition_hi,\n",
    "                                                value      = 1,\n",
    "                                                inplace    = True)\n",
    "\n",
    "original_df['out_AVG_CLICKS_PER_VISIT'].replace(to_replace = condition_lo,\n",
    "                                                value      = 1,\n",
    "                                                inplace    = True)\n",
    "\n",
    "# TOTAL_PHOTOS_VIEWED\n",
    "original_df['out_TOTAL_PHOTOS_VIEWED'] = 0\n",
    "condition_hi = original_df.loc[0:,'out_TOTAL_PHOTOS_VIEWED'][original_df['TOTAL_PHOTOS_VIEWED'] > TOTAL_PHOTOS_VIEWED_hi]\n",
    "condition_lo = original_df.loc[0:,'out_TOTAL_PHOTOS_VIEWED'][original_df['TOTAL_PHOTOS_VIEWED'] < TOTAL_PHOTOS_VIEWED_lo]\n",
    "\n",
    "original_df['out_TOTAL_PHOTOS_VIEWED'].replace(to_replace = condition_hi,\n",
    "                                               value      = 1,\n",
    "                                               inplace    = True)\n",
    "\n",
    "original_df['out_TOTAL_PHOTOS_VIEWED'].replace(to_replace = condition_lo,\n",
    "                                               value      = 1,\n",
    "                                               inplace    = True)\n",
    "\n",
    "# PRICE_PER_ORDER\n",
    "original_df['out_PRICE_PER_ORDER'] = 0\n",
    "condition_hi = original_df.loc[0:,'out_PRICE_PER_ORDER'][original_df['PRICE_PER_ORDER'] > PRICE_PER_ORDER_hi]\n",
    "\n",
    "original_df['out_PRICE_PER_ORDER'].replace(to_replace = condition_hi,\n",
    "                                               value      = 1,\n",
    "                                               inplace    = True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h2> Working with E-mail Addresses"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Working with Email Addresses to distinguish each groups\n",
    "\n",
    "# Step 1: Splitting personal emails \n",
    "\n",
    "# Placeholder list\n",
    "placeholder_lst = []  \n",
    "\n",
    "# Looping over each email address\n",
    "for index, col in original_df.iterrows(): \n",
    "    \n",
    "    # Splitting email domain at '@'\n",
    "    split_email = original_df.loc[index, 'EMAIL'].split(sep = '@') \n",
    "    \n",
    "    # Appending placeholder_lst with the results\n",
    "    placeholder_lst.append(split_email)\n",
    "    \n",
    "\n",
    "# Converting placeholder_lst into a DataFrame \n",
    "email_df = pd.DataFrame(placeholder_lst)\n",
    "\n",
    "\n",
    "# Displaying the results\n",
    "#email_df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Step 2: Concatenating with original DataFrame\n",
    "\n",
    "# Renaming column to concatenate\n",
    "email_df.columns = ['NAME' , 'EMAIL_DOMAIN']\n",
    "\n",
    "\n",
    "# Concatenating personal_email_domain with friends DataFrame \n",
    "original_df = pd.concat([original_df, email_df.loc[:, 'EMAIL_DOMAIN']], \n",
    "                   axis = 1)\n",
    "\n",
    "\n",
    "# Printing value counts of personal_email_domain\n",
    "#original_df.loc[: ,'EMAIL_DOMAIN'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Checking for new columns\n",
    "#original_df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "personal        861\n",
       "professional    696\n",
       "junk            389\n",
       "Name: DOMAIN_GROUP, dtype: int64"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# email domain types\n",
    "professional_email_domains = ['@mmm.com', '@amex.com', '@apple.com', \n",
    "                              '@boeing.com', '@caterpillar.com', '@chevron.com',\n",
    "                              '@cisco.com', '@cocacola.com', '@disney.com',\n",
    "                              '@dupont.com', '@exxon.com', '@ge.org',\n",
    "                              '@goldmansacs.com', '@homedepot.com', '@ibm.com',\n",
    "                              '@intel.com', '@jnj.com', '@jpmorgan.com',\n",
    "                              '@mcdonalds.com', '@merck.com', '@microsoft.com',\n",
    "                              '@nike.com', '@pfizer.com', '@pg.com',\n",
    "                              '@travelers.com', '@unitedtech.com', '@unitedhealth.com',\n",
    "                              '@verizon.com', '@visa.com', '@walmart.com']\n",
    "personal_email_domains     = ['@gmail.com', '@yahoo.com', '@protonmail.com']\n",
    "junk_email_domains         = ['@me.com', '@aol.com', '@hotmail.com',\n",
    "                              '@live.com', '@msn.com', '@passport.com']\n",
    "\n",
    "\n",
    "# placeholder list\n",
    "placeholder_lst = []  # good practice, overwriting the one above, everything in the workspace takes up place, we are renaming this - saves place on computer\n",
    "\n",
    "\n",
    "# looping to group observations by domain type\n",
    "for domain in original_df['EMAIL_DOMAIN']:\n",
    "        if '@' + domain in professional_email_domains: # has to be an exact match, that's why '@'\n",
    "            placeholder_lst.append('professional')\n",
    "            \n",
    "        elif '@' + domain in personal_email_domains:\n",
    "            placeholder_lst.append('personal')\n",
    "            \n",
    "        elif '@' + domain in junk_email_domains:\n",
    "            placeholder_lst.append('junk')\n",
    "            \n",
    "        else:\n",
    "            print('Unknown')\n",
    "\n",
    "\n",
    "# concatenating with original DataFrame\n",
    "original_df['DOMAIN_GROUP'] = pd.Series(placeholder_lst)\n",
    "\n",
    "\n",
    "# checking results\n",
    "original_df['DOMAIN_GROUP'].value_counts()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Finding:**    \n",
    "\n",
    "We have the majority of customers who registered with their personal emails."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h3>Exploring E-mail Domains"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Text(0, 0.5, 'Frequency of Subscription')"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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+vUjs3+F0wkwM2bPns306dNZc801m/SsNyU5ziLM3Llz6dmzpxcKnRBJ9OzZs1m1QS8YHGcRxwuFzktz49YLBsdxHKcILxictmNMj3A47ZqZM2ey5557MmDAAAYPHszo0aN54YUXWGKJJRgxYgSDBw9m33335Ysvvmh45sEHH2SjjTZi7bXXZu211+bCCy9suDd16lRGjRrFiBEjWGeddTjkkEMA+PTTT9l7771Zd911GTp0KJtvvjkff/xxRb1OP/10hgwZwrBhwxgxYgSPPfYYAKNGjWLQoEGMGDGCESNGsNtuuwEwZswYzjzzzEbuLL102vbMlfS+7LLLOPzww4tkR40a1TDZ9uOPP+ZHP/oRAwYMYMiQIWyxxRYNulay7bRp0xrsWziuuOIKAC655BLWXXddhg0bxtChQxk7diwAjz76KBtvvHGDfmPGjEkKVwre+ew4TgNmxi677MJ+++3HtddeC8CkSZN4++23GTBgAJMmTWL+/Plsu+22XH/99ey9997MnDmT73//+9x8882sv/76vPvuu2y//fasttpq7LTTThxxxBEcddRRfPvbYauJZ555BoBzzz2XlVZaqeF86tSpFUfPPPLII9x2221MnDiRxRdfnHfffZfPP1+4GeBVV13FyJEja2qLSnrncfDBB7Pmmmvy4osv0qVLF1555RWmTJlS1barr756g32zTJ8+ndNPP52JEyfSo0cPPv74Y2bNmgXAfvvtx/XXX8/w4cOZP38+U6dOrVnYvWBwHKeBe+65h27dunHooYc2XBsxYgTTpk1rOO/atSsbbbQRb775JgB//vOf2X///Vl//fUB6NWrF7///e8ZM2YMO+20EzNmzKBPnz4Nz6+77roAzJgxgzXWWKPh+qBBgyrqNWPGDHr16sXiiy/e4EdrU0nvarz88ss89thjXHXVVXTpEhpk+vfvT//+/fnf//5X1rZAkX2zvPPOOyyzzDINtZyll1664f8777zDKqusAoQ4GTx4cNMDWQFvSnIcp4Fnn32WDTbYoKrM3Llzeeyxx9hhhx0AmDx5cqNnRo4cyeTJkwE46qij2Hrrrdlxxx0555xz+OCDDwA48MAD+d3vfsemm27KSSedxIsvvljRz+2224433niDgQMH8pOf/IT77ruv6P7ee+/d0ARz9NFHNznc5aikdzUmT57MiBEj6Nq1a6N7ebZ9+eWXi5qSHnjgAYYPH85KK63EmmuuyQEHHMCtt95apN+gQYPYZZdd+Nvf/lbTuSheMDiOk0Qh4+rZsyd9+/Zl2LBhQGh+Kjf6pXDtgAMOYMqUKey+++7ce++9bLLJJsybN48RI0bwyiuvcPTRR/Pee++x4YYbMmXKlLJ+L7300kyYMIELL7yQ3r17873vfY/LLrus4f5VV13FpEmTmDRpEn/4wx9qEt5Kelca6dPS0V2FpqTC8fWvf52uXbty55138s9//pOBAwdy1FFHNfQl/OpXv2L8+PFst912XH311Q0FdS3wgsFxnAaGDBnChAkTyt4rZFwvvfQSjz76KLfcckvDM6WrHE+YMKGoaWPVVVflwAMPZOzYsSy22GI8++yzQMjwd911Vy644AJ+8IMfcMcdd1TUrWvXrowaNYpTTjmF888/nxtvvLGlwc2lnN49e/bk/fffL5J777336NWrF0OGDOGpp55iwYIFjdyqZttqSGKjjTbi+OOP59prry0K94ABA/jxj3/M3XffzVNPPcXs2bObHsgyeMHgOE4DW2+9NfPmzeOiiy5quPbEE0/w2msLV2leZZVVOOOMM/jtb38LwGGHHcZll13W0HE6e/Zsjj32WI455hgA7rzzzoYRTDNnzmT27NmsttpqPPTQQw0Z7Oeff85zzz1X1OeQZerUqUVNTZMmTaooWysq6b3hhhvy0EMPMXPmTADGjx/PvHnzGjqQR44cycknn0xh2+QXX3yRsWPHVrRtabNYlrfeeouJEyc2nGfDffvttxf50bVrV5ZbbrnaBN7MOuyxwQYbmNOBOHnZcDjthueee67RtTfffNN2331369+/vw0ePNhGjx5tL7zwgg0ZMqRBZsGCBTZs2DC7//77zczsvvvus5EjR9qgQYNs4MCBdsEFFzTIHnXUUTZw4EAbNmyYDRs2zK688kozM7v88stt3XXXtaFDh9rgwYPt6KOPtgULFpTVc/z48bbpppvaOuusY+uuu67tsssuNmvWLDMz23LLLW3gwIE2fPhwGz58uG2zzTZmZnbyySdbjx49bLXVVms4zMwkFV0766yzyvpZSW8zs5tvvtnWW289Gz58uG222WY2YcKEhnsffvihHXzwwda/f38bOnSobbnllvb4449Xte2rr75q3bt3bwjD8OHD7dxzz7Vp06bZVlttZYMGDbLh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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Visualizing the Subscription Frequency for Email Domain\n",
    "%matplotlib inline\n",
    "pd.crosstab(original_df.loc[:, 'EMAIL_DOMAIN'], original_df.loc[:, 'CROSS_SELL_SUCCESS']).plot(kind='bar')\n",
    "plt.title('Subscription Frequency for Email Domain')\n",
    "plt.xlabel('Email Domain')\n",
    "plt.ylabel('Frequency of Subscription')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Findings:**      \n",
    "\n",
    "Some of the email domains appear to be a strong predictor for the outcome variable -- gmail (personal), protonmail (personal), yahoo (personal)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>REVENUE</th>\n",
       "      <th>CROSS_SELL_SUCCESS</th>\n",
       "      <th>TOTAL_MEALS_ORDERED</th>\n",
       "      <th>UNIQUE_MEALS_PURCH</th>\n",
       "      <th>CONTACTS_W_CUSTOMER_SERVICE</th>\n",
       "      <th>PRODUCT_CATEGORIES_VIEWED</th>\n",
       "      <th>AVG_TIME_PER_SITE_VISIT</th>\n",
       "      <th>MOBILE_NUMBER</th>\n",
       "      <th>CANCELLATIONS_BEFORE_NOON</th>\n",
       "      <th>CANCELLATIONS_AFTER_NOON</th>\n",
       "      <th>TASTES_AND_PREFERENCES</th>\n",
       "      <th>MOBILE_LOGINS</th>\n",
       "      <th>PC_LOGINS</th>\n",
       "      <th>WEEKLY_PLAN</th>\n",
       "      <th>EARLY_DELIVERIES</th>\n",
       "      <th>LATE_DELIVERIES</th>\n",
       "      <th>PACKAGE_LOCKER</th>\n",
       "      <th>REFRIGERATED_LOCKER</th>\n",
       "      <th>FOLLOWED_RECOMMENDATIONS_PCT</th>\n",
       "      <th>AVG_PREP_VID_TIME</th>\n",
       "      <th>LARGEST_ORDER_SIZE</th>\n",
       "      <th>MASTER_CLASSES_ATTENDED</th>\n",
       "      <th>MEDIAN_MEAL_RATING</th>\n",
       "      <th>AVG_CLICKS_PER_VISIT</th>\n",
       "      <th>TOTAL_PHOTOS_VIEWED</th>\n",
       "      <th>PRICE_PER_ORDER</th>\n",
       "      <th>out_REVENUE</th>\n",
       "      <th>out_TOTAL_MEALS_ORDERED</th>\n",
       "      <th>out_UNIQUE_MEALS_PURCH</th>\n",
       "      <th>out_CONTACTS_W_CUSTOMER_SERVICE</th>\n",
       "      <th>out_AVG_TIME_PER_SITE_VISIT</th>\n",
       "      <th>out_CANCELLATIONS_BEFORE_NOON</th>\n",
       "      <th>out_CANCELLATIONS_AFTER_NOON</th>\n",
       "      <th>out_MOBILE_LOGINS</th>\n",
       "      <th>out_PC_LOGINS</th>\n",
       "      <th>out_WEEKLY_PLAN</th>\n",
       "      <th>out_EARLY_DELIVERIES</th>\n",
       "      <th>out_LATE_DELIVERIES</th>\n",
       "      <th>out_AVG_PREP_VID_TIME</th>\n",
       "      <th>out_LARGEST_ORDER_SIZE</th>\n",
       "      <th>out_MASTER_CLASSES_ATTENDED</th>\n",
       "      <th>out_MEDIAN_MEAL_RATING</th>\n",
       "      <th>out_AVG_CLICKS_PER_VISIT</th>\n",
       "      <th>out_TOTAL_PHOTOS_VIEWED</th>\n",
       "      <th>out_PRICE_PER_ORDER</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>DOMAIN_GROUP</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <td>junk</td>\n",
       "      <td>2055.353470</td>\n",
       "      <td>0.416452</td>\n",
       "      <td>71.197943</td>\n",
       "      <td>4.912596</td>\n",
       "      <td>7.035990</td>\n",
       "      <td>5.213368</td>\n",
       "      <td>98.475707</td>\n",
       "      <td>0.886889</td>\n",
       "      <td>1.370180</td>\n",
       "      <td>0.192802</td>\n",
       "      <td>0.691517</td>\n",
       "      <td>5.506427</td>\n",
       "      <td>1.488432</td>\n",
       "      <td>12.079692</td>\n",
       "      <td>1.516710</td>\n",
       "      <td>2.935733</td>\n",
       "      <td>0.334190</td>\n",
       "      <td>0.097686</td>\n",
       "      <td>28.971722</td>\n",
       "      <td>149.638303</td>\n",
       "      <td>4.403599</td>\n",
       "      <td>0.596401</td>\n",
       "      <td>2.760925</td>\n",
       "      <td>13.547558</td>\n",
       "      <td>107.048843</td>\n",
       "      <td>37.984108</td>\n",
       "      <td>0.025707</td>\n",
       "      <td>0.164524</td>\n",
       "      <td>0.010283</td>\n",
       "      <td>0.007712</td>\n",
       "      <td>0.020566</td>\n",
       "      <td>0.017995</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.046272</td>\n",
       "      <td>0.010283</td>\n",
       "      <td>0.223650</td>\n",
       "      <td>0.115681</td>\n",
       "      <td>0.056555</td>\n",
       "      <td>0.012853</td>\n",
       "      <td>0.020566</td>\n",
       "      <td>0.005141</td>\n",
       "      <td>0.069409</td>\n",
       "      <td>0.017995</td>\n",
       "      <td>0.706941</td>\n",
       "      <td>0.416452</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>personal</td>\n",
       "      <td>2117.465157</td>\n",
       "      <td>0.699187</td>\n",
       "      <td>75.435540</td>\n",
       "      <td>4.959350</td>\n",
       "      <td>6.934959</td>\n",
       "      <td>5.442509</td>\n",
       "      <td>101.363031</td>\n",
       "      <td>0.878049</td>\n",
       "      <td>1.409988</td>\n",
       "      <td>0.146341</td>\n",
       "      <td>0.722416</td>\n",
       "      <td>5.527294</td>\n",
       "      <td>1.469222</td>\n",
       "      <td>11.297329</td>\n",
       "      <td>1.449477</td>\n",
       "      <td>2.946574</td>\n",
       "      <td>0.363531</td>\n",
       "      <td>0.125436</td>\n",
       "      <td>36.051103</td>\n",
       "      <td>150.683391</td>\n",
       "      <td>4.441347</td>\n",
       "      <td>0.615563</td>\n",
       "      <td>2.817654</td>\n",
       "      <td>13.523810</td>\n",
       "      <td>106.736353</td>\n",
       "      <td>36.445319</td>\n",
       "      <td>0.022067</td>\n",
       "      <td>0.140534</td>\n",
       "      <td>0.010453</td>\n",
       "      <td>0.006969</td>\n",
       "      <td>0.023229</td>\n",
       "      <td>0.023229</td>\n",
       "      <td>0.001161</td>\n",
       "      <td>0.041812</td>\n",
       "      <td>0.013937</td>\n",
       "      <td>0.210221</td>\n",
       "      <td>0.098722</td>\n",
       "      <td>0.070848</td>\n",
       "      <td>0.018583</td>\n",
       "      <td>0.022067</td>\n",
       "      <td>0.004646</td>\n",
       "      <td>0.054588</td>\n",
       "      <td>0.011614</td>\n",
       "      <td>0.688734</td>\n",
       "      <td>0.411150</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>professional</td>\n",
       "      <td>2123.737787</td>\n",
       "      <td>0.800287</td>\n",
       "      <td>75.563218</td>\n",
       "      <td>4.833333</td>\n",
       "      <td>7.014368</td>\n",
       "      <td>5.406609</td>\n",
       "      <td>98.060388</td>\n",
       "      <td>0.872126</td>\n",
       "      <td>1.418103</td>\n",
       "      <td>0.175287</td>\n",
       "      <td>0.716954</td>\n",
       "      <td>5.507184</td>\n",
       "      <td>1.478448</td>\n",
       "      <td>10.951149</td>\n",
       "      <td>1.514368</td>\n",
       "      <td>3.020115</td>\n",
       "      <td>0.356322</td>\n",
       "      <td>0.106322</td>\n",
       "      <td>38.218391</td>\n",
       "      <td>150.912931</td>\n",
       "      <td>4.449713</td>\n",
       "      <td>0.594828</td>\n",
       "      <td>2.785920</td>\n",
       "      <td>13.466954</td>\n",
       "      <td>105.715517</td>\n",
       "      <td>35.735625</td>\n",
       "      <td>0.028736</td>\n",
       "      <td>0.129310</td>\n",
       "      <td>0.008621</td>\n",
       "      <td>0.004310</td>\n",
       "      <td>0.018678</td>\n",
       "      <td>0.024425</td>\n",
       "      <td>0.002874</td>\n",
       "      <td>0.044540</td>\n",
       "      <td>0.017241</td>\n",
       "      <td>0.195402</td>\n",
       "      <td>0.103448</td>\n",
       "      <td>0.074713</td>\n",
       "      <td>0.015805</td>\n",
       "      <td>0.033046</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.054598</td>\n",
       "      <td>0.004310</td>\n",
       "      <td>0.725575</td>\n",
       "      <td>0.392241</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                  REVENUE  CROSS_SELL_SUCCESS  TOTAL_MEALS_ORDERED  UNIQUE_MEALS_PURCH  CONTACTS_W_CUSTOMER_SERVICE  PRODUCT_CATEGORIES_VIEWED  AVG_TIME_PER_SITE_VISIT  MOBILE_NUMBER  CANCELLATIONS_BEFORE_NOON  CANCELLATIONS_AFTER_NOON  TASTES_AND_PREFERENCES  MOBILE_LOGINS  PC_LOGINS  WEEKLY_PLAN  EARLY_DELIVERIES  LATE_DELIVERIES  PACKAGE_LOCKER  REFRIGERATED_LOCKER  FOLLOWED_RECOMMENDATIONS_PCT  AVG_PREP_VID_TIME  LARGEST_ORDER_SIZE  MASTER_CLASSES_ATTENDED  MEDIAN_MEAL_RATING  AVG_CLICKS_PER_VISIT  TOTAL_PHOTOS_VIEWED  PRICE_PER_ORDER  out_REVENUE  out_TOTAL_MEALS_ORDERED  out_UNIQUE_MEALS_PURCH  out_CONTACTS_W_CUSTOMER_SERVICE  out_AVG_TIME_PER_SITE_VISIT  out_CANCELLATIONS_BEFORE_NOON  out_CANCELLATIONS_AFTER_NOON  out_MOBILE_LOGINS  out_PC_LOGINS  out_WEEKLY_PLAN  out_EARLY_DELIVERIES  out_LATE_DELIVERIES  out_AVG_PREP_VID_TIME  out_LARGEST_ORDER_SIZE  out_MASTER_CLASSES_ATTENDED  out_MEDIAN_MEAL_RATING  out_AVG_CLICKS_PER_VISIT  out_TOTAL_PHOTOS_VIEWED  out_PRICE_PER_ORDER\n",
       "DOMAIN_GROUP                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       \n",
       "junk          2055.353470            0.416452            71.197943            4.912596                     7.035990                   5.213368                98.475707       0.886889                   1.370180                  0.192802                0.691517       5.506427   1.488432    12.079692          1.516710         2.935733        0.334190             0.097686                     28.971722         149.638303            4.403599                 0.596401            2.760925             13.547558           107.048843        37.984108     0.025707                 0.164524                0.010283                         0.007712                     0.020566                       0.017995                      0.000000           0.046272       0.010283         0.223650              0.115681             0.056555               0.012853                0.020566                     0.005141                0.069409                  0.017995                 0.706941             0.416452\n",
       "personal      2117.465157            0.699187            75.435540            4.959350                     6.934959                   5.442509               101.363031       0.878049                   1.409988                  0.146341                0.722416       5.527294   1.469222    11.297329          1.449477         2.946574        0.363531             0.125436                     36.051103         150.683391            4.441347                 0.615563            2.817654             13.523810           106.736353        36.445319     0.022067                 0.140534                0.010453                         0.006969                     0.023229                       0.023229                      0.001161           0.041812       0.013937         0.210221              0.098722             0.070848               0.018583                0.022067                     0.004646                0.054588                  0.011614                 0.688734             0.411150\n",
       "professional  2123.737787            0.800287            75.563218            4.833333                     7.014368                   5.406609                98.060388       0.872126                   1.418103                  0.175287                0.716954       5.507184   1.478448    10.951149          1.514368         3.020115        0.356322             0.106322                     38.218391         150.912931            4.449713                 0.594828            2.785920             13.466954           105.715517        35.735625     0.028736                 0.129310                0.008621                         0.004310                     0.018678                       0.024425                      0.002874           0.044540       0.017241         0.195402              0.103448             0.074713               0.015805                0.033046                     0.000000                0.054598                  0.004310                 0.725575             0.392241"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Exploring the average proportions of the Domain Group\n",
    "original_df.groupby('DOMAIN_GROUP').mean()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Findings:**    \n",
    "\n",
    "Professional email domains on average contribute the most to the revenue and subscribe more to the promotion.  \n",
    "Professional email domains on average follow the meal recommendations more.  \n",
    "Personal email domains on average purchase more unique meals.   \n",
    "Personal email domains on average contact the customer service less.   \n",
    "Personal email domains on average spend more time on the site.   \n",
    "Personal email domains on average make less cancellations after noon.   \n",
    "Personal email domains on average specify more often their tastes and preferences in their profile.  \n",
    "Personal email domains on average attend more master classes.  \n",
    "Personal email domains on average rate meals better.  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Text(0, 0.5, 'Frequency')"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Checking for the frequency of each domain group \n",
    "original_df['DOMAIN_GROUP'].hist()\n",
    "plt.title('Histogram of Domain Group')\n",
    "plt.xlabel('Domain Group')\n",
    "plt.ylabel('Frequency')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Text(0, 0.5, 'Frequency of Subscription')"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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i5g7dy4Axx57LDfddBPbb789y5cvZ+rUqa347s2sOZVMCrVAbUQ8kU3fQkoKr0sakJ0lDCANFVk3f/nA6gNpYBDsiLgcuBygpqYmV7/fDz74IN27d+frX/96qWz48OHMmDGjNN21a1d23nlnXnstjUn+u9/9juOOO44dd0zt3f369eMXv/gFo0eP5oADDmD27NkMHDiwtPwnPvEJAGbPns0mm5TGAl/lV3J9s2fPpl+/fqy55pqlbVRaY3E35eWXX+aJJ55g7NixdOmSahyHDh3K0KFDeeCBBxrct8Aq+7fcG2+8Qa9evUpnTz179iw9f+ONNxgwINUadu3ala233rrlb9LMVlvF2hQiYg4wU1LdUXEv4EXgNuDYrOxYYFz2/DbgmOwqpF2Bt+uqmT6q559/nk9+8pNNzrN48WKeeOIJ9t13XwBeeOGFDy1TU1PDCy+8AMDpp5/OnnvuyX777ceFF17IggULABg5ciQ///nP+dSnPsWPf/xjpk2b1ug299lnH2bOnMkWW2zBN7/5TR566KFVXj/66KMZPnw4w4cP53vf+16L33dDGou7KS+88ALDhw+na9euH3qtuX378ssvl97D8OHDeeSRR9h+++3ZYIMN2HTTTTn++OO5/fbbV4lvyy235NBDD+Wyyy7zpadmVVbpq4++DYyV9CwwHPg/4Dxgb0nTSAOan5fNexdpgPjpwBXANyscG7DyoNW3b18GDx7MdtttB6Qqp4YufawrO/7445kyZQqHHXYY48ePZ9ddd2XJkiUMHz6cV155he9973vMnz+fnXbaiSlTpjS47Z49ezJx4kQuv/xy+vfvzxFHHMFVV11Ven3s2LFMnjyZyZMnc/7557fK+20s7sYu8/yol3/WVR/VPXbffXe6du3K3XffzS233MIWW2zB6aefXmo3+MlPfsKECRPYZ599uO6660pJ2syqo6JJISImR0RNRGwXEYdExFsRMS8i9oqIzbO/87N5IyJOiYjNIuITEdFqfWJvs802TJw4scHX6g5a06dP5/HHH+e2224rLVO/W+6JEyeuUp2x0UYbMXLkSMaNG0e3bt14/vnngXSw/8IXvsAll1zCV77yFe66665GY+vatSsjRozgnHPO4eKLL+bPf/7zR327zWoo7r59+/LWW2+tMt/8+fPp168f22yzDc888wwrVqz40Lqa2rdNkcTOO+/MD37wA2644YZV3vdmm23GN77xDe6//36eeeYZ5s2b1/I3aWarpcPd0dyQPffckyVLlnDFFVeUyp566ilefXVll+IDBgzgvPPO42c/+xkAp5xyCldddVWpkXTevHmceeaZfP/73wfg7rvv5oMPPgDSlU3z5s1j44035rHHHisdXJcuXcqLL764ShtDualTp65SvTR58uRG520tjcW900478dhjjzFnzhwAJkyYwJIlS0qNxTU1NZx99tnUDd86bdo0xo0b1+i+rV8VVm7WrFlMmjSpNF3+vu+8885VttG1a1d69+7dujvBzBrV4fo+aogkbr31Vk477TTOO+88evToUbpsstwhhxzC6NGjeeSRR9h999259tprOemkk1i4cCERwWmnncaBBx4IwD333MOpp55Kjx49ADj//PPZcMMNueeee/jGN75BRLBixQoOOOAAvvjFLzYY16JFi/j2t7/NggUL6NatG8OGDePyyy8vvX700UeXLknt168f9913HwA//elPV4m9traW9957b5UG5FGjRjFq1KgPbbOxuCFdObX//vuzYsUKevbsyfXXX19qWP7DH/7Ad7/7XYYNG8baa69duiS1uX1bVz1XZ+TIkRx88MGcccYZzJo1ix49etC/f38uvfRSAK655hpOP/101l57bbp168bYsWMbbMsws8pQ3a+y9qimpibqV/FMmTKFrbbaqqCIrFra/Ofs+xSsDZM0MSIavN69U1QfmZlZPp2i+qho8+bNY6+99vpQ+f3330/fvn0rss1zzz2Xm2++eZWyww47jB/96EcV2Z6ZdQxOClXQt2/fD93VW2k/+tGPnADMrMVcfWRmZiVOCmZmVuKkYGZmJW5TKNiQs+5s1fXNOO+AZue5++67OfXUU1m+fDknnngiZ53VVOe1Zp1QJ76k2GcKnczy5cs55ZRT+Nvf/saLL77I9ddfz4svvlh0WGbWRjgpdDJPPvkkw4YNY+jQoayxxhoceeSRjBs3rvkFzaxTcFLoZF577TUGDVo5bMXAgQNLY0iYmTkpdDINdWvyUbvHNrOOw0mhkxk4cCAzZ64cCru2tpaNNtqowIjMrC1xUuhkdtppJ6ZNm8a///1vli5dyg033MBBBx1UdFhm1kb4ktSC5bmEtDV169aNiy++mM997nMsX76ckSNHss0221Q1BjNru5wUOqH999+f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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Checking for the subscription frequency of each domain group \n",
    "pd.crosstab(original_df['DOMAIN_GROUP'], original_df['CROSS_SELL_SUCCESS']).plot(kind='bar')\n",
    "plt.title('Subscription Frequency for Domain Group')\n",
    "plt.xlabel('Domain Group')\n",
    "plt.ylabel('Frequency of Subscription')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Findings:**    \n",
    "\n",
    "Most of our customers use their personal emails when registering."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h3>One-Hot Encoding Categorical Variables"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [],
   "source": [
    "# One hot encoding categorical variables\n",
    "# Note: I decided to work with domain groups only as trends in individual cases can easily change.\n",
    "\n",
    "#one_hot_EMAIL_DOMAIN = pd.get_dummies(original_df['EMAIL_DOMAIN']) \n",
    "one_hot_DOMAIN_GROUP      = pd.get_dummies(original_df['DOMAIN_GROUP'])\n",
    "\n",
    "# Dropping categorical variables after they've been encoded\n",
    "#original_df = original_df.drop('EMAIL_DOMAIN', axis = 1)\n",
    "original_df = original_df.drop('DOMAIN_GROUP', axis = 1)\n",
    "\n",
    "# Joining codings together\n",
    "original_df = original_df.join([one_hot_DOMAIN_GROUP])\n",
    "#original_df = original_df.join([one_hot_EMAIL_DOMAIN])\n",
    "#original_df = original_df.join([one_hot_EMAIL_DOMAIN, one_hot_DOMAIN_GROUP])\n",
    "\n",
    "# Saving new columns\n",
    "new_columns = original_df.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Checking for the new columns\n",
    "#original_df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Descriptive statistics on dataset including new columns\n",
    "#original_df.describe()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h2>Working with Family Names"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "\"\\n# Working with Family Names because perhaps the same names indicate families where \\nmembers recommend services and offers to each other which could greatly impact our\\nprediction. \\n\\n# Counting the possible families based on the family name\\noriginal_df['FAMILY_NAME'].value_counts()\\n\\nOutput: There are many unknown values and a lot of individuals have the same family name.\\nFor example, 79 people have Frey as their last names. I am assuming that it is not possible \\nfor a family to be that big. Followed by a research that says that if people share the same \\nsurnames and it does not neccessarilly mean that they are related, I decided not to include the \\nFAMILY_NAME variable in my prediction.\\n\\n\""
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "\"\"\"\n",
    "# Working with Family Names because perhaps the same names indicate families where \n",
    "members recommend services and offers to each other which could greatly impact our\n",
    "prediction. \n",
    "\n",
    "# Counting the possible families based on the family name\n",
    "original_df['FAMILY_NAME'].value_counts()\n",
    "\n",
    "Output: There are many unknown values and a lot of individuals have the same family name.\n",
    "For example, 79 people have Frey as their last names. I am assuming that it is not possible \n",
    "for a family to be that big. Followed by a research that says that if people share the same \n",
    "surnames and it does not neccessarilly mean that they are related, I decided not to include the \n",
    "FAMILY_NAME variable in my prediction.\n",
    "\n",
    "\"\"\""
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h3>Guessing the Gender"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'  \\nI tried to guess the gender to analyze if male or female were more likely to subcribe \\nto the promotion. However, as the gender_guesser.detector labelled the majority of \\nthe data \"Unknown\", I did not see any value in proceeding.\\n\\nimport gender_guesser.detector as gender\\n\\n# guessing gender based on (given) name\\n\\n# placeholder list\\nplaceholder_lst = []\\n\\n\\n# looping to guess gender\\nfor name in original_df.loc[:, \\'FIRST_NAME\\']:\\n    guess = gender.Detector().get_gender(name)\\n    print(guess)\\n    placeholder_lst.append(guess)\\n\\n\\n# converting list into a series\\noriginal_df[\\'GENDER_GUESS\\'] = pd.Series(placeholder_lst)\\n\\n\\n# checking results\\noriginal_df.head(n = 5)\\n\\n'"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "\"\"\"  \n",
    "I tried to guess the gender to analyze if male or female were more likely to subcribe \n",
    "to the promotion. However, as the gender_guesser.detector labelled the majority of \n",
    "the data \"Unknown\", I did not see any value in proceeding.\n",
    "\n",
    "import gender_guesser.detector as gender\n",
    "\n",
    "# guessing gender based on (given) name\n",
    "\n",
    "# placeholder list\n",
    "placeholder_lst = []\n",
    "\n",
    "\n",
    "# looping to guess gender\n",
    "for name in original_df.loc[:, 'FIRST_NAME']:\n",
    "    guess = gender.Detector().get_gender(name)\n",
    "    print(guess)\n",
    "    placeholder_lst.append(guess)\n",
    "\n",
    "\n",
    "# converting list into a series\n",
    "original_df['GENDER_GUESS'] = pd.Series(placeholder_lst)\n",
    "\n",
    "\n",
    "# checking results\n",
    "original_df.head(n = 5)\n",
    "\n",
    "\"\"\""
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h1>Building Prediction Models "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {
    "code_folding": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "CROSS_SELL_SUCCESS                 1.00\n",
      "FOLLOWED_RECOMMENDATIONS_PCT       0.46\n",
      "professional                       0.19\n",
      "CANCELLATIONS_BEFORE_NOON          0.16\n",
      "MOBILE_NUMBER                      0.10\n",
      "TASTES_AND_PREFERENCES             0.08\n",
      "REFRIGERATED_LOCKER                0.07\n",
      "out_CANCELLATIONS_BEFORE_NOON      0.06\n",
      "MASTER_CLASSES_ATTENDED            0.04\n",
      "personal                           0.04\n",
      "PACKAGE_LOCKER                     0.04\n",
      "MOBILE_LOGINS                      0.04\n",
      "CONTACTS_W_CUSTOMER_SERVICE        0.04\n",
      "AVG_PREP_VID_TIME                  0.03\n",
      "out_CANCELLATIONS_AFTER_NOON       0.03\n",
      "MEDIAN_MEAL_RATING                 0.03\n",
      "out_EARLY_DELIVERIES               0.02\n",
      "out_LATE_DELIVERIES                0.02\n",
      "EARLY_DELIVERIES                   0.02\n",
      "LARGEST_ORDER_SIZE                 0.02\n",
      "out_AVG_PREP_VID_TIME              0.01\n",
      "LATE_DELIVERIES                    0.01\n",
      "AVG_TIME_PER_SITE_VISIT            0.01\n",
      "out_REVENUE                        0.01\n",
      "TOTAL_PHOTOS_VIEWED                0.01\n",
      "TOTAL_MEALS_ORDERED                0.01\n",
      "PRODUCT_CATEGORIES_VIEWED          0.00\n",
      "UNIQUE_MEALS_PURCH                 0.00\n",
      "REVENUE                            0.00\n",
      "out_MEDIAN_MEAL_RATING            -0.00\n",
      "out_WEEKLY_PLAN                    0.00\n",
      "WEEKLY_PLAN                       -0.01\n",
      "out_TOTAL_PHOTOS_VIEWED           -0.01\n",
      "PRICE_PER_ORDER                   -0.02\n",
      "out_UNIQUE_MEALS_PURCH            -0.02\n",
      "out_PC_LOGINS                     -0.02\n",
      "out_LARGEST_ORDER_SIZE            -0.02\n",
      "out_AVG_TIME_PER_SITE_VISIT       -0.03\n",
      "out_MOBILE_LOGINS                 -0.03\n",
      "out_TOTAL_MEALS_ORDERED           -0.03\n",
      "out_PRICE_PER_ORDER               -0.04\n",
      "AVG_CLICKS_PER_VISIT              -0.04\n",
      "out_MASTER_CLASSES_ATTENDED       -0.04\n",
      "PC_LOGINS                         -0.05\n",
      "CANCELLATIONS_AFTER_NOON          -0.05\n",
      "out_AVG_CLICKS_PER_VISIT          -0.06\n",
      "out_CONTACTS_W_CUSTOMER_SERVICE   -0.06\n",
      "junk                              -0.28\n",
      "Name: CROSS_SELL_SUCCESS, dtype: float64\n"
     ]
    }
   ],
   "source": [
    "# Creating a (Pearson) correlation matrix to display the relationships\n",
    "original_df_corr = original_df.corr().round(2)\n",
    "\n",
    "\n",
    "# Printing (Pearson) correlations with the response variable - CROSS_SELL_SUCCESS\n",
    "print(original_df_corr.loc['CROSS_SELL_SUCCESS'].sort_values(ascending = False))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Findings:**     \n",
    "\n",
    "The highest positive correlation is between CROSS_SELL_SUCCESS and FOLLOWED_RECOMMENDATIONS_PCT (0.46).  \n",
    "The second highest correlation, in this case negative, is with junk domain group (-0.28).  "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h2>Train/Test Split"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Declaring explanatory variables\n",
    "original_df_data = original_df.drop('CROSS_SELL_SUCCESS', axis = 1)\n",
    "\n",
    "# Declaring response variable\n",
    "original_df_target = original_df.loc[ : , 'CROSS_SELL_SUCCESS']\n",
    "\n",
    "# Train-test split with stratification\n",
    "X_train, X_test, y_train, y_test = train_test_split(\n",
    "            original_df_data,\n",
    "            original_df_target,\n",
    "            test_size = 0.25,\n",
    "            random_state = 222,\n",
    "            stratify = original_df_target)\n",
    "\n",
    "\n",
    "# Merging training data for statsmodels\n",
    "original_df_train = pd.concat([X_train, y_train], axis = 1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "REVENUE +\n",
      "NAME +\n",
      "EMAIL +\n",
      "FIRST_NAME +\n",
      "FAMILY_NAME +\n",
      "TOTAL_MEALS_ORDERED +\n",
      "UNIQUE_MEALS_PURCH +\n",
      "CONTACTS_W_CUSTOMER_SERVICE +\n",
      "PRODUCT_CATEGORIES_VIEWED +\n",
      "AVG_TIME_PER_SITE_VISIT +\n",
      "MOBILE_NUMBER +\n",
      "CANCELLATIONS_BEFORE_NOON +\n",
      "CANCELLATIONS_AFTER_NOON +\n",
      "TASTES_AND_PREFERENCES +\n",
      "MOBILE_LOGINS +\n",
      "PC_LOGINS +\n",
      "WEEKLY_PLAN +\n",
      "EARLY_DELIVERIES +\n",
      "LATE_DELIVERIES +\n",
      "PACKAGE_LOCKER +\n",
      "REFRIGERATED_LOCKER +\n",
      "FOLLOWED_RECOMMENDATIONS_PCT +\n",
      "AVG_PREP_VID_TIME +\n",
      "LARGEST_ORDER_SIZE +\n",
      "MASTER_CLASSES_ATTENDED +\n",
      "MEDIAN_MEAL_RATING +\n",
      "AVG_CLICKS_PER_VISIT +\n",
      "TOTAL_PHOTOS_VIEWED +\n",
      "PRICE_PER_ORDER +\n",
      "out_REVENUE +\n",
      "out_TOTAL_MEALS_ORDERED +\n",
      "out_UNIQUE_MEALS_PURCH +\n",
      "out_CONTACTS_W_CUSTOMER_SERVICE +\n",
      "out_AVG_TIME_PER_SITE_VISIT +\n",
      "out_CANCELLATIONS_BEFORE_NOON +\n",
      "out_CANCELLATIONS_AFTER_NOON +\n",
      "out_MOBILE_LOGINS +\n",
      "out_PC_LOGINS +\n",
      "out_WEEKLY_PLAN +\n",
      "out_EARLY_DELIVERIES +\n",
      "out_LATE_DELIVERIES +\n",
      "out_AVG_PREP_VID_TIME +\n",
      "out_LARGEST_ORDER_SIZE +\n",
      "out_MASTER_CLASSES_ATTENDED +\n",
      "out_MEDIAN_MEAL_RATING +\n",
      "out_AVG_CLICKS_PER_VISIT +\n",
      "out_TOTAL_PHOTOS_VIEWED +\n",
      "out_PRICE_PER_ORDER +\n",
      "EMAIL_DOMAIN +\n",
      "junk +\n",
      "personal +\n",
      "professional +\n"
     ]
    }
   ],
   "source": [
    "# Printing all the explanatory variables\n",
    "for val in original_df_data:\n",
    "    print(f\"{val} +\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Warning: Maximum number of iterations has been exceeded.\n",
      "         Current function value: 0.420513\n",
      "         Iterations: 35\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/Users/luuttami/anaconda3/lib/python3.7/site-packages/statsmodels/base/model.py:512: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals\n",
      "  \"Check mle_retvals\", ConvergenceWarning)\n",
      "/Users/luuttami/anaconda3/lib/python3.7/site-packages/statsmodels/base/model.py:1286: RuntimeWarning: invalid value encountered in sqrt\n",
      "  bse_ = np.sqrt(np.diag(self.cov_params()))\n",
      "/Users/luuttami/anaconda3/lib/python3.7/site-packages/scipy/stats/_distn_infrastructure.py:901: RuntimeWarning: invalid value encountered in greater\n",
      "  return (a < x) & (x < b)\n",
      "/Users/luuttami/anaconda3/lib/python3.7/site-packages/scipy/stats/_distn_infrastructure.py:901: RuntimeWarning: invalid value encountered in less\n",
      "  return (a < x) & (x < b)\n",
      "/Users/luuttami/anaconda3/lib/python3.7/site-packages/scipy/stats/_distn_infrastructure.py:1892: RuntimeWarning: invalid value encountered in less_equal\n",
      "  cond2 = cond0 & (x <= _a)\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<table class=\"simpletable\">\n",
       "<caption>Logit Regression Results</caption>\n",
       "<tr>\n",
       "  <th>Dep. Variable:</th>   <td>CROSS_SELL_SUCCESS</td> <th>  No. Observations:  </th>  <td>  1459</td>  \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Model:</th>                  <td>Logit</td>       <th>  Df Residuals:      </th>  <td>  1412</td>  \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Method:</th>                  <td>MLE</td>        <th>  Df Model:          </th>  <td>    46</td>  \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Date:</th>             <td>Wed, 05 Feb 2020</td>  <th>  Pseudo R-squ.:     </th>  <td>0.3304</td>  \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Time:</th>                 <td>23:29:26</td>      <th>  Log-Likelihood:    </th> <td> -613.53</td> \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>converged:</th>              <td>False</td>       <th>  LL-Null:           </th> <td> -916.19</td> \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Covariance Type:</th>      <td>nonrobust</td>     <th>  LLR p-value:       </th> <td>1.310e-98</td>\n",
       "</tr>\n",
       "</table>\n",
       "<table class=\"simpletable\">\n",
       "<tr>\n",
       "                 <td></td>                    <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Intercept</th>                       <td>   -1.8543</td> <td>      nan</td> <td>      nan</td> <td>   nan</td> <td>      nan</td> <td>      nan</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>REVENUE</th>                         <td>   -0.0003</td> <td>    0.000</td> <td>   -2.007</td> <td> 0.045</td> <td>   -0.001</td> <td>-7.79e-06</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>TOTAL_MEALS_ORDERED</th>             <td>    0.0031</td> <td>    0.003</td> <td>    0.938</td> <td> 0.348</td> <td>   -0.003</td> <td>    0.010</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>UNIQUE_MEALS_PURCH</th>              <td>    0.0133</td> <td>    0.031</td> <td>    0.424</td> <td> 0.672</td> <td>   -0.048</td> <td>    0.075</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>CONTACTS_W_CUSTOMER_SERVICE</th>     <td>    0.0804</td> <td>    0.048</td> <td>    1.685</td> <td> 0.092</td> <td>   -0.013</td> <td>    0.174</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>PRODUCT_CATEGORIES_VIEWED</th>       <td>   -0.0038</td> <td>    0.024</td> <td>   -0.158</td> <td> 0.874</td> <td>   -0.050</td> <td>    0.043</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>AVG_TIME_PER_SITE_VISIT</th>         <td>    0.0027</td> <td>    0.002</td> <td>    1.617</td> <td> 0.106</td> <td>   -0.001</td> <td>    0.006</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>MOBILE_NUMBER</th>                   <td>    0.7959</td> <td>    0.213</td> <td>    3.744</td> <td> 0.000</td> <td>    0.379</td> <td>    1.212</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>CANCELLATIONS_BEFORE_NOON</th>       <td>    0.2713</td> <td>    0.058</td> <td>    4.654</td> <td> 0.000</td> <td>    0.157</td> <td>    0.386</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>CANCELLATIONS_AFTER_NOON</th>        <td>   -0.3062</td> <td>    0.168</td> <td>   -1.826</td> <td> 0.068</td> <td>   -0.635</td> <td>    0.022</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>TASTES_AND_PREFERENCES</th>          <td>    0.3426</td> <td>    0.157</td> <td>    2.184</td> <td> 0.029</td> <td>    0.035</td> <td>    0.650</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>MOBILE_LOGINS</th>                   <td>    0.1822</td> <td>    0.121</td> <td>    1.501</td> <td> 0.133</td> <td>   -0.056</td> <td>    0.420</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>PC_LOGINS</th>                       <td>   -0.3341</td> <td>    0.136</td> <td>   -2.465</td> <td> 0.014</td> <td>   -0.600</td> <td>   -0.068</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>WEEKLY_PLAN</th>                     <td>   -0.0157</td> <td>    0.011</td> <td>   -1.478</td> <td> 0.139</td> <td>   -0.037</td> <td>    0.005</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>EARLY_DELIVERIES</th>                <td>    0.0442</td> <td>    0.051</td> <td>    0.863</td> <td> 0.388</td> <td>   -0.056</td> <td>    0.145</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>LATE_DELIVERIES</th>                 <td>    0.0545</td> <td>    0.035</td> <td>    1.542</td> <td> 0.123</td> <td>   -0.015</td> <td>    0.124</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>PACKAGE_LOCKER</th>                  <td>   -0.1176</td> <td>    0.171</td> <td>   -0.689</td> <td> 0.491</td> <td>   -0.452</td> <td>    0.217</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>REFRIGERATED_LOCKER</th>             <td>    0.4350</td> <td>    0.269</td> <td>    1.618</td> <td> 0.106</td> <td>   -0.092</td> <td>    0.962</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>FOLLOWED_RECOMMENDATIONS_PCT</th>    <td>    0.0607</td> <td>    0.004</td> <td>   14.262</td> <td> 0.000</td> <td>    0.052</td> <td>    0.069</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>AVG_PREP_VID_TIME</th>               <td>    0.0020</td> <td>    0.003</td> <td>    0.585</td> <td> 0.558</td> <td>   -0.005</td> <td>    0.009</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>LARGEST_ORDER_SIZE</th>              <td>    0.0211</td> <td>    0.081</td> <td>    0.260</td> <td> 0.795</td> <td>   -0.138</td> <td>    0.181</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>MASTER_CLASSES_ATTENDED</th>         <td>    0.1970</td> <td>    0.137</td> <td>    1.436</td> <td> 0.151</td> <td>   -0.072</td> <td>    0.466</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>MEDIAN_MEAL_RATING</th>              <td>   -0.1788</td> <td>    0.229</td> <td>   -0.780</td> <td> 0.435</td> <td>   -0.628</td> <td>    0.271</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>AVG_CLICKS_PER_VISIT</th>            <td>   -0.0790</td> <td>    0.063</td> <td>   -1.254</td> <td> 0.210</td> <td>   -0.203</td> <td>    0.045</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>TOTAL_PHOTOS_VIEWED</th>             <td> 1.133e-05</td> <td>    0.000</td> <td>    0.023</td> <td> 0.982</td> <td>   -0.001</td> <td>    0.001</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>PRICE_PER_ORDER</th>                 <td>    0.0171</td> <td>    0.009</td> <td>    1.921</td> <td> 0.055</td> <td>   -0.000</td> <td>    0.035</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>out_REVENUE</th>                     <td>    0.7726</td> <td>    0.670</td> <td>    1.153</td> <td> 0.249</td> <td>   -0.541</td> <td>    2.086</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>out_TOTAL_MEALS_ORDERED</th>         <td>   -0.8361</td> <td>    0.412</td> <td>   -2.029</td> <td> 0.042</td> <td>   -1.644</td> <td>   -0.028</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>out_UNIQUE_MEALS_PURCH</th>          <td>    0.4827</td> <td>    0.762</td> <td>    0.633</td> <td> 0.527</td> <td>   -1.012</td> <td>    1.977</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>out_CONTACTS_W_CUSTOMER_SERVICE</th> <td>   -1.0301</td> <td>    0.986</td> <td>   -1.045</td> <td> 0.296</td> <td>   -2.963</td> <td>    0.903</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>out_AVG_TIME_PER_SITE_VISIT</th>     <td>   -0.6728</td> <td>    0.597</td> <td>   -1.128</td> <td> 0.260</td> <td>   -1.842</td> <td>    0.497</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>out_CANCELLATIONS_BEFORE_NOON</th>   <td>   -0.6520</td> <td>    0.636</td> <td>   -1.025</td> <td> 0.305</td> <td>   -1.898</td> <td>    0.594</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>out_CANCELLATIONS_AFTER_NOON</th>    <td>   23.3919</td> <td> 7.87e+04</td> <td>    0.000</td> <td> 1.000</td> <td>-1.54e+05</td> <td> 1.54e+05</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>out_MOBILE_LOGINS</th>               <td>   -0.3213</td> <td>    0.325</td> <td>   -0.988</td> <td> 0.323</td> <td>   -0.959</td> <td>    0.316</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>out_PC_LOGINS</th>                   <td>    0.1529</td> <td>    0.567</td> <td>    0.270</td> <td> 0.787</td> <td>   -0.957</td> <td>    1.263</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>out_WEEKLY_PLAN</th>                 <td>    0.7742</td> <td>    0.354</td> <td>    2.185</td> <td> 0.029</td> <td>    0.080</td> <td>    1.469</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>out_EARLY_DELIVERIES</th>            <td>    0.1349</td> <td>    0.406</td> <td>    0.333</td> <td> 0.739</td> <td>   -0.660</td> <td>    0.930</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>out_LATE_DELIVERIES</th>             <td>   -0.2066</td> <td>    0.395</td> <td>   -0.523</td> <td> 0.601</td> <td>   -0.980</td> <td>    0.567</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>out_AVG_PREP_VID_TIME</th>           <td>    0.5500</td> <td>    0.675</td> <td>    0.814</td> <td> 0.415</td> <td>   -0.774</td> <td>    1.874</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>out_LARGEST_ORDER_SIZE</th>          <td>   -0.6077</td> <td>    0.466</td> <td>   -1.305</td> <td> 0.192</td> <td>   -1.521</td> <td>    0.305</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>out_MASTER_CLASSES_ATTENDED</th>     <td>   -1.4507</td> <td>    1.165</td> <td>   -1.245</td> <td> 0.213</td> <td>   -3.735</td> <td>    0.833</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>out_MEDIAN_MEAL_RATING</th>          <td>    0.2333</td> <td>    0.431</td> <td>    0.541</td> <td> 0.589</td> <td>   -0.612</td> <td>    1.079</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>out_AVG_CLICKS_PER_VISIT</th>        <td>   -1.5263</td> <td>    0.967</td> <td>   -1.579</td> <td> 0.114</td> <td>   -3.421</td> <td>    0.368</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>out_TOTAL_PHOTOS_VIEWED</th>         <td>   -0.1961</td> <td>    0.165</td> <td>   -1.187</td> <td> 0.235</td> <td>   -0.520</td> <td>    0.128</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>out_PRICE_PER_ORDER</th>             <td>   -0.5357</td> <td>    0.224</td> <td>   -2.386</td> <td> 0.017</td> <td>   -0.976</td> <td>   -0.096</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>junk</th>                            <td>   -1.7084</td> <td>      nan</td> <td>      nan</td> <td>   nan</td> <td>      nan</td> <td>      nan</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>personal</th>                        <td>   -0.4419</td> <td>      nan</td> <td>      nan</td> <td>   nan</td> <td>      nan</td> <td>      nan</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>professional</th>                    <td>    0.2960</td> <td>      nan</td> <td>      nan</td> <td>   nan</td> <td>      nan</td> <td>      nan</td>\n",
       "</tr>\n",
       "</table>"
      ],
      "text/plain": [
       "<class 'statsmodels.iolib.summary.Summary'>\n",
       "\"\"\"\n",
       "                           Logit Regression Results                           \n",
       "==============================================================================\n",
       "Dep. Variable:     CROSS_SELL_SUCCESS   No. Observations:                 1459\n",
       "Model:                          Logit   Df Residuals:                     1412\n",
       "Method:                           MLE   Df Model:                           46\n",
       "Date:                Wed, 05 Feb 2020   Pseudo R-squ.:                  0.3304\n",
       "Time:                        23:29:26   Log-Likelihood:                -613.53\n",
       "converged:                      False   LL-Null:                       -916.19\n",
       "Covariance Type:            nonrobust   LLR p-value:                 1.310e-98\n",
       "===================================================================================================\n",
       "                                      coef    std err          z      P>|z|      [0.025      0.975]\n",
       "---------------------------------------------------------------------------------------------------\n",
       "Intercept                          -1.8543        nan        nan        nan         nan         nan\n",
       "REVENUE                            -0.0003      0.000     -2.007      0.045      -0.001   -7.79e-06\n",
       "TOTAL_MEALS_ORDERED                 0.0031      0.003      0.938      0.348      -0.003       0.010\n",
       "UNIQUE_MEALS_PURCH                  0.0133      0.031      0.424      0.672      -0.048       0.075\n",
       "CONTACTS_W_CUSTOMER_SERVICE         0.0804      0.048      1.685      0.092      -0.013       0.174\n",
       "PRODUCT_CATEGORIES_VIEWED          -0.0038      0.024     -0.158      0.874      -0.050       0.043\n",
       "AVG_TIME_PER_SITE_VISIT             0.0027      0.002      1.617      0.106      -0.001       0.006\n",
       "MOBILE_NUMBER                       0.7959      0.213      3.744      0.000       0.379       1.212\n",
       "CANCELLATIONS_BEFORE_NOON           0.2713      0.058      4.654      0.000       0.157       0.386\n",
       "CANCELLATIONS_AFTER_NOON           -0.3062      0.168     -1.826      0.068      -0.635       0.022\n",
       "TASTES_AND_PREFERENCES              0.3426      0.157      2.184      0.029       0.035       0.650\n",
       "MOBILE_LOGINS                       0.1822      0.121      1.501      0.133      -0.056       0.420\n",
       "PC_LOGINS                          -0.3341      0.136     -2.465      0.014      -0.600      -0.068\n",
       "WEEKLY_PLAN                        -0.0157      0.011     -1.478      0.139      -0.037       0.005\n",
       "EARLY_DELIVERIES                    0.0442      0.051      0.863      0.388      -0.056       0.145\n",
       "LATE_DELIVERIES                     0.0545      0.035      1.542      0.123      -0.015       0.124\n",
       "PACKAGE_LOCKER                     -0.1176      0.171     -0.689      0.491      -0.452       0.217\n",
       "REFRIGERATED_LOCKER                 0.4350      0.269      1.618      0.106      -0.092       0.962\n",
       "FOLLOWED_RECOMMENDATIONS_PCT        0.0607      0.004     14.262      0.000       0.052       0.069\n",
       "AVG_PREP_VID_TIME                   0.0020      0.003      0.585      0.558      -0.005       0.009\n",
       "LARGEST_ORDER_SIZE                  0.0211      0.081      0.260      0.795      -0.138       0.181\n",
       "MASTER_CLASSES_ATTENDED             0.1970      0.137      1.436      0.151      -0.072       0.466\n",
       "MEDIAN_MEAL_RATING                 -0.1788      0.229     -0.780      0.435      -0.628       0.271\n",
       "AVG_CLICKS_PER_VISIT               -0.0790      0.063     -1.254      0.210      -0.203       0.045\n",
       "TOTAL_PHOTOS_VIEWED              1.133e-05      0.000      0.023      0.982      -0.001       0.001\n",
       "PRICE_PER_ORDER                     0.0171      0.009      1.921      0.055      -0.000       0.035\n",
       "out_REVENUE                         0.7726      0.670      1.153      0.249      -0.541       2.086\n",
       "out_TOTAL_MEALS_ORDERED            -0.8361      0.412     -2.029      0.042      -1.644      -0.028\n",
       "out_UNIQUE_MEALS_PURCH              0.4827      0.762      0.633      0.527      -1.012       1.977\n",
       "out_CONTACTS_W_CUSTOMER_SERVICE    -1.0301      0.986     -1.045      0.296      -2.963       0.903\n",
       "out_AVG_TIME_PER_SITE_VISIT        -0.6728      0.597     -1.128      0.260      -1.842       0.497\n",
       "out_CANCELLATIONS_BEFORE_NOON      -0.6520      0.636     -1.025      0.305      -1.898       0.594\n",
       "out_CANCELLATIONS_AFTER_NOON       23.3919   7.87e+04      0.000      1.000   -1.54e+05    1.54e+05\n",
       "out_MOBILE_LOGINS                  -0.3213      0.325     -0.988      0.323      -0.959       0.316\n",
       "out_PC_LOGINS                       0.1529      0.567      0.270      0.787      -0.957       1.263\n",
       "out_WEEKLY_PLAN                     0.7742      0.354      2.185      0.029       0.080       1.469\n",
       "out_EARLY_DELIVERIES                0.1349      0.406      0.333      0.739      -0.660       0.930\n",
       "out_LATE_DELIVERIES                -0.2066      0.395     -0.523      0.601      -0.980       0.567\n",
       "out_AVG_PREP_VID_TIME               0.5500      0.675      0.814      0.415      -0.774       1.874\n",
       "out_LARGEST_ORDER_SIZE             -0.6077      0.466     -1.305      0.192      -1.521       0.305\n",
       "out_MASTER_CLASSES_ATTENDED        -1.4507      1.165     -1.245      0.213      -3.735       0.833\n",
       "out_MEDIAN_MEAL_RATING              0.2333      0.431      0.541      0.589      -0.612       1.079\n",
       "out_AVG_CLICKS_PER_VISIT           -1.5263      0.967     -1.579      0.114      -3.421       0.368\n",
       "out_TOTAL_PHOTOS_VIEWED            -0.1961      0.165     -1.187      0.235      -0.520       0.128\n",
       "out_PRICE_PER_ORDER                -0.5357      0.224     -2.386      0.017      -0.976      -0.096\n",
       "junk                               -1.7084        nan        nan        nan         nan         nan\n",
       "personal                           -0.4419        nan        nan        nan         nan         nan\n",
       "professional                        0.2960        nan        nan        nan         nan         nan\n",
       "===================================================================================================\n",
       "\"\"\""
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Building a full model\n",
    "\n",
    "# Blueprinting a model type\n",
    "log_full = smf.logit(formula = \"\"\"CROSS_SELL_SUCCESS ~ REVENUE +\n",
    "TOTAL_MEALS_ORDERED +\n",
    "UNIQUE_MEALS_PURCH +\n",
    "CONTACTS_W_CUSTOMER_SERVICE +\n",
    "PRODUCT_CATEGORIES_VIEWED +\n",
    "AVG_TIME_PER_SITE_VISIT +\n",
    "MOBILE_NUMBER +\n",
    "CANCELLATIONS_BEFORE_NOON +\n",
    "CANCELLATIONS_AFTER_NOON +\n",
    "TASTES_AND_PREFERENCES +\n",
    "MOBILE_LOGINS +\n",
    "PC_LOGINS +\n",
    "WEEKLY_PLAN +\n",
    "EARLY_DELIVERIES +\n",
    "LATE_DELIVERIES +\n",
    "PACKAGE_LOCKER +\n",
    "REFRIGERATED_LOCKER +\n",
    "FOLLOWED_RECOMMENDATIONS_PCT +\n",
    "AVG_PREP_VID_TIME +\n",
    "LARGEST_ORDER_SIZE +\n",
    "MASTER_CLASSES_ATTENDED +\n",
    "MEDIAN_MEAL_RATING +\n",
    "AVG_CLICKS_PER_VISIT +\n",
    "TOTAL_PHOTOS_VIEWED +\n",
    "PRICE_PER_ORDER +\n",
    "out_REVENUE +\n",
    "out_TOTAL_MEALS_ORDERED +\n",
    "out_UNIQUE_MEALS_PURCH +\n",
    "out_CONTACTS_W_CUSTOMER_SERVICE +\n",
    "out_AVG_TIME_PER_SITE_VISIT +\n",
    "out_CANCELLATIONS_BEFORE_NOON +\n",
    "out_CANCELLATIONS_AFTER_NOON +\n",
    "out_MOBILE_LOGINS +\n",
    "out_PC_LOGINS +\n",
    "out_WEEKLY_PLAN +\n",
    "out_EARLY_DELIVERIES +\n",
    "out_LATE_DELIVERIES +\n",
    "out_AVG_PREP_VID_TIME +\n",
    "out_LARGEST_ORDER_SIZE +\n",
    "out_MASTER_CLASSES_ATTENDED +\n",
    "out_MEDIAN_MEAL_RATING +\n",
    "out_AVG_CLICKS_PER_VISIT +\n",
    "out_TOTAL_PHOTOS_VIEWED +\n",
    "out_PRICE_PER_ORDER +\n",
    "junk +\n",
    "personal +\n",
    "professional\"\"\", \n",
    "                               data = original_df_train)\n",
    "\n",
    "\n",
    "# Telling Python to run the data through the blueprint\n",
    "results_full = log_full.fit()\n",
    "\n",
    "\n",
    "# Printing the results\n",
    "results_full.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Optimization terminated successfully.\n",
      "         Current function value: 0.438201\n",
      "         Iterations 7\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<table class=\"simpletable\">\n",
       "<caption>Logit Regression Results</caption>\n",
       "<tr>\n",
       "  <th>Dep. Variable:</th>   <td>CROSS_SELL_SUCCESS</td> <th>  No. Observations:  </th>   <td>  1459</td>  \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Model:</th>                  <td>Logit</td>       <th>  Df Residuals:      </th>   <td>  1450</td>  \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Method:</th>                  <td>MLE</td>        <th>  Df Model:          </th>   <td>     8</td>  \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Date:</th>             <td>Wed, 05 Feb 2020</td>  <th>  Pseudo R-squ.:     </th>   <td>0.3022</td>  \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Time:</th>                 <td>23:29:26</td>      <th>  Log-Likelihood:    </th>  <td> -639.33</td> \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>converged:</th>              <td>True</td>        <th>  LL-Null:           </th>  <td> -916.19</td> \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Covariance Type:</th>      <td>nonrobust</td>     <th>  LLR p-value:       </th> <td>2.066e-114</td>\n",
       "</tr>\n",
       "</table>\n",
       "<table class=\"simpletable\">\n",
       "<tr>\n",
       "                <td></td>                  <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Intercept</th>                    <td>   -2.6782</td> <td>    0.351</td> <td>   -7.633</td> <td> 0.000</td> <td>   -3.366</td> <td>   -1.990</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>MOBILE_NUMBER</th>                <td>    0.7557</td> <td>    0.203</td> <td>    3.727</td> <td> 0.000</td> <td>    0.358</td> <td>    1.153</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>CANCELLATIONS_BEFORE_NOON</th>    <td>    0.2368</td> <td>    0.050</td> <td>    4.775</td> <td> 0.000</td> <td>    0.140</td> <td>    0.334</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>TASTES_AND_PREFERENCES</th>       <td>    0.3327</td> <td>    0.150</td> <td>    2.211</td> <td> 0.027</td> <td>    0.038</td> <td>    0.628</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>PC_LOGINS</th>                    <td>   -0.2949</td> <td>    0.130</td> <td>   -2.272</td> <td> 0.023</td> <td>   -0.549</td> <td>   -0.040</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>FOLLOWED_RECOMMENDATIONS_PCT</th> <td>    0.0580</td> <td>    0.004</td> <td>   14.470</td> <td> 0.000</td> <td>    0.050</td> <td>    0.066</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>out_PRICE_PER_ORDER</th>          <td>   -0.3129</td> <td>    0.141</td> <td>   -2.226</td> <td> 0.026</td> <td>   -0.588</td> <td>   -0.037</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>personal</th>                     <td>    1.2427</td> <td>    0.177</td> <td>    7.002</td> <td> 0.000</td> <td>    0.895</td> <td>    1.591</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>professional</th>                 <td>    1.8973</td> <td>    0.194</td> <td>    9.759</td> <td> 0.000</td> <td>    1.516</td> <td>    2.278</td>\n",
       "</tr>\n",
       "</table>"
      ],
      "text/plain": [
       "<class 'statsmodels.iolib.summary.Summary'>\n",
       "\"\"\"\n",
       "                           Logit Regression Results                           \n",
       "==============================================================================\n",
       "Dep. Variable:     CROSS_SELL_SUCCESS   No. Observations:                 1459\n",
       "Model:                          Logit   Df Residuals:                     1450\n",
       "Method:                           MLE   Df Model:                            8\n",
       "Date:                Wed, 05 Feb 2020   Pseudo R-squ.:                  0.3022\n",
       "Time:                        23:29:26   Log-Likelihood:                -639.33\n",
       "converged:                       True   LL-Null:                       -916.19\n",
       "Covariance Type:            nonrobust   LLR p-value:                2.066e-114\n",
       "================================================================================================\n",
       "                                   coef    std err          z      P>|z|      [0.025      0.975]\n",
       "------------------------------------------------------------------------------------------------\n",
       "Intercept                       -2.6782      0.351     -7.633      0.000      -3.366      -1.990\n",
       "MOBILE_NUMBER                    0.7557      0.203      3.727      0.000       0.358       1.153\n",
       "CANCELLATIONS_BEFORE_NOON        0.2368      0.050      4.775      0.000       0.140       0.334\n",
       "TASTES_AND_PREFERENCES           0.3327      0.150      2.211      0.027       0.038       0.628\n",
       "PC_LOGINS                       -0.2949      0.130     -2.272      0.023      -0.549      -0.040\n",
       "FOLLOWED_RECOMMENDATIONS_PCT     0.0580      0.004     14.470      0.000       0.050       0.066\n",
       "out_PRICE_PER_ORDER             -0.3129      0.141     -2.226      0.026      -0.588      -0.037\n",
       "personal                         1.2427      0.177      7.002      0.000       0.895       1.591\n",
       "professional                     1.8973      0.194      9.759      0.000       1.516       2.278\n",
       "================================================================================================\n",
       "\"\"\""
      ]
     },
     "execution_count": 33,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Building a fit model\n",
    "\n",
    "# Blueprinting a model type\n",
    "log_fit = smf.logit(formula = \"\"\"CROSS_SELL_SUCCESS ~ MOBILE_NUMBER +\n",
    "CANCELLATIONS_BEFORE_NOON +\n",
    "TASTES_AND_PREFERENCES +\n",
    "PC_LOGINS +\n",
    "FOLLOWED_RECOMMENDATIONS_PCT +\n",
    "out_PRICE_PER_ORDER +\n",
    "personal +\n",
    "professional\"\"\", \n",
    "                               data = original_df_train)\n",
    "\n",
    "\n",
    "# Telling Python to run the data through the blueprint\n",
    "results_fit = log_fit.fit()\n",
    "\n",
    "\n",
    "# Printing the results\n",
    "results_fit.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Creating sets of explanatory variables based on significance and research\n",
    "\n",
    "sig_variables_initial = ['MOBILE_NUMBER', 'CANCELLATIONS_BEFORE_NOON', \n",
    "               'TASTES_AND_PREFERENCES', 'PC_LOGINS', \n",
    "               'FOLLOWED_RECOMMENDATIONS_PCT', 'AVG_PREP_VID_TIME', \n",
    "               'personal', 'professional']\n",
    "\n",
    "sig_variables_10 = ['AVG_TIME_PER_SITE_VISIT', 'MOBILE_NUMBER', 'CANCELLATIONS_BEFORE_NOON',\n",
    "                  'TASTES_AND_PREFERENCES', 'PC_LOGINS', 'WEEKLY_PLAN', 'EARLY_DELIVERIES',\n",
    "                  'FOLLOWED_RECOMMENDATIONS_PCT', 'out_WEEKLY_PLAN', 'out_PRICE_PER_ORDER',\n",
    "                  'personal', 'professional']\n",
    "\n",
    "sig_variables_5 = ['MOBILE_NUMBER', 'CANCELLATIONS_BEFORE_NOON',\n",
    "                  'TASTES_AND_PREFERENCES', 'PC_LOGINS',\n",
    "                  'FOLLOWED_RECOMMENDATIONS_PCT', 'out_PRICE_PER_ORDER',\n",
    "                  'personal', 'professional']\n",
    "\n",
    "x_variables = ['MOBILE_NUMBER', 'CANCELLATIONS_BEFORE_NOON',\n",
    "               'TASTES_AND_PREFERENCES', 'PC_LOGINS',\n",
    "               'FOLLOWED_RECOMMENDATIONS_PCT', 'out_PRICE_PER_ORDER',\n",
    "               'personal', 'professional', 'REFRIGERATED_LOCKER',\n",
    "               'CANCELLATIONS_AFTER_NOON', 'MOBILE_LOGINS', 'AVG_PREP_VID_TIME']\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h2>Final Models"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h3>Logistic Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training ACCURACY: 0.7601\n",
      "Testing  ACCURACY: 0.7474\n",
      "AUC Value: 0.7091\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/Users/luuttami/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:947: ConvergenceWarning: lbfgs failed to converge. Increase the number of iterations.\n",
      "  \"of iterations.\", ConvergenceWarning)\n"
     ]
    }
   ],
   "source": [
    "# train/test split with the full model\n",
    "original_df_data   =  original_df.loc[ : , x_variables]\n",
    "original_df_target =  original_df.loc[ : , 'CROSS_SELL_SUCCESS']\n",
    "\n",
    "# train-test split with the scaled data\n",
    "X_train, X_test, y_train, y_test = train_test_split(\n",
    "            original_df_data,\n",
    "            original_df_target,\n",
    "            random_state = 222,\n",
    "            test_size = 0.25,\n",
    "            stratify = original_df_target)\n",
    "\n",
    "# INSTANTIATING a logistic regression model\n",
    "logreg = LogisticRegression(solver = 'lbfgs',\n",
    "                            C = 1,\n",
    "                            random_state = 222)\n",
    "\n",
    "\n",
    "# FITTING the training data\n",
    "logreg_fit = logreg.fit(X_train, y_train)\n",
    "\n",
    "\n",
    "# PREDICTING based on the testing set\n",
    "logreg_pred = logreg_fit.predict(X_test)\n",
    "\n",
    "\n",
    "# SCORING the results\n",
    "print('Training ACCURACY:', logreg_fit.score(X_train, y_train).round(4))\n",
    "print('Testing  ACCURACY:', logreg_fit.score(X_test, y_test).round(4))\n",
    "print('AUC Value:', roc_auc_score(y_true  = y_test,\n",
    "                                  y_score = logreg_pred).round(4))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h4>Confusion Matrix (Logistic Model)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[ 94  62]\n",
      " [ 61 270]]\n"
     ]
    }
   ],
   "source": [
    "# creating a confusion matrix\n",
    "print(confusion_matrix(y_true = y_test,\n",
    "                       y_pred = logreg_pred))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [],
   "source": [
    "############################################\n",
    "# Creating user-defined function: visual_cm\n",
    "############################################\n",
    "def visual_cm(true_y, pred_y, labels = None):\n",
    "\n",
    "    # visualizing the confusion matrix\n",
    "\n",
    "    # setting labels\n",
    "    lbls = labels\n",
    "    \n",
    "    # declaring a confusion matrix object\n",
    "    cm = confusion_matrix(y_true = true_y,\n",
    "                          y_pred = pred_y)\n",
    "\n",
    "\n",
    "    # heatmap\n",
    "    sns.heatmap(cm,\n",
    "                annot       = True,\n",
    "                xticklabels = lbls,\n",
    "                yticklabels = lbls,\n",
    "                cmap        = 'Blues',\n",
    "                fmt         = 'g')\n",
    "\n",
    "\n",
    "    plt.xlabel('Predicted')\n",
    "    plt.ylabel('Actual')\n",
    "    plt.title('Confusion Matrix of the Classifier')\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAW4AAAErCAYAAAD+N2lQAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4xLjEsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy8QZhcZAAAgAElEQVR4nO3debzc0/3H8df7JkqskYgtYi2jthKqlmp1b1WF0qKook37+9GqotZa68dPa61WRVFCqfVHWy2qVC0JotZy7SQiJAhJLBE+vz/OuYzbu8y9me+d+d77fnp8H2a+35lzzsydfObM53vO+SoiMDOz8mhpdAPMzKxnHLjNzErGgdvMrGQcuM3MSsaB28ysZBy4zcxKxoG7wSQNkfRHSa9Kumw+ytlF0vX1bFsjSPqLpN0LKHc7SZMlzZa0QQ2P31LSlHq3o4v6QtKHCyr7A58NSZtLeiy/F9sW9Z5bgSLCWw0b8E3gbmA28DzwF+ATdSh3N+BOYHCjX2Mn7dsSCODKdvs/mvffXGM5RwEXNvB1PAGM6eJ4AB9u97qn1LH+5YBz8mdnFvAIcDSwSEf1F/xe3Ajs2+jPlrfeb+5x10DSj4FTgf8BlgFWBH4NjKlD8SsBj0bEvDqUVZTpwGaShlft2x14tF4VKCny87gS8FCB5XdK0jDgDmAIsGlELAZ8HhgKrNaAJtXlvZA0uA5tsd5o9DdHs2/AEqRe9te7eMyCpMA+NW+nAgvmY1sCU4D9gRdJPa498rGjgbnA27mOvWjXMwVWJvXGBuf73waeJPXangJ2qdp/a9XzNgPuAl7N/9+s6tjNwLHAbbmc64GlOnltbe3/DbB33jco7zuCqh43cBowGXgNmARskfd/qd3rvK+qHcfldrwBfDjv+04+fiZweVX5/0vqLaqDdrYAhwPP5Pf5gvy3WzDXGcAc4IkOnntL1fHZwI5d/d2q/ua/AJ4FXsjvz5BO3sOfAQ8ALV18ht7rcQNfAf6V38fJwFFVj1sIuBB4CZiZ/7bL1PrZIP3yeDe/37Pz63jvPc+P2RN4GHgFuA5YqV079wYeA55q9L/Pgbo1vAHNvuWgM48uUhnAMcAEYGlgBHA7cGw+tmV+/jHAAsBWwOvAkvn4UXwwULe/v3L+xzIYWCT/Y67kY8sBa+fb1f84h+V/dLvl5+2c7w/Px2/O/4DXIPUCbwZO6OS1tQWwzYCJed9W+R/0d/hg4N4VGJ7r3B+YBizU0euqasezwNr5OQvwwcC9MKlX/21gC2AGsEIn7dwTeBxYFVgUuBIYX3W8y1RE++M1/N1OBa7J7/ViwB+B4zspewJwdDefs+rAvSWwLunLaD3SF8O2+dj3cl0Lk75ANwQWr/Wzke8/DXyu3d+h7T3fNr+PH8l/k8OB29u184b8ujv8ovJW/OZUSfeGAzOi61TGLsAxEfFiREwn9aR3qzr+dj7+dkRcS+rpVHrZnneBdSQNiYjnI6Kjn7xfAR6LiPERMS8iLiblVL9a9ZjzIuLRiHgDuBRYv6tKI+J2YJikCvAtUo+2/WMujIiXcp0nkXpz3b3O30XEQ/k5b7cr73XSl8HJpF7mDyKisxOGuwAnR8STETEbOATYaT5/znf4d5Mk4LvAfhHxckTMIqXRduqknOGkHntNIuLmiHggIt6NiPuBi4FPVbVpOCnIvxMRkyLitXysls9Gd75H+gJ6OH/m/wdYX9JKVY85Pr/uN3pRvtWBA3f3XgKW6iYALE/6id7mmbzvvTLaBf7XSb3CHomIOaSf8d8Hnpf0Z0lr1tCetjaNrLo/rRftGQ/sA3wauKr9QUn7S3o4j5CZSUpVLNVNmZO7OhgRd5J+/ov0BdOZjv4Gg0nnJHqrs7/bCFKPd5Kkmfm1/jXv77AcUg+4JpI+LukmSdMlvUr6e7e9j+NJv3YukTRV0omSFujBZ6M7KwGnVb2ul0nvffVnp8u/mRXPgbt7dwBvkn5CdmYq6QPfZsW8rzfmkIJCm2WrD0bEdRHxeVIgeAQ4u4b2tLXpuV62qc144L+Ba3Nv+D2StgAOAr5BSicMJeXX1db0TsrscnlKSXuTeu5TgZ908dCO/gbzSGmGeptByhGvHRFD87ZERHT25fc3YLsenHz9PSkNMyoiliDlzwWQe/9HR8RapPTV1qRfQLV+NrozGfhe1esaGhFD8i+uNl5StMEcuLsREa+STsL9Ko95XVjSApK+LOnE/LCLgcMljZC0VH78hb2s8l7gk5JWlLQE6Sc/AJKWkbSNpEWAt0g/3d/poIxrgTUkfVPSYEk7AmsBf+plmwCIiKdIP9kP6+DwYqRAOR0YLOkIUu61zQvAyj0ZOSJpDdKJvV1JqaefSOospXMxsJ+kVSQtSvqJ/4duUlzVXiDlx7sVEe+SguIpkpbObR0p6YudPOVk0ntxflvKIT/+ZEnrdfD4xYCXI+JNSRuThqKSn/dpSetKGkTKab8NvNODz0Z3fgMcImntXN8Skr7ei3KsQA7cNYiIk4Efk07UTCf1SvYB/i8/5GekMd73k0YP3JP39aauG4A/5LIm8cFg20I66TeV9BP2U6QecPsyXiL1xPYn/Uz/CbB1RMzoTZvalX1rRHT0a+I60tj2R0lpijf54E/qtslFL0m6p7t6cmrqQuB/I+K+iHgMOBQYL2nBDp5yLukXwS2kERVvAj+o7VUB6eTp+TlF8I0aHn8Q6STeBEmvkXrVHebzI+JlUu/4bWCipFmk0TGv5jLa+2/gmPy4I/hgimhZ4HJS0H4Y+Afpfarps9GdiLiKNHrnkvy6HgS+3NNyrFiK8K8eM7My8QD6JlOpVPYljVgQcHZra+upVccOAH4OjGhtbZ3v3rOVR6VSGQr8FliHlGPeE/gaaaTQXNLwzj1aW1tnNqyR1mecKmkilUplHVLQ3pg0pXzrSqWyej42ijTb7tnGtdAa6DTgr62trWuSPhsPk8ZTr9Pa2roeKUV1SBfPt37EPe7m8hFgQmtr6+sAlUrlH8B2wInAKaRc9dWNa541QqVSWRz4JGkiDa2trXNJvezqRcUmADv0eeOsIeoeuPMJlU4T5xGxeGfHjAeB4yqVynDScLOtgLsrlco2wHOtra33VSq9nbdjJbYq6aT4eZVK5aOkk9b7tra2zql6zJ6kk9o2ANQ9VRIRi+XgfCpwMGng/gqks/C9GmkxULS2tj5MOqN/A2lCx32kIXaHkUYX2MA0GBgNnNna2roBaaz/wW0HK5XKYaTPyUWNaZ71tcJGlUiaGBEf725fu+NjgbEAR5/4yw133G3PQtpWFheM+yVDhw3nsvG/ZcGFFgJgxvQXGTZ8BCf9ZjxLDu9uUmL/s+Lwhbt/UD8zY8Z0dvvmjvzl+r8DcM+kuzn3t+M448xxXHP1VVx+6SWc9dvfMWTIkAa3tHGGLPDeRK/el7HBPjUHwzf+dcZ81zc/isxxvyNpF+ASUupkZ7qZEBAR44BxAK3TXh+Q4xRnvvIyQ5ccxvQXnueOf/6dn//6fLbZ4b35F3xnx604+ayLWHzokg1spfWlpZYawbLLLsvTTz3JyqusysQJd7Dqaqtx26238Ltzzua3v7twQAftuil0VeH6KjJwf5N0Jvw0UuC+jaoZYNaxE356ALNem8mgwYP5/o8OZtHFfErA4KBDf8qhBx3A22+/zchRozjm2OPZZacdmDt3Lt//7h4ArLfeRzn8yGMa3NISU0M70T3StBNwBmqP27o2EFMl1r26pEo22q/2VMndpzQ0yhf220DSGpJulPRgvr+epMOLqs/MbL5ItW8NVmRS52zShIC3AfK6wp2tV2xm1lhqqX1rsCJz3AtHxJ364LdTM19X0cwGspZBjW5BzYoM3DMkrUaejCNpB3pwFRAzsz7VBCmQWhUZuPcmDe1bU9Jz5IuXFlifmVnvNUEKpFaFBe6IeBL4XF7YvSVfl8/MrDm5xw2ShgNHAp8AQtKtpAuvvlRUnWZmvVaiHneRLb2EtDDO9qRVy6bjRXDMrFmVaDhgkTnuYRFxbNX9n0nq6oK7ZmaN01KeVa6L7HHfJGknSS15+wbw5wLrMzPrvRbVvjVYketxi3SB3fH50CDSlaePrHedZmbzrUQ57roH7ohYrN5lmpkVrgly17Uqose9ZkQ8Iml0R8cj4p5612lmNt8Gco+blB4ZC5zUwbEAPlNAnWZm82cgT3mPiLGSWoDDI+K2epdvZlaIEqVKCvltEBHvAr8oomwzs0KUaHXAIltwvaTtpRJ9jZnZwOUJOEDKdS8CzJP0Jml4YOQrwJuZNZcm6EnXqshFpjws0MzKowl60rUq8tJl20laour+UE95N7Om1TK49q3RTS2w7CMj4tW2OxExE8+aNLNm5Rw30PGXQuO/qszMOlKiHHeRLb1b0smSVpO0qqRTgEkF1mdm1nsl6nEXGbh/AMwlrcF9GfAm6XJmZmbNp07juCWNknSTpIclPSRp33bHD5AUkpbK9yXpdEmPS7q/s+VCqhU5qmQOcHBu2CBgkbzPzKz51K8nPQ/YPyLukbQYMEnSDRHxb0mjgM8Dz1Y9/svA6nn7OHBm/n+nihxV8ntJi+drTj4EtEo6sKj6zMzmR0tLS81bVyLi+bbF9PK1dh8GRubDpwA/Ia3b1GYMcEEkE4Chkpbrsq29fI21WCsiXgO2Ba4FVgR2K7A+M7PeUw+2WouUVgY2ACZK2gZ4LiLua/ewkcDkqvtTeD/Qd6jIUR4LSFqAFLjPiIi3JUV3TzIza4SerM4haSxpFdQ24yJiXLvHLApcAfyIlD45DPhCR8V1sK/LWFlk4D4LeBq4D7hF0krAawXWZ2bWaz0J3DlIj+vseO60XgFcFBFXSloXWAW4L9ezAnCPpI1JPexRVU9fAZjaVf2FpUoi4vSIGBkRW+XczTPAp4uqz8xsfkiqeeumHAHnAA9HxMkAEfFARCwdEStHxMqkYD06IqYB1wDfyqNLNgFejYjnu6qjsB63pOGkmZKfIHX7bwWOAV4qqk4zs95S/S4CvDnpfN4Dku7N+w6NiGs7efy1wFbA48DrwB7dVVBkquQS4BZg+3x/F9KY7s8VWKeZWa/UawXqiLiVbk5h5l532+2gh3NcigzcwyLi2Kr7P/MiU2bWrMp06YAihwPeJGknSS15+wbw5wLrMzPrtXrluPtCEVd5n0XKaYt0MYUL86EWYDZeIdDMmlAzBORaFXGxYF9AwczKpzxxu9BRJZ/saH9E3FJUnWZmvdXdVPZmUuTJyep1SRYCNiYt6/qZAus0M+uVAZ0qaRMRX62+n1fFOrGo+szM5kt54nafXpFmCrBOH9ZnZlYz97gBSb/k/YVSWoD1SeuWmJk1HQfu5O6q2/OAiyPitgLrMzPrNZ+cBCLifHhvlax1gOeKqsvMbL6Vp8Nd/5mTkn4jae18ewlSeuQC4F+Sdq53fWZm9VCmmZNF/DbYIiIeyrf3AB6NiHWBDUmX7DEzazplCtxFpErmVt3+POkK70TEtGZ4wWZmHSlTfCoicM+UtDUpp705sBeApMHAkALqMzObf+WJ24UE7u8BpwPLAj/KV3gA+CxeHdDMmtSAHlUSEY8CX+pg/3XAdfWuz8ysHgZ6qsTMrHQcuM3MyqY8cbu4K+BIWqWWfWZmzaBMwwGLzMZf0cG+ywusz8ys18oUuIu4dNmawNrAEpK+VnVocdK63GZmTaelpfEBuVZF5LgrwNbAUKB6Te5ZwHcLqM/MbL41QUe6ZkUMB7wauFrSphFxR73LNzMrQjOkQGpVZI57sqSrJL0o6QVJV0haocD6zMx6Tap9a7QiA/d5wDXA8sBI4I95n5lZ02lpUc1boxUZuJeOiPMiYl7efgeMKLA+M7Nec+BOpkvaVdKgvO0KvFRgfWZmveZUSbIn8A1gGvA8sEPeZ2bWdAb0OO42EfEssE1R5ZuZ1VMzBORaFTEB54guDkdEHFvvOs3M5leJ4nYhPe45HexbhHRBheGAA7eZNZ0B3eOOiJPabktaDNiXdO3JS4CTOnuemVkjNcNokVoVcnJS0jBJPwPuJ305jI6IgyLixSLqMzObX/UaVSJplKSbJD0s6SFJ++b9wyTdIOmx/P8l835JOl3S45LulzS6u7bWPXBL+jlwF2ltknUj4qiIeKXe9ZiZ1VMdR5XMA/aPiI8AmwB7S1oLOBi4MSJWB27M9wG+DKyet7HAmd1VUESPe3/SbMnDgamSXsvbLEmvFVCfmdl8q1ePOyKej4h78u1ZwMOk2eNjgPPzw84Hts23xwAXRDIBGCppua7qKCLHXZ4rbpqZZUWcnJS0MrABMBFYJiKehxTcJS2dHzYSmFz1tCl53/Odlesga2ZGz3rcksZKurtqG/uf5WlR0gVlfhQRXWUbOvrGiK7a6mtOmpnRs1ElETEOGNfZcUkLkIL2RRFxZd79gqTlcm97OaBtsMYUYFTV01cApnbZ1ppbambWj9Xr5KTSA84BHo6Ik6sOXQPsnm/vDlxdtf9beXTJJsCrbSmVzrjHbWZGXWdObg7sBjwg6d6871DgBOBSSXsBzwJfz8euBbYCHgdeJ8176ZIDt5kZ9Ts5GRG30nHeGuCzHTw+gL17UocDt5kZA3zKu5lZGZVpyrsDt5kZXh3QzKx0nCoxMyuZEsVtB24zM4CWEkVuB24zM9zjNjMrnUEeVWJmVi4+OWlmVjIlitsO3GZmAOp0lnrzceA2MwNKlOJ24DYzA+e4zcxKx6NKzMxKpkQdbgduMzNwqsTMrHRKFLcduM3MoJ+sVSLpj3RxifiI2KaQFpmZNUC/CNzAL/qsFWZmDVaiQSWdB+6I+EdfNsTMrJH61clJSasDxwNrAQu17Y+IVQtsl5lZnypR3KalhsecB5wJzAM+DVwAjC+yUWZmfU1SzVuj1RK4h0TEjYAi4pmIOAr4TLHNMjPrWy2qfWu0WoYDvimpBXhM0j7Ac8DSxTbLzKxvlWlUSS097h8BCwM/BDYEdgN2L7JRZmZ9rUWqeWu0bnvcEXFXvjkb2KPY5piZNUYTxOOa1TKq5CY6mIgTEc5zm1m/0QwnHWtVS477gKrbCwHbk0aYmJn1GyWK2zWlSia123WbJE/OMbN+pRly17WqJVUyrOpuC+kE5bKFtShbaamFi67CSmjJj+3T6CZYE3rjX2fMdxktzTDOr0a1pEomkXLcIqVIngL2KrJRZmZ9rZYhds2ilsD9kYh4s3qHpAULao+ZWUOU6eRkLV8yt3ew7456N8TMrJHqOXNS0rmSXpT0YLv9P5DUKukhSSdW7T9E0uP52Be7K7+r9biXBUYCQyRtQEqVACxOmpBjZtZv1DnF/TvgDNLaTgBI+jQwBlgvIt6StHTevxawE7A2sDzwN0lrRMQ7nRXeVarki8C3gRWAk3g/cL8GHNrLF2Nm1pTqeZX3iLhF0srtdv8XcEJEvJUf82LePwa4JO9/StLjwMZ0kdnoaj3u84HzJW0fEVf0/iWYmTW/nqS4JY0FxlbtGhcR47p52hrAFpKOA94EDsgz00cCE6oeNyXv61QtJyc3lHRjRMzMDV4S2D8iDq/huWZmpdCTcdw5SHcXqNsbDCwJbAJ8DLhU0qq8n834QBVdFVTLyckvtwVtgIh4Bdiq9raamTW/lh5svTQFuDKSO4F3gaXy/lFVj1sBmNpdW7szqHr4n6QhgIcDmlm/ItW+9dL/ka9lIGkN4EPADOAaYCdJC0paBVgduLOrgmpJlVwI3CjpvHx/D+D8XjbczKwp1XPKu6SLgS2BpSRNAY4EzgXOzUME5wK7R0QAD0m6FPg3aZLj3l2NKIHa1io5UdL9wOdIuZi/Aiv1/iWZmTWfQXWcOhkRO3dyaNdOHn8ccFyt5dfS4waYRsrHfIM05d2jTMysX+kXi0zlHMxOwM7AS8AfSNed/HQftc3MrM+UKG532eN+BPgn8NWIeBxA0n590iozsz5WosUBuxxVsj0pRXKTpLMlfZaOxxuamZWeevBfo3UauCPiqojYEVgTuBnYD1hG0pmSvtBH7TMz6xODW2rfGq3bJkTEnIi4KCK2Jg0Mvxc4uPCWmZn1IUk1b43Wo++OiHg5Is7yhYLNrL+p57KuRat1OKCZWb/WBB3pmjlwm5nRT8Zxm5kNJM2QAqmVA7eZGTDIPW4zs3IpUdx24DYzA6dKzMxKxycnzcxKpkRx24HbzAzc4zYzK51B5YnbDtxmZkBTrEFSKwduMzPKtWa1A7eZGc5xm5mVTnnCtgO3mRkALSWagePAbWZGDy9O0GAO3GZmeFSJmVnplCdsO3CbmQHucZuZlY5z3GZmJeNx3GZmJVOiuO3AbWYG0FKi05MO3GZmuMdtZlY6KlGPu0wnUs3MCiPVvnVfls6V9KKkB6v2/VzSI5Lul3SVpKFVxw6R9LikVklf7K58B24zM2CQVPNWg98BX2q37wZgnYhYD3gUOARA0lrATsDa+Tm/ljSoq8IduM3MqG+POyJuAV5ut+/6iJiX704AVsi3xwCXRMRbEfEU8DiwcVflO3CbmZFy3LX+Vwd7An/Jt0cCk6uOTcn7OuXAbWYGtKj2TdJYSXdXbWNrrUfSYcA84KK2XR08LLoqw6NKzMzo2aiSiBgHjOtxHdLuwNbAZyOiLThPAUZVPWwFYGpX5bjHbWZGmvJe69Ybkr4EHARsExGvVx26BthJ0oKSVgFWB+7sqiz3uM3MSCmQepF0MbAlsJSkKcCRpFEkCwI35JUIJ0TE9yPiIUmXAv8mpVD2joh3uirfgdvMjPpOwImInTvYfU4Xjz8OOK7W8h24zczwlHczs9IpUdx24DYzA6/HbWZWOiWK2w7cZmZQrtUBHbjNzHCP28ysdEoUtx24zcyAUkVuB24zM5zjNjMrnXpOeS+aA7eZGThVYmZWNk6VmJmVjIcDmpmVTInitgO3mRmAStTlduA2M8OpEjOz0ilR3HbgNjMDShW5HbjNzPBwQDOz0nGO28ysZBy4zcxKxqkSM7OScY/bzKxkShS3HbjNzIBSRW4HbjMzoKVEuRIHbjMzStXhduA2MwNKFbkduM3M8HBAM7PSKVGK24HbzAxKlSlx4DYzA19IwcysdEoUt2lpdAPMzJqBerB1W5a0n6SHJD0o6WJJC0laRdJESY9J+oOkD/W2rQ7cZmakHnetW9flaCTwQ2CjiFgHGATsBPwvcEpErA68AuzV27Y6cJuZAfXtczMYGCJpMLAw8DzwGeDyfPx8YNvettSB28yM+vW4I+I54BfAs6SA/SowCZgZEfPyw6YAI3vbVgduMzOgRbVvksZKurtqG9tWjqQlgTHAKsDywCLAlzuoMnrbVo8qMTOjZzMnI2IcMK6Tw58DnoqI6QCSrgQ2A4ZKGpx73SsAU3vbVve4zcygninuZ4FNJC2sNDj8s8C/gZuAHfJjdgeu7m1THbjNzKhf3I6IiaSTkPcAD5Di7DjgIODHkh4HhgPn9LatTpWYmVHfCTgRcSRwZLvdTwIb16N8B24zMzzl3cysdMoTth24zcyAcq1V4sBtZoYvpGBmVjpl6nF7OKCZWcm4x21mBrSUqMvtwG1mRrlSJQ7cZmZ4OKCZWfmUKHI7cDeh1157jaOPOJzHH38USRx97P/wwgvTOPNXZ/DUk09w0SWXsfY66za6mVagFZYZym+P/RbLDF+cdyM494rb+NXFNzP+hD1YfeVlABi62BBmznqDTXY6AYAD9vwC3x6zKe+8+y77n3g5f7vj4Ua+hNLxcECbLycefxybf2ILTjr1dN6eO5c33nyTxRZbnFNO+yXHHt1++QPrj+a98y4Hn3wl9z4yhUUXXpDbf38QN058hN0OPu+9x5zw4+14dfYbAKy56rJ8/YujGb3DcSw3Ygmu/c0+rLvtMbz7bq+XfB5wypTj9nDAJjN79mwmTbqL7bZPqz8u8KEPsfjii7Pqaqux8iqrNrh11lemzXiNex+ZAsDs19/ikaemsfyIoR94zPafH82lf50EwNZbrsdl193D3Lfn8czUl3hi8gw+ts7Kfd3sUqvXFXD6QiE9bkk/7up4RJxcRL39wZTJk1lyyWEccdghtLY+wlprr81PDj6MhRdeuNFNswZZcblhrF9ZgbsefPq9fZuPXo0XXp7FE89OB2DkiCWY+MD7x5978RWWX3qJPm5puZUpVaKI+v+UktT2e74CfAy4Jt//KnBLRHynk+eNBdouATQuX2ViQKlUKhsBE4DNW1tbJ1YqldPmzJmz2pQpU7bOx28GDmhtbb27ke20PrMo8A/gOODKqv1nXn755UvtsMMOX8/3fwXcAVyY758DXAtc0VcNtb5TSI87Io4GkHQ9MDoiZuX7RwGXdfG8ri4HNFBMAaa0trZOzPcvl/THRjbIGmYBUuC9iA8G7cHA1w466KBpO+zQdkEVpgCjqh4zX5fGsuZWdI57RWBu1f25wMoF11lqra2t04DJlUqlknd9du7cuW82sk3WECL1mh8G2qcWPwc88uSTT75dte8aYCdgQdJFalcH7uyDdloDFD2qZDxwp6SrSFc03g64oOA6+4MfABdVKpUPAU++9NJLz1cqle2AXwIjgD9XKpV7W1tbv9jQVlqRNgd2I1366t6871BS+mMn4GKgOuX4EHAp6dqG84C9gXf6qrHWtwrJcX+gAmk0sEW+e0tE/KvQCvshSWMHYr7fuubPxcDVF4H7E8DqEXGepBHAohHxVKGVmpn1Y4UG7jy6ZCOgEhFrSFoeuCwiNi+sUjOzfq7ok5PbAdsAcwAiYiqwWMF1mpn1a0UH7rmRuvQBIGmRguubb5JC0klV9w/Iwxi7es62ktbq5FhF0s2S7pX0sKRuc5KSZve44d2X+du2Ns5v+UW0rz+RdJikhyTdn//uH+/isUdJOqDO9W8k6fR6lF9E+2z+FT2q5FJJZwFDJX0X2BM4u+A659dbwNckHR8RM2p8zrbAn0hn9Ns7HTglIq4GkNTnq0NJGtTZpCerL0mbAluT5i+8JWkp4EN9WP/giLgb8AStfqzQHndE/AK4nDSJYA3giIj4ZZF11sE80iSg/dofkLSSpBtzT+pGSStK2oyUDvp57l2t1u5py5EmRwAQEQ/ksr4t6Yyqsv8kacuq+ydJuifXMyLv+6Gkf3WP0gUAAAjaSURBVOf6L8n7FpV0nqQH8v7t8/7Zko6RNBHYNPf6N+qm/NUk/VXSJEn/lLRm3r+KpDsk3SXp2Pl5cweA5YAZEfEWQETMiIipkp7OQbytR3xz1XM+Kunvkh7LHRwkLSfplvyZelDSFnn/l/Lf7T5JN+Z9R0kalye8XSBpS0l/6qr8/LwD89/0fklHV+0/TFKrpL+RZj9bk+mLRaYeAP4J3JJvl8GvgF0ktV/s4QzggohYjzSb7fSIuJ00+eHAiFg/Ip5o95xTgL9L+ouk/SQNpXuLAPdExGjSdOe2JQQOBjbI9X8/7/sp8GpErJv3/72qjAcj4uMRcWuN5Y8DfhARGwIHAL/O+08DzoyIjwHTamj/QHY9MErSo5J+LelTNTxnPeArwKbAEfkk/jeB6yJifeCjwL35C/ZsYPuI+Cjw9aoyNgTGRMQ3aylf0hdIk3Q2BtYHNpT0SUkbksaJbwB8jbRkhTWZQgO3pO+QZm99DdgBmCBpzyLrrIeIeI00UeiH7Q5tCvw+3x4PfKKGss4DPkKa6r8l6T1YsJunvQv8Id++sKqe+4GLJO1K+mUAaRbdr6rqeyXffIfO16n4j/IlLQpsBlwm6V7gLFLvEdJkkIvz7fHdtH1Ai4jZpCA6FpgO/EHSt7t52tUR8UZOzd1ECqZ3AXsonV9ZNy8bsQlpLsRTua6Xq8q4JiLe6EH5X8jbv4B7gDVJgXwL4KqIeD3/O7imkzKtgYrOcR9I6iG+BCBpOHA7cG7B9dbDqaQP9HldPKamsZR5NM25wLmSHgTWIQXe6i/OhWqo5yvAJ0mpmZ9KWps0NbqjdrwZEbXOnIvclpm5h9dVG6wb+X2/GbhZ0gPA7nzw793+b93+vY2IuEXSJ0l/8/GSfg7M7OCxbeZ01aQO7gs4PiLOqj4g6Udd1GFNouhUyRRgVtX9WcDkguusi9ybuRTYq2r37aSfkQC7AG0piFl0Mswx5yQXyLeXBYYDzwFPA+tLapE0itQLatNC+oUC6SfzrZJagFERcRPwE2AoaeW464F9qupbsoaX9x/l597VU5K+nsuRpI/mx9zW7nVbJ5RGEa1etWt94BnS33vDvG/7dk8bI2mh3LHZErhL0krAixFxNmnNktGk1f8+JWmVXNewGpv1H+UD1wF75l9aSBopaWlSSnM7SUMkLUZa0dOaTNHrcT8HTJR0NelbfAzlWvjmJKqCIil1cq6kA0k/g/fI+y8Bzpb0Q2CHdnnuLwCnSWpbKOrAiJgm6QXgKVLe/0FS777NHGBtSZOAV4EdgUHAhTnvLtJIlZmSfgb8Kvfk3wGO5oMryXWko/IhBeUzJR1OWpnuEuA+YF/g95L2xcuEdmdR4Jf5XMY84HFS2uQjwDmSDgUmtnvOncCfSYuyHZtPZu4OHCjpbWA28K2ImK609PGV+Yv8ReDzNbTpP8oHpkr6CHCH0pUBZgO7RsQ9kv5AWh/lGdL5KWsyRa/H3aG2ZV/NzKznCl+rxMzM6quoVMmpEfEjpQsA/Mc3Q0RsU0S9ZmYDQVGjStqGjP2ioPLNzAaswlIlkgYB50fEroVUYGY2QBU2HDCPZR0hqc/WaTAzGwiKnoDzNHCbpGuomiAQEe2voWf9iKR3SMMcB5Oumbh7RLzey7K2BA6IiK0lbQOsFREndPLYocA3I+LXHR3voo6jgNl5bR2zplf0BJyppFXzWkgTVNo269/eyOu2rEO6QPT3qw/myT09/uxFxDWdBe1sKPDfPS3XrGwK7XF7vLaRJnCsJ2ll4C+ktTI2BbaVVCFNGFoQeALYIyJmS/oSacmBGVRNTMprfmwUEftIWgb4DbBqPvxfpAlSq+W1Vm6IiAPzZKlv5Dquiogjc1mHAd8izeSdDkwq7B0wq7OiF5m6oXo1PElLSrquyDqteUgaDHyZ91eFrJBWV9yAlDo7HPhcXqXwbuDHkhYirYD3VdKCR8t2UvzpwD/yKnmjSVc5Pxh4Ivf2D/QKeNZfFZ3jHhERM9vuRMQreT0E69+G5F4vpB73OcDywDMRMSHv3wRYi3QOBNLFBu4grVL3VEQ8BiDpQtKU8fY+Q+oxt50If7WDdVqqV8CDNB19dVK67qq2vHs+B2NWGkUH7nckrRgRz0K6EAFeeWwgeKP9KoM5OFevYCdSOmPndo9bn/p9RrwCnvVLRZ+cPIy0st14SeNJK48dUnCdVg4TgM0lfRhA0sKS1gAeAVbR+1cS2rmT599IymsjaZCkxfnPVRq9Ap71S0WfnPyrpNGkn8UC9uvBdRytH8sr3X0buLjqwhKHR8SjeQW8P0uaQVo6d50OitgXGCdpL9KqiP8VEXdIui2vlPiXnOf2CnjW7xS6yJSkzYF7I2KO0lVbRgOnRcQzhVVqZtbPFZ0qORN4PS/IfyCpd3NBwXWamfVrRQfueZG69GNIF9Y9DU/AMTObL0WPKpkl6RBgN2CLvPDUAgXXaWbWrxXd494ReAvYMyKmASOBnxdcp5lZv1b4FXDyBXI3Jo2bvSsHcDMz66Wip7x/h3Sh0q+Rrio+QdKeRdZpZtbfFT0csBXYLCJeyveHA7dHRKWwSs3M+rmic9xTSLPZ2swircZmZma9VNTFgn+cbz4HTJR0NSnHPYaUOjEzs14qajhg21jtJ/LW5uqC6jMzGzAKH1ViZmb1VegEHEk30cHymRHxmSLrNTPrz4qeOXlA1e2FgO2BeQXXaWbWr/V5qkTSPyLiU31aqZlZP1J0qmRY1d0WYCM6v4agmZnVoOhUySTez3HPA54G9iq4TjOzfq2ocdwfAyZHxCr5/u6k/PbTwL+LqNPMbKAoaubkWcBcAEmfBI4HzgdeBcYVVKeZ2YBQVKpkUES8nG/vCIyLiCuAKyTdW1CdZmYDQlE97kGS2r4UPgv8vepY0Xl1M7N+raggejHwj3yV7jfIV9GW9GFSusTMzHqpsHHckjYBlgOuj4g5ed8awKIRcU8hlZqZDQBeq8TMrGSKXo/bzMzqzIHbzKxkHLjNzErGgdvMrGQcuM3MSub/AYruNarvHvOUAAAAAElFTkSuQmCC\n",
      "text/plain": [
       "<Figure size 432x288 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# calling the visual_cm function for visualized confusion matrix\n",
    "visual_cm(true_y = y_test,\n",
    "          pred_y = logreg_pred,\n",
    "          labels = ['Not Subscribed', 'Subscribed'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Correct predictions: 363\n",
      "Incorrect predictions: 124\n",
      "Accuracy: 0.7453798767967146\n",
      "Misclassification Rate: 0.2546201232032854\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.61      0.60      0.60       156\n",
      "           1       0.81      0.82      0.81       331\n",
      "\n",
      "    accuracy                           0.75       487\n",
      "   macro avg       0.71      0.71      0.71       487\n",
      "weighted avg       0.75      0.75      0.75       487\n",
      "\n"
     ]
    }
   ],
   "source": [
    "# Calculating correct and incorrect predictions\n",
    "\n",
    "correct_predictions = 99 + 264\n",
    "incorrect_predictions = 57 + 67\n",
    "accuracy = (99 + 264) / 487\n",
    "misclassification_rate = (57 + 67) / 487\n",
    "\n",
    "print(\"Correct predictions:\", correct_predictions)\n",
    "print(\"Incorrect predictions:\", incorrect_predictions)\n",
    "print(\"Accuracy:\", accuracy)\n",
    "print(\"Misclassification Rate:\", misclassification_rate)  \n",
    "\n",
    "# Printing the classification report\n",
    "print(classification_report(y_test, logreg_pred))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Findings:**  \n",
    "\n",
    "The result is telling us that we have 363 (75%) correct predictions and 124 (25%) incorrect predictions.   \n",
    "\n",
    "The classifier has an Accuracy of 75% -- how often the classifier is correct.  \n",
    "The classifier has a True Positive Rate/Sensitivity/Recall of 80% -- when it's actually yes, how often it predicts yes.  \n",
    "The classifier has a True Negative Rate/Specificity of 63% -- when it's actually no, how often it predicts no.  \n",
    "The classifier has a Precision of 82% -- when it predicts yes, how often it is correct.  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['Model', 'Training Accuracy', 'Testing Accuracy', 'AUC Value']\n",
      "['Logistic Regression', 0.7601, 0.7474, 0.7091]\n"
     ]
    }
   ],
   "source": [
    "# creating an empty list\n",
    "model_performance = [['Model', 'Training Accuracy',\n",
    "                      'Testing Accuracy', 'AUC Value']]\n",
    "\n",
    "\n",
    "# train accuracy\n",
    "logreg_train_acc  = logreg_fit.score(X_train, y_train).round(4)\n",
    "\n",
    "\n",
    "# test accuracy\n",
    "logreg_test_acc   = logreg_fit.score(X_test, y_test).round(4)\n",
    "\n",
    "\n",
    "# auc value\n",
    "logreg_auc = roc_auc_score(y_true  = y_test,\n",
    "                           y_score = logreg_pred).round(4)\n",
    "\n",
    "\n",
    "# saving the results\n",
    "model_performance.append(['Logistic Regression',\n",
    "                          logreg_train_acc,\n",
    "                          logreg_test_acc,\n",
    "                          logreg_auc])\n",
    "\n",
    "\n",
    "# checking the results\n",
    "for model in model_performance:\n",
    "    print(model)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h3>KNN Classification Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {},
   "outputs": [],
   "source": [
    "# creating lists for training set accuracy and test set accuracy\n",
    "training_accuracy = []\n",
    "test_accuracy = []\n",
    "\n",
    "\n",
    "# building a visualization of 1 to 50 neighbors\n",
    "neighbors_settings = range(1, 21)\n",
    "\n",
    "\n",
    "for n_neighbors in neighbors_settings:\n",
    "    # Building the model\n",
    "    clf = KNeighborsClassifier(n_neighbors = n_neighbors)\n",
    "    clf.fit(X_train, y_train)\n",
    "    \n",
    "    # Recording the training set accuracy\n",
    "    training_accuracy.append(clf.score(X_train, y_train))\n",
    "    \n",
    "    # Recording the generalization accuracy\n",
    "    test_accuracy.append(clf.score(X_test, y_test))\n",
    "\n",
    "\n",
    "# plotting the visualization\n",
    "#fig, ax = plt.subplots(figsize=(12,8))\n",
    "#plt.plot(neighbors_settings, training_accuracy, label = \"training accuracy\")\n",
    "#plt.plot(neighbors_settings, test_accuracy, label = \"test accuracy\")\n",
    "#plt.ylabel(\"Accuracy\")\n",
    "#plt.xlabel(\"n_neighbors\")\n",
    "#plt.legend()\n",
    "#plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The optimal number of neighbors is 9\n"
     ]
    }
   ],
   "source": [
    "# finding the optimal number of neighbors\n",
    "opt_neighbors = test_accuracy.index(max(test_accuracy)) + 1\n",
    "print(f\"\"\"The optimal number of neighbors is {opt_neighbors}\"\"\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training ACCURACY: 0.8026\n",
      "Testing  ACCURACY: 0.7721\n",
      "AUC Score        : 0.7459\n"
     ]
    }
   ],
   "source": [
    "# INSTANTIATING a KNN classification model with optimal neighbors\n",
    "knn_opt = KNeighborsClassifier(n_neighbors = opt_neighbors)\n",
    "\n",
    "\n",
    "# FITTING the training data\n",
    "knn_fit = knn_opt.fit(X_train, y_train)\n",
    "\n",
    "\n",
    "# PREDICTING based on the testing set\n",
    "knn_pred = knn_fit.predict(X_test)\n",
    "\n",
    "\n",
    "# SCORING the results\n",
    "print('Training ACCURACY:', knn_fit.score(X_train, y_train).round(4))\n",
    "print('Testing  ACCURACY:', knn_fit.score(X_test, y_test).round(4))\n",
    "print('AUC Score        :', roc_auc_score(y_true  = y_test,\n",
    "                                          y_score = knn_pred).round(4))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h4>Confusion Matrix (KNN)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[105  51]\n",
      " [ 60 271]]\n"
     ]
    }
   ],
   "source": [
    "# creating a confusion matrix\n",
    "print(confusion_matrix(y_true = y_test,\n",
    "                       y_pred = knn_pred))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# calling the visual_cm function\n",
    "visual_cm(true_y = y_test,\n",
    "          pred_y = knn_pred,\n",
    "          labels = ['Not Subscribed', 'Subscribed'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Correct predictions: 391\n",
      "Incorrect predictions: 96\n",
      "Accuracy: 0.8028747433264887\n",
      "Misclassification Rate: 0.1971252566735113\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.64      0.67      0.65       156\n",
      "           1       0.84      0.82      0.83       331\n",
      "\n",
      "    accuracy                           0.77       487\n",
      "   macro avg       0.74      0.75      0.74       487\n",
      "weighted avg       0.78      0.77      0.77       487\n",
      "\n"
     ]
    }
   ],
   "source": [
    "# Calculating correct and incorrect predictions\n",
    "\n",
    "correct_predictions = 102 + 289\n",
    "incorrect_predictions = 42 + 54\n",
    "accuracy = (102 + 289) / 487\n",
    "misclassification_rate = (42 + 54) / 487\n",
    "\n",
    "print(\"Correct predictions:\", correct_predictions)\n",
    "print(\"Incorrect predictions:\", incorrect_predictions)\n",
    "print(\"Accuracy:\", accuracy)\n",
    "print(\"Misclassification Rate:\", misclassification_rate)  \n",
    "\n",
    "# Printing the classification report\n",
    "print(classification_report(y_test, knn_pred))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Findings:**  \n",
    "\n",
    "The result is telling us that we have 391 (80%) correct predictions and 96 (20%) incorrect predictions.   \n",
    "\n",
    "The classifier has an Accuracy of 80% -- how often the classifier is correct.  \n",
    "The classifier has a True Positive Rate/Sensitivity/Recall of 87% -- when it's actually yes, how often it predicts yes.  \n",
    "The classifier has a True Negative Rate/Specificity of 65% -- when it's actually no, how often it predicts no.  \n",
    "The classifier has a Precision of 84% -- when it predicts yes, how often it is correct.  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['Model', 'Training Accuracy', 'Testing Accuracy', 'AUC Value']\n",
      "['Logistic Regression', 0.7601, 0.7474, 0.7091]\n",
      "['KNN Classification', 0.8026, 0.7721, 0.7459]\n"
     ]
    }
   ],
   "source": [
    "# train accuracy\n",
    "knn_train_acc = knn_fit.score(X_train, y_train).round(4)\n",
    "\n",
    "\n",
    "# test accuracy\n",
    "knn_test_acc  = knn_fit.score(X_test, y_test).round(4)\n",
    "\n",
    "\n",
    "# auc value\n",
    "knn_auc       = roc_auc_score(y_true  = y_test,\n",
    "                              y_score = knn_pred).round(4)\n",
    "\n",
    "\n",
    "# saving the results\n",
    "model_performance.append(['KNN Classification',\n",
    "                          knn_train_acc,\n",
    "                          knn_test_acc,\n",
    "                          knn_auc])\n",
    "\n",
    "\n",
    "# checking the results\n",
    "for model in model_performance:\n",
    "    print(model)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h3>CART Models"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h4>Decision Tree Classifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training ACCURACY: 0.9993\n",
      "Testing  ACCURACY: 0.7413\n",
      "AUC Score        : 0.7012\n"
     ]
    }
   ],
   "source": [
    "# INSTANTIATING a classification tree object\n",
    "full_tree = DecisionTreeClassifier()\n",
    "\n",
    "\n",
    "# FITTING the training data\n",
    "full_tree_fit = full_tree.fit(X_train, y_train)\n",
    "\n",
    "\n",
    "# PREDICTING on new data\n",
    "full_tree_pred = full_tree_fit.predict(X_test)\n",
    "\n",
    "\n",
    "# SCORING the model\n",
    "print('Training ACCURACY:', full_tree_fit.score(X_train, y_train).round(4))\n",
    "print('Testing  ACCURACY:', full_tree_fit.score(X_test, y_test).round(4))\n",
    "print('AUC Score        :', roc_auc_score(y_true  = y_test,\n",
    "                                          y_score = full_tree_pred).round(4))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {},
   "outputs": [],
   "source": [
    "##############################################\n",
    "# Creating user-defined fucntion: display_tree\n",
    "##############################################\n",
    "def display_tree(tree, feature_df, height = 500, width = 800):\n",
    "\n",
    "    # visualizing the tree\n",
    "    dot_data = StringIO()\n",
    "\n",
    "    \n",
    "    # exporting tree to graphviz\n",
    "    export_graphviz(decision_tree      = tree,\n",
    "                    out_file           = dot_data,\n",
    "                    filled             = True,\n",
    "                    rounded            = True,\n",
    "                    special_characters = True,\n",
    "                    feature_names      = feature_df.columns)\n",
    "\n",
    "\n",
    "    # declaring a graph object\n",
    "    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())\n",
    "\n",
    "\n",
    "    # creating image\n",
    "    img = Image(graph.create_png(),\n",
    "                height = height,\n",
    "                width  = width)\n",
    "    \n",
    "    return img"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'\\n# calling display_tree\\ndisplay_tree(tree       = full_tree_fit,\\n            feature_df = X_train)\\n            \\n'"
      ]
     },
     "execution_count": 50,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "\"\"\"\n",
    "# calling display_tree\n",
    "display_tree(tree       = full_tree_fit,\n",
    "            feature_df = X_train)\n",
    "            \n",
    "\"\"\""
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Findings:**\n",
    "\n",
    "FOLLOWED_RECOMMENDATIONS_PCT variable has the most splitting power."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h4>Confusion Matrix (Decision Tree)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[ 92  64]\n",
      " [ 62 269]]\n"
     ]
    }
   ],
   "source": [
    "# creating a confusion matrix\n",
    "print(confusion_matrix(y_true = y_test,\n",
    "                       y_pred = full_tree_pred))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Correct predictions: 359\n",
      "Incorrect predictions: 128\n",
      "Accuracy: 0.7371663244353183\n",
      "Misclassification Rate: 0.26283367556468173\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.60      0.59      0.59       156\n",
      "           1       0.81      0.81      0.81       331\n",
      "\n",
      "    accuracy                           0.74       487\n",
      "   macro avg       0.70      0.70      0.70       487\n",
      "weighted avg       0.74      0.74      0.74       487\n",
      "\n"
     ]
    }
   ],
   "source": [
    "# Calculating correct and incorrect predictions\n",
    "\n",
    "correct_predictions = 105 + 254\n",
    "incorrect_predictions = 77 + 51\n",
    "accuracy = (105 + 254) / 487\n",
    "misclassification_rate = (77 + 51) / 487\n",
    "\n",
    "print(\"Correct predictions:\", correct_predictions)\n",
    "print(\"Incorrect predictions:\", incorrect_predictions)\n",
    "print(\"Accuracy:\", accuracy)\n",
    "print(\"Misclassification Rate:\", misclassification_rate)  \n",
    "\n",
    "# Printing the classification report\n",
    "print(classification_report(y_test, full_tree_pred))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Findings:**  \n",
    "\n",
    "The result is telling us that we have 359 (74%) correct predictions and 128 (26%) incorrect predictions.   \n",
    "\n",
    "The classifier has an Accuracy of 74% -- how often the classifier is correct.  \n",
    "The classifier has a True Positive Rate/Sensitivity/Recall of 77% -- when it's actually yes, how often it predicts yes.  \n",
    "The classifier has a True Negative Rate/Specificity of 67% -- when it's actually no, how often it predicts no.  \n",
    "The classifier has a Precision of 83% -- when it predicts yes, how often it is correct.  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['Model', 'Training Accuracy', 'Testing Accuracy', 'AUC Value']\n",
      "['Logistic Regression', 0.7601, 0.7474, 0.7091]\n",
      "['KNN Classification', 0.8026, 0.7721, 0.7459]\n",
      "['Full Tree', 0.9993, 0.7413, 0.7012]\n"
     ]
    }
   ],
   "source": [
    "# train accuracy\n",
    "full_tree_train_acc = full_tree_fit.score(X_train, y_train).round(4)\n",
    "\n",
    "\n",
    "# test accuracy\n",
    "full_tree_test_acc  = full_tree_fit.score(X_test, y_test).round(4)\n",
    "\n",
    "\n",
    "# auc value\n",
    "full_tree_auc       = roc_auc_score(y_true  = y_test,\n",
    "                                    y_score = full_tree_pred).round(4)\n",
    "\n",
    "\n",
    "# saving the results\n",
    "model_performance.append(['Full Tree',\n",
    "                          full_tree_train_acc,\n",
    "                          full_tree_test_acc,\n",
    "                          full_tree_auc])\n",
    "\n",
    "\n",
    "# checking the results\n",
    "for model in model_performance:\n",
    "    print(model)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h4>Pruned Decision Tree"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training ACCURACY: 0.806\n",
      "Testing  ACCURACY: 0.7823\n",
      "AUC Score        : 0.7755\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "'\\n# calling display_tree\\ndisplay_tree(tree       = tree_pruned_fit,\\n             feature_df = X_train)\\n             \\n'"
      ]
     },
     "execution_count": 54,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# PRUNING THE TREE\n",
    "\n",
    "# INSTANTIATING a classification tree object\n",
    "tree_pruned      = DecisionTreeClassifier(max_depth = 5,\n",
    "                                          min_samples_leaf = 25,\n",
    "                                          random_state = 222)\n",
    "\n",
    "\n",
    "# FITTING the training data\n",
    "tree_pruned_fit  = tree_pruned.fit(X_train, y_train)\n",
    "\n",
    "\n",
    "# PREDICTING on new data\n",
    "tree_pred = tree_pruned_fit.predict(X_test)\n",
    "\n",
    "\n",
    "# SCORING the model\n",
    "print('Training ACCURACY:', tree_pruned_fit.score(X_train, y_train).round(4))\n",
    "print('Testing  ACCURACY:', tree_pruned_fit.score(X_test, y_test).round(4))\n",
    "print('AUC Score        :', roc_auc_score(y_true  = y_test,\n",
    "                                          y_score = tree_pred).round(4))\n",
    "\n",
    "\"\"\"\n",
    "# calling display_tree\n",
    "display_tree(tree       = tree_pruned_fit,\n",
    "             feature_df = X_train)\n",
    "             \n",
    "\"\"\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['Model', 'Training Accuracy', 'Testing Accuracy', 'AUC Value']\n",
      "['Logistic Regression', 0.7601, 0.7474, 0.7091]\n",
      "['KNN Classification', 0.8026, 0.7721, 0.7459]\n",
      "['Full Tree', 0.9993, 0.7413, 0.7012]\n",
      "['Pruned Tree', 0.806, 0.7823, 0.7755]\n"
     ]
    }
   ],
   "source": [
    "# train accuracy\n",
    "p_tree_train_acc = tree_pruned_fit.score(X_train, y_train).round(4)\n",
    "\n",
    "\n",
    "# test accuracy\n",
    "p_tree_test_acc  = tree_pruned_fit.score(X_test, y_test).round(4)\n",
    "\n",
    "\n",
    "# auc value\n",
    "p_tree_auc       = roc_auc_score(y_true  = y_test,\n",
    "                                 y_score = tree_pred).round(4)\n",
    "\n",
    "\n",
    "# saving the results\n",
    "model_performance.append(['Pruned Tree',\n",
    "                          p_tree_train_acc,\n",
    "                          p_tree_test_acc,\n",
    "                          p_tree_auc])\n",
    "\n",
    "# checking the results\n",
    "for model in model_performance:\n",
    "    print(model)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h4>Confusion Matrix (Pruned Decision Tree)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[118  38]\n",
      " [ 68 263]]\n"
     ]
    }
   ],
   "source": [
    "# creating a confusion matrix\n",
    "print(confusion_matrix(y_true = y_test,\n",
    "                       y_pred = tree_pred))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Correct predictions: 378\n",
      "Incorrect predictions: 109\n",
      "Accuracy: 0.7761806981519507\n",
      "Misclassification Rate: 0.22381930184804927\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.63      0.76      0.69       156\n",
      "           1       0.87      0.79      0.83       331\n",
      "\n",
      "    accuracy                           0.78       487\n",
      "   macro avg       0.75      0.78      0.76       487\n",
      "weighted avg       0.80      0.78      0.79       487\n",
      "\n"
     ]
    }
   ],
   "source": [
    "# Calculating correct and incorrect predictions\n",
    "\n",
    "correct_predictions = 124 + 254\n",
    "incorrect_predictions = 77 + 32\n",
    "accuracy = (124 + 254) / 487\n",
    "misclassification_rate = (77 + 32) / 487\n",
    "\n",
    "print(\"Correct predictions:\", correct_predictions)\n",
    "print(\"Incorrect predictions:\", incorrect_predictions)\n",
    "print(\"Accuracy:\", accuracy)\n",
    "print(\"Misclassification Rate:\", misclassification_rate)  \n",
    "\n",
    "# Printing the classification report\n",
    "print(classification_report(y_test, tree_pred))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Findings:**  \n",
    "\n",
    "The result is telling us that we have 378 (78%) correct predictions and 109 (22%) incorrect predictions.   \n",
    "\n",
    "The classifier has an Accuracy of 78% -- how often the classifier is correct.  \n",
    "The classifier has a True Positive Rate/Sensitivity/Recall of 77% -- when it's actually yes, how often it predicts yes.  \n",
    "The classifier has a True Negative Rate/Specificity of 79% -- when it's actually no, how often it predicts no.  \n",
    "The classifier has a Precision of 89% -- when it predicts yes, how often it is correct.  "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h4>Plotting Feature Importances"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "metadata": {},
   "outputs": [],
   "source": [
    "##########################################################\n",
    "# Creating user-defined funtion: plot_feature_importances\n",
    "##########################################################\n",
    "def plot_feature_importances(model, train, export = False):\n",
    "    \n",
    "    # declaring the number\n",
    "    n_features = X_train.shape[1]\n",
    "    \n",
    "    # setting plot window\n",
    "    fig, ax = plt.subplots(figsize=(12,9))\n",
    "    \n",
    "    plt.barh(range(n_features), model.feature_importances_, align='center')\n",
    "    plt.yticks(pd.np.arange(n_features), train.columns)\n",
    "    plt.xlabel(\"Feature importance\")\n",
    "    plt.ylabel(\"Feature\")\n",
    "    \n",
    "    if export == True:\n",
    "        plt.savefig('Tree_Leaf_50_Feature_Importance.png')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 864x648 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# plotting feature importance\n",
    "plot_feature_importances(tree_pruned_fit,\n",
    "                         train = X_train,\n",
    "                         export = False)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Findings:**  \n",
    "\n",
    "FOLLOWED_RECOMMENDATIONS_PCT variable is the most important, followed by professional and personal e-mails."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h4>Tuned Decision Tree Classifier using GridSearchCV "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Tuned Parameters  : {'criterion': 'gini', 'max_depth': 8, 'min_samples_leaf': 4}\n",
      "Tuned CV AUC      : 0.6434\n"
     ]
    }
   ],
   "source": [
    "from sklearn.model_selection import GridSearchCV     # hyperparameter tuning\n",
    "from sklearn.metrics import make_scorer              # customizable scorer\n",
    "\n",
    "# declaring a hyperparameter space\n",
    "depth_space          = pd.np.arange(1, 10, 1)\n",
    "samples_leaf_space   = pd.np.arange(1,10,1)\n",
    "criterion_space      = ['gini']\n",
    "\n",
    "\n",
    "# creating a hyperparameter grid\n",
    "param_grid = {'criterion' : criterion_space,\n",
    "              'max_depth' : depth_space,\n",
    "              'min_samples_leaf' : samples_leaf_space\n",
    "              }\n",
    "\n",
    "\n",
    "# INSTANTIATING the model object without hyperparameters\n",
    "tree_tuned = DecisionTreeClassifier(random_state = 222)\n",
    "\n",
    "\n",
    "# GridSearchCV object (due to cross-validation)\n",
    "tree_tuned_cv = GridSearchCV(estimator  = tree_tuned,\n",
    "                           param_grid = param_grid,\n",
    "                           cv         = 3,\n",
    "                           scoring    = make_scorer(roc_auc_score,\n",
    "                                                    needs_threshold = False))\n",
    "\n",
    "\n",
    "# FITTING to the FULL DATASET\n",
    "tree_tuned_cv.fit(original_df_data, original_df_target)\n",
    "\n",
    "\n",
    "# printing the optimal parameters and best score\n",
    "print(\"Tuned Parameters  :\", tree_tuned_cv.best_params_)\n",
    "print(\"Tuned CV AUC      :\", tree_tuned_cv.best_score_.round(4))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=8,\n",
       "                       max_features=None, max_leaf_nodes=None,\n",
       "                       min_impurity_decrease=0.0, min_impurity_split=None,\n",
       "                       min_samples_leaf=4, min_samples_split=2,\n",
       "                       min_weight_fraction_leaf=0.0, presort=False,\n",
       "                       random_state=222, splitter='best')"
      ]
     },
     "execution_count": 61,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Calling for the best estimator\n",
    "tree_tuned_cv.best_estimator_"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training ACCURACY: 0.8485\n",
      "Testing  ACCURACY: 0.846\n",
      "AUC Score        : 0.8274\n"
     ]
    }
   ],
   "source": [
    "# building a model based on hyperparameter tuning results\n",
    "\n",
    "# INSTANTIATING a logistic regression model with tuned values\n",
    "tree_tuned = tree_tuned_cv.best_estimator_\n",
    "\n",
    "# PREDICTING based on the testing set\n",
    "tree_tuned_pred = tree_tuned.predict(X_test)\n",
    "\n",
    "# SCORING the results\n",
    "print('Training ACCURACY:', tree_tuned.score(X_train, y_train).round(4))\n",
    "print('Testing  ACCURACY:', tree_tuned.score(X_test, y_test).round(4))\n",
    "print('AUC Score        :', roc_auc_score(y_true  = y_test,\n",
    "                                          y_score = tree_tuned_pred).round(4))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h4>Plotting Feature Importance"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 864x648 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# plotting feature importance\n",
    "plot_feature_importances(tree_tuned,\n",
    "                         train = X_train,\n",
    "                         export = False)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h4>Confusion Matrix (Tuned Decision Tree)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[121  35]\n",
      " [ 40 291]]\n"
     ]
    }
   ],
   "source": [
    "# Creating a confusion matrix\n",
    "print(confusion_matrix(y_true = y_test,\n",
    "                       y_pred = tree_tuned_pred))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Correct predictions: 412\n",
      "Incorrect predictions: 75\n",
      "Accuracy: 0.8459958932238193\n",
      "Misclassification Rate: 0.1540041067761807\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.75      0.78      0.76       156\n",
      "           1       0.89      0.88      0.89       331\n",
      "\n",
      "    accuracy                           0.85       487\n",
      "   macro avg       0.82      0.83      0.82       487\n",
      "weighted avg       0.85      0.85      0.85       487\n",
      "\n"
     ]
    }
   ],
   "source": [
    "# Calculating correct and incorrect predictions\n",
    "\n",
    "correct_predictions = 121 + 291\n",
    "incorrect_predictions = 40 + 35\n",
    "accuracy = (121 + 291) / 487\n",
    "misclassification_rate = (40 + 35) / 487\n",
    "\n",
    "print(\"Correct predictions:\", correct_predictions)\n",
    "print(\"Incorrect predictions:\", incorrect_predictions)\n",
    "print(\"Accuracy:\", accuracy)\n",
    "print(\"Misclassification Rate:\", misclassification_rate)  \n",
    "\n",
    "# Printing the classification report\n",
    "print(classification_report(y_test, tree_tuned_pred))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Findings:**  \n",
    "\n",
    "The result is telling us that we have 412 (85%) correct predictions and 75 (15%) incorrect predictions.   \n",
    "\n",
    "The classifier has an Accuracy of 85% -- how often the classifier is correct.  \n",
    "The classifier has a True Positive Rate/Sensitivity/Recall of 88% -- when it's actually yes, how often it predicts yes.  \n",
    "The classifier has a True Negative Rate/Specificity of 78% -- when it's actually no, how often it predicts no.  \n",
    "The classifier has a Precision of 89% -- when it predicts yes, how often it is correct.  "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h3>Model Performance Comparison"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 66,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Model</th>\n",
       "      <th>Training Accuracy</th>\n",
       "      <th>Testing Accuracy</th>\n",
       "      <th>AUC Value</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <td>0</td>\n",
       "      <td>Logistic Regression</td>\n",
       "      <td>0.7601</td>\n",
       "      <td>0.7474</td>\n",
       "      <td>0.7091</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>1</td>\n",
       "      <td>KNN Classification</td>\n",
       "      <td>0.8026</td>\n",
       "      <td>0.7721</td>\n",
       "      <td>0.7459</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>2</td>\n",
       "      <td>Full Tree</td>\n",
       "      <td>0.9993</td>\n",
       "      <td>0.7413</td>\n",
       "      <td>0.7012</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>3</td>\n",
       "      <td>Pruned Tree</td>\n",
       "      <td>0.8060</td>\n",
       "      <td>0.7823</td>\n",
       "      <td>0.7755</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>4</td>\n",
       "      <td>Tuned Tree</td>\n",
       "      <td>0.8485</td>\n",
       "      <td>0.8460</td>\n",
       "      <td>0.8274</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                 Model  Training Accuracy  Testing Accuracy  AUC Value\n",
       "0  Logistic Regression             0.7601            0.7474     0.7091\n",
       "1   KNN Classification             0.8026            0.7721     0.7459\n",
       "2            Full Tree             0.9993            0.7413     0.7012\n",
       "3          Pruned Tree             0.8060            0.7823     0.7755\n",
       "4           Tuned Tree             0.8485            0.8460     0.8274"
      ]
     },
     "execution_count": 66,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# train accuracy\n",
    "tuned_tree_train_acc = tree_tuned.score(X_train, y_train).round(4)\n",
    "\n",
    "\n",
    "# test accuracy\n",
    "tuned_tree_test_acc  = tree_tuned.score(X_test, y_test).round(4)\n",
    "\n",
    "\n",
    "# auc value\n",
    "tuned_tree_auc       = roc_auc_score(y_true  = y_test,\n",
    "                                     y_score = tree_tuned_pred).round(4)\n",
    "\n",
    "\n",
    "# saving the results\n",
    "model_performance.append(['Tuned Tree',\n",
    "                          tuned_tree_train_acc,\n",
    "                          tuned_tree_test_acc,\n",
    "                          tuned_tree_auc])\n",
    "\n",
    "# converting to DataFrame and checking the results\n",
    "pd.DataFrame(model_performance[1:], columns = model_performance[0])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h3>Other: Hyperparameter Tuning"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'\\n\\n##########################################\\n# Hyperparameter Tuning with GridsearchCV\\n##########################################\\n\\n# declaring a hyperparameter space\\ncriterion_space = [\\'gini\\', \\'entropy\\']\\nsplitter_space = [\\'best\\', \\'random\\']\\ndepth_space = pd.np.arange(1, 25)\\nleaf_space  = pd.np.arange(1, 100)\\n\\n\\n# creating a hyperparameter grid\\nparam_grid = {\\'criterion\\'        : criterion_space,\\n              \\'splitter\\'         : splitter_space,\\n              \\'max_depth\\'        : depth_space,\\n              \\'min_samples_leaf\\' : leaf_space}\\n\\n\\n# INSTANTIATING the model object without hyperparameters\\ntuned_tree = DecisionTreeClassifier(random_state = 802)\\n\\n\\n# GridSearchCV object\\ntuned_tree_cv = GridSearchCV(estimator  = tuned_tree,\\n                             param_grid = param_grid,\\n                             cv         = 3,\\n                             scoring    = make_scorer(roc_auc_score,\\n                                                      needs_threshold = False))\\n\\n\\n# FITTING to the FULL DATASET (due to cross-validation)\\ntuned_tree_cv.fit(original_df_data, original_df_target)\\n\\n\\n# PREDICT step is not needed\\n\\n\\n# printing the optimal parameters and best score\\nprint(\"Tuned Parameters  :\", tuned_tree_cv.best_params_)\\nprint(\"Tuned Training AUC:\", tuned_tree_cv.best_score_.round(4))\\n\\n'"
      ]
     },
     "execution_count": 67,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "\"\"\"\n",
    "\n",
    "##########################################\n",
    "# Hyperparameter Tuning with GridsearchCV\n",
    "##########################################\n",
    "\n",
    "# declaring a hyperparameter space\n",
    "criterion_space = ['gini', 'entropy']\n",
    "splitter_space = ['best', 'random']\n",
    "depth_space = pd.np.arange(1, 25)\n",
    "leaf_space  = pd.np.arange(1, 100)\n",
    "\n",
    "\n",
    "# creating a hyperparameter grid\n",
    "param_grid = {'criterion'        : criterion_space,\n",
    "              'splitter'         : splitter_space,\n",
    "              'max_depth'        : depth_space,\n",
    "              'min_samples_leaf' : leaf_space}\n",
    "\n",
    "\n",
    "# INSTANTIATING the model object without hyperparameters\n",
    "tuned_tree = DecisionTreeClassifier(random_state = 802)\n",
    "\n",
    "\n",
    "# GridSearchCV object\n",
    "tuned_tree_cv = GridSearchCV(estimator  = tuned_tree,\n",
    "                             param_grid = param_grid,\n",
    "                             cv         = 3,\n",
    "                             scoring    = make_scorer(roc_auc_score,\n",
    "                                                      needs_threshold = False))\n",
    "\n",
    "\n",
    "# FITTING to the FULL DATASET (due to cross-validation)\n",
    "tuned_tree_cv.fit(original_df_data, original_df_target)\n",
    "\n",
    "\n",
    "# PREDICT step is not needed\n",
    "\n",
    "\n",
    "# printing the optimal parameters and best score\n",
    "print(\"Tuned Parameters  :\", tuned_tree_cv.best_params_)\n",
    "print(\"Tuned Training AUC:\", tuned_tree_cv.best_score_.round(4))\n",
    "\n",
    "\"\"\""
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h3>Other: Ensemble Modelling"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 68,
   "metadata": {
    "code_folding": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "\"\\n\\n##########################################\\n# Ensemble Modelling\\n##########################################\\n\\n# new packages\\nfrom sklearn.ensemble import RandomForestClassifier     # random forest\\nfrom sklearn.ensemble import GradientBoostingClassifier # gbm\\n\\n# INSTANTIATING a random forest model with default values\\nrf_default = RandomForestClassifier(n_estimators     = 10,\\n                                    criterion        = 'gini',\\n                                    max_depth        = None,\\n                                    min_samples_leaf = 1,\\n                                    bootstrap        = True,\\n                                    warm_start       = False,\\n                                    random_state     = 802)\\n\\n# FITTING the training data\\nrf_default_fit = rf_default.fit(X_train, y_train)\\n\\n\\n# PREDICTING based on the testing set\\nrf_default_fit_pred = rf_default_fit.predict(X_test)\\n\\n\\n# SCORING the results\\nprint('Training ACCURACY:', rf_default_fit.score(X_train, y_train).round(4))\\nprint('Testing  ACCURACY:', rf_default_fit.score(X_test, y_test).round(4))\\nprint('AUC Score        :', roc_auc_score(y_true  = y_test,\\n                                          y_score = rf_default_fit_pred).round(4))\\n                                          \\n\""
      ]
     },
     "execution_count": 68,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "\"\"\"\n",
    "\n",
    "##########################################\n",
    "# Ensemble Modelling\n",
    "##########################################\n",
    "\n",
    "# new packages\n",
    "from sklearn.ensemble import RandomForestClassifier     # random forest\n",
    "from sklearn.ensemble import GradientBoostingClassifier # gbm\n",
    "\n",
    "# INSTANTIATING a random forest model with default values\n",
    "rf_default = RandomForestClassifier(n_estimators     = 10,\n",
    "                                    criterion        = 'gini',\n",
    "                                    max_depth        = None,\n",
    "                                    min_samples_leaf = 1,\n",
    "                                    bootstrap        = True,\n",
    "                                    warm_start       = False,\n",
    "                                    random_state     = 802)\n",
    "\n",
    "# FITTING the training data\n",
    "rf_default_fit = rf_default.fit(X_train, y_train)\n",
    "\n",
    "\n",
    "# PREDICTING based on the testing set\n",
    "rf_default_fit_pred = rf_default_fit.predict(X_test)\n",
    "\n",
    "\n",
    "# SCORING the results\n",
    "print('Training ACCURACY:', rf_default_fit.score(X_train, y_train).round(4))\n",
    "print('Testing  ACCURACY:', rf_default_fit.score(X_test, y_test).round(4))\n",
    "print('AUC Score        :', roc_auc_score(y_true  = y_test,\n",
    "                                          y_score = rf_default_fit_pred).round(4))\n",
    "                                          \n",
    "\"\"\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "\"\\n\\n# INSTANTIATING the model object without hyperparameters\\nfull_gbm_default = GradientBoostingClassifier(loss          = 'deviance',\\n                                              learning_rate = 0.1,\\n                                              n_estimators  = 100,\\n                                              criterion     = 'friedman_mse',\\n                                              max_depth     = 3,\\n                                              warm_start    = False,\\n                                              random_state  = 802)\\n\\n\\n# FIT step is needed as we are not using .best_estimator\\nfull_gbm_default_fit = full_gbm_default.fit(X_train, y_train)\\n\\n\\n# PREDICTING based on the testing set\\nfull_gbm_default_pred = full_gbm_default_fit.predict(X_test)\\n\\n\\n# SCORING the results\\nprint('Training ACCURACY:', full_gbm_default_fit.score(X_train, y_train).round(4))\\nprint('Testing ACCURACY :', full_gbm_default_fit.score(X_test, y_test).round(4))\\nprint('AUC Score        :', roc_auc_score(y_true  = y_test,\\n                                          y_score = full_gbm_default_pred).round(4))\\n                                          \\n\""
      ]
     },
     "execution_count": 69,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "\"\"\"\n",
    "\n",
    "# INSTANTIATING the model object without hyperparameters\n",
    "full_gbm_default = GradientBoostingClassifier(loss          = 'deviance',\n",
    "                                              learning_rate = 0.1,\n",
    "                                              n_estimators  = 100,\n",
    "                                              criterion     = 'friedman_mse',\n",
    "                                              max_depth     = 3,\n",
    "                                              warm_start    = False,\n",
    "                                              random_state  = 802)\n",
    "\n",
    "\n",
    "# FIT step is needed as we are not using .best_estimator\n",
    "full_gbm_default_fit = full_gbm_default.fit(X_train, y_train)\n",
    "\n",
    "\n",
    "# PREDICTING based on the testing set\n",
    "full_gbm_default_pred = full_gbm_default_fit.predict(X_test)\n",
    "\n",
    "\n",
    "# SCORING the results\n",
    "print('Training ACCURACY:', full_gbm_default_fit.score(X_train, y_train).round(4))\n",
    "print('Testing ACCURACY :', full_gbm_default_fit.score(X_test, y_test).round(4))\n",
    "print('AUC Score        :', roc_auc_score(y_true  = y_test,\n",
    "                                          y_score = full_gbm_default_pred).round(4))\n",
    "                                          \n",
    "\"\"\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 70,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "\"\\n\\n# INSTANTIATING the model object without hyperparameters\\ngbm_tuned = GradientBoostingClassifier(learning_rate = 0.1,\\n                                       max_depth     = 2,\\n                                       n_estimators  = 100,\\n                                       random_state  = 802)\\n\\n\\n# FIT step is needed as we are not using .best_estimator\\ngbm_tuned_fit = gbm_tuned.fit(X_train, y_train)\\n\\n\\n# PREDICTING based on the testing set\\ngbm_tuned_pred = gbm_tuned_fit.predict(X_test)\\n\\n\\n# SCORING the results\\nprint('Training ACCURACY:', gbm_tuned_fit.score(X_train, y_train).round(4))\\nprint('Testing  ACCURACY:', gbm_tuned_fit.score(X_test, y_test).round(4))\\nprint('AUC Score        :', roc_auc_score(y_true  = y_test,\\n                                          y_score = gbm_tuned_pred).round(4))\\n                                          \\n\""
      ]
     },
     "execution_count": 70,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "\"\"\"\n",
    "\n",
    "# INSTANTIATING the model object without hyperparameters\n",
    "gbm_tuned = GradientBoostingClassifier(learning_rate = 0.1,\n",
    "                                       max_depth     = 2,\n",
    "                                       n_estimators  = 100,\n",
    "                                       random_state  = 802)\n",
    "\n",
    "\n",
    "# FIT step is needed as we are not using .best_estimator\n",
    "gbm_tuned_fit = gbm_tuned.fit(X_train, y_train)\n",
    "\n",
    "\n",
    "# PREDICTING based on the testing set\n",
    "gbm_tuned_pred = gbm_tuned_fit.predict(X_test)\n",
    "\n",
    "\n",
    "# SCORING the results\n",
    "print('Training ACCURACY:', gbm_tuned_fit.score(X_train, y_train).round(4))\n",
    "print('Testing  ACCURACY:', gbm_tuned_fit.score(X_test, y_test).round(4))\n",
    "print('AUC Score        :', roc_auc_score(y_true  = y_test,\n",
    "                                          y_score = gbm_tuned_pred).round(4))\n",
    "                                          \n",
    "\"\"\""
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.3"
  },
  "toc": {
   "base_numbering": 1,
   "nav_menu": {},
   "number_sections": true,
   "sideBar": true,
   "skip_h1_title": false,
   "title_cell": "Table of Contents",
   "title_sidebar": "Contents",
   "toc_cell": false,
   "toc_position": {},
   "toc_section_display": true,
   "toc_window_display": false
  },
  "varInspector": {
   "cols": {
    "lenName": 16,
    "lenType": 16,
    "lenVar": 40
   },
   "kernels_config": {
    "python": {
     "delete_cmd_postfix": "",
     "delete_cmd_prefix": "del ",
     "library": "var_list.py",
     "varRefreshCmd": "print(var_dic_list())"
    },
    "r": {
     "delete_cmd_postfix": ") ",
     "delete_cmd_prefix": "rm(",
     "library": "var_list.r",
     "varRefreshCmd": "cat(var_dic_list()) "
    }
   },
   "types_to_exclude": [
    "module",
    "function",
    "builtin_function_or_method",
    "instance",
    "_Feature"
   ],
   "window_display": false
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
