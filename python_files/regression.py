"""Module with regression functions"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import numpy as np
from sklearn.model_selection import train_test_split
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms

def shrink_data(df):
    """Creates dataframe with continuous columns"""
    x_cols_all = ['population',
 'poverty-rate',
 'renter-occupied-households',
 'pct-renter-occupied',
 'median-gross-rent',
 'median-household-income',
 'median-property-value',
 'rent-burden',
 'pct-white',
 'pct-af-am',
 'pct-hispanic',
 'pct-am-ind',
 'pct-asian',
 'pct-nh-pi',
 'pct-multiple',
 '% Affordable Units']
    puds_to_transform = df[x_cols_all]
    puds_to_transform = pd.concat([puds_to_transform,df['eviction-rate']],axis=1)
    puds_to_transform = puds_to_transform.fillna(0)
    return puds_to_transform 
    
def drop_outliers(df):
    """Returns X and y datasets with outliers removed"""
    puds_cleaned = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]
    puds_cleaned = puds_cleaned.fillna(0)
    X = puds_cleaned.drop('eviction-rate',axis=1)
    y = pd.DataFrame(puds_cleaned['eviction-rate'])
    return X, y

def segment_test_data(X, y, split=0.1):
    """Creates train test split and conducts log and scaler transformation on training data""" 
    X_train , X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=12)
    return X_train, X_test, y_train, y_test


def log_and_scale(X_train, y_train):
    scaler = StandardScaler()
    X_train = X_train.transform(lambda x: np.log(x + 1))
    X_train = scaler.fit_transform(X_train)
    y_train = y_train.transform(lambda x: np.log(x + 1))
    y_train = scaler.fit_transform(np.array(y_train).reshape(-1, 1))
    return X_train, y_train
        

def transform_arrays_to_df(X_train, y_train, X_labels, y_labels='eviction-rate'):
    X = pd.DataFrame(X_train)
    y = pd.DataFrame(y_train)
    X.columns = X_labels
    y.columns = [y_labels]
    return X, y

def feature_histogram(X, y):
    """Display grid of features' distributions"""
    sns.set_context('notebook')
    scaled = pd.concat([X, y], axis=1)
    pd.DataFrame(scaled).hist(figsize  = [12, 12], color='gray'); 
    plt.subplots_adjust(wspace=.5, hspace=.5)
    return plt.show();

def multicolinearity_check1(X):
    """Display visuals to identify potential issues of multicolinearity"""
    sns.set_context('notebook')
    sns.set_palette('gray',1);
    axs = pd.plotting.scatter_matrix(X,figsize  = [12, 20]);
    n = len(X.columns)
    for x in range(n):
        for y in range(n):
            # to get the axis of subplots
            ax = axs[x, y]
            # to make x axis name vertical  
            ax.xaxis.label.set_rotation(90)
            # to make y axis name horizontal 
            ax.yaxis.label.set_rotation(0)
            # to make sure y axis names are outside the plot area
            ax.yaxis.labelpad = 50
    return plt.show();

def multicolinearity_check2(X):
    return X.corr()

def multicolinearity_check(X):
    sns.set_context('notebook')
    fig = plt.figure(figsize=(12,10))
    ax = sns.heatmap(X.corr(), center=0, vmin=-1, cmap=(sns.diverging_palette(240, 10, sep=80, n=20)), annot=round(X.corr(),2));
    
    return plt.show(); 

def big_lin_reg(X,y,X_labels):
    """Fits linear regression model"""
    X = sm.add_constant(X)
    model = sm.OLS(y, X, hasconst=True )
    result = model.fit()
    labels = ['intercept'] + X_labels
    return result.summary(xname=labels)

# def residual_check1(X,y):
#     sns.set_context('talk')
#     fig = plt.figure(figsize=(8,20))
#     for num in range(len(X.columns)):
#         num += 1
#         ax = fig.add_subplot((len(X.columns),1,num)
#         num - = 1
#         sns.residplot(x=X[num], y=Y, data=X, color='gray', scatter_kws={"s": 20});
#         plt.subplots_adjust(wspace=.5, hspace=.5);
#     return plt.show();

def big_lin_reg_return_residuals(X,y):
    X = sm.add_constant(X)
    model = sm.OLS(y, X, hasconst=True )
    result = model.fit()
    return result.resid, result.model.exog

    
def residual_checks(X,y):
    results, result_m_e = big_lin_reg_return_residuals(X,y)
    
    name = ['Jarque-Bera', 'p-value', 'Skew', 'Kurtosis']
    test = sms.jarque_bera(results);
    [print('Tests of normality of residuals:')]
    [print("   - "+str(el[0])+": "+str(round(el[1],3))) for el in list(zip(name, test))];
    name = ['Lagrange multiplier statistic', 'p-value',
        'f-value', 'f p-value']
    test = sms.het_breuschpagan(results, result_m_e);
    [print("\n")]
    [print('Tests of heteroskedasticty of residuals:')]
    [print("   - "+str(el[0])+": "+str(round(el[1],3))) for el in list(zip(name, test))];

def residual_plot(X,y):
    sns.set_context('talk')
    scaled = pd.concat([X, y], axis=1)
    outcome = scaled['eviction-rate']
    fig = plt.figure(figsize=(6,3))
    plt.title('Plot of Residuals');
    ax = sns.residplot(x='pct-white', y=outcome, data=scaled, color='gray', scatter_kws={"s": 20, 'alpha': .8}, label='Percent White');
    ax = sns.residplot(x='pct-renter-occupied', y=outcome, data=scaled, color='blue', scatter_kws={"s": 20, 'alpha': .4}, label='Percent Renter-Occupied');
    ax = sns.residplot(x='poverty-rate', y=outcome, data=scaled, color='red', scatter_kws={"s": 20,'alpha': .3}, label='Poverty Rate');
    ax.legend()
    plt.legend(bbox_to_anchor=(1.02,1.05), loc="upper left", prop={'size': 12})
    ax.set_xlabel('Feature Value', size=14);
    ax.set_ylabel('Eviction Value', size=14);
    return plt.show();

    
    
