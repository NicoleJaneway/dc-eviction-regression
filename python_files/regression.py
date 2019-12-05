"""Module with regression functions"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
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


def log_and_scale(df):
    assert type(df) == pd.core.frame.DataFrame,"Input is not a dataframe"
    scaler = MinMaxScaler()
    df = df.transform(lambda x: np.log(x + 1))
    if df.shape[-1] == 1:
        df = np.array(df).reshape(-1, 1)
    else:
        df = np.array(df)
    df = scaler.fit_transform(df)
    return df
        

def transform_arrays_to_df(X_train, y_train, X_labels, y_labels='eviction-rate'):
    X = pd.DataFrame(X_train)
    y = pd.DataFrame(y_train)
    X.columns = X_labels
    y.columns = [y_labels]
    return X, y

def feature_histogram(X, y=None):
    """Display grid of features' distributions"""
    sns.set_context('notebook')
    scaled = pd.concat([X, y], axis=1)
    pd.DataFrame(scaled).hist(figsize  = [12, 12], color='gray'); 
    plt.subplots_adjust(wspace=.5, hspace=.5)
    return plt.show();

def feature_bar_chart(result, X_labels):
    X_labels.insert(0,'intercept')
    viz0 = pd.DataFrame(result.params)
    viz0.columns = ['coef']
    sns.set_context('talk')
    viz0['col_names'] = X_labels
    viz0['coef_transformed'] = viz0['coef'].map(lambda x: abs(x))
    viz0.sort_values(by='coef_transformed', ascending=False, inplace=True)
    viz0['color'] = viz0['coef'].map(lambda x: "'gray'" if x > 0 else "'blue'")
    ax = sns.barplot(x=viz0['col_names'], y=viz0['coef_transformed'], color='grey');
    plt.xticks(rotation=90, size=12);
    ax.set_xlabel('Columns', size=14);
    ax.set_ylabel('Coefficient', size=14);
    ax.set_title('Relative Importance of Features');
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

def lin_reg(X,y):
    """Fits linear regression model"""
    model = sm.OLS(y, X, hasconst=True )
    result = model.fit()
    return model, result

def create_summary(result,X_labels=[],X=None):
    if X_labels == []:
        X_labels = [el for el in range(len(X))]
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

def residual_checks(X,y):
    X_labels = [el for el in range(X.shape[-1])]

    model, result = lin_reg(X,y)
    
    name = ['Jarque-Bera', 'p-value', 'Skew', 'Kurtosis']
    test = sms.jarque_bera(result.resid);
    [print('Tests of normality of residuals:')]
    [print("   - "+str(el[0])+": "+str(round(el[1],3))) for el in list(zip(name, test))];
    name = ['Lagrange multiplier statistic', 'p-value',
        'f-value', 'f p-value']
    test = sms.het_breuschpagan(result.resid, result.model.exog);
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

import statsmodels.api as sm

def stepwise_selection(X, y, 
                       initial_list=[], 
                       threshold_in=0.01, 
                       threshold_out = 0.05, 
                       verbose=True):
    """ Perform a forward-backward feature selection 
    based on p-value from statsmodels.api.OLS
    Arguments:
        X - pandas.DataFrame with candidate features
        y - list-like with the target
        initial_list - list of features to start with (column names of X)
        threshold_in - include a feature if its p-value < threshold_in
        threshold_out - exclude a feature if its p-value > threshold_out
        verbose - whether to print the sequence of inclusions and exclusions
    Returns: list of selected features 
    Always set threshold_in < threshold_out to avoid infinite looping.
    See https://en.wikipedia.org/wiki/Stepwise_regression for the details
    """
    included = list(initial_list)
    while True:
        changed=False
        # forward step
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed=True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

        # backward step
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() # null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.argmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included    


def create_values_table(X, X_var, y, y_hat):
    values_table = pd.concat([X['pct-white'], y['eviction-rate'].reset_index(), pd.DataFrame(y_hat)], axis=1)
    values_table.drop('index', axis=1, inplace=True)
    values_table.columns = [X_var, 'y', 'y_hat']
    values_table['residual'] = (values_table['y']-values_table['y_hat'])
    return values_table

def y_vs_y_hat_scatter(x, y, y_hat):
    fig, ax = plt.subplots(sharey=True)
    sns.scatterplot(x=x, y=y,color='gray', label='Actual');
    sns.scatterplot(x=x, y=y_hat,color='blue', marker='.', label='Predicted');
    ax.set_title('Plot of Results')
    ax.set_ylabel('Eviction Rates (%)', size=14)
    ax.set_xlabel('White Population (%)', size=14)
    plt.legend(bbox_to_anchor=(1.02,1.05), loc="upper left", prop={'size': 12})
    return plt.show();

