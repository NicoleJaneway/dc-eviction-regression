"""Module with feature selection functions"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def shrink(df):
    x_cols_shrink = ['GEOID','year','name','population', 'poverty-rate', 'renter-occupied-households', 'pct-renter-occupied', 'median-gross-rent', 
              'median-household-income','median-property-value','rent-burden','pct-white','pct-af-am','pct-hispanic','pct-am-ind','pct-asian',
'pct-nh-pi',
'pct-multiple',
'pct-other',
'eviction-filings',
'evictions',
'eviction-rate',
'eviction-filing-rate',
'low-flag',
'imputed',
'subbed',
'ward',
'PUD_NAME',
'PUD_WEB_URL',
'PUD_CHANGE_NARRATIVE',
'PUD_ZONING',
'PUD_STATUS',
'PUD_CASE_NUMBER',
'GLOBALID',
'WARD',
'geometry_ft',
'Zone_Cat',
'ADDRESS',
'PROJECT_NAME',
'STATUS_PUBLIC',
'AGENCY_CALCULATED',
'AFFORDABLE_UNITS_AT_0_30_AMI',
'AFFORDABLE_UNITS_AT_31_50_AMI',
'AFFORDABLE_UNITS_AT_51_60_AMI',
'AFFORDABLE_UNITS_AT_61_80_AMI',
'AFFORDABLE_UNITS_AT_81_AMI',
'TOTAL_AFFORDABLE_UNITS',
'MAR_WARD',
'Type',
'ANC',
'Total # Residential Units',
'% Affordable Units',
'Total # of 3 bedroom+ units (Fam sized units)',
'Ownership (rental vs. condo or mix)',
'Affordability notes (What levels of AMI% are avail)',
'FULLADDRESS',
'GIS_LAST_MOD_DTTM',
'CASE_ID_update']
    return df[x_cols_shrink]


def create_demo_col(df):
    x_cols_demo = ['pct-white','pct-af-am','pct-hispanic','pct-am-ind','pct-asian','pct-nh-pi','pct-multiple','pct-other']
    df['pct-non-white'] = df[x_cols_demo].sum(axis=1)-df['pct-white']
    df = df[['GEOID','year','name','population', 'poverty-rate', 'renter-occupied-households', 'pct-renter-occupied', 'median-gross-rent', 
              'median-household-income','median-property-value','rent-burden','pct-white','pct-af-am','pct-hispanic','pct-am-ind','pct-asian',
'pct-nh-pi',
'pct-multiple',
'pct-other',
'pct-non-white',
'eviction-filings',
'evictions',
'eviction-rate',
'eviction-filing-rate',
'low-flag',
'imputed',
'subbed',
'ward',
'PUD_NAME',
'PUD_WEB_URL',
'PUD_CHANGE_NARRATIVE',
'PUD_ZONING',
'PUD_STATUS',
'PUD_CASE_NUMBER',
'GLOBALID',
'WARD',
'geometry_ft',
'Zone_Cat',
'ADDRESS',
'PROJECT_NAME',
'STATUS_PUBLIC',
'AGENCY_CALCULATED',
'AFFORDABLE_UNITS_AT_0_30_AMI',
'AFFORDABLE_UNITS_AT_31_50_AMI',
'AFFORDABLE_UNITS_AT_51_60_AMI',
'AFFORDABLE_UNITS_AT_61_80_AMI',
'AFFORDABLE_UNITS_AT_81_AMI',
'TOTAL_AFFORDABLE_UNITS',
'MAR_WARD',
'Type',
'ANC',
'Total # Residential Units',
'% Affordable Units',
'Total # of 3 bedroom+ units (Fam sized units)',
'Ownership (rental vs. condo or mix)',
'Affordability notes (What levels of AMI% are avail)',
'FULLADDRESS',
'GIS_LAST_MOD_DTTM',
'CASE_ID_update']]
    return df

def agg_puds(df):
    def count_puds(df):
        df_count = df[['GEOID','year','name','population', 'poverty-rate', 'renter-occupied-households',
       'pct-renter-occupied', 'median-gross-rent', 'median-household-income',
       'median-property-value', 'rent-burden', 'pct-white', 'pct-af-am',
       'pct-hispanic', 'pct-am-ind', 'pct-asian', 'pct-nh-pi', 'pct-multiple',
       'pct-other', 'eviction-filings', 'evictions',
       'eviction-rate', 'eviction-filing-rate', 'low-flag', 'imputed',
       'subbed', 'ward', 'PUD_NAME']].groupby(by=['GEOID','year','name','population', 'poverty-rate', 'renter-occupied-households',
       'pct-renter-occupied', 'median-gross-rent', 'median-household-income',
       'median-property-value', 'rent-burden', 'pct-white', 'pct-af-am',
       'pct-hispanic', 'pct-am-ind', 'pct-asian', 'pct-nh-pi', 'pct-multiple',
       'pct-other', 'eviction-filings', 'evictions',
       'eviction-rate', 'eviction-filing-rate', 'low-flag', 'imputed',
       'subbed', 'ward']).count().reset_index()
        df_count.rename(columns={'PUD_NAME':'pud_count'}, inplace=True)
        return df_count
    
    def avg_puds(df):
        df_avg = df[['GEOID','year','name','population', 'poverty-rate', 'renter-occupied-households',
       'pct-renter-occupied', 'median-gross-rent', 'median-household-income',
       'median-property-value', 'rent-burden', 'pct-white', 'pct-af-am',
       'pct-hispanic', 'pct-am-ind', 'pct-asian', 'pct-nh-pi', 'pct-multiple',
       'pct-other', 'eviction-filings', 'evictions',
       'eviction-rate', 'eviction-filing-rate', 'low-flag', 'imputed',
       'subbed', 'ward', '% Affordable Units']].groupby(by=['GEOID','year','name','population', 'poverty-rate', 'renter-occupied-households',
       'pct-renter-occupied', 'median-gross-rent', 'median-household-income',
       'median-property-value', 'rent-burden', 'pct-white', 'pct-af-am',
       'pct-hispanic', 'pct-am-ind', 'pct-asian', 'pct-nh-pi', 'pct-multiple',
       'pct-other', 'eviction-filings', 'evictions',
       'eviction-rate', 'eviction-filing-rate', 'low-flag', 'imputed',
       'subbed', 'ward']).mean().reset_index()
        df_avg['% Affordable Units'] = df_avg['% Affordable Units'].replace(np.nan, 0)
        return df_avg
    
    return count_puds(df).merge(avg_puds(df))

def chart_of_affordable_housing(x1, x2, y):
    sns.set_context('talk')
    ax = sns.scatterplot(x = x1, 
                    y = y, 
                    hue = [0 if el == 0 else 1 for el in x2], 
                palette=(sns.diverging_palette(10, 220, sep=80, n=2)), markers='.', alpha=.8)
    plt.legend(bbox_to_anchor=(1.02,1.05), loc="upper left", prop={'size': 12})
    leg = ax.axes.get_legend()
    # ax.legend()
    new_labels = ['No Affordable Housing', 'Affordable Housing']
    for t, l in zip(leg.texts, new_labels): t.set_text(l)
    params = {'legend.fontsize': 10}
    plt.rcParams.update(params)
    ax.set_xlabel('Count of Puds by Block Group', size=14);
    ax.set_ylabel('Eviction Rate (%)', size=14);
    ax.set_title('Plot of Affordable Housing', size=16)
    plt.xticks(size=14)
    plt.yticks(size=14)
    return plt.show()

