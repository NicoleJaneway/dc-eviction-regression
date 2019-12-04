"""Module with data visualization functions"""

from shapely.geometry import Point, Polygon
import fiona 
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def years_dist(x=6,y=3):
    """Returns distribution of eviction rates for years active in Eviction Lab dataset"""
    evict = pd.read_csv('data/eviction_lab/csv/block-groups.csv')
    sns.set_palette(sns.color_palette("Blues_r", 8,))
    plt.figure(figsize=(x,y))
    ax = sns.kdeplot(evict['eviction-rate'].loc[evict['year'] == 2016], color='red');
    sns.kdeplot(evict['eviction-rate'].loc[evict['year'] == 2012]);
    sns.kdeplot(evict['eviction-rate'].loc[evict['year'] == 2011]);
    sns.kdeplot(evict['eviction-rate'].loc[evict['year'] == 2010]);
    sns.kdeplot(evict['eviction-rate'].loc[evict['year'] == 2009]);
    sns.kdeplot(evict['eviction-rate'].loc[evict['year'] == 2008]);
    sns.kdeplot(evict['eviction-rate'].loc[evict['year'] == 2007]);
    sns.kdeplot(evict['eviction-rate'].loc[evict['year'] == 2006]);
    ax.set(xlim=(0, 15));
    plt.legend(labels = [2016, 2012, 2011, 2010, 2009, 2008, 2007, 2006], prop={'size': 10});
    plt.title('Distribution of Eviction Rate by Year', size=16);
    ax.yaxis.set_visible(False);
    ax.set_xlabel('Eviction Rate (%)', size=14);
    plt.show();
    return plt.show();

def pud_count_by_type(x=6,y=3):
    """Returns bar graph of PUD count by zoning type"""
    sns.set_context('notebook')
    puds = pd.read_csv('data/final_datasets/master_puds_blocks.csv')
    plt.figure(figsize=(x,y))
    ax = sns.countplot(x=puds.Zone_Cat,palette=sns.diverging_palette(10, 240, sep=80, n=3, center="dark"));
    ax.set_title('Count by PUD Zoning Category', size=16);
    ax.set_xlabel("");
    ax.set_ylabel('Count of PUDs', size=12);
    return plt.show();

def affordable_units_count(x=6,y=3):
    """Returns bar graph of PUD count by zoning type"""
    puds = pd.read_csv('data/final_datasets/master_puds_blocks.csv')
    puds['% Affordable Units'] = np.where(puds['% Affordable Units'].isnull(), 0, puds['% Affordable Units'])
    puds_g = puds.loc[~puds['Total # Residential Units'].isnull()][['OBJECTID_PUD', 'PUD_NAME', 'PUD_WEB_URL', 'PUD_CHANGE_NARRATIVE',
       'PUD_ZONING', 'PUD_STATUS', 'PUD_CASE_NUMBER', 'SHAPEAREA', 'SHAPELEN',
       'GLOBALID', 'WARD', 'Name', 'Description', 'Zone_Cat',
       'CASE_ID_orig', 'ADDRESS', 'PROJECT_NAME', 'STATUS_PUBLIC',
       'AGENCY_CALCULATED', 'AFFORDABLE_UNITS_AT_0_30_AMI',
       'AFFORDABLE_UNITS_AT_31_50_AMI', 'AFFORDABLE_UNITS_AT_51_60_AMI',
       'AFFORDABLE_UNITS_AT_61_80_AMI', 'AFFORDABLE_UNITS_AT_81_AMI',
       'TOTAL_AFFORDABLE_UNITS', 'MAR_WARD', 'Type', 'ANC',
       'Total # Residential Units', '% Affordable Units',
       'Total # of 3 bedroom+ units (Fam sized units)',
       'Ownership (rental vs. condo or mix)',
       'Affordability notes (What levels of AMI% are avail)', 'FULLADDRESS',
       'GIS_LAST_MOD_DTTM', 'CASE_ID_update', 'GEOID', 'ward']]
    plt.figure(figsize=(x,y)) 
    viz0 = pd.DataFrame(pd.concat([puds_g['% Affordable Units'], pd.cut(puds_g['% Affordable Units'], bins=[-.01,0, 0.25, .5, .75, 1])], axis=1))
    viz0.columns = ['% Affordable Units', 'bin0']
    renaming1 = {'(-0.01, 0.0]': '0%',
'(0.0, 0.25]': '0-25%',
 '(0.25, 0.5]':'25-50%',
 '(0.5, 0.75]':'50-75%',
 '(0.75, 1.0]':'75-100%'}
    viz0['bin1'] = viz0.bin0.astype(str).map(renaming1)
    renaming2 = {'(-0.01, 0.0]': '0%',
'(0.0, 0.25]': '0-25%',
 '(0.25, 0.5]':'25-100%',
 '(0.5, 0.75]':'25-100%',
 '(0.75, 1.0]':'25-100%'}
    viz0['bin2'] = viz0.bin0.astype(str).map(renaming2)
    ax = sns.countplot(viz0['bin2'], order=['0%','0-25%','25-100%'], palette="coolwarm_r")
    ax.set_ylabel('Count of PUDs', size=12);
    ax.set_xlabel('');
    ax.set_title('Affordable Units out of Total Units', size=16);
    plt.xticks(size=14)
    return plt.show();

def lin_reg_puds(x=6,y=3):
    plt.figure(figsize=(x,y))
    ax = sns.regplot(x='pud_count', y=outcome, data=minipuds, color='gray', marker='.');
    plt.title('Plot of PUDs against Eviction Rate');
    ax.set_xlabel('Count of Puds by Block Group', size=14);
    ax.set_ylabel('Eviction Rate (%)', size=14);
    return plt.show();



