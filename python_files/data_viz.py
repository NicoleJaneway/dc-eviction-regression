"""Module with data visualization functions"""

from shapely.geometry import Point, Polygon
import fiona 
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def years_dist(x=6,y=3):
    """Returns distribution of eviction rates for years active in Eviction Lab dataset"""
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
    plt.figure(figsize=(x,y))
    ax = sns.countplot(x=puds.Zone_Cat,palette=sns.diverging_palette(10, 220, sep=80, n=3, center="dark"));
    ax.set_title('Count by PUD Zoning Category');
    ax.set_xlabel("");
    ax.set_ylabel('Count of PUDs', size=14);
    return plt.show();

def pud_count_by_type(x=6,y=3):
    """Returns bar graph of PUD count by zoning type"""
    plt.figure(figsize=(x,y))
    ax = sns.countplot(x=puds.Zone_Cat,palette=sns.diverging_palette(10, 220, sep=80, n=3, center="dark"));
    ax.set_title('Count by PUD Zoning Category');
    ax.set_xlabel("");
    ax.set_ylabel('Count of PUDs', size=14);
    return plt.show();
