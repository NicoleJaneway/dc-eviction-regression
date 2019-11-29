"""Module with data cleaning functions"""

from shapely.geometry import Point, Polygon
import fiona 
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def merge_gis(gis_df, data_df):
    df = data_df.merge(gis_df, left_index=True, right_index=True)
    df = gpd.GeoDataFrame(df, geometry=df.geometry)
    return df

def clean_eviction_data(df):
    df = df.loc[df.year == 2016]
    return df
    
def merge_tracts(gis_df, data_df):
    df = pd.concat([data_df, gis_df], axis = 1)
    df = gpd.GeoDataFrame(df, geometry=df.geometry)
    return df

def merge_eviction_data(tracts_df, eviction_df):
    df = eviction_df.merge(tracts_df, on='GEOID')
    return df

def clean_grassroots_pud_data(gr_df):
    gr_case_id_update = pd.read_csv('../data/grassroots_dc/csv/case_id_update.csv')
    gr_case_id_update = gr_case_id_update[['CASE_ID','PROJECT_NAME']]
    gr_df = gr_df.merge(gr_case_id_update, on="PROJECT_NAME", suffixes=('_orig','_update'))
    return gr_df

def add_pud_ft(pud_df,zoning_xwalk, gr_xwalk):
    pud_df = pud_df.merge(zoning_xwalk['Zone_Cat'], how='left', left_on='PUD_ZONING', right_on=zoning_xwalk['Zone'])
    pud_df = pud_df.merge(gr_xwalk, how='left', left_on='PUD_CASE_NUMBER', right_on='CASE_ID_update', suffixes=('_PUD', '_gr'))
    return pud_df

def add_gis_ft1(super_df, sub_df, super_unit):
    for index, row in sub_df.iterrows():
        for super_index, super_row in super_df.iterrows():
            if row['geometry'].centroid.within(super_row['geometry']):
                sub_df.loc[index, super_unit] = super_row['Name']
    return sub_df
            
def add_gis_ft2(super_df, sub_df, super_unit):
    for index, row in sub_df.iterrows():
        for super_index, super_row in super_df.iterrows():
            try:
                if row['geometry'].centroid.within(super_row['geometry']):
                    sub_df.loc[index, super_unit] = int(super_row['GEOID'])
            except:
                pass
    return sub_df

def create_master(ft_df, eviction_df):
    df = eviction_df.merge(ft_df, how='outer', on="GEOID", suffixes=('_Tract','_ft'))
    return df

def clean_master_puds(df):
    df = df[['GEOID',
'year',
'name',
'population',
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
'pct-other',
'eviction-filings',
'evictions',
'eviction-rate',
'eviction-filing-rate',
'low-flag',
'imputed',
'subbed',
'OBJECTID',
'TRACT',
'FAGI_TOTAL_2010',
'FAGI_MEDIAN_2010',
'FAGI_TOTAL_2013',
'FAGI_MEDIAN_2013',
'FAGI_TOTAL_2011',
'FAGI_MEDIAN_2011',
'FAGI_TOTAL_2012',
'FAGI_MEDIAN_2012',
'FAGI_TOTAL_2014',
'FAGI_MEDIAN_2014',
'FAGI_TOTAL_2015',
'FAGI_MEDIAN_2015',
'geometry_Tract',
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

def export_master(df, name):
    df = df.astype(object)
    df.to_csv(fr'../data/final_datasets/{name}.csv', index=False)
    return "exported %s" %name