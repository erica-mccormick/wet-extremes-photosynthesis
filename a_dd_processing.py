
import glob
import os
from tqdm import tqdm
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import seaborn as sns
import subprocess
from utils import paths
from utils import event_identification_tools
from utils import fluxnet_tools
import time
import pickle
import datetime
import math

def main():
    print('Running a_dd_processing.py')    
    t0 = time.time()
    
    
    ### ------------- FOR TESTING ONLY, LIMIT THE NUMBER OF SITES  -------------
    #sites_for_testing = ['SN-Dhr', 'AU-RDF', 'US-WHs', 'IT-Ro1', 'US-Var', 'AU-Dry', 'US-Goo', 'FR-LBr']

    # ARGUMENTS
    with open(paths.PATH_TO_SWC_SITES, 'rb') as f:
        swc_sites = pickle.load(f)

    findevents_dir = 'output_findevents/v3'
    event_filename = 'allsites_event.csv'
    training_filename = 'allsites_training.csv'

    cols_to_normalize = ['RH', 'SWC_F_MDS_1', 'GPP_DT_VUT_REF', 'GPP_NT_VUT_REF',
                             'SW_IN_F', 'TA_F', 'LE_F_MDS', 'VPD_F', 'WS_F']

    # CALCULATE WARM DAYS (ie GROW SEASON)
    growseason_dict = calculate_growing_season(swc_sites, 'TA_F')

    # LOAD EVENT AND TRAINING HH DATA
    allsites_event = pd.read_csv(os.path.join(findevents_dir, event_filename))
    allsites_training = pd.read_csv(os.path.join(findevents_dir, training_filename))

    # Initialize for later
    allsites_event_hh = pd.DataFrame()
    allsites_training_hh = pd.DataFrame()
    allsites_event_hh_g = pd.DataFrame()
    allsites_training_hh_g = pd.DataFrame()
    allsites_event_dd = pd.DataFrame()
    allsites_training_dd = pd.DataFrame()
    allsites_event_dd_g = pd.DataFrame()
    allsites_training_dd_g = pd.DataFrame()
    allsites_event_hh_clipped= pd.DataFrame()
    allsites_training_hh_clipped= pd.DataFrame()
    allsites_event_hh_clipped_g= pd.DataFrame()
    allsites_training_hh_clipped_g= pd.DataFrame()
    allsites_event_clipped_dd= pd.DataFrame()
    allsites_training_clipped_dd= pd.DataFrame()
    allsites_event_dd_clipped_g= pd.DataFrame()
    allsites_training_dd_clipped_g= pd.DataFrame()


    for site_name in swc_sites:
        print(f"Processing {site_name}")
        
        ### ------------- ADD FEATURES BASED ON HALF-HOURLY DATA -------------
        # Load daily data and site-specific event and training data
        dd_data = load_dd_data(site_name)
        
        event = allsites_event[allsites_event['SITE_ID'] == site_name]
        training = allsites_training[allsites_training['SITE_ID'] == site_name]

        event['day'] = pd.to_datetime(event['day'])
        training['day'] = pd.to_datetime(training['day'])

        event['TIMESTAMP_START'] = pd.to_datetime(event['TIMESTAMP_START'])
        training['TIMESTAMP_START'] = pd.to_datetime(training['TIMESTAMP_START'])
        
        # Add summed precipitation over specific range of months
        sum_days = 60
        skipped_days = 30
        event = sum_p_months(event, dd_data, sum_days, skipped_days)
        training = sum_p_months(training, dd_data, sum_days, skipped_days)
        
        # Add mean GPP over the first week of April (for no rain days)
        event = april_gpp(event, dd_data, doy_start = 92, doy_end = 100, p_thresh = 1)
        training = april_gpp(training, dd_data, doy_start = 92, doy_end = 100, p_thresh = 1)


        ### ------------- CLIP TO WARM DAYS -------------
        event_clipped = clip_to_growing_season(event, site_name, growseason_dict, time_col_for_doy = 'TIMESTAMP_START')
        training_clipped = clip_to_growing_season(training, site_name, growseason_dict, time_col_for_doy = 'TIMESTAMP_START')


        ### ------------- CONVERT  HH TO DAILY -------------
        event_dd = calc_daytime_avg_df(event, cols_to_normalize)
        training_dd = calc_daytime_avg_df(training, cols_to_normalize)

        event_clipped_dd = calc_daytime_avg_df(event_clipped, cols_to_normalize)
        training_clipped_dd = calc_daytime_avg_df(training_clipped, cols_to_normalize)      


        ### ------------- CALCULATE AND SUBTRACT DOY AVG GPP FOR SEASONAL ADJUSTMENT -------------
        hh_data = load_hh_data(site_name) 

        event_g = growingseason_normalized_features(training, event, cols_to_normalize, 'hh')
        training_g = growingseason_normalized_features(training, training, cols_to_normalize, 'hh')

        event_clipped_g = growingseason_normalized_features(training, event_clipped, cols_to_normalize, 'hh')
        training_clipped_g = growingseason_normalized_features(training, training_clipped, cols_to_normalize, 'hh')

        event_dd_g = growingseason_normalized_features(training, event_dd, cols_to_normalize, 'dd')
        training_dd_g = growingseason_normalized_features(training, training_dd, cols_to_normalize, 'dd')

        event_clipped_dd_g = growingseason_normalized_features(training, event_clipped_dd, cols_to_normalize, 'dd')
        training_clipped_dd_g = growingseason_normalized_features(training, training_clipped_dd, cols_to_normalize, 'dd')

        ### ------------- FIGURES SHOWING SEASONAL SUBTRACTION -------------
        #print(training_clipped_dd_g.columns)
        plt.figure(dpi=300)
        plt.plot(training_clipped_dd_g['day'], training_clipped_dd_g['GPP_NT_VUT_REF'], 'o', label = 'GPP_NT_VUT_REF', ms = 2, color = 'black')
        plt.plot(training_clipped_dd_g['day'], training_clipped_dd_g['GPP_NT_VUT_REF_mean_doy'], 'o', label = 'GPP_NT_VUT_REF_mean_doy', ms = 2, color = '#e00016')
        #plt.plot(training_clipped_dd_g['day'], training_clipped_dd_g['GPP_NT_VUT_REF_adj'], '-o', label = 'GPP_NT_VUT_REF_adj')
        plt.legend(loc = 'best')
        plt.xticks(rotation = 90)
        plt.title(site_name)
        plt.tight_layout()
        plt.savefig('figs/seasons/' + site_name + '.png')
        plt.close()

        ### ------------- CONCATENATE SITE DF INTO FULL DF -------------

        # HH and daily with grow/nogrow season adjustment
        allsites_event_hh = pd.concat([allsites_event_hh, event])
        allsites_training_hh = pd.concat([allsites_training_hh, training])

        allsites_event_hh_g = pd.concat([allsites_event_hh_g, event_g])
        allsites_training_hh_g = pd.concat([allsites_training_hh_g, training_g])

        allsites_event_dd = pd.concat([allsites_event_dd, event_dd])
        allsites_training_dd = pd.concat([allsites_training_dd, training_dd])

        allsites_event_dd_g = pd.concat([allsites_event_dd_g, event_dd_g])
        allsites_training_dd_g = pd.concat([allsites_training_dd_g, training_dd_g])

        # HH and daily CLIPPED with grow/nogrow season adjustment 
        allsites_event_hh_clipped = pd.concat([allsites_event_hh_clipped, event_clipped])
        allsites_training_hh_clipped = pd.concat([allsites_training_hh_clipped, training_clipped])

        allsites_event_hh_clipped_g = pd.concat([allsites_event_hh_clipped_g, event_clipped_g])
        allsites_training_hh_clipped_g = pd.concat([allsites_training_hh_clipped_g, training_clipped_g])

        allsites_event_clipped_dd = pd.concat([allsites_event_clipped_dd, event_clipped_dd])
        allsites_training_clipped_dd = pd.concat([allsites_training_clipped_dd, training_clipped_dd])

        allsites_event_dd_clipped_g = pd.concat([allsites_event_dd_clipped_g, event_clipped_dd_g])
        allsites_training_dd_clipped_g = pd.concat([allsites_training_dd_clipped_g, training_clipped_dd_g])
    

    # Drop rows that are missing features (CURRENTLY ONLY DOING THIS FOR TWO OF THE DAILY CSVS)
    necessary_features = ['P_F_lag_sum', 'GPP_April']

    allsites_event_clipped_dd = remove_nan_rows(allsites_event_clipped_dd, necessary_features)
    allsites_event_dd_clipped_g = remove_nan_rows(allsites_event_dd_clipped_g, necessary_features)
    allsites_training_clipped_dd = remove_nan_rows(allsites_training_clipped_dd, necessary_features)
    allsites_training_dd_clipped_g = remove_nan_rows(allsites_training_dd_clipped_g, necessary_features)


    # Save all dataframes

    #allsites_event_hh.to_csv(os.path.join(findevents_dir,'allsites_event_hh.csv'))
    #allsites_training_hh.to_csv(os.path.join(findevents_dir, 'allsites_training_hh.csv'))
    #allsites_event_hh_g.to_csv(os.path.join(findevents_dir, 'allsites_event_hh_g.csv'))
    #allsites_training_hh_g.to_csv(os.path.join(findevents_dir,'allsites_training_hh_g.csv')) 
    #allsites_event_dd.to_csv(os.path.join(findevents_dir,'allsites_event_dd.csv'))
    #allsites_training_dd.to_csv(os.path.join(findevents_dir,'allsites_training_dd.csv'))
    #allsites_event_dd_g.to_csv(os.path.join(findevents_dir,'allsites_event_dd_g.csv'))
    #allsites_training_dd_g.to_csv(os.path.join(findevents_dir,'allsites_training_dd_g.csv'))
    #allsites_event_hh_clipped.to_csv(os.path.join(findevents_dir,'allsites_event_hh_clipped.csv'))
    #allsites_training_hh_clipped.to_csv(os.path.join(findevents_dir,'allsites_training_hh_clipped.csv'))
    #allsites_event_hh_clipped_g.to_csv(os.path.join(findevents_dir,'allsites_event_hh_clipped_g.csv'))
    #allsites_training_hh_clipped_g.to_csv(os.path.join(findevents_dir,'allsites_training_hh_clipped_g.csv'))
    allsites_event_clipped_dd.to_csv(os.path.join(findevents_dir,'allsites_event_dd_clipped.csv'))
    allsites_training_clipped_dd.to_csv(os.path.join(findevents_dir,'allsites_training_dd_clipped.csv'))
    allsites_event_dd_clipped_g.to_csv(os.path.join(findevents_dir,'allsites_event_dd_clipped_g.csv'))
    allsites_training_dd_clipped_g.to_csv(os.path.join(findevents_dir,'allsites_training_dd_clipped_g.csv'))

    t1 = time.time()
    print(f"Elapsed time: {round((t1-t0), 2)} seconds")



# Helper functions --------------------------------------------------------------------------------------------------
def remove_nan_rows(df, necessary_features):
    num_days_per_site = df.groupby("SITE_ID")['day'].nunique().reset_index()
    df = df.dropna(subset = necessary_features)
    num_days_after_site = df.groupby("SITE_ID")['day'].nunique().reset_index()
    num_days_per_site['after'] = num_days_after_site['day']
    num_days_per_site['days removed'] = num_days_per_site['day'] - num_days_per_site['after']
    num_days_per_site = num_days_per_site[num_days_per_site['days removed'] > 0]
    print('allsites_event_dd_clipped_g')
    print(num_days_per_site[['SITE_ID', 'days removed']])
    return df
    
def load_dd_data(site_name):
    dd_data = pd.read_csv(os.path.join(paths.FLUXNET_DD_DIR, site_name + '.csv'), parse_dates = ['TIMESTAMP'])
    dd_data['TIMESTAMP'] = pd.to_datetime(dd_data['TIMESTAMP'])
    dd_data['SITE_ID'] = site_name
    dd_data['day'] = dd_data['TIMESTAMP'].dt.floor('d') #overkill but its fine
    return dd_data

def load_hh_data(site_name, usecols = 'all'):
    hour_file = paths.FLUXNET_HH_DIR + '/' + site_name + '.csv'
    if usecols == 'all':
        df_hh = pd.read_csv(hour_file, parse_dates = ['TIMESTAMP_START']) 
    else:
        df_hh = pd.read_csv(hour_file, parse_dates = ['TIMESTAMP_START'], usecols = usecols)  
    df_hh['day'] = df_hh['TIMESTAMP_START'].dt.floor('d')
    df_hh['SITE_ID'] = site_name
    return df_hh


def sum_p_months(event_or_training_df, dd_fluxnet, sum_days = 60, skipped_days = 30):
    # Check for datetimeindex and set if necessary
    if not isinstance(dd_fluxnet.index, pd.DatetimeIndex):
        dd_fluxnet['TIMESTAMP'] = pd.to_datetime(dd_fluxnet['TIMESTAMP'])
        dd_fluxnet = dd_fluxnet.set_index("TIMESTAMP")

    # Simplify by just grabbing necessary column
    daily_fluxnet_df_temp = dd_fluxnet[['P_F']].copy()
    # Roll index back by skipped_days to start summing precip over days before day-of-interest
    daily_fluxnet_df_temp.index = daily_fluxnet_df_temp.index - datetime.timedelta(days=skipped_days)
    # Roll and sum starting from t-skipped_days 
    freq = str(sum_days) + 'D'
    daily_fluxnet_df_temp['P_F_lag_sum'] = daily_fluxnet_df_temp['P_F'].rolling(freq, min_periods=sum_days).sum()
    # Roll index back to original
    daily_fluxnet_df_temp.index = daily_fluxnet_df_temp.index + datetime.timedelta(days=skipped_days)
    # Merge P_F_lag_sum to event_or_training_df
    event_or_training_df = event_or_training_df.merge(daily_fluxnet_df_temp[['P_F_lag_sum']], how = 'left', left_on = 'day', right_index = True)
    return event_or_training_df


def april_gpp(event_or_test_df, df_dd, doy_start = 92, doy_end = 100, p_thresh = 1):
    df = fluxnet_tools.add_time_columns_to_df(df_dd, timestamp_col = 'TIMESTAMP')
    df = df[df['P_F'] < p_thresh]
    df = df[df['DOY'] >= doy_start]
    df = df[df['DOY'] <= doy_end]
    gpps = df.groupby('Year')['GPP_NT_VUT_REF'].mean() 
    gpps = gpps.to_frame().rename(columns={'GPP_NT_VUT_REF':'GPP_April'}).reset_index()
    event_or_test_df['Year'] = event_or_test_df['day'].dt.year
    event_or_test_df = event_or_test_df.merge(gpps, how = 'left', on = 'Year')
    return event_or_test_df



def calculate_growing_season(sites, temperature_col):
    growseason_dict = {}
    for site in sites:
        # Load daily data
        df = pd.read_csv(os.path.join(paths.FLUXNET_DD_DIR, site + '.csv'), parse_dates = ['TIMESTAMP'])
        df['doy'] = df['TIMESTAMP'].dt.dayofyear
        df = df.sort_values(by='doy')

        # Get rid of nan -9999 days
        df = df[df[temperature_col] > -100]

        # How many years are in the record?
        recordlength = df.groupby('doy')[temperature_col].count().reset_index()
        num_years = recordlength[temperature_col].max()

        # Count how many years (by doy) temperatures are greater than 1 degree
        dftemp = pd.DataFrame(df[df[temperature_col] > 1].groupby('doy')[temperature_col].count())
        dftemp = dftemp.rename(columns = {temperature_col:"n_years"})
        dftemp['dayofyear'] = dftemp.index

        # Get a column where '1' means every year on that doy the temp was >1 deg, else 0
        dftemp['count'] = np.where(dftemp['n_years'] < num_years, 0, 1)
        
        # Cumsum on the count column so that the first_doy and last_doy are the same throughout
        # an entire period of continuous days where temp > 1 deg
        dftemp['id']=dftemp['count'].eq(0).cumsum()
        dftemp['max_consec_days']=dftemp.groupby('id')['count'].transform('cumsum')
        dftemp['first_doy']=dftemp.groupby('id')['dayofyear'].transform('first')
        dftemp['last_doy']=dftemp.groupby('id')['dayofyear'].transform('last')

        # The start and end are the mode because that represents the longest consecutive period >1 deg
        # If the temperature is never below 1 deg, then set the entire year as the growing season
        start = dftemp['first_doy'].mode()[0]
        end = dftemp['last_doy'].mode()[0]
        if start == end:
            start = 0
            end = 365

        growseason_dict[site] = {
            'start': start, 
            'stop': end,
            'length': dftemp['max_consec_days'].max()}

    return growseason_dict




def growingseason_normalized_features(training_df, event_or_training_df, cols_to_normalize, time_freq):
    if time_freq == 'dd':
        date_col = 'day'
        merge_cols = ['doy']
        window = 30
    elif time_freq == 'hh':
        date_col = 'TIMESTAMP_START'
        merge_cols = ['hour', 'minute', 'doy']
        training_df['hour'] = training_df.TIMESTAMP_START.dt.hour
        training_df['minute'] = training_df.TIMESTAMP_START.dt.minute
        window = 1 # effectively no window
        event_or_training_df['hour'] = event_or_training_df.TIMESTAMP_START.dt.hour
        event_or_training_df['minute'] = event_or_training_df.TIMESTAMP_START.dt.minute
        
    else: raise ValueError(f"time_freq must be 'dd' or 'hh', got {time_freq}")
    if 'doy' not in training_df.columns: training_df['doy'] = pd.to_datetime(training_df[date_col]).dt.dayofyear
    if 'doy' not in event_or_training_df.columns: event_or_training_df['doy'] = pd.to_datetime(event_or_training_df[date_col]).dt.dayofyear

    # Go through columns and add
    for col in cols_to_normalize:
        if col in training_df.columns:
            adj_col_name = col + '_adj'
            mean_col_name = col + '_mean_doy'
            temp = training_df.groupby(merge_cols)[col].mean().reset_index()
            temp = temp.rename(columns = {col: mean_col_name})
            temp[mean_col_name] = temp[mean_col_name].rolling(window = window, center = True, min_periods = 1).mean()
            event_or_training_df = event_or_training_df.merge(temp, how = 'left', on = merge_cols)
            event_or_training_df[adj_col_name] = event_or_training_df[col] - event_or_training_df[mean_col_name]
        else:
            print(f"{col} not available for growing season adjustment")
    return event_or_training_df


def clip_to_growing_season(df, site_name, growseason_dict, time_col_for_doy = 'TIMESTAMP_START'):
    if 'doy' not in df.columns:
        df['doy'] = df[time_col_for_doy].dt.dayofyear
    df = df[df['doy'] >= growseason_dict[site_name]['start']]
    df = df[df['doy'] <= growseason_dict[site_name]['stop']]
    return df


def calc_daytime_avg_df(event_or_training_df, cols_to_normalize):
    """
    For each column name in col_names, calculate the daytime average of this column
    for every day in df_hh and return a new dataframe with just this daily data.
    Note that col_names is a dictionary, where the key is the column name and the value is
    the number of 'pieces' of the column name to keep for the new daytime avg column, which takes the
    form col_name + _dayavg in lowercase.
    
    Args:
        df_hh (df): input dataframe in hh format with only days and times of day desired
        col_names (dict): dictionary with key: column name, value: pieces of name to keep for final col names
    
    Returns:
        df: dataframe of daily data for all columns in col_names keys
    """
    days_all_daytime_avg = pd.DataFrame()
    days_all_daytime_avg['day'] = event_or_training_df.groupby(['day'])['day'].first()
    days_all_daytime_avg.reset_index(drop=True, inplace=True)
    for col in event_or_training_df.columns:
        if col == 'day':
            pass
        elif col in cols_to_normalize:
            daytime_avg_col = pd.DataFrame({col : event_or_training_df.groupby(['day'])[col].mean()}).reset_index()
            days_all_daytime_avg = days_all_daytime_avg.merge(daytime_avg_col, how='left', on ='day')
        else:
            daytime_avg_col = pd.DataFrame({col : event_or_training_df.groupby(['day'])[col].first()}).reset_index()
            days_all_daytime_avg = days_all_daytime_avg.merge(daytime_avg_col, how='left', on ='day')
    #event_or_training_df = event_or_training_df.drop(cols_to_normalize, axis=1)
    #event_or_training_df = event_or_training_df.merge(days_all_daytime_avg, how = 'left', on = 'day')
    return days_all_daytime_avg

if __name__ == '__main__':
    main()