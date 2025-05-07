
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
    #sites_for_testing = ['AU-Wom', 'AU-GWW']#, 'IT-Isp', 'US-KS2', 'AU-Rob', 'GF-Guy',
     #  'CN-Din', 'AU-DaS', 'AU-Rig']

    # ARGUMENTS
    with open(paths.PATH_TO_SWC_SITES, 'rb') as f:
        swc_sites = pickle.load(f)

    args = import_args()
    
    findevents_dir = args.findevents_dir #'output_findevents/v3' is original submission
    event_filename = args.event_filename #'allsites_event.csv'
    training_filename = args.training_filename #'allsites_training.csv'

    cols_to_convert_to_daily = ['RH', 'SWC_F_MDS_1', 'GPP_DT_VUT_REF', 'GPP_NT_VUT_REF',
                             'SW_IN_F', 'TA_F', 'LE_F_MDS', 'VPD_F', 'WS_F']

    # CALCULATE WARM DAYS (ie GROW SEASON)
    clipping_dict = calculate_warm_days(swc_sites, 'TA_F')

    # LOAD EVENT AND TRAINING HH DATA
    allsites_event = pd.read_csv(os.path.join(findevents_dir, event_filename))
    allsites_training = pd.read_csv(os.path.join(findevents_dir, training_filename))

    # Initialize for later
    allsites_event_clipped_dd= pd.DataFrame()
    allsites_training_clipped_dd= pd.DataFrame()

    for site_name in swc_sites:
        print(f"Processing {site_name}")
        
        ### ------------- ADD FEATURES BASED ON HALF-HOURLY DATA -------------
        # Load daily data and site-specific event and training data
        dd_data = load_dd_data(site_name)
        
        event = allsites_event[allsites_event['SITE_ID'] == site_name]
        training = allsites_training[allsites_training['SITE_ID'] == site_name]
        
        ## Convert 'day' (added during a_findevents.py) to datetime
        event['day'] = pd.to_datetime(event['day'])
        training['day'] = pd.to_datetime(training['day'])

        ## Convert 'TIMETAMP_START (hourly) to datetime
        #event['TIMESTAMP_START'] = pd.to_datetime(event['TIMESTAMP_START'])
        #training['TIMESTAMP_START'] = pd.to_datetime(training['TIMESTAMP_START'])

        ### ------------- CONVERT  HH TO DAILY -------------
        event_dd = calc_daytime_avg_df(event, cols_to_convert_to_daily)
        training_dd = calc_daytime_avg_df(training, cols_to_convert_to_daily)
        
        ### ------------- CLIP TO WARM DAYS -------------
        event_clipped_dd = clip_to_warm_days(event_dd, site_name, clipping_dict, time_col_for_doy = 'day')
        training_clipped_dd = clip_to_warm_days(training_dd, site_name, clipping_dict, time_col_for_doy = 'day')

        ### ------------- ADD MORE COLUMNS -------------
        ## Add summed precipitation over specific range of months
        sum_days = 30
        skipped_days = 30
        avg_days = 30
        
        event_clipped_dd = sum_p_months(event_clipped_dd, dd_data, sum_days, skipped_days)
        training_clipped_dd = sum_p_months(training_clipped_dd, dd_data, sum_days, skipped_days)     

        ## Add mean daily GPP for the previous month (skipping the most recent month)
        event_clipped_dd = lagged_gpp(event_clipped_dd, dd_data, avg_days, skipped_days)
        training_clipped_dd = lagged_gpp(training_clipped_dd, dd_data, avg_days, skipped_days)
        
        ## Add column for LAI climatology
        event_clipped_dd = lai_climatology(event_clipped_dd, site_name, daily_lai_df_path = 'site_metadata/lai_appears_download/Wet-extremes-LAI-MOD15A2H-061-results.csv')
        training_clipped_dd = lai_climatology(training_clipped_dd, site_name, daily_lai_df_path = 'site_metadata/lai_appears_download/Wet-extremes-LAI-MOD15A2H-061-results.csv')

        ### ------------- CONCATENATE SITE DF INTO FULL DF -------------
        allsites_event_clipped_dd = pd.concat([allsites_event_clipped_dd, event_clipped_dd])
        allsites_training_clipped_dd = pd.concat([allsites_training_clipped_dd, training_clipped_dd])

    # Drop rows that are missing features (CURRENTLY ONLY DOING THIS FOR TWO OF THE DAILY CSVS)
    necessary_features = ['SW_IN_F', 'TA_F', 'RH', 'WS_F', 'LAI', 'P_F_lag_sum', 'GPP_lag_mean']
    allsites_event_clipped_dd = remove_nan_rows(allsites_event_clipped_dd, necessary_features)
    allsites_training_clipped_dd = remove_nan_rows(allsites_training_clipped_dd, necessary_features)

    # Save all dataframes
    print(f"Saving to {findevents_dir}")
    allsites_event_clipped_dd.to_csv(os.path.join(findevents_dir,'allsites_event_dd_clipped.csv'))
    allsites_training_clipped_dd.to_csv(os.path.join(findevents_dir,'allsites_training_dd_clipped.csv'))
    
    t1 = time.time()
    print(f"Elapsed time: {round((t1-t0), 2)} seconds")



# Helper functions --------------------------------------------------------------------------------------------------
def import_args():
    parser = argparse.ArgumentParser('Process half-hourly fluxnet data for extreme SWC')
    parser.add_argument('-findevents_dir', type=str, default='output_findevents/v3')
    parser.add_argument('-event_filename', type=str, default='allsites_event.csv')
    parser.add_argument('-training_filename', type=str, default='allsites_training.csv')

    args = parser.parse_args()
    with open(os.path.join(args.findevents_dir, 'args_ddprocessing.txt'), 'w') as f: json.dump(args.__dict__, f, indent=2)
    return args


def remove_nan_rows(df, necessary_features):
    for col in necessary_features:
        print(f"{col}: {df[df[col].isna()].shape[0]}")
    df = df.dropna(subset=necessary_features)
    #num_days_per_site = df.groupby("SITE_ID")['day'].nunique().reset_index()
    #df = df.dropna(subset = necessary_features)
    #num_days_after_site = df.groupby("SITE_ID")['day'].nunique().reset_index()
    #num_days_per_site['after'] = num_days_after_site['day']
    #num_days_per_site['days removed'] = num_days_per_site['day'] - num_days_per_site['after']
    #num_days_per_site = num_days_per_site[num_days_per_site['days removed'] > 0]
    #print(num_days_per_site[['SITE_ID', 'days removed']])
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

def lai_climatology(event_or_training_df, site_name, daily_lai_df_path):
    """
    Downloaded daily LAI from Terra MODIS LAI MOD15A2H.061, 500m, 8-day (2000-02-18 to 2025-01-01):
    Myneni, R., Y. Knyazikhin, T. Park. MODIS/Terra Leaf Area Index/FPAR 8-Day L4 Global 500m SIN Grid V061. 2021, 
    distributed by NASA EOSDIS Land Processes Distributed Active Archive Center, 
    https://doi.org/10.5067/MODIS/MOD15A2H.061. Accessed 2025-01-16.
    
    Used NASA Appears
    
    """
    lai = pd.read_csv(daily_lai_df_path, parse_dates = ['Date']).rename(columns = {"ID":"SITE_ID", "MOD15A2H_061_Lai_500m":"LAI"})
    lai_site = lai[lai['SITE_ID'] == site_name].copy()
    lai_site = lai_site[lai_site['MOD15A2H_061_FparLai_QC_MODLAND_Description']== 'Good quality (main algorithm with or without saturation)']
    lai_site = lai_site[['Date', 'LAI']]
    if lai_site.shape[0] == 0:
        # This is true for one site, IT-Ro1 which has no high-quality LAI
        # To avoid errors in RF, just make it a column of 1s (which won't impact model)
        print("\tNo valid LAI data")
        event_or_training_df['LAI'] = 1
    else:
        # interpolate to daily
        lai_site = lai_site.set_index('Date').resample('D').interpolate().reset_index()
        # need DOY climatology
        lai_site['doy'] = lai_site['Date'].dt.dayofyear.astype(int)
        lai_clim = lai_site.groupby('doy')['LAI'].mean().reset_index()
        # merge climatology with event or training df
        if not 'doy' in event_or_training_df.columns:
            event_or_training_df['TIMESTAMP'] = pd.to_datetime(event_or_training_df['TIMESTAMP'])
            event_or_training_df['doy'] = event_or_training_df['TIMESTAMP'].dt.dayofyear.astype(int)
        event_or_training_df = event_or_training_df.merge(lai_clim[['doy', 'LAI']], on = ['doy'], how = 'left')
    return event_or_training_df


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

def lagged_gpp(event_or_training_df, dd_fluxnet, avg_days = 30, skipped_days=30):
    # Check for datetimeindex and set if necessary
    if not isinstance(dd_fluxnet.index, pd.DatetimeIndex):
        dd_fluxnet['TIMESTAMP'] = pd.to_datetime(dd_fluxnet['TIMESTAMP'])
        dd_fluxnet = dd_fluxnet.set_index("TIMESTAMP")
    # Simplify by just grabbing necessary column
    daily_fluxnet_df_temp = dd_fluxnet[['GPP_DT_VUT_REF']].copy()
    # Roll index back by skipped_days to start summing precip over days before day-of-interest
    daily_fluxnet_df_temp.index = daily_fluxnet_df_temp.index - datetime.timedelta(days=skipped_days)
    # Roll and sum starting from t-skipped_days 
    freq = str(avg_days) + 'D'
    daily_fluxnet_df_temp['GPP_lag_mean'] = daily_fluxnet_df_temp['GPP_DT_VUT_REF'].rolling(freq, min_periods=avg_days).mean()
    # Roll index back to original
    daily_fluxnet_df_temp.index = daily_fluxnet_df_temp.index + datetime.timedelta(days=skipped_days)
    # Merge P_F_lag_sum to event_or_training_df
    event_or_training_df = event_or_training_df.merge(daily_fluxnet_df_temp[['GPP_lag_mean']], how = 'left', left_on = 'day', right_index = True)
    return event_or_training_df


'''
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
'''


def calculate_warm_days(sites, temperature_col):
    clipping_dict = {}
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

        clipping_dict[site] = {
            'start': start, 
            'stop': end,
            'length': dftemp['max_consec_days'].max()}

    return clipping_dict




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


def clip_to_warm_days(df, site_name, clipping_dict, time_col_for_doy):
    if 'doy' not in df.columns:
        df['doy'] = df[time_col_for_doy].dt.dayofyear
    df = df[df['doy'] >= clipping_dict[site_name]['start']]
    df = df[df['doy'] <= clipping_dict[site_name]['stop']]
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
