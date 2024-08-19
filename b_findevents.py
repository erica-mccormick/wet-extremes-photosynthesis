
import glob
import os
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

#### MODIFIED FROM b_HH_PROCESSING.PY ####

def main():
    print('Running a_findevents.py')

    t0 = time.time()
    args = import_args()

    
    # ARGUMENTS
    CSV_OUTPUT_DIR = args.csv_output_dir
    
    if os.path.exists(CSV_OUTPUT_DIR) == False:
        print(f'Making directory {CSV_OUTPUT_DIR} to save logging files.')
        subprocess.call('mkdir ' + os.path.join(CSV_OUTPUT_DIR), shell=True)
        create_logging_files(CSV_OUTPUT_DIR)
        
        # Save args to txt file too
        with open(os.path.join(args.csv_output_dir, 'args_findevents.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)

    percentile_thresh = args.percentile_thresh #0.95
    consec_timestep_thresh = args.consec_timestep_thresh #96
    am = args.am #'9:00'
    pm = args.pm #'17:00'
    p_thresh_hh = args.p_thresh_hh #0.1 # < 
    p_pos_per_daytime_thresh = args.p_pos_per_daytime_thresh # 2 # <=
    qc_allowed_timesteps = qc_frac_to_num(args.am, args.pm, args.qc_frac_allowed)
    
    # Column name: how many pieces of name to keep for the daytime avg column
    features = {'SWC_F_MDS_1':1, 
                'GPP_DT_VUT_REF':2, 
                'GPP_NT_VUT_REF':2,
                'SW_IN_F':2, # Shortwave radiation, incoming 
                'TA_F':1, 
                'LE_F_MDS':1, # Latent heat flux, gapfilled using MDS method
                'VPD_F':1, # Vapor Pressure Deficit consolidated from VPD_F_MDS and VPD_ERA
                'RH':1,
                #'PPFD_IN':2,  # Photosynthetic photon flux density, incoming (MANY SITES DONT HAVE)
                'WS_F':1}
    

    # Column name : threshold, GREATER THAN will be assigned as 'gapfilled'
    qc_dict = {'VPD_F_QC': 1, # 0 = measured; 1 = good quality gapfill; 2 = downscaled from ERA
                'SW_IN_F_QC': 1, # 0 = measured; 1 = good quality gapfill; 2 = downscaled from ERA
                'SWC_F_MDS_1_QC': 2, # 0 = measured; 1 = good quality gapfill; 2 = medium; 3 = poor
                'TA_F_QC': 1, # 0 = measured; 1 = good quality gapfill; 2 = downscaled from ERA
                'LE_F_MDS_QC': 2, # 0 = measured; 1 = good quality gapfill; 2 = medium; 3 = poor
                'WS_F_QC':2} # 0=measured; 2=downscaled from ERA, currently not excluding anything

    ### DFs TO SAVE AT END
    allsites_training = pd.DataFrame()
    allsites_event = pd.DataFrame()
    
    ### STEP 0: GET THE LIST OF SITES TO PROCESS
    
    # Manually removed sites because GPP data is all bad
    skip_sites_gpp = ['IT-Ro2', 
                    'CA-TP1',
                    'CA-TP2',
                    'CA-TP3',
                    'JP-MBF',
                    'US-GLE',
                    'US-Syv',
                    'US-WCr',
                    'SD-Dem',
                    'US-GLE',
                    'IT-Cpz',
                    'CN-Du2',
                    'DE-Lnf',
                    'DK-Eng',
                    'IT-Ren',
                    'US-NR1',
                    'MY-PSO', # NO RH
                    'US-Me1', # swc bad
                    'BR-Sa3', # swc bad
                    'AU-Lox', # swc bad
                    ]
        
    swc_sites = fluxnet_tools.get_sites_with_variable(folder_of_site_csvs = paths.FLUXNET_HH_DIR, 
                                                      col_of_interest = 'SWC_F_MDS_1',
                                                      out_txt_file_path = paths.PATH_TO_SWC_SITES,
                                                      skip_sites_list = skip_sites_gpp, 
                                                      verbose = True)
    
        ### ------------- FOR TESTING ONLY, LIMIT THE NUMBER OF SITES  -------------
    #sites_for_testing = ['US-Me2', 'US-Me3', 'US-Me6', 'US-Me4', 'US-Me5']
    #swc_sites = sites_for_testing


    for file in glob.glob(paths.FLUXNET_HH_DIR + '/*'):
        site_name = file.split('/')[9][0:6] 
        if site_name in swc_sites:

            print(f"Processing {site_name}")
            
            ### STEP 1: LOAD AND PREPARE DATA
            
            # Calculate growing season start and end DOY
            ###growseason_dict = calculate_growing_season(swc_sites, temperature_col = 'TA_F')

            # Load csv of half-hourly raw FLUXNET data
            df_hh = load_hh_data(site_name = site_name, usecols = 'all')


            ### STEP 2: FIND SWC EVENTS
            
            # Find events that meet thresholds and append dataframe of hh timesteps during events
            events = get_events_df(df_hh, site_name, percentile_thresh, consec_timestep_thresh, CSV_OUTPUT_DIR)
            
            # If there are events for the site, continue processing:
            if events.shape[0] > 0:

                
                # Get a dataframe of all of the days between the start-end of each event
                days_events = get_event_days(events, site_name)
                
                ###df_hh = clip_to_growing_season(df_hh, site_name, growseason_dict)

                # Isolate just the daytime for all hh data (am and pm are inclusive)
                df_hh_daytime = df_hh_daytime_only(df_hh, am, pm)

                # Remove days with precipitation (>=p_thresh_hh for >p_pos_per_daytime_thresh timesteps)
                df_hh_no_p = df_hh_no_precip(df_hh_daytime, p_thresh_hh, p_pos_per_daytime_thresh, site_name)
                
                # Remove gapfilled days
                df_hh_nogapfill = drop_gapfilled(df_hh_no_p, qc_dict, features, qc_allowed_timesteps)


                ### STEP 3: Get HH dataset for event vs training/testing

                # Make sure both days_events and df_hh have doy column and date column
                if 'doy' not in days_events.columns:
                    days_events['doy'] = pd.to_datetime(days_events['day']).dt.dayofyear
                if 'doy' not in df_hh_nogapfill.columns:
                    df_hh_nogapfill['doy'] = pd.to_datetime(df_hh_nogapfill['day']).dt.dayofyear
                if 'day' not in df_hh_nogapfill.columns:
                    df_hh_nogapfill['day'] = pd.to_datetime(df_hh_nogapfill['TIMESTAMP_START']).dt.day

                # Separate df_hh_nogapfill into training and event datasets
                combined = pd.merge(df_hh_nogapfill, days_events, on = 'day', how="left", indicator=True)
                hh_train = combined.loc[combined._merge == 'left_only'].drop(columns=['_merge'])   
                hh_event = combined.loc[combined._merge == 'both'].drop(columns=['_merge'])   

                # Make sure you don't have doy_x and doy_y
                hh_train = delete_x_y_from_merge(hh_train, true_col_name = 'doy')
                hh_event = delete_x_y_from_merge(hh_event, true_col_name = 'doy')


                # Choose columns to keep
                keys = []
                for k, v in features.items(): keys.append(k)
                cols_to_keep = keys + ['day', 'doy', 'TIMESTAMP_START', 'start', 'end']
                hh_train = hh_train[cols_to_keep]
                hh_event = hh_event[cols_to_keep]


                # Make sure they each have a SITE_ID column             
                hh_train['SITE_ID'] = site_name
                hh_event['SITE_ID'] = site_name

                # Add daytime-avg features
                #site_days_all = calc_daytime_avg_df(df_hh_nogapfill, col_names = features)

                allsites_training = pd.concat([allsites_training, hh_train])
                allsites_event = pd.concat([allsites_event, hh_event])
            
            else:
                print('\tNo events')  
                
    allsites_training.to_csv(os.path.join(CSV_OUTPUT_DIR, 'allsites_training.csv'))        
    allsites_event.to_csv(os.path.join(CSV_OUTPUT_DIR, 'allsites_event.csv'))     

    t1 = time.time()
    print(f"Elapsed time: {round((t1-t0)/60, 2)} minutes")
    



def import_args():
    parser = argparse.ArgumentParser('Process half-hourly fluxnet data for extreme SWC')
    parser.add_argument('-percentile_thresh', type=float, default=0.95)
    parser.add_argument('-consec_timestep_thresh', type=int, default=96)
    parser.add_argument('-am', type=str, default='9:00')
    parser.add_argument('-pm', type=str, default='17:00')
    parser.add_argument('-p_thresh_hh', type=float, default=0.1)
    parser.add_argument('-p_pos_per_daytime_thresh', type=int, default=1)
    parser.add_argument('-qc_frac_allowed', type=float, default=0.15)
    parser.add_argument('-csv_output_dir', type=str, default='output_hhprocess/test')
    args = parser.parse_args()
    with open(os.path.join(args.csv_output_dir, 'args_findevents.txt'), 'w') as f: json.dump(args.__dict__, f, indent=2)
    return args

def create_logging_files(CSV_OUTPUT_DIR):
    file_names = ['rows_after_loading_csv.txt',
                  'rows_after_dropping_bad_swc.txt',
                  'num_days_events_unfiltered.txt',
                  'num_days_removeprecip_before.txt',
                  'num_days_removeprecip_after.txt',
                  'num_days_training_final.txt',
                  'num_days_events_final.txt',
                  'event_swc_perc_mm.txt']
    for file in file_names:
        with open(os.path.join(CSV_OUTPUT_DIR, file), 'w') as f:
            f.write('SITE_ID, value')
    with open(os.path.join(CSV_OUTPUT_DIR, 'num_days_gapfill_remaining.txt'), 'w') as f:
        f.write('SITE_ID, pre_gapfill_removal, swc_removed, temp_removed, le_removed, vpd_removed')


def delete_x_y_from_merge(df, true_col_name):
    fake_col_name = true_col_name + '_x'
    if fake_col_name in df.columns:
        df[true_col_name] = df[fake_col_name]
        del df[fake_col_name]
        del df[true_col_name + '_y']
    return df

def qc_frac_to_num(am, pm, qc_frac_allowed):
    """
    Allow user to specify the fraction of hh daytime timesteps which can have
    poor quality gapfilling by converting the fraction into the actual number
    of allowed timesteps (the argument taken by the drop_gapfill() function).
    The qc_allowed_timesteps is rounded down and if this rounding is necessary,
    a message prints the new fraction and ratio of allowed/actual timesteps.
    """
    am_int = int(am.split(':')[0])
    pm_int = int(pm.split(':')[0])
    num_timesteps = (pm_int - am_int) * 2 + 1
    qc_allowed_timesteps = math.floor(qc_frac_allowed * num_timesteps)
    frac_actual = qc_allowed_timesteps / num_timesteps
    if frac_actual != qc_frac_allowed:
        print(f"qc_frac rounded down to {round(frac_actual,2)} ({qc_allowed_timesteps}/{num_timesteps} timesteps).")
    return qc_allowed_timesteps

def load_hh_data(site_name, usecols = 'all'):
    hour_file = paths.FLUXNET_HH_DIR + '/' + site_name + '.csv'
    if usecols == 'all':
        df_hh = pd.read_csv(hour_file, parse_dates = ['TIMESTAMP_START']) 
    else:
        df_hh = pd.read_csv(hour_file, parse_dates = ['TIMESTAMP_START'], usecols = usecols)  
    df_hh['day'] = df_hh['TIMESTAMP_START'].dt.floor('d')
    df_hh['SITE_ID'] = site_name
    return df_hh

def load_dd_data(site_name):
    dd_data = pd.read_csv(os.path.join(paths.FLUXNET_DD_DIR, site_name + '.csv'), parse_dates = ['TIMESTAMP'])
    dd_data['TIMESTAMP'] = pd.to_datetime(dd_data['TIMESTAMP'])
    dd_data['SITE_ID'] = site_name
    dd_data['day'] = dd_data['TIMESTAMP'].dt.floor('d') #overkill but its fine
    return dd_data



def clip_months(df, start_month = 4, end_month = 10):
    months = np.arange(start_month, end_month+1)
    if 'day' in df.columns:
        df['Month'] = df['day'].dt.month
    if 'Month' not in df.columns:
        df = fluxnet_tools.add_time_columns_to_df(df, wy_vars = None, timestamp_col = 'TIMESTAMP_START')
    df = df[df['Month'].isin(months)]
    del df['Month']
    return df

    


def find_days_non_null_single_var(df_hh, gapfill_col, qc_allowed_timesteps = 0):
    gapfill_col_newname = gapfill_col + '_qcthresh'
    df_hh[gapfill_col_newname] = np.where(df_hh[gapfill_col].lt(-100), 1, 0)
    qc_sum = pd.DataFrame({'qc_sum' :  df_hh.groupby(['day'])[gapfill_col_newname].sum()}).reset_index()
    df_passes_qc = qc_sum[qc_sum['qc_sum'] <= qc_allowed_timesteps]
    days = df_passes_qc['day'].values
    return days   

def find_days_non_gapfilled_single_var(df_hh, qc_dict, gapfill_col, qc_allowed_timesteps):
    """
    For a given column (ex SWC_F_MDS_1_QC), return an array of days where the value of the qc 
    column is >0 and less than the "bad gapfilling threshold" specified in the dictionary qc_dict 
    for <= the number of hh timesteps (qc_allowed_timesteps).
    The input dataframe (hh timescale) is not returned.
    """
    gapfill_col_newname = gapfill_col + '_qcthresh'
    #df_hh[gapfill_col_newname]= pd.cut(df_hh[gapfill_col],
    #            [-10000, -1, qc_dict[gapfill_col], np.inf],
    #            labels=[1,0,1], #1:bad, 0:good = Negative vals and vals > qc_dict[col] are bad
    #            ordered = False)
    #df_hh[gapfill_col_newname] = pd.factorize(df_hh[gapfill_col_newname], sort=True)[0] # convert labels to ints

    df_hh[gapfill_col_newname] = np.where(df_hh[gapfill_col].gt(qc_dict[gapfill_col]), 1, 0)
    qc_sum = pd.DataFrame({'qc_sum' :  df_hh.groupby(['day'])[gapfill_col_newname].sum()}).reset_index()
    df_passes_qc = qc_sum[qc_sum['qc_sum'] <= qc_allowed_timesteps]
    days = df_passes_qc['day'].values
    return days


def drop_gapfilled(df_hh, qc_dict, features_dict, qc_allowed_timesteps):
    """
    The interesction of each array of days returned by find_days_non_gapfilled_single_var()
    results in the days which meet the criteria for gapfilling. The original df (df_hh) is
    returned for all of the rows (hh timestep) which are in the days array. 
    """
    days = df_hh['day'].values
    
    for key, val in features_dict.items():
        if key in df_hh.columns:
            day_non_null = find_days_non_null_single_var(df_hh, key)
            days = set(days) & set(day_non_null)  # type: ignore  # I'm pretty sure this is working despite the error
    for k, v in qc_dict.items():
        if k in df_hh.columns:
            day_non_gapfill = find_days_non_gapfilled_single_var(df_hh, qc_dict, k, qc_allowed_timesteps)
            days = set(days) & set(day_non_gapfill) # type: ignore  # I'm pretty sure this is working despite the error
        else:
            print(f'\tqc col {k} not found.')
    df_hh = df_hh[df_hh['day'].isin(days)]
    return df_hh

def get_events_df(df, site_name, percentile_thresh, consec_timestep_thresh, CSV_OUTPUT_DIR):
    # Get rid of bad SWC values
    df['SWC_cleaned'] = df['SWC_F_MDS_1'].replace(-9999, np.nan).dropna(axis = 0)
    # Calculate the percentile cutoff threshold
    threshold = df['SWC_cleaned'].quantile(percentile_thresh)
    # Get a column of just the 'extreme' SWC values and how many there are
    df['SWC_EXTREME'] = (df['SWC_cleaned'] >= threshold).astype(int)
    df['SWC_EXTREME_COUNTER'] = df.groupby(df['SWC_EXTREME'].eq(0).cumsum()).cumcount()
    # Find events and add summary columns
    events = event_identification_tools.find_events(df = df, consec_timestep_thresh = consec_timestep_thresh)
    if events.shape[0] > 0:
        events['SITE_ID'] = site_name
        events['SWC_threshold'] = threshold
        events['event_count'] = events.shape[0]
        events['percentile_thresh'] = percentile_thresh
        events['consec_timestep_thresh'] = consec_timestep_thresh
    
        with open(os.path.join(CSV_OUTPUT_DIR, 'event_swc_perc_mm.txt'), 'a') as f:
            f.write('\n' + site_name + ',' +str(threshold))
        return events
    # Return empty dataframe if no events were found
    else:
        blank = pd.DataFrame()
        return blank

def get_event_days(events_df, site_name):
    """
    Because the get_events_df() function returns a row for each
    event with start and end dates, we need a function to get the
    days that happen between the start and end dates, which constitute 
    the "event" dataset. This function accomplishes that by iterating through
    a range of days between the start and end value of each row. It returns
    a dataframe of just the days that are classified as an 'event' for this single
    site, and also writes the number of days to a file, 'num_days_events_unfiltered.txt'.
    """
    event_days = pd.DataFrame()
    events_df['start_day'] = events_df['start'].dt.floor('d')
    events_df['end_day'] = events_df['end'].dt.floor('d')
    for _, row in events_df.iterrows():
        temp = pd.DataFrame()
        temp['day'] = pd.date_range(row['start_day'], row['end_day'], freq = 'd')
        temp['start'] = row['start']
        temp['end'] = row['end']
        event_days = pd.concat([event_days, temp])
    return event_days


def df_hh_daytime_only(df_hh, am, pm):
    """
    Keep only the rows where TIMESTAMP_START is within am and pm to get the daytime data only.
    
    Args:
        df_hh: dataframe with half-hourly data for a single site
        am (str): starting hour, such as '9:00', inclusive
        pm (str): ending hour, such as '17:00', inclusive
    Returns:
        df, original dataframe but with only rows between am and pm
    """
    df_hh.set_index(df_hh['TIMESTAMP_START'], inplace=True)
    df_hh = df_hh.between_time(am, pm)
    return df_hh


def df_hh_no_precip(df_hh, p_thresh_hh, p_pos_per_daytime_thresh, site_name):
    """
    This function finds the days that have fewer than p_pos_per_daytime_thresh (#) of 
    hh timesteps with less than p_thresh_hh [mm] of precipitation. It writes the total number
    of unique DAYS in the existing df_hh dataset before and after removing the days with precipitation
    to two txt files. It then inner merges the dataframe of days to the original df_hh dataframe
    to return a df_hh dataframe with half-hour data for days that passed the precipitation filtering.
    
    Args:
        p_thresh_hh (float): mm of precipitation, LESS THAN THIS VALUE PASSES
        p_pos_per_daytime_thresh (int): LESS THAN OR EQUAL TO THIS VALUE PASSES
         
    """
    df_hh['p_positive'] = np.where(df_hh['P_F'].lt(p_thresh_hh), 0, 1)
    days_all = pd.DataFrame({'p_pos_timesteps' :  df_hh.groupby(['day'])['p_positive'].sum()}).reset_index()
    days_no_p = days_all[days_all['p_pos_timesteps'] <= p_pos_per_daytime_thresh]

    rows_nop = days_no_p.shape[0]
    rows_all = days_all.shape[0]

    df_hh = pd.merge(df_hh, days_no_p, on = 'day', how="inner")
    return df_hh


def calc_daytime_avg_df(df_hh, col_names):
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
    days_all_daytime_avg['day'] = df_hh.groupby(['day'])['day'].first()
    days_all_daytime_avg.reset_index(drop=True, inplace=True)
    for k, v in col_names.items():
        if k in df_hh.columns:
            col_name_new = ''
            for i in np.arange(v):
                col_name_new = col_name_new + str.lower(k.split('_')[i]) + '_'
            col_name_new = col_name_new + 'dayavg'
            daytime_avg_col = pd.DataFrame({col_name_new : df_hh.groupby(['day'])[k].mean()}).reset_index()
            days_all_daytime_avg = days_all_daytime_avg.merge(daytime_avg_col, how='left', on ='day')
        else:
            print(f'\t{k} feature not found.')
    return days_all_daytime_avg




#### FOR TAKING OUT SEASONAL TREND
def calculate_growing_season(sites, temperature_col):
    growseason_dict = {}
    for site in sites:
        # Load half-hourly data
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


def clip_to_growing_season(df, site_name, growseason_dict, time_col_for_doy = 'TIMESTAMP_START'):
    if 'doy' not in df.columns:
        df['doy'] = df[time_col_for_doy].dt.dayofyear
    df = df[df['doy'] >= growseason_dict[site_name]['start']]
    df = df[df['doy'] <= growseason_dict[site_name]['stop']]
    return df


def normalize_to_growing_season(df, cols_to_normalize, site_name, time_col_for_doy = 'day', save_dir = ''):
    if 'doy' not in df.columns:
        df['doy'] = pd.to_datetime(df[time_col_for_doy]).dt.dayofyear
    for col in cols_to_normalize:
        if col in df.columns:
            new_col_name = col + '_mean_doy'
            #final_col_name = col + '_final' # Used for making plots, but want _dayavg for final
            temp = pd.DataFrame(df.groupby('doy')[col].mean())
            temp = temp.rename(columns = {col: new_col_name})
            temp[new_col_name] = temp[new_col_name].rolling(window = 30, center = True, min_periods = 1).mean()
            df = df.merge(temp, how = 'left', on = 'doy')
            og_col_name = col + '_original'
            df[og_col_name] = df[col]
            df[col] = df[col] - df[new_col_name]

            #plt.figure()
            #plt.title(site_name) 
            #plt.plot(df['doy'], df[col], 'o', ms = 2, alpha = 0.75, label = 'Seasonally corrected')
            #plt.plot(df['doy'], df[og_col_name], 'o', ms = 2, alpha = 0.75,  label = col)
            #plt.legend()
            #plt.xlabel('DOY')
            #plt.ylabel(col)
            #plt.savefig(os.path.join(save_dir, site_name + '_' + col + '_growseason.png'))
            #plt.close()

            #file_name = site_name + '_' + col + '_growseason.csv'
            #temp.to_csv(os.path.join(save_dir, file_name))

        else:
            raise ValueError(f"Column '{col}' not in dataframe.")

    return df

def ORIGINAL_normalize_to_growing_season(df, cols_to_normalize, site_name, time_col_for_doy = 'day', save_dir = ''):
    if 'doy' not in df.columns:
        df['doy'] = pd.to_datetime(df[time_col_for_doy]).dt.dayofyear
    for col in cols_to_normalize:
        if col in df.columns:
            new_col_name = col + '_mean_doy'
            #final_col_name = col + '_final' # Used for making plots, but want _dayavg for final
            temp = pd.DataFrame(df.groupby('doy')[col].median())
            temp = temp.rename(columns = {col: new_col_name})
            temp[new_col_name] = temp[new_col_name].rolling(window = 60, center = True, min_periods = 1).mean()
            df = df.merge(temp, how = 'left', on = 'doy')
            df[col] = df[col] - df[new_col_name]

            file_name = site_name + '_' + col + '_growseason.csv'
            temp.to_csv(os.path.join(save_dir, file_name))

        else:
            raise ValueError(f"Column '{col}' not in dataframe.")

    return df
            

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



if __name__ == '__main__':
    main()