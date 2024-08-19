from utils import fluxnet_tools
from utils import paths
import glob
import os
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
import datetime
import pickle

pd.options.mode.chained_assignment = None  # type: ignore 



def find_events(df, count_col = 'SWC_EXTREME_COUNTER', consec_timestep_thresh = 96): #, allowed_timesteps_below_thresh = 0
    """
    Make a dataframe of the start, end, and duration of events that last at
    least as long as <cons_timestep_thresh>. Currently it is set up for HALF HOURLY DATA ONLY.

    Args:
        df: the dataframe with the timeseries.
        count_col (str): the column that has 0s and 1s corresponding to whether the criteria is met
        consec_timestep_thresh (int): how many timesteps qualify as 'extreme' and will be saved
    Returns:
        df: df of the extreme events start time, end time, and duration
    """
    if 'TIMESTAMP_START' in df.columns.unique():
        df = df.set_index(df['TIMESTAMP_START'])
    
    events = pd.DataFrame()
    event_starts, event_ends, event_durations = [], [], []

    # Isolate the longest storm
    counter_max = df[count_col].max()

    while counter_max >= consec_timestep_thresh:
        end = df[df[count_col] == counter_max].index.tolist()[0]

        start = pd.to_datetime(end) - datetime.timedelta(minutes=(int(counter_max)*30)) + datetime.timedelta(minutes = 30)

        # Remove that range from the df to find the next storm
        df = df.loc[df.index.difference(df.index[df.index.slice_indexer(start, end)])]

        # Save the event information
        event_starts.append(start)
        event_ends.append(end)
        event_durations.append(counter_max)
        
        # Calculate the next biggest storm and continue
        counter_max = df[count_col].max()

    events['start'] = event_starts
    events['end'] = event_ends
    events['duration'] = event_durations

    return events



def find_all_events(path_to_swc_sites, folder_of_site_csvs, events_csv_dir, out_csv_name, percentile_thresh, consec_timestep_thresh, bad_data_df = None, verbose = False):

    events_all = pd.DataFrame()

    # Load in lists of sites that have SWC data
    with open(path_to_swc_sites, 'rb') as fp:
        swc_sites = pickle.load(fp)

    timestamp_col = 'TIMESTAMP_START'
    time_cols = ['TIMESTAMP_START', 'TIMESTAMP_END']
    col_of_interest = 'SWC_F_MDS_1'
    drop_vals = -9999
    counter = 1
    for file in tqdm(glob.glob(folder_of_site_csvs + '/*')):
        site_name = file.split('/')[9][0:6] ### THIS LINE MAY NEED TO BE CHANGED FOR DIFF NAME FORMATS
        if site_name in swc_sites:
            

            if verbose: print(f"{site_name} ({counter}/{len(os.listdir(folder_of_site_csvs))})")
            counter+= 1
            # Read df and parse dates
            df = pd.read_csv(file, parse_dates = time_cols, index_col = timestamp_col)
            df['TIMESTAMP_START'] = df.index
            df['day'] = df['TIMESTAMP_START'].dt.floor('d')

            if bad_data_df is not None:
                bad_data = bad_data_df[bad_data_df['Site'] == site_name]
                bad_data = bad_data[['Site', 'day']]
                df = pd.merge(df, bad_data, on = 'day', how="left", indicator=True)
                df = df.loc[df._merge == 'left_only'].drop(columns=['_merge'])

            # Get rid of bad SWC values
            df['SWC_cleaned'] = df[col_of_interest].replace(drop_vals, np.nan).dropna(axis = 0)

            # Calculate the percentile cutoff threshold
            threshold = df['SWC_cleaned'].quantile(percentile_thresh)

            # Print out the threshold and how many timesteps are above it
            if verbose: print(f"\tSWC {percentile_thresh} perc.: {threshold}\n\tTimesteps > thresh: {len(df['SWC_cleaned']>threshold)}")

            # Get a column of just the 'extreme' SWC values and how many there are
            df['SWC_EXTREME'] = (df['SWC_cleaned'] >= threshold).astype(int)
            df['SWC_EXTREME_COUNTER'] = df.groupby(df['SWC_EXTREME'].eq(0).cumsum()).cumcount()

            # See if there are places where the percentile stays > thresh for 2 days
            events = find_events(df = df, consec_timestep_thresh = consec_timestep_thresh)
            if events.shape[0] > 0:
                events['SITE_ID'] = site_name
                events['SWC_threshold'] = threshold
                events['event_count'] = events.shape[0]
                events_all = pd.concat([events_all, events])
            if verbose: print(f"\tNumber of extreme events: {events.shape[0]}")
        
    events_all['percentile_thresh'] = percentile_thresh
    events_all['consec_timestep_thresh'] = consec_timestep_thresh
    events_all.to_csv(os.path.join(events_csv_dir, out_csv_name))
    return events_all


def calc_events_from_threshold_lists(perc_list, timestep_list, override_existing_files = False):
    existing_files = [i.split('/')[-1] for i in glob.glob(os.path.join(paths.OUT_CSVS_DIR, 'events_lists_2', '*'))]
    for p in perc_list:
        for t in timestep_list:
            print(f"Finding events that exceed {p*100}th % SWC for {t} hh timesteps...")
            out_csv_name = f'extreme_swc_events_{p*100}hh_{t}perc.csv'
            # Check if file is already present unless override = True
            if out_csv_name not in existing_files:
                find_all_events(
                    path_to_swc_sites = paths.PATH_TO_SWC_SITES,
                    folder_of_site_csvs = paths.FLUXNET_HH_DIR,
                    events_csv_dir = paths.EVENTS_CSV_DIR,
                    out_csv_name = out_csv_name,
                    percentile_thresh = p,
                    consec_timestep_thresh = t, 
                    verbose = False)
            else:
                if override_existing_files == False:
                    continue
                else:
                    find_all_events(
                        path_to_swc_sites = paths.PATH_TO_SWC_SITES,
                        folder_of_site_csvs = paths.FLUXNET_HH_DIR,
                        events_csv_dir = paths.EVENTS_CSV_DIR,
                        out_csv_name = out_csv_name,
                        percentile_thresh = p,
                        consec_timestep_thresh = t, 
                        verbose = False)



def how_many_events(events_all_growing, timethresh_1, percthresh_1, timethresh_2 = None, percthresh_2 = None):
    """
    Get a dataframe for just one (or a combination) of thresholds. Also calculates some summary statistics.
    """
    events_all_growing['idx'] = events_all_growing.index
    combo1 = events_all_growing[events_all_growing['consec_timestep_thresh'] == timethresh_1]
    combo1 = combo1[combo1['percentile_thresh'] == percthresh_1]
    if timethresh_2 is not None:
        if percthresh_2 is not None:
            combo2 = events_all_growing[events_all_growing['consec_timestep_thresh'] == timethresh_2]
            combo2 = combo2[combo2['percentile_thresh'] == percthresh_2]
            combo = combo1.merge(combo2, how = 'left', on = ['SITE_ID']).query('start_y >= start_x and start_y <= end_x')
            print(f"[{timethresh_1}hh > {percthresh_1*100}%]: {combo1.shape[0]} events; [{timethresh_2}hh > {percthresh_2*100}%]: {combo2.shape[0]} events.\n\tCombined: {combo.shape[0]} events at {len(combo.SITE_ID.unique())} sites")
            
            # Add a column that says how many mini-storms were in the big storm
            combo['num_events_mini'] = combo.groupby(['SITE_ID', 'start_x'])['SITE_ID'].transform(len)#.plot.bar()
            combo = combo.drop_duplicates(subset=['SITE_ID', 'start_x'], keep = 'first')
            
            # Add in 'normal' columns for plotting script
            combo['start'] = combo['start_x']
            combo['end'] = combo['end_x']
            combo['percentile_thresh'] = combo['percentile_thresh_x']
            combo['SWC_threshold'] = combo['SWC_threshold_x']      
    else:
        combo = combo1.copy()
        print(f"[{timethresh_1}hh > {percthresh_1*100}%]: {combo1.shape[0]} events at {len(combo.SITE_ID.unique())} sites")
    return combo #type: ignore



def plot_timeseries_for_thresh_category(percentile_thresh, 
                                        consec_timestep_thresh, 
                                        extreme_event_df_all,
                                        out_fig_path, 
                                        percentile_thresh_2 = None,
                                        consec_timestep_thresh_2 = None,
                                        plot_days_before_event = 35,
                                        plot_days_after_event = 25,
                                        plot_gpp = True, 
                                        plot_grid = False,
                                        p_percentile_thresh = 0.99,
                                        just_plot_full_timeseries = True):
    
    if not os.path.exists(out_fig_path):
        os.makedirs(out_fig_path)
        os.makedirs(os.path.join(out_fig_path, 'full_timeseries'))

    
    df_one_thresh = extreme_event_df_all[extreme_event_df_all['percentile_thresh'] == percentile_thresh]
    df_one_thresh = df_one_thresh[df_one_thresh['consec_timestep_thresh'] == consec_timestep_thresh]

    if percentile_thresh_2 is not None:
        if consec_timestep_thresh_2 is not None:
            # Isolate second df that meets the second criteria
            combo2 = extreme_event_df_all[extreme_event_df_all['consec_timestep_thresh'] == consec_timestep_thresh_2]
            combo2 = combo2[combo2['percentile_thresh'] == percentile_thresh_2]
            combo = df_one_thresh.merge(combo2, how = 'left', on = ['SITE_ID']).query('start_y >= start_x and start_y <= end_x')
            
            # Print out summary
            print(f"[{percentile_thresh}, {consec_timestep_thresh}]: {df_one_thresh.shape[0]} events; [{percentile_thresh_2}, {consec_timestep_thresh_2}]: {combo2.shape[0]} events. COMBINED: {combo.shape[0]} events")

            # Add a column that says how many mini-storms were in the big storm
            combo['num_events_mini'] = combo.groupby(['SITE_ID', 'start_x'])['SITE_ID'].transform(len)#.plot.bar()
            combo = combo.drop_duplicates(subset=['SITE_ID', 'start_x'], keep = 'first')

            # Add in 'normal' columns for plotting script
            combo['start'] = combo['start_x']
            combo['end'] = combo['end_x']
            combo['percentile_thresh'] = combo['percentile_thresh_x']
            combo['SWC_threshold'] = combo['SWC_threshold_x']
            df_one_thresh = combo.copy()

        else:
            print('2nd threshold not fully specified, so only using the first...')
    else:
        print(f"Plotting {percentile_thresh} percentile, {consec_timestep_thresh} timeseps: {df_one_thresh.shape[0]} events.")

    fluxnet_tools.plot_all_swc_event_timeseries(
        extreme_df = df_one_thresh,
        site_list = df_one_thresh['SITE_ID'].unique(),
        fig_dir = out_fig_path, 
        swc_days_before_event = plot_days_before_event,
        swc_days_after_event = plot_days_after_event,
        p_perc_thresh = p_percentile_thresh,
        fluxnet_dir_dd = paths.FLUXNET_DD_DIR,
        fluxnet_dir_hh = paths.FLUXNET_HH_DIR,
        plot_gpp = plot_gpp,
        plot_grid = plot_grid,
        just_plot_full_timeseries = just_plot_full_timeseries)
