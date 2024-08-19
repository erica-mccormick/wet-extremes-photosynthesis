
import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import paths
from utils import event_identification_tools
from utils import fluxnet_tools
import time
import datetime
import argparse
from scipy.stats import gaussian_kde, percentileofscore
import json
from scipy import stats

### Goals:
# Pull in best model for each site
# Merge with original data from hh_processing
# Do bias reduction bootstrapping
# Aggregate into individual storms
# Export dataframe for future plotting in a python notebook

def main():
    t0 = time.time()
    args = import_args()
    print(f'Running anomalies.py. Saving to output_anomalies folder with out_name {args.out_name}')

    # ------------- Load event and test data with anomalies -------------
    event = pd.read_csv(args.best_event_path, parse_dates = ['day'])
    test = pd.read_csv(args.best_test_path, parse_dates = ['day'])
    # Make sure the hour isn't included
    event['day'] = pd.to_datetime(event['day'].dt.date)
    test['day'] = pd.to_datetime(test['day'].dt.date)

    #print('Event and test length start: ', event.shape[0], test.shape[0])


    # ------------- Merge with metadata -- this is done later -------------
    #metadata = pd.read_csv(args.metadata_path).drop_duplicates(subset='SITE_ID',keep='first')
    #event = event.merge(metadata, how='left', on = ['SITE_ID'])


    # ------------- Do event bootstrapping on anomalies from testing distribution -------------
    print('Bootstrapping events')
    event_with_bootstrap = pd.DataFrame()
    for site in event['SITE_ID'].unique():

        # Separate dataframe by site
        event_site = event[event['SITE_ID']==site]
        test_site = test[test['SITE_ID']==site]

        # Get bootstrapped event day anomalies and save to new dataframe
        event_site['actual - predicted (bootstrapped)'] = debias_anomaly(site, test_site, event_site, num_iters = 1, bins = 15, anomaly_col = 'actual - predicted')
        event_with_bootstrap = pd.concat([event_with_bootstrap, event_site])

    # Save new anomaly column
    event['actual - predicted (bootstrapped)'] = event_with_bootstrap['actual - predicted (bootstrapped)']


    # ------------- Add column of event anomaly percentileof test -------------
    event_new = pd.DataFrame()
    for site in event['SITE_ID'].unique():
        # Separate dataframe by site
        event_site = event[event['SITE_ID']==site]
        test_site = test[test['SITE_ID']==site]
        # Add percentileof column
        event_site = add_event_percentile_of_test(event_site, test_site)
        event_new = pd.concat([event_new, event_site])
    event = event_new


    # ------------- Make dataframe with 25th and 75th percentile of testing distribution per site -------------
    print('Making dataframe with test distribution uncertainty')
    uncertainty = pd.DataFrame()
    for site in event['SITE_ID'].unique():
        test_site = test[test['SITE_ID']==site]
        twentyfifth = np.percentile(test_site['actual - predicted'], 25)
        seventyfifth = np.percentile(test_site['actual - predicted'], 75)
        uncertainty_temp = pd.DataFrame({"SITE_ID":[site],"p25":[twentyfifth], "p75":[seventyfifth]})
        uncertainty = pd.concat([uncertainty, uncertainty_temp])


    # ------------- Add storm information to event dataframe -------------
    print('Adding storm information to event dataframe (takes a few mins)')
    event_data_all = pd.DataFrame()
    for site_id in event['SITE_ID'].unique():
    #for site_id in ["AU-Rig", "CA-SF2", "US-Var"]:
        print(site_id)
        event_data = event[event['SITE_ID'] == site_id]

        ######event_data['day'] = pd.to_datetime(event_data['day'])
        _, dd_event = load_data(site_id = site_id)

        # Calculate storms from daily data and merge with event days. Also save the total storm df.
        storm_df = calculate_storms(dd_event, daily_p_thresh = 0.1) #default 0.1
        #plot_precip_timeseries(storm_df, site_id)
        storm_df['SITE_ID'] = site_id
        event_data = merge_storms_to_events(event_data, storm_df)

        # Add on the storm percentile
        storm_df_null = storm_df[storm_df['storm_length'] > 0] # get rid of non-storm days
        storm_df_null = add_storm_percentile(storm_df_null)

        event_data = event_data.merge(storm_df_null, how = 'left', on = ['storm_start', 'storm_end', 'storm_length', 'storm_amount'])
        event_data = event_data.drop_duplicates(subset=['day']) # doesn't usually do anything
        
        ### Get the maximum value of wind and swc from the beginning of the storm to the anomaly day
        # First add swc_percentile and wind_percentile to dd_event
        dd_event = add_daily_percentile_to_fluxnet_dd(dd_fluxnet = dd_event, column = 'WS_F', new_column_name = 'wind_percentile')
        dd_event = add_daily_percentile_to_fluxnet_dd(dd_fluxnet = dd_event, column = 'SWC_F_MDS_1', new_column_name = 'swc_percentile')
        dd_event = add_daily_percentile_to_fluxnet_dd(dd_fluxnet = dd_event, column = 'P_F', new_column_name = 'p_percentile')
        dd_event = add_daily_percentile_to_fluxnet_dd(dd_fluxnet = dd_event, column = 'TA_F', new_column_name = 'temp_percentile')

        # Get the percentile min, max, and mean from the start of the storm to the anomaly day (inclusive)
        event_data = max_value_to_date_during_storm(dd_fluxnet = dd_event, event_or_training_df = event_data, col_of_interest = 'wind_percentile')
        event_data = max_value_to_date_during_storm(dd_fluxnet = dd_event, event_or_training_df = event_data, col_of_interest = 'swc_percentile')
        event_data = max_value_to_date_during_storm(dd_fluxnet = dd_event, event_or_training_df = event_data, col_of_interest = 'p_percentile')
        event_data = max_value_to_date_during_storm(dd_fluxnet = dd_event, event_or_training_df = event_data, col_of_interest = 'temp_percentile')
        

        # Get the absolute magnitude of the min, max, and mean from the start of the storm to the anomaly day (inclusive)
        event_data = max_value_to_date_during_storm(dd_fluxnet = dd_event, event_or_training_df = event_data, col_of_interest = 'WS_F')
        event_data = max_value_to_date_during_storm(dd_fluxnet = dd_event, event_or_training_df = event_data, col_of_interest = 'SWC_F_MDS_1')
        event_data = max_value_to_date_during_storm(dd_fluxnet = dd_event, event_or_training_df = event_data, col_of_interest = 'P_F')
        event_data = max_value_to_date_during_storm(dd_fluxnet = dd_event, event_or_training_df = event_data, col_of_interest = 'TA_F')
        
        event_data_all = pd.concat([event_data_all, event_data])

    # ------------- Make dataframe with seasonal PET/P -------------
    print('Making dataframe with seasonal PET/P (takes a few mins)')
    all_seasonal = pd.DataFrame()
    for s in event['SITE_ID'].unique():
        _, dd_event = load_data(site_id = s)
        seasonal_mean = mean_seasonal(dd_event)
        seasonal_mean['SITE_ID'] = s
        all_seasonal = pd.concat([all_seasonal, seasonal_mean])



    # ------------- Save dataframes -------------
        # Save a copy of all arguments
    print('Saving 3 dataframes')
    out_args = 'output_anomalies/' + args.out_name + '_args.txt'
    with open(out_args, 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    event_data_all.to_csv('output_anomalies/' + args.out_name + '_event.csv') # event dataframe is also storm dataframe
    uncertainty.to_csv('output_anomalies/'+ args.out_name + '_testuncertainty.csv')
    all_seasonal.to_csv('output_anomalies/'+ args.out_name + '_seasonalclimate.csv')
    t1 = time.time()
    print(f"Elapsed time: {round((t1-t0)/60,2)} minutes")

def import_args():
    parser = argparse.ArgumentParser('Add columns to event and feature data that are not available from FLUXNET')
    parser.add_argument('-best_event_path', type=str, default='output_nn/event_best_march14.csv', help='Output directory from multiple NN runs')
    parser.add_argument('-best_test_path', type=str, default='output_nn/test_best_march14.csv', help='Output directory from multiple NN runs')
    parser.add_argument('-output_findevents_dir', type=str, default='output_findevents/v2', help='Output directory from multiple NN runs')
    parser.add_argument('-metadata_path', type=str, default='metadata.csv', help='Output directory from multiple NN runs')
    parser.add_argument('-out_name', type=str, default='LE', help='Output directory from multiple NN runs')
    args = parser.parse_args()
    return args

# ----------------------------- Add column with percentile of event within test distribution -----------------------------

def add_event_percentile_of_test(event_site_df, test_site_df):
    ps = []
    for _, row in event_site_df.iterrows():
        ps.append(percentileofscore(a = test_site_df['actual - predicted'].values, score = row['actual - predicted']))
    event_site_df['event_percentileof_test'] = ps
    return event_site_df




# ----------------------------- Bootstrapping to debias anomaly -----------------------------


def draw_test_sample(test_site_df, num_vals_to_generate, bins = 15, anomaly_col = 'actual - predicted'):
    """
    Return values drawn from a distribution built from the test anomaly for a single site.

    Args:
        test_site_df (df): dataframe of testing data for a single site
        num_vals_to_generate (int): how many datapoints to return
        bins (int, optional): the bins size when creating histogram from data
        bandwidth (float, optional): bw_method for scipy.stats.gaussian_kde
        anomaly_col (str, optional): the name of the anomaly column to be used as data from test_site_df
    Returns:
        np.array: Random sample from test of size num_vals_to_generate

    """

    # Code modified from: https://stackoverflow.com/questions/17821458/random-number-from-histogram

    data = np.array(test_site_df[anomaly_col])
    hist, bins = np.histogram(data, bins)
    x_grid = np.linspace(min(data), max(data), num_vals_to_generate)

    # Get gaussian kde pdf
    kde = gaussian_kde(data, bw_method = 'scott') # this is default bw_method
    kdepdf = kde.evaluate(x_grid)

    # Randomly draw from cdf
    cdf = np.cumsum(kdepdf)
    cdf = cdf / cdf[-1]
    values = np.random.rand(num_vals_to_generate)
    value_bins = np.searchsorted(cdf, values)
    rand_draw_from_cdf = x_grid[value_bins]

    return rand_draw_from_cdf

def debias_anomaly(site, test_site_df, event_site_df, num_iters, bins = 15, anomaly_col = 'actual - predicted', make_plots = True):
    num_events = event_site_df.shape[0]
    anomaly_corrected = list(np.zeros(num_events))
    anomaly_event = np.array(event_site_df[anomaly_col])

    if make_plots:
        plt.figure(dpi=200, figsize = (7,2))
        plt.plot(event_site_df['actual - predicted'],[0]*len(event_site_df['actual - predicted']), '*', label = 'original', alpha = 0.75)
    
    for i in range(num_iters):
        noise = draw_test_sample(test_site_df, num_events, bins, anomaly_col)
        event_minus_noise = anomaly_event - noise
        anomaly_corrected = anomaly_corrected + event_minus_noise

        if make_plots:
            plt.plot(event_minus_noise,[i+1]*len(event_minus_noise), 'o', alpha = 0.4)

    anomaly_corrected = anomaly_corrected / num_iters

    if make_plots:
        plt.plot(anomaly_corrected,[i+2]*len(anomaly_corrected), '^',  alpha = 0.4)
        plt.ylabel('Iteration')
        plt.xlabel('actual - predicted after subtraction')
        plt.axvline(-0.00005, 0.005, color = 'black')
        plt.title(f'{num_iters} iters with {bins} bins at {site}')
        if not os.path.exists('figs/anomaly_avg_iteration'): os.makedirs('figs/anomaly_avg_iteration')
        plt.savefig(os.path.join('figs','anomaly_avg_iteration', site + '.png'))
        plt.close()

    return anomaly_corrected


# ----------------------------- Functions associated with storms -----------------------------
def load_data(site_id):
    # half-hour fluxnet data
    hh_data = pd.read_csv(os.path.join(paths.FLUXNET_HH_DIR, site_id + '.csv'), parse_dates = ['TIMESTAMP_START'])
    hh_data['TIMESTAMP'] = pd.to_datetime(hh_data['TIMESTAMP_START'])
    hh_data['SITE_ID'] = site_id

    # daily fluxnet data
    dd_data = pd.read_csv(os.path.join(paths.FLUXNET_DD_DIR, site_id + '.csv'), parse_dates = ['TIMESTAMP'])
    dd_data['TIMESTAMP'] = pd.to_datetime(dd_data['TIMESTAMP'])
    dd_data['SITE_ID'] = site_id

    return hh_data, dd_data


def plot_precip_timeseries(storm_df_site, site):
    plt.plot(storm_df_site['TIMESTAMP'], storm_df_site['P_F'], color = 'black')
    #for idx, row in storm_df_site.iterrows():
    #    plt.axvspan(xmin = row['storm_start'], xmax = row['storm_end'], color = 'red', alpha = 0.25)
    plt.title(site)
    plt.ylabel('P_F (daily) [mm]')
    plt.xticks(rotation = 90)
    plt.savefig('figs/storm_timeseries/' + site + '.png', dpi = 300)
    plt.close()


def calculate_storms(dd_fluxnet, daily_p_thresh = 0.1):
    df = dd_fluxnet.copy()
    df['groupId1']=df['P_F'].lt(daily_p_thresh).cumsum()

    df['storm_amount']=df.groupby('groupId1').P_F.transform('sum')

    # Must subtract 1 from storm_length because groupby includes the first
    # day (with no P) as part of the storm group. The total amount is correct, though.
    df['storm_length'] = df.groupby('groupId1')['P_F'].transform('count')
    df['storm_length'] = df['storm_length'] - 1

    # Add the start and end day and add 1 to start day for reason listed above
    df['storm_start']=df.groupby('groupId1').TIMESTAMP.transform('first')
    #print(df['storm_start'].values)

    df['storm_end']=df.groupby('groupId1').TIMESTAMP.transform('last')
    df['storm_start'] = df['storm_start'] + datetime.timedelta(days=1)

    # If the storm-start is one day *after* storm_end, set storm_start = storm_end
    df['storm_start'].where(df['storm_start'] <= df['storm_end'], df['storm_end'], inplace=True)

    # The TIMESTEMP_PLUS1 is the default to align the storm with the event day
    # The PLUSXX is to get storms that are more than one day before the events
    df['TIMESTAMP_PLUS1'] = df['TIMESTAMP'] + datetime.timedelta(days=1)
    df['TIMESTAMP_PLUS2'] = df['TIMESTAMP'] + datetime.timedelta(days=2)
    df['TIMESTAMP_PLUS3'] = df['TIMESTAMP'] + datetime.timedelta(days=3)
    df['TIMESTAMP_PLUS4'] = df['TIMESTAMP'] + datetime.timedelta(days=4)
    df['TIMESTAMP_PLUS5'] = df['TIMESTAMP'] + datetime.timedelta(days=5)
    df['TIMESTAMP_PLUS6'] = df['TIMESTAMP'] + datetime.timedelta(days=6)
    df['TIMESTAMP_PLUS7'] = df['TIMESTAMP'] + datetime.timedelta(days=7)
    df['TIMESTAMP_PLUS8'] = df['TIMESTAMP'] + datetime.timedelta(days=8)
    df['TIMESTAMP_PLUS9'] = df['TIMESTAMP'] + datetime.timedelta(days=9)
    df['TIMESTAMP_PLUS10'] = df['TIMESTAMP'] + datetime.timedelta(days=10)
    df['TIMESTAMP_PLUS11'] = df['TIMESTAMP'] + datetime.timedelta(days=11)
    df['TIMESTAMP_PLUS12'] = df['TIMESTAMP'] + datetime.timedelta(days=12)
    df['TIMESTAMP_PLUS13'] = df['TIMESTAMP'] + datetime.timedelta(days=13)
    df['TIMESTAMP_PLUS14'] = df['TIMESTAMP'] + datetime.timedelta(days=14)
    df['TIMESTAMP_PLUS15'] = df['TIMESTAMP'] + datetime.timedelta(days=15)
    df['TIMESTAMP_PLUS16'] = df['TIMESTAMP'] + datetime.timedelta(days=16)
    df['TIMESTAMP_PLUS17'] = df['TIMESTAMP'] + datetime.timedelta(days=17)
    df['TIMESTAMP_PLUS18'] = df['TIMESTAMP'] + datetime.timedelta(days=18)
    df['TIMESTAMP_PLUS19'] = df['TIMESTAMP'] + datetime.timedelta(days=19)
    df['TIMESTAMP_PLUS20'] = df['TIMESTAMP'] + datetime.timedelta(days=20)
    df['TIMESTAMP_PLUS21'] = df['TIMESTAMP'] + datetime.timedelta(days=21)
    df['TIMESTAMP_PLUS22'] = df['TIMESTAMP'] + datetime.timedelta(days=22)
    df['TIMESTAMP_PLUS23'] = df['TIMESTAMP'] + datetime.timedelta(days=23)
    df['TIMESTAMP_PLUS24'] = df['TIMESTAMP'] + datetime.timedelta(days=24)
    df['TIMESTAMP_PLUS25'] = df['TIMESTAMP'] + datetime.timedelta(days=25)
    df['TIMESTAMP_PLUS26'] = df['TIMESTAMP'] + datetime.timedelta(days=26)
    df['TIMESTAMP_PLUS27'] = df['TIMESTAMP'] + datetime.timedelta(days=27)
    df['TIMESTAMP_PLUS28'] = df['TIMESTAMP'] + datetime.timedelta(days=28)
    df['TIMESTAMP_PLUS29'] = df['TIMESTAMP'] + datetime.timedelta(days=29)
    df['TIMESTAMP_PLUS30'] = df['TIMESTAMP'] + datetime.timedelta(days=30)
    df['TIMESTAMP_PLUS31'] = df['TIMESTAMP'] + datetime.timedelta(days=31)
    df['TIMESTAMP_PLUS32'] = df['TIMESTAMP'] + datetime.timedelta(days=32)
    df['TIMESTAMP_PLUS33'] = df['TIMESTAMP'] + datetime.timedelta(days=33)
    df['TIMESTAMP_PLUS34'] = df['TIMESTAMP'] + datetime.timedelta(days=34)
    df['TIMESTAMP_PLUS35'] = df['TIMESTAMP'] + datetime.timedelta(days=35)
    df['TIMESTAMP_PLUS36'] = df['TIMESTAMP'] + datetime.timedelta(days=36)
    df['TIMESTAMP_PLUS37'] = df['TIMESTAMP'] + datetime.timedelta(days=37)
    df['TIMESTAMP_PLUS38'] = df['TIMESTAMP'] + datetime.timedelta(days=38)
    df['TIMESTAMP_PLUS39'] = df['TIMESTAMP'] + datetime.timedelta(days=39)
    df['TIMESTAMP_PLUS40'] = df['TIMESTAMP'] + datetime.timedelta(days=40)
    df['TIMESTAMP_PLUS41'] = df['TIMESTAMP'] + datetime.timedelta(days=41)
    df['TIMESTAMP_PLUS42'] = df['TIMESTAMP'] + datetime.timedelta(days=42)
    df['TIMESTAMP_PLUS43'] = df['TIMESTAMP'] + datetime.timedelta(days=42)
    df['TIMESTAMP_PLUS44'] = df['TIMESTAMP'] + datetime.timedelta(days=44)
    df['TIMESTAMP_PLUS45'] = df['TIMESTAMP'] + datetime.timedelta(days=45)
    df['TIMESTAMP_PLUS46'] = df['TIMESTAMP'] + datetime.timedelta(days=46)
    df['TIMESTAMP_PLUS47'] = df['TIMESTAMP'] + datetime.timedelta(days=47)
    df['TIMESTAMP_PLUS48'] = df['TIMESTAMP'] + datetime.timedelta(days=48)
    df['TIMESTAMP_PLUS49'] = df['TIMESTAMP'] + datetime.timedelta(days=49)
    df['TIMESTAMP_PLUS50'] = df['TIMESTAMP'] + datetime.timedelta(days=50)
    df['TIMESTAMP_PLUS51'] = df['TIMESTAMP'] + datetime.timedelta(days=51)
    df['TIMESTAMP_PLUS52'] = df['TIMESTAMP'] + datetime.timedelta(days=52)


    df = df[['storm_amount','TIMESTAMP', 'TIMESTAMP_PLUS1', 'TIMESTAMP_PLUS2', 
             'TIMESTAMP_PLUS3', 'TIMESTAMP_PLUS4','TIMESTAMP_PLUS5',
             'TIMESTAMP_PLUS6', 'TIMESTAMP_PLUS7', 'TIMESTAMP_PLUS8', 'TIMESTAMP_PLUS9',
             'TIMESTAMP_PLUS10', 'TIMESTAMP_PLUS11', 'TIMESTAMP_PLUS12', 'TIMESTAMP_PLUS13',
             'TIMESTAMP_PLUS14', 'TIMESTAMP_PLUS15', 'TIMESTAMP_PLUS16', 'TIMESTAMP_PLUS17',
             'TIMESTAMP_PLUS18', 'TIMESTAMP_PLUS19', 'TIMESTAMP_PLUS20', 'TIMESTAMP_PLUS21',
             'TIMESTAMP_PLUS22', 'TIMESTAMP_PLUS23', 'TIMESTAMP_PLUS24', 'TIMESTAMP_PLUS25',
             'TIMESTAMP_PLUS26', 'TIMESTAMP_PLUS27', 'TIMESTAMP_PLUS28', 'TIMESTAMP_PLUS29',
             'TIMESTAMP_PLUS30', 'TIMESTAMP_PLUS31', 'TIMESTAMP_PLUS32', 'TIMESTAMP_PLUS33', 
             'TIMESTAMP_PLUS34', 'TIMESTAMP_PLUS35', 'TIMESTAMP_PLUS36', 'TIMESTAMP_PLUS37', 
             'TIMESTAMP_PLUS38', 'TIMESTAMP_PLUS39', 'TIMESTAMP_PLUS40', 'TIMESTAMP_PLUS41',
             'TIMESTAMP_PLUS42', 'TIMESTAMP_PLUS43', 'TIMESTAMP_PLUS44', 'TIMESTAMP_PLUS45', 
             'TIMESTAMP_PLUS46', 'TIMESTAMP_PLUS47', 'TIMESTAMP_PLUS48', 'TIMESTAMP_PLUS49', 
             'TIMESTAMP_PLUS50', 'TIMESTAMP_PLUS51', 'TIMESTAMP_PLUS52',
             'P_F', 'storm_length', 'storm_start', 'storm_end']]
    return df

def merge_storms_to_events(event_or_training_df, storm_df):
    """
    Because an event can be associated with a storm that happened >1 day before, we roll the
    end day of the storm and save how many days we must subtract 'day_no_rain_pre' while still
    associating the event day with a storm.
    """
    
    def roll_storm(previous_temp_nostorm, timestamp_plusX, day_no_rain_pre, storm_df = storm_df):
        temp_nostorm_NEXT = previous_temp_nostorm[previous_temp_nostorm['storm_amount']==0]
        temp_nostorm_NEXT = temp_nostorm_NEXT[['day']].merge(storm_df[merge_cols + [timestamp_plusX]], how = 'left', left_on = 'day', right_on = timestamp_plusX)
        temp_storm_roll = temp_nostorm_NEXT[temp_nostorm_NEXT['storm_amount']>0]
        temp_storm_roll['day_no_rain_pre'] = day_no_rain_pre
        return temp_storm_roll, temp_nostorm_NEXT
    
    merge_cols = ['storm_amount', 'storm_length', 'storm_start','storm_end']
    event = event_or_training_df[['day']].merge(storm_df[merge_cols + ['TIMESTAMP_PLUS1']], how = 'left', left_on = 'day', right_on = 'TIMESTAMP_PLUS1')

    temp_storm = event[event['storm_amount']>0]
    temp_storm['day_no_rain_pre'] = 0


    temp_nostorm = event[event['storm_amount']==0]
    temp_nostorm = temp_nostorm[['day']].merge(storm_df[merge_cols + ['TIMESTAMP_PLUS2']], how = 'left', left_on = 'day', right_on = 'TIMESTAMP_PLUS2')
    temp_storm_roll1 = temp_nostorm[temp_nostorm['storm_amount']>0]
    temp_storm_roll1['day_no_rain_pre'] = 1

    event_storm_stats = pd.concat([temp_storm, temp_storm_roll1])
    temp_next = temp_nostorm

    for i in np.arange(3,52):
        timestasmp_pluxX = f'TIMESTAMP_PLUS{i}'
        temp_storm_roll, temp_next = roll_storm(temp_next, timestasmp_pluxX, i-1)
        event_storm_stats = pd.concat([event_storm_stats, temp_storm_roll])[['day', 'storm_amount','storm_length','day_no_rain_pre', 'storm_start','storm_end']]

    # If there's still event days without a storm, just assign them a really big day_no_rain_pre number
    temp_next = temp_next[temp_next['storm_amount']==0]
    temp_next['day_no_rain_pre'] = 99
    event_storm_stats = pd.concat([event_storm_stats, temp_next])

    # Merge new columns with original dataframe
    event_or_training_df = event_or_training_df.merge(event_storm_stats, how = 'left', on = 'day', suffixes = ['_x', ''])
    #event_or_training_df = event_or_training_df.merge(event_storm_stats, how = 'left', on = 'day')
    return event_or_training_df


def add_storm_percentile(storm_df):
    p_length, p_amount = [], []
    for _, row in storm_df.iterrows():
        p_length.append(stats.percentileofscore(storm_df['storm_length'], row['storm_length']))
        p_amount.append(stats.percentileofscore(storm_df['storm_amount'], row['storm_amount']))
    storm_df['storm_length_percentile'] = p_length
    storm_df['storm_amount_percentile'] = p_amount
    return storm_df[['storm_start', 'storm_end', 'storm_length', 'storm_amount', 'storm_length_percentile', 'storm_amount_percentile']]

# ----------------------------- Seasonal PET/P -----------------------------

def mean_annuals(dd_fluxnet, event_or_training_df):
    dd_fluxnet['TIMESTAMP'] = pd.to_datetime(dd_fluxnet['TIMESTAMP'])
    dd_fluxnet['Year'] = dd_fluxnet['TIMESTAMP'].dt.year
    MAP = dd_fluxnet.groupby('Year')['P_F'].sum().mean()
    MAT = dd_fluxnet.groupby('Year')['TA_F'].sum().mean()
    dd_fluxnet['ET'] = dd_fluxnet['LE_F_MDS'] / 2.45e6  * 86400 # LE in W m-2	 # lambda in J/kg
    ET_annual = dd_fluxnet.groupby('Year')['ET'].sum().mean()
    event_or_training_df['MAP'] = MAP
    event_or_training_df['MAT'] = MAT
    event_or_training_df['ET_annual_mean'] = ET_annual
    return MAP, MAT, ET_annual



def single_season(season, variable):
    if variable == 'TA_F': output = season.groupby('Year')[variable].mean().mean()
    else: output = season.groupby('Year')[variable].sum().mean()
    return output



def mean_seasonal(dd_fluxnet):
    dd_fluxnet['TIMESTAMP'] = pd.to_datetime(dd_fluxnet['TIMESTAMP'])
    dd_fluxnet['Year'] = dd_fluxnet['TIMESTAMP'].dt.year
    dd_fluxnet['Month'] = dd_fluxnet['TIMESTAMP'].dt.month
    dd_fluxnet['ET'] = dd_fluxnet['LE_F_MDS'] / 2.45e6  * 86400 # LE in W m-2	 # lambda in J/kg

    winter = dd_fluxnet[dd_fluxnet['Month'].isin([1,2,3])]
    spring = dd_fluxnet[dd_fluxnet['Month'].isin([4,5,6])]
    summer = dd_fluxnet[dd_fluxnet['Month'].isin([7,8,9])]
    fall = dd_fluxnet[dd_fluxnet['Month'].isin([10,11,12])]

    save_var = []
    save_season  = []
    save_value = []

    season_df = [winter, spring, summer, fall]
    season_name = ['winter', 'spring', 'summer', 'fall']
    for var in ['ET', 'P_F', 'TA_F']:
        for s in range(len(season_df)):
            save_value.append(single_season(season_df[s], var))
            save_var.append(var)
            save_season.append(season_name[s])
    seasonal_means = pd.DataFrame({"variable":save_var, "season":save_season, 'value':save_value})
    return seasonal_means

    
# ----------------------------- From 'add_columns.py in V3: add max/min/percentile columns -----------------------------

# for each day in event, go between the start of the previous storm and that day and find the max value in the dd_event file
def max_value_to_date_during_storm(dd_fluxnet, event_or_training_df, col_of_interest):
    # Check for datetimeindex and set if necessary
    dd_fluxnet['TIMESTAMP'] = pd.to_datetime(dd_fluxnet['TIMESTAMP'])
    maximum_value = []
    minimum_value = []
    mean_value  = []
    max_day = []
    for _, row in event_or_training_df.iterrows():
        start = row['storm_start']
        end = row['day']

        # If the storm isn't working
        #dd_fluxnet_filtered = dd_fluxnet[dd_fluxnet['TIMESTAMP'].isin(pd.date_range(start, end))]

        try:
            dd_fluxnet_filtered = dd_fluxnet[dd_fluxnet['TIMESTAMP'].isin(pd.date_range(start, end))]
        except:
            print("Storm analysis not working (exception triggered on line 514)")
            maximum_value.append(np.nan)
            minimum_value.append(np.nan)
            mean_value.append(np.nan)
            max_day.append(np.nan)
        
        else:
            # Get the daily fluxnet data for just this time period
            value_max = dd_fluxnet_filtered[col_of_interest].max()
            value_min = dd_fluxnet_filtered[col_of_interest].min()
            value_mean = dd_fluxnet_filtered[col_of_interest].mean()
            maximum_value.append(value_max)
            minimum_value.append(value_min)
            mean_value.append(value_mean)

            # Get the day that the max occured on
            day_max = dd_fluxnet_filtered[dd_fluxnet_filtered[col_of_interest]==value_max]
            if day_max.shape[0] > 0:
                day_max = day_max['TIMESTAMP'].values[0]
                max_day.append(day_max)
            else: max_day.append(np.nan)
            

    event_or_training_df[col_of_interest + '_max'] = maximum_value
    event_or_training_df[col_of_interest + '_min'] = minimum_value
    event_or_training_df[col_of_interest + '_mean'] = mean_value
    event_or_training_df[col_of_interest + '_maxday'] = max_day
    return event_or_training_df

def add_daily_percentile_to_fluxnet_dd(dd_fluxnet, column, new_column_name):
    """
    Just add percentile to the daily fluxnet dataframe, and don't merge into event data
    """
    percentile = []
    for _, row in dd_fluxnet.iterrows():
        perc = stats.percentileofscore(dd_fluxnet[column], row[column])
        percentile.append(perc) 
    dd_fluxnet[new_column_name] = percentile
    return dd_fluxnet



if __name__ == '__main__':
    main()