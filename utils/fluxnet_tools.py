
import datetime
import glob
import os
import pickle
import time
import zipfile

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import stats
from tqdm import tqdm


def plot_all_swc_event_timeseries(extreme_df, site_list, fig_dir, swc_days_before_event = 30, swc_days_after_event = 10, p_perc_thresh = 0.99, bad_data_df = None, fluxnet_dir_dd = '', fluxnet_dir_hh = '', plot_gpp = False, plot_grid = False, just_plot_full_timeseries = False):
    """
    Plots timeseries of precipitation, GPP, and SWC for each "event" at all sites in site_list. Both full timeseries with 
    events highlighted and zoomed-in event timeseries are saved in folders titled by the name of the site, in the folder specified in
    fig_dir.
    
    Args:
        extreme_df: df with one event per row and SITE_ID column
        site_list (``list`` of ``str``): list of site names that correspond to SITE_ID
        fig_dir (``str``): directory to save all figures in
        swc_days_before_event (int): For zoomed-in plots, how many days before SWC event to show (default: 30)
        swc_days_after_event (int): For zoomed-in plots, how many days after SWC event to show (default: 10)
        p_perc_thresh (float): What percentile precipitation to mark as 'extreme' with black line (default: 0.99)
        bad_data_df (optional, df): df with days and corresponding sites to be removed for not pasing QC check when plotting (default: None)
        fluxnet_dir_dd (str): directory where daily fluxnet unzipped csvs are stored (default: '')
        fluxnet_dir_hh (str): directory where half-hourly fluxnet unzipped csvs are stored (default: '')
        plot_gpp (optional, bool): Show GPP on plot or not (default: False)
        plot_grid (optional, bool): Plot gridlines (default: False)
        just_plot_full_timeseries (optional, bool): If True, skip plotting the zoomed-in plots and just save full-site timeseries (default: False)
        
    """
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    if not os.path.exists(os.path.join(fig_dir, 'full_timeseries')):
        os.makedirs(os.path.join(fig_dir, 'full_timeseries'))

    # Set grid parameter
    plt.rc('grid', linestyle='-', color='black', alpha = 0.5, linewidth=1)

    for file in tqdm(glob.glob(fluxnet_dir_dd + '/*')):
        site_name = file.split('/')[9][0:6] 
        if site_name in site_list:
            # READ IN FILES AND DO NECESSARY CLEANING
            #print('Parsing ' + site_name)
            day_file = fluxnet_dir_dd + '/' + site_name + '.csv'
            hour_file = fluxnet_dir_hh+ '/' + site_name + '.csv'

            # Read df and parse dates
            df_day = pd.read_csv(day_file, usecols = ['TIMESTAMP', 'P_F', 'GPP_DT_VUT_REF'], parse_dates = ['TIMESTAMP'], index_col = 'TIMESTAMP')
            df_hh = pd.read_csv(hour_file, usecols = ['TIMESTAMP_START', 'SWC_F_MDS_1'], parse_dates = ['TIMESTAMP_START'], index_col = 'TIMESTAMP_START')
            
            df_day['TIMESTAMP'] = df_day.index
            df_day['day'] = df_day['TIMESTAMP'].dt.floor('d')
                         
            df_hh['TIMESTAMP_START'] = df_hh.index
            df_hh['day'] = df_hh['TIMESTAMP_START'].dt.floor('d')
            
            # Make sure xlims are the same in the top and bottom
            # Save the xmin and xmax before removing bad data
            xmin = df_hh.index.min()
            xmax = df_hh.index.max()

            ls_main = '-'
            if bad_data_df is not None:
                bad_data = bad_data_df[bad_data_df['Site'] == site_name]
                bad_data = bad_data[['Site', 'day']]
                df_day = pd.merge(df_day, bad_data, on = 'day', how="left", indicator=True)
                df_day = df_day.loc[df_day._merge == 'left_only'].drop(columns=['_merge'])
                df_hh = pd.merge(df_hh, bad_data, on = 'day', how="left", indicator=True)
                df_hh = df_hh.loc[df_hh._merge == 'left_only'].drop(columns=['_merge'])
                df_day = df_day.set_index('TIMESTAMP')
                df_hh = df_hh.set_index('TIMESTAMP_START')
                ls_main = '-'
    

            # Get rid of bad SWC values
            df_hh['SWC_cleaned'] = df_hh['SWC_F_MDS_1'].replace(-9999, np.nan).dropna(axis = 0)

            # Calculate P_F percentile
            df_day['P_positive'] = df_day['P_F'].replace(0, np.nan)
            p_percentile = df_day['P_positive'].quantile(p_perc_thresh)

            # Choose just the events for this site
            ex_site = extreme_df[extreme_df['SITE_ID'] == site_name]
            ex_site.reset_index(drop=True,inplace=True)
            ex_site['start'] = pd.to_datetime(ex_site.loc[:,'start'])
            ex_site['end'] = pd.to_datetime(ex_site.loc[:,'end']) 
            
            if 'percentile_thresh' in ex_site.columns:
                swc_label = str(ex_site['percentile_thresh'][0] * 100) + ' %'
            else: swc_label = None
            
            ##### MAKING FIGURES #####
            
            ###### FULL TIMESERIES ######
            #print('Plotting ' + site_name)
            # Set up axes
            fig, ax = plt.subplots(dpi=300)
            divider = make_axes_locatable(ax)
            ax_top = divider.append_axes("top", size="40%", pad=.1)

            # Optionally plot GPP on a middle axis
            if plot_gpp:
                ax_middle = divider.append_axes("top", size="40%", pad=.1)
                ax_middle.plot(df_day.index, df_day['GPP_DT_VUT_REF'], ls=ls_main, color = '#713199', lw = 0.4)
                ax_middle.tick_params(labelbottom=False)
                ax_middle.set_ylabel('GPP_DT_REF\n(gC $m^{-2}$ $d^{-1}$)')
                if plot_grid: ax_middle.grid(axis='x')

            # Plot SWC stuff
            ax.plot(df_hh.index, df_hh['SWC_cleaned'], ls=ls_main, color = '#757268', lw = 0.4)
            ax.axhline(y= ex_site['SWC_threshold'][0], c = 'black', ls = '--', label = swc_label)
            ax.set_ylabel('SWC (%)\n(half-hourly)')
            if plot_grid: ax.grid(axis='x')

            # Plot P_F stuff
            ax_top.plot(df_day.index, df_day['P_F'], ls=ls_main, color = '#68acd9', lw = 0.4)
            ax_top.axhline(y = p_percentile, c = 'black', ls = '--', label = str(p_perc_thresh*100) + ' percentile')
            ax_top.tick_params(labelbottom=False)
            ax_top.legend(loc = 'best')
            ax_top.set_ylabel('P_F\n(mm/day)')
            if plot_grid: ax_top.grid(axis='x')

            ax.set_xlim(xmin, xmax)
            ax_top.set_xlim(xmin, xmax)
            if plot_gpp: ax_middle.set_xlim(xmin, xmax) #type: ignore

            ###### EVENT SPECIFIC ######
            for i in range(ex_site.shape[0]):
                
                # Add shading to full timeseries plot
                ax.axvspan(ex_site['start'][i], ex_site['end'][i] , color = '#e3869c', alpha = 0.4)
                ax_top.axvspan(ex_site['start'][i], ex_site['end'][i], color = '#e3869c', alpha = 0.4)
                if plot_gpp: ax_middle.axvspan(ex_site['start'][i], ex_site['end'][i], color = '#e3869c', alpha = 0.4) #type: ignore

                #### ACTUAL EVENT TIMESERIES ####
                if just_plot_full_timeseries == False:

                    # SET UP AXES
                    fig2, ax2 = plt.subplots(dpi=300)
                    divider2 = make_axes_locatable(ax2)
                    if plot_gpp: ax_middle2 = divider2.append_axes("top", size="40%", pad=.1)
                    ax_top2 = divider2.append_axes("top", size="40%", pad=.05)
                    
            
                    # Add shading for all storms (in case theyre overlapping)
                    for j in range(ex_site.shape[0]):
                        ax2.axvspan(ex_site['start'][j], ex_site['end'][j] , color = '#e3869c', alpha = 0.4)
                        ax_top2.axvspan(ex_site['start'][j], ex_site['end'][j], color = '#e3869c', alpha = 0.4)
                        if plot_gpp: ax_middle2.axvspan(ex_site['start'][j], ex_site['end'][j], color = '#e3869c', alpha = 0.4) #type: ignore

                    # Optionally plot GPP on a middle axis
                    if plot_gpp:
                        ax_middle2.plot(df_day.index, df_day['GPP_DT_VUT_REF'], ls=ls_main, color = '#713199', lw = 0.4) #type: ignore
                        ax_middle2.tick_params(labelbottom=False) #type: ignore
                        ax_middle2.set_ylabel('GPP_DT_REF\n(gC $m^{-2}$ $d^{-1}$)') #type: ignore

                    # SWC stuff
                    ax2.plot(df_hh.index, df_hh['SWC_cleaned'], ls=ls_main, color = '#757268', lw = 0.4)
                    ax2.axhline(y= ex_site['SWC_threshold'][0], c = 'black', ls = '--', label = swc_label)
                    ax2.set_ylabel('SWC (%)\n(half-hourly)')

                    # P stuff
                    ax_top2.plot(df_day.index, df_day['P_F'], ls=ls_main, color = '#68acd9', lw = 0.4)
                    ax_top2.axhline(y = p_percentile, c = 'black', ls = '--', label = str(p_perc_thresh*100) + ' percentile')
                    ax_top2.tick_params(labelbottom=False)
                    ax_top2.legend(loc = 'best')
                    ax_top2.set_ylabel('P_F\n(mm/day)')
                    
                    # Set xlims
                    xmin2 = ex_site['start'][i] - datetime.timedelta(days = swc_days_before_event)
                    xmax2 = ex_site['end'][i] + datetime.timedelta(days = swc_days_after_event)
                    ax2.set_xlim(xmin2, xmax2)
                    ax_top2.set_xlim(xmin2, xmax2)
                    if plot_gpp: ax_middle2.set_xlim(xmin2, xmax2) #type: ignore
                    ax2.tick_params(axis='x', labelrotation = 90)

                    ax_top2.set_title(site_name)

                    fig2.tight_layout()
                    fig2.savefig(fig_dir + '/' + site_name + '_' + str(pd.to_datetime(ex_site['start'][i])) + '.png')
                    plt.close(fig2)

            ax.tick_params(axis='x', labelrotation = 90)
            ax_top.set_title(site_name)
            fig.tight_layout()
            fig.savefig(fig_dir + '/full_timeseries/' + site_name + '_full.png')
            plt.close(fig)    

def get_sites_with_variable(folder_of_site_csvs, col_of_interest, out_txt_file_path = '', skip_sites_list = [], verbose = True):
    """
    For a given <col_of_interest> (e.g. "P_F" or "SWC_F_MDS_1"), 
    return a list (also saved to <out_txt_file_path> using pickle) 
    of the site csvs in <folder_of_site_csvs> that contain that column.
    
    Args:
        folder_of_site_csvs (str): folder with csvs for each site
        col_of_interest (str): column for a variable to be checked for
        out_txt_file_path (str): path to the location to save the list of site names
        
    Returns:
        list of str: sites_to_keep
    """
    sites_to_keep = []
    
    for file in glob.glob(folder_of_site_csvs + '/*'):
        site_name = file.split('/')[9][0:6] ### THIS LINE MAY NEED TO BE CHANGED FOR DIFF NAME FORMATS
        df_cols = pd.read_csv(file, index_col=0, nrows=0).columns.tolist()
        if col_of_interest in df_cols:
            if site_name not in skip_sites_list:
                sites_to_keep.append(site_name)
    if verbose:
        print(f'{len(sites_to_keep)}/{len(os.listdir(folder_of_site_csvs))} sites contain {col_of_interest} data and are not in skip_sites_list.')
    
    with open(out_txt_file_path,'wb') as f: # needs to be wb bc load needs bytes (b)
        pickle.dump(sites_to_keep,f)
    return sites_to_keep


def add_percentile_col_to_df(df, column, drop_vals  = None, stats_kind = 'strict'):
    new_col_name = column + '_percentile'
    try:
        if drop_vals is not None:
            for val in drop_vals:
                df[column] = df[column].replace(val, np.nan).dropna(axis = 0)
        values = df[column].values
        df[new_col_name] = [stats.percentileofscore(values, i, kind=stats_kind) for i in values]
        return df
    except KeyError:
        print(f"{column} column not in {df}. Percentile column not calculated.")

def add_time_columns_to_df(df, wy_vars = None, timestamp_col = 'TIMESTAMP'):

    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df['DOY'] = df[timestamp_col].dt.dayofyear
    df['Year'] = df[timestamp_col].dt.year
    df['Month'] = df[timestamp_col].dt.month
    df['Week'] = df[timestamp_col].dt.isocalendar().week
    df['Wateryear'] = np.where(~df[timestamp_col].dt.month.isin([10,11,12]),df['Year'], df['Year']+1)
   
    # Function to add wy_vars
    def add_wy(df, col):
        try:
            new_col_name = col + '_wy_cumsum'
            df[new_col_name] = df.groupby('Wateryear')[col].cumsum()
        except KeyError:
            print(f"wy_vars column ('{col}') not in dataframe.")
        return df
    
    # Add wy_vars columns with extreme error handling
    if wy_vars is not None:
        if type(wy_vars) == list:
            for col in wy_vars:
                df = add_wy(df, col)
        elif type(wy_vars) == str:
                df = add_wy(df, wy_vars)
        else:
            raise TypeError(f'wy_vars must be list of str, str, or None. Got {type(wy_vars)}.')
    return df


def filter_by_igbp(df, exclude_igbp): 
    # IGBP descriptions: https://fluxnet.org/data/badm-data-templates/igbp-classification/
    df_filtered = df[~df['IGBP'].isin(exclude_igbp)]
    print(f'Filtering by IGBP... ({df_filtered.shape[0]}/{df.shape[0]}) remaining.')
    return df_filtered


def filter_by_country(df, include_countries = ['USA', 'Canada']):
    df_filtered = df[df['COUNTRY'].isin(include_countries)]
    print(f'Filtering by country... ({df_filtered.shape[0]}/{df.shape[0]}) remaining.')
    return df_filtered


def filter_by_variables(metadata, var_name_list):
    # Set up df with site ID column
    df = pd.DataFrame()
    igbp = metadata[metadata['VARIABLE'] == 'IGBP']
    df['SITE_ID'] = igbp['SITE_ID'].unique()
    
    # Get a single variable as a column in the dataframe
    def grab_single_variable(df, metadata, var_name):
        temp = metadata[metadata['VARIABLE'] == var_name]
        temp = temp.rename(columns={"DATAVALUE": var_name})
        temp2 = temp.groupby(['SITE_ID']).first().reset_index()
        df = df.merge(temp2[['SITE_ID', var_name]], how = 'left', on = 'SITE_ID')
        return df
    
    # Loop through all vars in list
    for i in range(len(var_name_list)):
        df = grab_single_variable(df, metadata, var_name_list[i])
    
    # If any columns have 'DATE', change to datetime
    date_cols = [col for col in df.columns if 'DATE' in col]
    for col in date_cols: df[col] = pd.to_datetime(df[col])
    
    return df


def extract_zipped_timestep_files(dir_fluxnet_raw, path_selected_sites, freq, out_dir, verbose):
    """
    For all of the zip folders in the Fluxnet data directory (dir_fluxnet_raw), check
    if the sites are in the SITE_ID column of the csv at path_selected_sites. If so, 
    grab the csv from within the associated zipped fluxnet folder that corresponds to 
    the desired frequency (ie 'DD' for daily, 'HH' for hourly, etc)
    and save to a new directory (dir_new). One csv for each site is saved to out_dir, which 
    is created if it does not yet exist. If the corresponding zipped folder for a site in
    path_selected_sites does not exist in dir_fluxnet_raw, it will be skipped.
    
    Args:
        dir_fluxnet_raw (str): path to raw fluxnet directory
        path_selected_sites (str): path to df of filtered sites from filter_sites.py
        freq (str): temporal frequency string in file names (ie 'DD', 'HH', 'YY', etc)
        out_dir (str): name of directory where csvs should be saved

    """
    # Make sure dir_new exists, and if not, make it
    if not os.path.exists(out_dir):
        print(f'Making new directory {out_dir}...')
        os.mkdir(out_dir)
        
    # Sites filtered by landcover and country (see filter_sites.py)
    selected_sites = pd.read_csv(path_selected_sites)
    site_id_keep = selected_sites['SITE_ID'].unique()
    # Loop through zip folders in dir_fluxnet_raw directory
    for i in glob.glob(dir_fluxnet_raw + '/*'):
        site_id_files = i.split('/')[9][4:10] # used to be i.split('/')[2][4:10]
        # Check if site is in selected_sites
        if site_id_files in site_id_keep:
            try:
                # Open zipped directory 
                with zipfile.ZipFile(i, 'r') as z:
                    for filename in z.namelist():
                        # Find, open, and save file with freq string (ie DD, etc)
                        if freq in filename:
                            with z.open(filename) as f:
                                df = pd.read_csv(f)
                                if verbose: print(f'Saving {site_id_files}')
                                df.to_csv(os.path.join(out_dir, site_id_files + '.csv'))      
            # Skip directories that aren't zipped (their zipped versions are still found)
            except IsADirectoryError:
                print(f"Skipping {site_id_files}... zipped folder not found.")
                continue
    return 0
            