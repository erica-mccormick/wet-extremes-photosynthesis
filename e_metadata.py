
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import os
import ee
ee.Initialize()
from functools import reduce
import geemap

##########################################
########### IMPORT COORDINATES ###########
##########################################

# Make a folder for geemap byproducts to be saved
if not os.path.exists('temp_files'):
    os.makedirs('temp_files')
    
# Paths
path_to_fluxnet_metadata = 'metadata_fluxnet.csv' # original metadata from fluxnet
path_to_fluxnet_hh_dir = 'FLUXNET_UNZIPPED/HH'
path_to_fluxnet_dd_dir = 'FLUXNET_UNZIPPED/DD'
path_to_anomalies_with_storm_info = 'output_anomalies/RF_DT_targetscaled_event.csv'

# Pandas
meta = pd.read_csv(path_to_fluxnet_metadata)
site_list = meta.SITE_ID.unique()
tier_two = ['RU-Sam', 'RU-SkP', 'RU-Tks', 'RU-Vrk', 'SE-St1', 'ZA-Kru'] # remove tier 2 sites
site_list = [i for i in site_list if i not in tier_two]

# Geopandas
meta_geometry = [Point(xy) for xy in zip(meta['LOCATION_LONG'], meta['LOCATION_LAT'])]
meta_gdf = gpd.GeoDataFrame(meta, geometry=meta_geometry)
meta_gdf = meta_gdf.set_crs('epsg:4326') 

# GEE
def make_points(gdf):
    features=[]
    for _, row in gdf.iterrows():
        x,y = row['geometry'].coords.xy
        cords = np.dstack((x,y)).tolist()
        double_list = reduce(lambda x,y: x+y, cords)
        single_list = reduce(lambda x,y: x+y, double_list)
        g=ee.Geometry.Point(single_list)
        feature = ee.Feature(g)
        feature = feature.set('SITE_ID', row['SITE_ID'])
        features.append(feature)
        ee_object = ee.FeatureCollection(features)
    return ee_object

meta_fc = make_points(meta_gdf)






###############################################################################################
################################# GET FLUXNET DATA FOR A SITE #################################
###############################################################################################

def load_daily_fluxnet(site_id):
    # daily fluxnet data
    dd_data = pd.read_csv(os.path.join(path_to_fluxnet_dd_dir, site_id + '.csv'), parse_dates = ['TIMESTAMP'])
    dd_data['TIMESTAMP'] = pd.to_datetime(dd_data['TIMESTAMP'])
    dd_data['Year'] = dd_data['TIMESTAMP'].dt.year
    dd_data['SITE_ID'] = site_id
    return dd_data


####################################################################################
################################# SOIL INFORMATION #################################
####################################################################################

def extract_soilgrids_gee():
    # https://git.wur.nl/isric/soilgrids/soilgrids.notebooks/-/blob/master/markdown/access_on_gee.md
    layers = ['bdod_mean', 'cfvo_mean', 'clay_mean', 'sand_mean', 'silt_mean', 'soc_mean']
    conversion_factor = {'bdod_mean': 100, 'cfvo_mean': 10, 'clay_mean': 10, 'sand_mean': 10, 'silt_mean': 10, 'soc_mean': 10}
    all_soil = meta_gdf[['SITE_ID']]
    for l in layers:
        print(f"Downloading {l} from SoilGrids for all depths")
        image = ee.Image("projects/soilgrids-isric/" + l)
        band_names = image.bandNames().getInfo()
        path = 'temp_files/' + l.split('_')[0] + '.csv'
        geemap.extract_values_to_points(meta_fc, image, path)
        extracted = pd.read_csv(path)[band_names + ['SITE_ID']]
        extracted[str(l) + '_alldepth'] = extracted[band_names].mean(axis = 1)
        extracted = extracted[['SITE_ID', str(l) + '_alldepth', str(l.split('_')[0]) + '_0-5cm_mean']] # comment out if you want all depth increments instead of avg and surface
        # Apply conversion factor
        extracted[str(l) + '_alldepth'] = extracted[str(l) + '_alldepth'] / conversion_factor[l]
        extracted[str(l.split('_')[0]) + '_0-5cm_mean'] = extracted[str(l.split('_')[0]) + '_0-5cm_mean'] / conversion_factor[l]
        all_soil = all_soil.merge(extracted, how = 'left', on = 'SITE_ID')
    return all_soil

def soil_weight_to_percent(all_soil):
    ### Convert sand, silt, and clay from g/kg to percent texture
    all_soil['total'] = all_soil['sand_mean_alldepth'] + all_soil['silt_mean_alldepth'] + all_soil['clay_mean_alldepth']
    all_soil['sand'] = (all_soil['sand_mean_alldepth'] / all_soil['total']) * 100
    all_soil['silt'] = (all_soil['silt_mean_alldepth'] / all_soil['total']) * 100
    all_soil['clay'] = (all_soil['clay_mean_alldepth'] / all_soil['total']) * 100
    # Just the surface (0-5cm)
    all_soil['total_surface'] = all_soil['sand_0-5cm_mean'] + all_soil['silt_0-5cm_mean'] + all_soil['clay_0-5cm_mean']
    all_soil['sand_surface'] = (all_soil['sand_0-5cm_mean'] / all_soil['total']) * 100
    all_soil['silt_surface'] = (all_soil['silt_0-5cm_mean'] / all_soil['total']) * 100
    all_soil['clay_surface'] = (all_soil['clay_0-5cm_mean'] / all_soil['total']) * 100
    # Only keep columns with percentage units
    all_soil = all_soil[['SITE_ID', 'sand', 'silt', 'clay', 'sand_surface', 'silt_surface', 'clay_surface',
                         'bdod_mean_alldepth', 'soc_mean_alldepth', 'cfvo_mean_alldepth']]
    return all_soil
    
    
def calculate_ksat(all_soil):
    # Crosby et al., 1984    
    # https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/WR020i006p00682?src=getftr
    # mean log ksat in inches per hour
    # sand in percent of total
    # clay in percent of total
    all_soil['ksat_inches'] = 25.4 + 10**(-0.6 + 0.012 * all_soil['sand_surface'] - 0.0064 * all_soil['clay_surface'])
    all_soil['ksat_surface'] =  all_soil['ksat_inches'] * (1/60) * (1/60) * (25.4) # 1hr/60min, 1min/60sec, 25.4mm/1inch = mm/second
    del all_soil['ksat_inches']
    return all_soil
    

def soil_stats(site_list):
    mean_swc = []
    median_swc = []
    max_swc = []
    min_swc = []
    skew = []
    swc_99, swc_90, swc_perc_diff = [], [] ,[]
    s = []
    for site in site_list:
        try:
            dd_fluxnet = load_daily_fluxnet(site)[['SITE_ID', 'SWC_F_MDS_1']]
            dd_fluxnet = dd_fluxnet[dd_fluxnet['SWC_F_MDS_1'] >= 0]
            #try:
            skewness = dd_fluxnet['SWC_F_MDS_1'].skew()
            skew.append(skewness)
            #:
             #   print('skew not working')
            #    skew.append(np.nan)
             #   continue
            try:
                swc_99_temp = np.percentile(dd_fluxnet['SWC_F_MDS_1'].values, 99)
                swc_99.append(swc_99_temp)
            except IndexError:
                swc_99.append(np.nan)
                
            try:
                swc_90_temp = np.percentile(dd_fluxnet['SWC_F_MDS_1'].values, 90)
                swc_90.append(swc_90_temp)
            except IndexError:
                swc_90.append(np.nan)
                
            mean_swc.append(dd_fluxnet['SWC_F_MDS_1'].mean())
            median_swc.append(dd_fluxnet['SWC_F_MDS_1'].median())
            max_swc.append(dd_fluxnet['SWC_F_MDS_1'].max())
            min_swc.append(dd_fluxnet['SWC_F_MDS_1'].min())
            swc_perc_diff.append(swc_99_temp - swc_90_temp)
            s.append(site)
            
        # If a site comes up that doesn't have SWC data 
        except KeyError:
            continue

    
    swc_stats = pd.DataFrame({"SITE_ID":s, "mean_swc":mean_swc, "median_swc":median_swc, 'max_swc':max_swc, 'min_swc':min_swc,
                              'swc_99':swc_99, 'swc_90':swc_90, 'swc_99_min_90':swc_perc_diff, "soil_skew":skew})
    return swc_stats
    
###########################################################################
################################# ARIDITY #################################
###########################################################################

def mean_annual_climate(site_list):
    climate = pd.DataFrame()
    for site in site_list:
        dd_fluxnet = load_daily_fluxnet(site)
        MAP = dd_fluxnet.groupby('Year')['P_F'].sum().mean()
        MAT = dd_fluxnet.groupby('Year')['TA_F'].mean().mean()
        dd_fluxnet['ET'] = dd_fluxnet['LE_F_MDS'] / 2.45e6  * 86400 # LE in W m-2	 # lambda in J/kg
        ET_annual = dd_fluxnet.groupby('Year')['ET'].sum().mean()
        aridity = ET_annual / MAP
        climate_temp = pd.DataFrame({"SITE_ID":site,
                                     "MAP": MAP,
                                     "MAT":MAT,
                                     "ET_annual_mean": ET_annual,
                                     "P/ET": aridity}, index = [0])
        climate = pd.concat([climate, climate_temp])
    return climate


def mean_seasonal_precipitation(site_list):
    precip, season, site_name = [], [], []
    for site in site_list:
        dd_fluxnet = load_daily_fluxnet(site)
        dd_fluxnet['Month'] = dd_fluxnet['TIMESTAMP'].dt.month
        winter = dd_fluxnet[dd_fluxnet['Month'].isin([1,2,3])]
        spring = dd_fluxnet[dd_fluxnet['Month'].isin([4,5,6])]
        summer = dd_fluxnet[dd_fluxnet['Month'].isin([7,8,9])]
        fall = dd_fluxnet[dd_fluxnet['Month'].isin([10,11,12])]
        season_df = [winter, spring, summer, fall]
        season_name = ['winter', 'spring', 'summer', 'fall']
        for i in range(len(season_df)):
            precip.append(season_df[i].groupby('Year')['P_F'].sum().mean())
            season.append(season_name[i])
            site_name.append(site)
    seasonal_precip = pd.DataFrame({"SITE_ID":site_name, "P_F": precip, "season":season})
    return seasonal_precip


def extract_modis_pet(start_month = '01', end_month = '01', final_col_name = 'PET'):
    # Start month inclusive, end month exclusive
    if not isinstance(start_month, str):
        raise TypeError("start and end month must be string, such as '01'")
    print(f"Downloading MODIS PET from GEE")
    # Get mean annual PET from 8-day GEE
    modis_gee = ee.ImageCollection("MODIS/NTSG/MOD16A2/105").select('PET')
    all_years = ee.Image(1)
    for start in np.arange(2000, 2013):
        # Make sure its the same year if we're doing a single season
        if start_month == end_month:
            end_year = pd.to_datetime(str(start + 1) + '-' + end_month + '-01')
        elif end_month == '01':
            end_year = pd.to_datetime(str(start + 1) + '-' + end_month + '-01')
        else:
            end_year = pd.to_datetime(str(start) + '-' + end_month + '-01')
            
        start_year = pd.to_datetime(str(start) + '-' + start_month + '-01')
        modis_annual = modis_gee.filterDate(start_year, end_year)
        modis_annual = modis_annual.reduce(ee.Reducer.sum())
        
        # Add a band for each year
        all_years = all_years.addBands(modis_annual)
        
    # Take image with one band per year and make into a collection
    bands = all_years.bandNames()
    image_list = bands.map(lambda i : all_years.select(i))
    pet = ee.ImageCollection.fromImages(image_list)
    
    # Get mean of all years and apply scaling factor
    pet = all_years.reduce(ee.Reducer.mean()).multiply(0.1) # scaling factor, in kg/m2
    pet = pet.reproject(crs = 'EPSG:4326', scale = 500)
    # Extract values for metadata coordinates
    path = 'temp_files/pet.csv'
    geemap.extract_values_to_points(meta_fc, pet, path)
    extracted_pet = pd.read_csv(path)
    extracted_pet[final_col_name] = extracted_pet['first']
    del extracted_pet['first']
    del extracted_pet['system:index']
    return extracted_pet


def create_aridity_dataset(site_list):
    print("Creating aridity dataset using MODIS PET and FLUXNET P.")
    # MODIS PET
    pet_annual = extract_modis_pet('01', '01', 'PET') 
    pet_winter = extract_modis_pet('01', '04', 'PET_winter') 
    pet_spring = extract_modis_pet('04', '07', 'PET_spring') 
    pet_summer= extract_modis_pet('07', '10', 'PET_summer') 
    pet_fall = extract_modis_pet('10', '01', 'PET_fall') 
    
    ### ANNUAL ###
    annual_p = mean_annual_climate(site_list)
    annual_p = annual_p[['SITE_ID', 'MAP']]
    aridity_annual = pet_annual.merge(annual_p, how = 'inner', on = 'SITE_ID')
    aridity_annual['aridity_annual'] = aridity_annual['MAP'] / aridity_annual['PET']
    aridity_annual = aridity_annual[['SITE_ID', 'aridity_annual', 'PET']]

    ### SEASONAL ###
    # Convert each PET seasonal df to one that matches the seasonal P
    pet_seasonal = pet_winter.merge(pet_spring, how = 'left', on = 'SITE_ID').merge(pet_summer, how = 'left', on = 'SITE_ID').merge(pet_fall, how = 'left', on = 'SITE_ID')
    pet_seasonal_melt = pet_seasonal.melt(id_vars = 'SITE_ID')
    pet_seasonal_melt['season'] = [i.split('_')[1] for i in pet_seasonal_melt['variable'].values]
    pet_seasonal_melt['PET'] = pet_seasonal_melt['value']
    del pet_seasonal_melt['variable']
    del pet_seasonal_melt['value']

    # FLUXNET P and aridity calculation (P/PET)
    seasonal_p = mean_seasonal_precipitation(site_list)
    aridity = pet_seasonal_melt.merge(seasonal_p, how = 'inner', on = ['SITE_ID', 'season'])
    aridity['PET/P'] = aridity['PET'] / aridity['P_F']

    # Convert back to wide format required for future metadata
    aridity_pivot = aridity[['SITE_ID', 'PET/P', 'season']].pivot_table(index = 'SITE_ID', columns = 'season', values = 'PET/P', aggfunc = 'first')
    aridity_pivot.columns = [f'aridity_{i}' for i in aridity_pivot.columns]
    aridity_pivot = aridity_pivot.reset_index()
    aridity_pivot = aridity_pivot.merge(aridity_annual, how = 'inner', on = 'SITE_ID')
    
    return aridity_pivot
    
    
###########################################################################
############################# TOPOGRAPHY ##################################
###########################################################################
    
def extract_twi_gee():
    merit_hydro = ee.Image("MERIT/Hydro/v1_0_1")
    slope_angle = ee.Terrain.slope(merit_hydro.select('elv'))
    # Extract slope angle and twi to temporary files folder
    geemap.extract_values_to_points(meta_fc, slope_angle, 'temp_files/merit_slope_angle_gee.csv')
    geemap.extract_values_to_points(meta_fc, merit_hydro, 'temp_files/merit_twi_gee.csv')   
    # Pull back in  
    slope = pd.read_csv('temp_files/merit_slope_angle_gee.csv')[['SITE_ID', 'first']].rename(columns={"first":"slope"})
    twi = pd.read_csv('temp_files/merit_twi_gee.csv')[['SITE_ID', 'elv', 'dir', 'upa', 'hnd']]
    # Merge
    twi = twi.merge(slope, how = 'left', on = 'SITE_ID')  
    # Convert slope to radian, then calculate tangent, and finally, TWI
    twi['slope_radian'] = twi['slope'] * (np.pi / 180)
    twi['tan_slope_radian'] = np.tan(twi['slope_radian'])
    twi['twi'] = np.log(twi['upa'] / twi['tan_slope_radian'])
    twi = twi[['SITE_ID', 'elv', 'hnd', 'twi']]
    return twi



###########################################################################
############################### KOPPEN ####################################
###########################################################################

def extract_koppen_climate_gee():
    # Imported to GEE from: https://people.eng.unimelb.edu.au/mpeel/koppen.html
    koppen_img = ee.Image('users/ericaelmstead/20_RockMoisture/KoppenClimate/Koppen_Global')
    geemap.extract_values_to_points(meta_fc, koppen_img, 'temp_files/koppen.csv')
    koppen = pd.read_csv('temp_files/koppen.csv')[['SITE_ID', 'first']].rename(columns={"first":"koppen"})
    
    # Labels for koppen values, in order!
    labels = ['Af', 'Am', 'Aw', 
                'BWh', 'BSh', 'BWk', 'BSk', 
                'Csa', 'Csb', 'Csc', 'Cwa', 'Cwb', 'Cwc', 'Cfa', 'Cfb', 'Cfc', 
                'Dsa', 'Dsb', 'Dsc','Dsd', 'Dwa', 'Dwb', 'Dwc', 'Dwd', 'Dfa', 'Dfb', 'Dfc', 'Dfd',
                'ET', 'EF', 'ET high elev', 'EF high elev']
    ids = np.arange(1,33)
    koppen_dict = dict(zip(ids, labels))
    
    print(koppen.head())
    koppen['koppen_value'] = koppen['koppen'] # save this column
    koppen = koppen.replace({"koppen":koppen_dict})
    koppen = koppen.rename({"koppen":"koppen_name"})
    return koppen


###########################################################################
########################## STORM STATISTICS ###############################
###########################################################################

def storm_stats(anomaly_path):

    # Read in anomaly dataframe for storm information
    anomaly_df = pd.read_csv(anomaly_path)
    print(anomaly_df.columns)
    # Mean 'extreme storm' properties for each site
    storm_means = anomaly_df.drop_duplicates(
        subset=['SITE_ID', 'storm_start']).groupby(
            "SITE_ID")[['storm_amount', 'storm_length_percentile', 'storm_amount_percentile', 'storm_length', 'day_no_rain_pre']].mean()

    # Rename columsn to reflect that it is a mean
    storm_means_newcols = [i + '_pluvial_event_mean' for i in storm_means.columns]
    newnamedict = {i:j for i, j in zip(storm_means.columns, storm_means_newcols)}
    storm_means = storm_means.rename(columns = newnamedict)
    storm_means = storm_means.reset_index()
    return storm_means


    

### Aridity
aridity = create_aridity_dataset(site_list)
climate = mean_annual_climate(site_list)

### TWI, elevation, and HAND
twi = extract_twi_gee()
#print(twi.head())

### Koppen
koppen = extract_koppen_climate_gee()
#print(koppen.head())

### Soil
all_soil = extract_soilgrids_gee()
all_soil = soil_weight_to_percent(all_soil)
all_soil = calculate_ksat(all_soil)
#print(all_soil.head())
soil_wetness_stats = soil_stats(site_list)

### Storm stats
storms = storm_stats(path_to_anomalies_with_storm_info)
print(storms.head())

### MERGE ALL TOGETHER

meta = meta.merge(
    all_soil, how = 'left', on = 'SITE_ID').merge(
        soil_wetness_stats, how = 'left', on = 'SITE_ID').merge(
        aridity, how = 'left', on = 'SITE_ID').merge(
            twi, how = 'left', on = 'SITE_ID').merge(
                koppen, how = 'left', on = 'SITE_ID').merge(
                    storms, how = 'left', on = 'SITE_ID').merge(
                        climate, how = 'left', on = 'SITE_ID')
                
print(meta.head())
print(meta.columns)

meta.to_csv('metadata_total.csv')