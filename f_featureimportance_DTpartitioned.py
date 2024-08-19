

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
from sklearn import preprocessing
from sklearn.model_selection import RandomizedSearchCV, KFold, cross_val_score
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV, RandomizedSearchCV
import seaborn as sns
import numpy as np
import pandas as pd
rng = np.random.default_rng()
import itertools
import warnings
import sys
import time
########################################################################################################################
############################## RANDOM FOREST #######################################################
########################################################################################################################
t0 = time.time()

def rf_explain_anom(features, name, search_iters):
    
    n_estimators = [5, 25, 50, 75, 100, 200]
    max_features = [1]
    max_depth = [None, 4, 8, 16, 32]
    min_samples_split = [4, 8, 16, 32]
    min_samples_leaf = [4, 8, 16, 32, 0.1]
    bootstrap = [True, False]
    random_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}

    events_iqr['mean_lai'] = events_iqr['mean_lai'].fillna(events_iqr['mean_lai'].mean())

    # Apply Minmax scaler to X and y
    
    scaler_features = preprocessing.MinMaxScaler()
    scaler_target = preprocessing.MinMaxScaler()

    # Split the data into training and testing sets
    column_names = events_iqr[features].columns
    X = scaler_features.fit_transform(events_iqr[features])
    y = scaler_target.fit_transform(events_iqr['anomaly_norm_gpp'].values.reshape(-1, 1))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle = True, test_size=0.25, random_state=42)
    #print(X_train.shape, y_train.shape)

    y_train = y_train.ravel()
    y_test = y_test.ravel()
    #y = y.ravel()
    print(X.shape, y.shape)
    

    # Inner and outer loop cross-validation
    inner_cv = KFold(n_splits = 3, shuffle = True, random_state = 1)
    outer_cv = KFold(n_splits = 5, shuffle = True, random_state = 1)

    estimator = RandomForestRegressor()
    
        # Search the space defined by the gridsearch random_grid dictionary
    search = RandomizedSearchCV(estimator = estimator,
                                param_distributions = random_grid,
                                cv = inner_cv,
                                n_iter = search_iters,
                                random_state = 42,
                                verbose = 1)


    # Perform cross-validation with inner and outer cv
    scores = cross_val_score(search, X_train, y_train, scoring = 'r2', cv = outer_cv)
    #cv_score_mean.append(scores.mean())
    #cv_score_std.append(scores.std())
    
    # Fit search with X and y
    reg = search.fit(X_train, y_train)
    
    performance = pd.DataFrame({"name":name,
                       "features":[features],
                       "params":[reg.best_params_],
                       "r2":reg.best_score_,
                      # "train_score":reg.score(X, y_train),
                       #"test_score":reg.score(X_test, y_test),
                       'cv_score_mean':scores.mean(),
                       'cv_score_std':scores.std()}, index = [0])
    
    # Get feature importances
    r = permutation_importance(reg, X_test, y_test, n_repeats=10, random_state=1)
    importances = pd.DataFrame({"importance":r['importances_mean'], "feature":column_names})
    importances['group'] = importances['feature'].map(groups)
    importances = importances.sort_values(by = 'importance', ascending = False)
    #print(f"Name: {name}, R2:{reg.best_score_}")
    
    return importances, performance



groups = {'porosity':"site", 'twi':"site", 'MAT_y':"site", "MAP_y":"site", 'soil_skew':"site", 'elv':"site", 'hnd':"site",
       'bdod_mean_alldepth':"site", 'ksat_surface':"site", 'aridity_new':"site", "WTD":"site","mean_lai":"site",
       'WS_F_max':"storm", 'SWC_F_MDS_1':"storm", 'storm_amount':"storm", 'storm_length':"storm", 'P_F_lag_sum':"RF",
       'GPP_April':"RF", 'LAI':"RF", 'VPD_F':"RF", 'RH':"RF", 'WS_F':"RF", 
       'pre_storm_swc':"storm", "swc_over_porosity":"storm"}

########################################################################################################################
############################## GET COMBINATIONS OF FEATURES #######################################################
########################################################################################################################

# Train a random forest for all combinations of features to show that no matter what you pick, the site characteristics are the most important

soil = ['porosity', 'bdod_mean_alldepth', 'ksat_surface', 'soil_skew']
topo = ['elv', 'twi', 'hnd']
climate = ['aridity_new', 'MAP_y', 'MAT_y']
veg = ['mean_lai']

storm_soil = ['SWC_F_MDS_1', 'pre_storm_swc']
storm_intensity = ['storm_amount', 'storm_length']
wind = ['WS_F_max']
#rf_features = ['TA_F', 'VPD_F','RH']

# Get all combinations with one thing from each soil, topo, climate, and veg category
site_combinations = list(itertools.product(soil, topo, climate, veg))
storm_combinations = list(itertools.product(storm_soil, storm_intensity, wind))#, rf_features))

print(f"{len(site_combinations)} site combinations, {len(storm_combinations)} site combinations")

# Get all of the combinations of the site_combinations and storm_combinations lists added together
all_combinations = list(itertools.product(site_combinations, storm_combinations))
total_combos = []

storm_combos = [list(i) for i in storm_combinations]
site_combos = [list(i) for i in site_combinations]


for i in all_combinations:
    total_combos.append((list(i[0])) + list(i[1]))

print(f"{len(total_combos)} total combinations for site + storm")

# Use ALL of the site and storm combinations to make rf
all_site = soil + topo + climate + veg
all_storm = storm_soil + storm_intensity + wind
all_all = all_site + all_storm


########################################################################################################################
############################## pull in data #######################################################
########################################################################################################################

events_iqr = pd.read_csv('events_iqr.csv')

########################################################################################################################
############################## RUN IT FOR ALL FEATURES #######################################################
########################################################################################################################

which_version = sys.argv[1]

"""
if which_version == 'features':
    # site and storm
    importances = pd.DataFrame()
    performances = pd.DataFrame()
    combo_count = 0
    importance, r2 = rf_explain_anom(all_all, 'allall', search_iters = 50)
    importance['r2'] = r2['r2'].tolist()[0]

    importances = pd.concat([importances, importance])
    performances = pd.concat([performances, r2])
    combo_count += 1
    importances.to_csv('runs/rf_explain_anomaly/importances_allall_notest.csv')
    performances.to_csv('runs/rf_explain_anomaly/performances_allall_notest.csv')
   
    # just site
    importances = pd.DataFrame()
    performances = pd.DataFrame()
    combo_count = 0
    importance, r2 = rf_explain_anom(all_site, 'allsite', search_iters = 50)
    importance['r2'] = r2['r2'].tolist()[0]

    importances = pd.concat([importances, importance])
    performances = pd.concat([performances, r2])
    combo_count += 1
    importances.to_csv('runs/rf_explain_anomaly/importances_allsite_notest.csv')
    performances.to_csv('runs/rf_explain_anomaly/performances_allsite_notest.csv')
      

    # just storm
    importances = pd.DataFrame()
    performances = pd.DataFrame()
    combo_count = 0
    importance, r2 = rf_explain_anom(all_storm, 'allstorm', search_iters = 50)
    importance['r2'] = r2['r2'].tolist()[0]

    importances = pd.concat([importances, importance])
    performances = pd.concat([performances, r2])
    combo_count += 1
    importances.to_csv('runs/rf_explain_anomaly/importances_allstorm_notest.csv')
    performances.to_csv('runs/rf_explain_anomaly/performances_allstorm_notest.csv')
      
 
"""     
if which_version == 'all':

    importances = pd.DataFrame()
    performances = pd.DataFrame()
    combo_count = 0
    for i in total_combos:
        print(f"WORKING ON COMBO: {combo_count + 1}")
        importance, r2 = rf_explain_anom(i, 'All', search_iters = 50)
        importance['r2'] = r2['r2'].tolist()[0]
        importances = pd.concat([importances, importance])
        performances = pd.concat([performances, r2])
        combo_count += 1
        print(r2['r2'])
        importances.to_csv('runs/rf_explain_anomaly/importances_all_smallersearch.csv')
        performances.to_csv('runs/rf_explain_anomaly/performances_all_smallersearch.csv')

elif which_version == 'site':

    # Just use site features
    print(f"{len(site_combos)} total combinations for site only")
    importances_site = pd.DataFrame()
    performances_site = pd.DataFrame()
    combo_count = 0
    for i in site_combos:
        print(f"WORKING ON COMBO: {combo_count + 1}")
        importance, r2 = rf_explain_anom(i, 'Site', search_iters = 50)
        importance['r2'] = r2['r2'].tolist()[0]
        importances_site = pd.concat([importances_site, importance])
        performances_site = pd.concat([performances_site, r2])
        combo_count += 1
        print(r2['r2'])
        importances_site.to_csv('runs/rf_explain_anomaly/importances_site_smallersearch.csv')
        performances_site.to_csv('runs/rf_explain_anomaly/performances_site_smallersearch.csv')


elif which_version == 'storm':
    # Just use storm features 
    print(f"{len(storm_combos)} total combinations for storm only")
    importances_storm = pd.DataFrame()
    performances_storm = pd.DataFrame()
    combo_count = 0
    for i in storm_combos:
        print(f"WORKING ON COMBO: {combo_count + 1}")
        importance, r2 = rf_explain_anom(i,'Storm', search_iters = 50)
        importance['r2'] = r2['r2'].tolist()[0]
        importances_storm = pd.concat([importances_storm, importance])
        performances_storm = pd.concat([performances_storm, r2])
        combo_count += 1
        print(r2['r2'])
        importances_storm.to_csv('runs/rf_explain_anomaly/importances_storm_smallersearch.csv')
        performances_storm.to_csv('runs/rf_explain_anomaly/performances_storm_smallersearch.csv')

t1 = time.time()
print(f"Elapsed time: {t1-t0} secs")