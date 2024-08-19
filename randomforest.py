import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import json
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV, RandomizedSearchCV
rng = np.random.default_rng()
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, KFold, cross_val_score

import sys


def main():

    features = ['SW_IN_F', 'TA_F', 'RH', 'WS_F', 'LAI', 'P_F_lag_sum', 'GPP_April']
    target = 'GPP_DT_VUT_REF'
    
    run_name = sys.argv[1]
    #run_name = 'RF_DT_targetscaled_mega1'

    train_path = 'output_findevents/v3/allsites_training_dd_clipped.csv'
    event_path = 'output_findevents/v3/allsites_event_dd_clipped.csv'
    train_all = pd.read_csv(train_path)
    event_all = pd.read_csv(event_path)

    test_split_frac = 0.25
    search_iters = 100
    random_state = 1
    inner_cv_splits = 3
    outer_cv_splits = 10

    # Make a directory to hold the results
    rf_runs_dir = 'runs/RF'
    dir_name = rf_runs_dir + '/' + run_name
    print(f"Saving random forest results to {dir_name}")
    if not os.path.exists(dir_name): os.makedirs(dir_name)

    # Save a text file with the necessary parameters
    args = {"dir_name":dir_name,
            "run_name":run_name,
            "features":features,
            "target":target,
            "train_path":train_path,
            "event_path":event_path,
            "random_state":random_state,
            "search_iters":search_iters,
            "test_split_frac":test_split_frac,
            "inner_cv_splits": inner_cv_splits,
            "outer_cv_splits": outer_cv_splits
            }
    with open(dir_name + '/arguments.txt', 'w') as f: f.write(json.dumps(args))


    # Make the random grid to search over
    ## Gridsearch from https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
    n_estimators = [5, 10, 50, 75, 100, 150, 200, 250, 300, 400]
    max_features = [2, 4, 6, 1.0, 'sqrt']
    max_depth = [None, 3, 6, 9, 12, 15, 18]
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4, 8, 16]
    bootstrap = [True, False]
    random_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}
    with open(dir_name + '/gridsearch.txt', 'w') as f: f.write(json.dumps(random_grid))

    print("Number of NaNs of each feature:")
    print(train_all[features].isnull().sum(), '\n')

    test_final, event_final, scores = randomforest_all_sites(train_all, event_all, features, target = target,
                                                                       random_grid = random_grid, inner_cv_splits = inner_cv_splits, outer_cv_splits = outer_cv_splits,
                                                                         test_split_frac = test_split_frac, search_iters = search_iters, random_state = random_state)
    
   
    # save rf results
    test_final.to_csv(f'{rf_runs_dir}/{run_name}/test.csv')
    event_final.to_csv(f'{rf_runs_dir}/{run_name}/event.csv')
    scores.to_csv(f'{rf_runs_dir}/{run_name}/performance.csv')




def randomforest_all_sites(train_df, event_df, features, target, random_grid, inner_cv_splits, outer_cv_splits, test_split_frac = 0.25, search_iters = 25, random_state = 1):
    # Lists for storing results
    site = []
    test_score = []
    train_score = []
    event_score = []
    cv_score_mean = []
    cv_score_std = []
    best_model = []
    
    # Final train df with all sites
    test_final = pd.DataFrame()
    event_final = pd.DataFrame()

    # Iterate through sites
    sites = train_df.SITE_ID.unique()
    for s in sites:
        train = train_df[train_df['SITE_ID'] == s].copy()
        event = event_df[event_df['SITE_ID'] == s].copy()

        if event.shape[0] > 2:
            site.append(s)
            print(s)

            # Apply Minmax scaler to X and y
            scaler_features = MinMaxScaler()
            scaler_target = MinMaxScaler()

            X = scaler_features.fit_transform(train[features])
            y = scaler_target.fit_transform(np.array(train[target]).reshape(-1,1)) # because there's only one column, need to be array of shape (-1,1)

            X_event = scaler_features.transform(event[features])
            y_event = scaler_target.transform(np.array(event[target]).reshape(-1,1))

             # Split into training and testing
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split_frac, shuffle = True, random_state=random_state)
            
            # Make sure ys are good shape
            y_train = y_train.ravel()
            y_test = y_test.ravel()
            y_event = y_event.ravel()

            print(X_train.shape, y_train.shape)

            # Inner and outer loop cross-validation
            inner_cv = KFold(n_splits = 10, shuffle = True, random_state = random_state)
            outer_cv = KFold(n_splits = outer_cv_splits, shuffle = True, random_state = random_state)

            # Define the model
            estimator = RandomForestRegressor()

            # Search the space defined by the gridsearch random_grid dictionary
            search = RandomizedSearchCV(estimator = estimator,
                                        param_distributions = random_grid,
                                        cv = inner_cv,
                                        n_iter = search_iters,
                                        random_state = random_state,
                                        verbose = 1)

            # Perform cross-validation with inner and outer cv
            scores = cross_val_score(search, X_train, y_train, scoring = 'r2', cv = outer_cv)
            cv_score_mean.append(scores.mean())
            cv_score_std.append(scores.std())

            # Fit search with X and y
            reg = search.fit(X_train, y_train)

            # Save the best model and model score
            best_model.append(reg.best_params_)            
            train_score.append(reg.score(X_train, y_train))
            test_score.append(reg.score(X_test, y_test))
            event_score.append(reg.score(X_event, y_event))

            # Get an event and test dataframe to add predictions onto
            # Use train_test_split with the same random seed to get the full dataframe for testing
            # Do train_test_split to get train and test datasets to begin with
            _, test_results = train_test_split(train, test_size=test_split_frac, shuffle = True, random_state=random_state)
            event_results = event.copy()

            # Save predictions for test and inverse scale
            test_results['predicted_scaled'] = reg.predict(X_test)
            test_results['predicted'] = scaler_target.inverse_transform(reg.predict(X_test).reshape(-1,1))
            test_results['actual_scaled'] = y_test
            test_results['actual'] = scaler_target.inverse_transform(y_test.reshape(-1,1))

            # Save predictions for event and inverse scale
            event_results['predicted_scaled'] = reg.predict(X_event)
            event_results['predicted'] = scaler_target.inverse_transform(reg.predict(X_event).reshape(-1,1))
            event_results['actual_scaled'] = y_event
            event_results['actual'] = scaler_target.inverse_transform(y_event.reshape(-1,1))

            test_final = pd.concat([test_final, test_results])
            event_final = pd.concat([event_final, event_results])
        

    # Make actual - predicted column
    test_final['actual - predicted'] = test_final['actual'] - test_final['predicted']
    event_final['actual - predicted'] = event_final['actual'] - event_final['predicted']

    # Make df of scores
    scores = pd.DataFrame({"SITE_ID":site, "train_score":train_score, "test_score":test_score, "event_score":event_score, "cv_score_mean":cv_score_mean, "cv_score_std":cv_score_std, "best_params":best_model})
    return test_final, event_final, scores

if __name__ == '__main__':
    main()