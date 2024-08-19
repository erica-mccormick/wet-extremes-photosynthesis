
# Code for "Extreme soil wetness from storms commonly leads to photosynthesis reductions"

Submitted to <i>Nature Ecology and Evolution</i> August, 2024.

Data will be uploaded to Figshare upon acceptance.

## Authors: 

Erica L. McCormick<sup>1</sup>, Caroline A. Famiglietti<sup>1,2</sup>, Dapeng Feng<sup>1,3</sup>, Anna M. Michalak<sup>1,4</sup>, and Alexandra G. Konings<sup>1</sup>

<sup>1</sup>Department of Earth System Science, Stanford University

<sup>2</sup>Hydrosat, Inc.

<sup>3</sup>Stanford Institute for Human-Centered 
Artificial Intelligence (HAI), Stanford University

<sup>4</sup>Department of Global Ecology, Carnegie 
Institution for Science

## Steps to reproducing analysis:

All figures and statistics are in analyses.ipynb. The order of analysis prior to that is:

1.  a_dd_processing.py --> Clean and prepare Fluxnet data

2. b_findevents.py --> Identify extreme wet event days

3. c_randomforest.py --> Train random forest model for each site (with hyperparameter tuning) and apply model to extreme wet event days. 

4. d_calc_anomalies_add_columns.py --> Calculate GPP anomalies and additional attributes (such as cumulative storm metrics)

6. e_metadata.py --> Combine Fluxnet metadata with ancillary datasets (many extracted using GEE in this script)

5. f_random_forest_anomaly.py --> Train random forest models on the GPP anomalies to assess feature importance (Figure 3). There is one version for the daytime partitioned GPP (DT) and one for nighttime.


7. analyses.ipynb --> This is where remaining analysis and all statistics and figures (SI and main text) are printed.

There is also a folder called <i>utils</i> which contains folder paths and small scripts used multiple times.


