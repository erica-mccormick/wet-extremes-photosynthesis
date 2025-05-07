
# Code and data for "Susceptibility to photosynthesis suppression from extreme storms is highly site-dependent"


<i>Global Change Biology</i>, 2025.


## Authors: 

Erica L. McCormick<sup>1</sup>, Caroline A. Famiglietti<sup>1,2</sup>, Dapeng Feng<sup>1,3</sup>, Anna M. Michalak<sup>1,4</sup>, and Alexandra G. Konings<sup>1</sup>

<sup>1</sup>Department of Earth System Science, Stanford University

<sup>2</sup>Hydrosat, Inc.

<sup>3</sup>Stanford Institute for Human-Centered 
Artificial Intelligence (HAI), Stanford University

<sup>4</sup>Department of Global Ecology, Carnegie 
Institution for Science

## Steps for reproducing analysis:

All figures and statistics are in e_main_figure_analyses.py. 

To run the entire analysis from the beginning:
1) Download the FLUXNET2015 ([FLUXNET2015](https://fluxnet.org/data/fluxnet2015-dataset/)) dataset (Pastorello et al., 2020). 
2) Specify the correct paths to this dataset in utils/paths.py.
3) Create a conda environment using environment.txt.
4) Follow all steps in ``all_analysis.sh``. Note that each analysis step is completed twice: once using GPP from the daytime partitioning method and once using the GPP from the nighttime partitioning method.

All data produced is provided in the folders ``output_anomalies``, ``output_findevents``, and ``runs``.

## Summary of files:

1.  ``a_dd_processing.py`` --> Clean and prepare FLUXNET data

2. ``b_findevents.py`` --> Identify extreme wet event days

3. ``c_randomforest.py`` --> Train random forest model for each site (with hyperparameter tuning) and apply model to extreme wet event days. 

4. ``d_calc_anomalies_add_columns.py`` --> Calculate GPP anomalies and additional attributes (such as cumulative storm metrics)

5. ``e_metadata`` --> Colate all site metadata, such as topography, mean annual precipitation, etc.

6. ``f_main_figure_analyses.py`` --> Generates all figures and statistics found in paper.

5. ``g_random_forest_anomaly.py`` --> Train random forest models on the GPP anomalies to assess feature importance. After this, run e_main_figure_analyses.py again to generate relevant figures. 




