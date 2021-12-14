# ABCD
Dynamical relationships between cognition, environment, personality, educational performance and psychopathological symptoms

This repository contains data, scripts, and notebooks used to prepare, clean and analyse the Adolescent Brain Cognitive Development study data (https://nda.nih.gov/abcd for more info). 

# Structure

* raw: raw data csv files downloaded from NIMH website using NDADownloaderManager
* derived: preliminary and final outputs, organised into dated versions
* notebooks: Jupyter notebooks in python and R - run these to extract relevant variables, clean data, and perform analysis
* scripts: helper python scripts used in notebooks

# Data preparation

The `data_preparation.ipynb` notebook contains all the code to extract and rename columns from ~60 raw csv files. This involves ~500 variables across four timepoints for 11878 subjects (at baseline, this reduces by three-year follow-up). Minimal pre-processing is applied to remove void values, calculate means/ratios, and exclude unacceptable performance on cognitive tasks. The file uses functions defined in `data_preparation.py` specifications defined in `specifications.py`.

# Data cleaning

The `data_cleaning.ipynb` notebook contains all the code for post-processing of the extracted dataset. This involves four steps:
1. remove outliers using a) interquartile ratio with cutoff of 2.5 and b) "natural" bounds from other research or experimental design
2. standardize continuous variables using StandardScaler
3. calculate interaction terms based, e.g. between anxiety and depression
4. propogate fixed demographic variables captured at baseline across subsequent time points

The target variables for each step are specified in `data_cleaning.py` and interaction functions are defined in `interactions.py`. It is straightforward to edit either of these files to add/remove variables and interactions.

# Data analysis

* `WIP_graphicalVAR.ipynb` contains code for preliminary network analysis usig Graphical vector-autoregressive models to investigate dynamical relationships between psychopathogical symptoms (Vars1) and dimensional diagnoses (Vars2) 
* `WIP_subject_clusters_PCA_kmeans.ipynb` contains code for preliminary segmentation analysis of subjects using PCA to reduce dimensionality and k-means clustering
* There are four notebooks containing code for outdated analysis on previous versions of the dataset including drift-diffusion modelling on forced two-choice tasks (EZ, Hierarchical DDM, Generalised DDM a.k.a. PyDDM) in `__archive_ddm.ipynb`, data-driven ontology discovery using EFA and hierarchical clustering in `__archive_DV_EFA_Hclust.ipynb`, gaussian graphical models adapted from nature network model primer in `__archive_EBICglasso_R.ipynb`, and missing value imputation using missForest in `__archive_missForest_imputation_R.ipynb`
