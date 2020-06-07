# Machine Learning Engineer Nanodegree
## Capstone Project
### Arvato: Customer Segmentation and Prediction


### Table of Contents

1. [Installation](#installation)
2. [Introduction](#introduction)
3. [Project Motivation](#motivation)
4. [Files](#files)
5. [Instructions](#instructions)
6. [Results](#results)
7. [Licensing, Authors, and Acknowledgements](#licensing)


## Installation <a name="installation"></a>
Besides the libraries included in the Anaconda distribution for Python 3.6 the following libraries have been included in this project:
* `LGBM` -  gradient boosting framework that uses tree based learning algorithms.
* `XGBoost` - optimized distributed gradient boosting library designed to be highly efficient, flexible and portable.
* `skopt` - simple and efficient library to minimize (very) expensive and noisy black-box functions.


## Introduction <a name="introduction"></a>
This project was made available for Udacity by Arvato.
The goal is to find if there are particular patterns in individuals based on collected data that makes them more likely to be responsive to a mail-order campaign by Arvato for the sale of organic products.

The project is divided in 2 sections:
1. Customer Segmentation Report (Unsupervised Learning): For this portion of the project I performed EDA, PCA and clustering analysis (KMeans) to identify clusters that are good descriptors for what makes a core customer for this particular Arvato's client.

2. Predict Customer Report (Supervised Learning): using the provided data on how customers responded to a marketing campaign I created a model that predicts how particular individuals would respond to campaign.


## Project Motivation <a name="motivation"></a>
This project provided by Arvato Financial Solutions was one of the available capstone projects. I chose this project mainly for several:
* It is a real-world problem in which the data had nearly no transformation, making it a chance for an intensive experience with data processing.
* Customer segmentation is one of the fields in data science and machine learning that is continuously growing and progressing, having this hands-on experience can be very valuable for future problems.
* Since it has a Kaggle competition portion it allows me to measure the success of my efforts against others.


## Files <a name="files"></a>
Provided by Arvato:

•Udacity_AZDIAS_Subset.csv: Demographics data for the general population of Germany; 891211 persons (rows) x 85 features (columns).

•Udacity_CUSTOMERS_Subset.csv: Demographics data for customers of a mail-order company; 191652 persons (rows) x 85 features (columns).

•Data_Dictionary.md: Detailed information file about the features in the provided datasets.

•AZDIAS_Feature_Summary.csv: Summary of feature attributes for demographics data; 85 features (rows) x 4 columns

Created by me:

•utils.py : includes all the helper functions to perform data preprocessing and running the prediction models

•ELBOW METHOD.ipynb: notebook where I used an elbow plot to select the number of ks to use in KMeans

•MODEL TESTING.ipynb: notebook where I used a bayesian approach for hyperparamater tunning.


## Instructions <a name="instructions"></a>
The dataset used in this project is proprietary. As so this project is not usable for those outside of this nanodegree, it does serve as a snapshot of the strategies I chose to use to approach this challenge.


## Results <a name="results"></a>
I did identify clusters of relevance of future customers and identified positive responders to the mail-order campaign successfuly, pleased read the attached report for an extensive discussion.


## Licensing, Authors, Acknowledgements<a name="licensing"></a>
I am greatly thankful for the incredible challenge provided by Arvato and Udacity.