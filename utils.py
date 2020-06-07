import numpy as np
import pandas as pd
import collections
import matplotlib.pyplot as plt
from pylab import *

# sklearn
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import confusion_matrix,precision_recall_fscore_support
from sklearn.utils.multiclass import unique_labels

'''
################## Checking balance between dataframes and types of features ##################
'''

#function to check categorical variable counts
def categorical_checker(df, attributes_df):
    '''
    Takes in a feature dataframe and a demographic dataframe and prints the counts for categorical variables
    Args:
    df: demographics dataframe
    attributes_df: dataframe with the summary of all the features
    returns:
    nothing
    '''
    categorical = attributes_df[attributes_df['type'] == 'categorical']['attribute'].values
    categorical = [x for x in categorical if x in df.columns] 
    binary = [x for x in categorical if df[x].nunique()==2]
    multilevel = [x for x in categorical if df[x].nunique()>2]
    print(df[categorical].nunique())
    

# function to determine if 2 dataframes are balanced in terms of number and type of features
def balance_checker(df1, df2):
    '''
    Takes in 2 dataframes and checks if attributes match between the 2 dataframes match
    Args: any 2 dataframes 
    prints True or False if the dataframes match or not
    '''
    features_list_df1 = df1.columns.values
    features_list_df2 = df2.columns.values
    equal = collections.Counter(features_list_df1) == collections.Counter(features_list_df2)
    
    print('Feature balance between dfs?: ', equal)
    
    if equal == False:
        print('Your first argument df differs from the second on the following columns: ')
        print(set(features_list_df1) - set(features_list_df2))
        
        print('Your second argument df differs from the first on the following columns: ')
        print(set(features_list_df2) - set(features_list_df1))
        
'''
################## Checking missing values ##################
'''

# creating a function to determine percentage of missing values
def percentage_of_missing(df):
    '''
    This function calculates the percentage of missing values in a dataframe and splits it on a defined
    percentage boundary
    inputs: dataframe
    output: missing values dataframe
    '''
    percent_missing = df.isnull().sum()* 100/len(df)
    percent_missing_df = pd.DataFrame({'column_name': df.columns,
                                 'percent_missing': percent_missing})
    return percent_missing_df


#creating a function to split a percentage of missing values dataframe for visualization purposes
def split_on_percentage(df, boundary, conditional):
    '''
    This function takes in a dataframe and splits it on a particular percentage boundary with a particular conditional
    Args: dataframe created with percentage_of_missing
    boundary: percentage value we want to upper bound or lower bound values for
    conditional: determines if we are getting greater or less than values
    '''
    if conditional == '>':
        split_df = df[df.percent_missing > boundary]
    elif conditional == '>=':
        split_df = df[df.percent_missing >= boundary]
    elif conditional == '<=':
        split_df = df[df.percent_missing <= boundary]
    else:
        split_df = df[df.percent_missing > boundary]
    
    return split_df  

#function to delete columns with too much missing data
def columns_to_delete(df):
    '''
    Fuction goes through dataframe created with split_on_percentage() and saves column names over
    the chosen boundary and saves column names to a list
    Args: dataframe created using split_on_percentage
    returns: list of the columns we want to exclude
    '''
    cols_del = df.index.values.tolist()
    
    return cols_del

#function to plot/visualize histogram of data missing in rows
def row_hist(df1, df2, bins):
    '''
    This function takes in the azdias and the customers dataframe, and the number of bins
    we want the data to be distributed by and plots a histogram of nulls distribution accross
    rows
    '''
    rcParams['figure.figsize'] = 8, 8
   
    plt.hist(df1.isnull().sum(axis=1), bins, color = 'cyan')

    plt.hist(df2.isnull().sum(axis=1), bins, color = 'grey')

    plt.title('Distributions of null values in Azdias and Customers rows')
    plt.xlabel('# Null Values')
    plt.ylabel('Rows')

    plt.show()
    

#function to delete rows with too much missing data
def row_dropper(df, boundary):
    '''
    This function identifies rows missing more than a threshold amount of data specified with boundary
    and drops them
    Args: 
    dataframe: already cleaned up of columns missing more than a boundary defined percentage 
    boundary: number of missing entries limit to droppable rows
    
    returns:
    dataframe with dropped rows with more than a certain amount of missing values
    '''
    df = df.dropna(thresh=df.shape[1]-boundary)
    df = df.reset_index()
    del df['index']
    
    return df

'''
################## Data Cleaning and feature engineering ##################
'''

#function to handle special feature columns
def special_feature_handler(df):
    '''
    This function deals with the special characters in the cameo columns
    Finds the special characters and replaces them with nan
    Args: azdias or customer dataframe
    returns: dataframe with X or XX replaced with nan
    '''
    
    #drop the unnamed column
    if 'Unnamed: 0' in df:
        df.drop('Unnamed: 0', axis = 1, inplace = True)
        
        
    #dealing with the X and XX that appear in the place of NAN
    #'CAMEO_DEU_2015'
    cols = ['CAMEO_DEUG_2015', 'CAMEO_INTL_2015']
    df[cols] = df[cols].replace({'X': np.nan, 'XX':np.nan, '': np.nan, ' ':np.nan})
    df[cols] = df[cols].astype(float)
       
    return df

#function to deal with all the missing and unknown entries
def unknowns_to_NANs(df, xls):
    '''
    This function uses the information in the Dias files to help identify values that are missing or unknown
    Replaces missing or unknown value with nan
    Args: customer or azdias dataframe and dias dataframe
    '''
    
    #using the DIAs xls file lets save meanings that might indicate unknown values
    unknowns = xls['Meaning'].where(xls['Meaning'].str.contains('unknown')).value_counts().index
    
    #I will now create a list of all the unknown values for each attribute and replace them on my azdias and customers
    missing_unknowns = xls[xls['Meaning'].isin(unknowns)]
    
    for row in missing_unknowns.iterrows():
        missing_values = row[1]['Value']
        attribute = row[1]['Attribute']
        
        #dealing with columns that only exist in df
        if attribute not in df.columns:
            continue
        
        #dealing with strings or ints
        if isinstance(missing_values,int): 
            df[attribute].replace(missing_values, np.nan, inplace=True)
        elif isinstance(missing_values,str):
            eval("df[attribute].replace(["+missing_values+"], np.nan, inplace=True)")

#function for features engineering: creating novel features
def feat_eng(df):
    '''
    This function takes in either the azdias dataframe or the customers dataframe to create new features
    and encode select categorical features
    Args: customer or azdias dataframe
    returns: dataframe with novel features and encoded categorical features
    '''
    
    #dropping columns that appear in customers but not azdias if present
    if 'REGIOTYP' in df:
        df.drop('REGIOTYP', axis = 1, inplace = True)
    if 'KKK' in df:
        df.drop('KKK', axis = 1, inplace = True)
    
    #OST_WEST_KZ is a binary feature that needs encoding it takes the values array(['W', 'O'], dtype=object)
    o_w_k_dict = {'OST_WEST_KZ': {'W':0, 'O':1}}
    df = df.replace(o_w_k_dict)
    
    #label encoding for Cameo_deu_2015
    cameo_fill = df['CAMEO_DEU_2015'].value_counts().idxmax()
    df['CAMEO_DEU_2015'] = df['CAMEO_DEU_2015'].fillna(cameo_fill)
    df['CAMEO_DEU_2015'] = df['CAMEO_DEU_2015'].replace(['XX'], cameo_fill)
    data = df['CAMEO_DEU_2015']
    values = array(data)
    label_encoder = LabelEncoder()
    int_encoder = label_encoder.fit_transform(values)
    df['CAMEO_DEU_2015'] = int_encoder


    #extract the time,and keep the year for column with date/time information
    df['EINGEFUEGT_AM']=pd.to_datetime(df['EINGEFUEGT_AM']).dt.year
    
    #creating the dictionaries for mapping in PRAEGENDE_JUGENDJAHRE
    #decades:
    decades_dict = {1: 40, 2: 40, 3: 50, 4: 50, 5: 60, 6: 60, 7: 60,
           8: 70, 9: 70, 10: 80, 11: 80, 12: 80, 13: 80, 14: 90,
           15: 90, 0: np.nan}
    df['PRAEGENDE_JUGENDJAHRE_DECADE'] = df['PRAEGENDE_JUGENDJAHRE'].map(decades_dict)
    print('Creating PRAEGENDE_JUGENDJAHRE_DECADE feature')
    
    #mainstream or avant-garde movement
    movement_dict = {1: 0, 2: 1, 3: 0, 4: 1, 5: 0, 6: 1, 7: 1, 8: 0,
           9: 1, 10: 0, 11: 1, 12: 0, 13: 1, 14: 0, 15: 1, 0: np.nan}
    df['PRAEGENDE_JUGENDJAHRE_MOVEMENT'] = df['PRAEGENDE_JUGENDJAHRE'].map(movement_dict)
    
    print('Creating PRAEGENDE_JUGENDJAHRE_MOVEMENT feature')
       
    # WOHNLAGE refers to neighborhood area, from very good to poor; rural
    #creating dictionaries for WOHNLAGE
    area_dict = {1.0:0, 2.0:0, 3.0:0, 4.0:0, 5.0:0, 7.0:1, 8.0:1}
    #creating a feature for borough quality
    df['WOHNLAGE_QUALITY'] = df[(df['WOHNLAGE'] > 0) & (df['WOHNLAGE'] < 7)]['WOHNLAGE']
    
    print('Creating WOHNLAGE_QUALITY feature')
    
    #creating a feature for rural/urban division
    df['WOHNLAGE_AREA'] = df['WOHNLAGE'].map(area_dict)
    print('Creating WOHNLAGE_AREA feature')
    
    
    #Using CAMEO to create a wealth and family type feature
    df['WEALTH'] = df['CAMEO_INTL_2015'].apply(lambda x: np.floor_divide(float(x), 10) if float(x) else np.nan)
    df['FAMILY'] = df['CAMEO_INTL_2015'].apply(lambda x: np.mod(float(x), 10) if float(x) else np.nan)
    print('Creating Wealth and Family feature')
     
    #dealing with LP_LEBENSPHASE_FEIN
    life_stage = {1: 'younger_age', 2: 'middle_age', 3: 'younger_age',
              4: 'middle_age', 5: 'advanced_age', 6: 'retirement_age',
              7: 'advanced_age', 8: 'retirement_age', 9: 'middle_age',
              10: 'middle_age', 11: 'advanced_age', 12: 'retirement_age',
              13: 'advanced_age', 14: 'younger_age', 15: 'advanced_age',
              16: 'advanced_age', 17: 'middle_age', 18: 'younger_age',
              19: 'advanced_age', 20: 'advanced_age', 21: 'middle_age',
              22: 'middle_age', 23: 'middle_age', 24: 'middle_age',
              25: 'middle_age', 26: 'middle_age', 27: 'middle_age',
              28: 'middle_age', 29: 'younger_age', 30: 'younger_age',
              31: 'advanced_age', 32: 'advanced_age', 33: 'younger_age',
              34: 'younger_age', 35: 'younger_age', 36: 'advanced_age',
              37: 'advanced_age', 38: 'retirement_age', 39: 'middle_age',
              40: 'retirement_age'}

    fine_scale = {1: 'low', 2: 'low', 3: 'average', 4: 'average', 5: 'low', 6: 'low',
              7: 'average', 8: 'average', 9: 'average', 10: 'wealthy', 11: 'average',
              12: 'average', 13: 'top', 14: 'average', 15: 'low', 16: 'average',
              17: 'average', 18: 'wealthy', 19: 'wealthy', 20: 'top', 21: 'low',
              22: 'average', 23: 'wealthy', 24: 'low', 25: 'average', 26: 'average',
              27: 'average', 28: 'top', 29: 'low', 30: 'average', 31: 'low',
              32: 'average', 33: 'average', 34: 'average', 35: 'top', 36: 'average',
              37: 'average', 38: 'average', 39: 'top', 40: 'top'}
    
    df['LP_LEBENSPHASE_FEIN_life_stage'] = df['LP_LEBENSPHASE_FEIN'].map(life_stage)
    df['LP_LEBENSPHASE_FEIN_fine_scale'] = df['LP_LEBENSPHASE_FEIN'].map(fine_scale)
    
    life_dict = {'younger_age': 1, 'middle_age': 2, 'advanced_age': 3,
            'retirement_age': 4}
    scale_dict = {'low': 1, 'average': 2, 'wealthy': 3, 'top': 4}

    df['LP_LEBENSPHASE_FEIN_life_stage'] = df['LP_LEBENSPHASE_FEIN_life_stage'].map(life_dict)
    df['LP_LEBENSPHASE_FEIN_fine_scale'] = df['LP_LEBENSPHASE_FEIN_fine_scale'].map(scale_dict)
    
    print('Creating LP_LEBENSPHASE_FEIN_life_stage and LP_LEBENSPHASE_FEIN_fine_scale feature')
    
    #one hot encoding of remaining features
    cat_features = ['ANREDE_KZ']
    df = pd.get_dummies(df, columns = cat_features, prefix = cat_features, dummy_na = True, drop_first = True)

    
    #dropping columns used to create new features, have object types or duplicated information (ie. grob/fein)
    cols = ['PRAEGENDE_JUGENDJAHRE', 'WOHNLAGE', 'CAMEO_INTL_2015','LP_LEBENSPHASE_GROB', 'LP_LEBENSPHASE_FEIN',
            'D19_LETZTER_KAUF_BRANCHE']
    
    df.drop(cols, axis = 1, inplace = True)
            
    #imputing nans with most frequent value
    imputer = SimpleImputer(strategy= 'most_frequent')
    imputed_df = pd.DataFrame(imputer.fit_transform(df))
    imputed_df.columns = df.columns
    imputed_df.index = df.index
       
    return imputed_df

#function to scale and normalize the dataframes features
def feature_scaling(df, type_scale):
    '''
    This function takes in either the azdias or the customers dataframe and applyes the selected feature scaler
    Args: customer or azdias dataframe and a string representing the type of scaling intended
    returns: scaled dataframe
    '''
    
    features_list = df.columns
    
    if type_scale == 'StandardScaler':
        df_scaled = StandardScaler().fit_transform(df)
        
    if type_scale == 'MinMaxScaler':
        df_scaled = MinMaxScaler().fit_transform(df)
    
    df_scaled = pd.DataFrame(df_scaled)
    df_scaled.columns = features_list
    
    return df_scaled

'''
################## Models and model viz ##################
'''

#pca model
def pca_model(df, n_components):
    '''
    This function defines a model that takes in a previously scaled dataframe and returns the result of 
    the transformation. The output is an onject created post data fitting
    '''
    pca = PCA(n_components)
    pca_df = pca.fit(df)
    
    return pca_df

#scree plots for PCA
def scree_plots(SS,MMS, dataname):
    '''
    This function takes in the transformed data using PCA and plots it in scree plots
    '''
    subplot(2,1,1)

    plt.plot(np.cumsum(SS.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained Variance Ratio vs Number of Components SS' + dataname)
    plt.grid(b=True)

    subplot(2,1,2)
    plt.plot(np.cumsum(MMS.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained Variance Ratio vs Number of Components MMS' + dataname)
    plt.grid(b=True)

    plot = tight_layout()
    plot = plt.show()

#function to help interpret the pca results
def interpret_pca(df, n_components, component):
    '''
    Maps each weight to its corresponding feature name and sorts according to weight.
    Args:
        df (dataframe): dataframe on which pca has been used on.
        pca (pca): pca object.
        component (int): which principal compenent to return
    Returns:
        df_pca (dataframe): dataframe for specified component containing the explained variance
                            and all features and weights sorted according to weight.
    '''
    rcParams['figure.figsize'] = 8, 8
    pca = PCA(n_components)
    df_pca = pca.fit_transform(df)
    
    df_pca = pd.DataFrame(columns=list(df.columns))
    df_pca.loc[0] = pca.components_[component]
    dim_index = "Dimension: {}".format(component + 1)

    df_pca.index = [dim_index]
    df_pca = df_pca.loc[:, df_pca.max().sort_values(ascending=False).index]

    ratio = np.round(pca.explained_variance_ratio_[component], 4)
    df_pca['Explained Variance'] = ratio

    cols = list(df_pca.columns)
    cols = cols[-1:] + cols[:-1]
    df_pca = df_pca[cols]

    return df_pca

#function to display interesting features
def display_interesting_features(df, pca, dimensions):
    '''
    This function displays interesting features of the selected dimension
    '''
    
    features = df.columns.values
    components = pca.components_
    feature_weights = dict(zip(features, components[dimensions]))
    sorted_weights = sorted(feature_weights.items(), key = lambda kv: kv[1])
    
    print('Lowest: ')
    for feature, weight, in sorted_weights[:3]:
        print('\t{:20} {:.3f}'.format(feature, weight))
    
    print('Highest: ')
    for feature, weight in sorted_weights[-3:]:
        print('\t{:20} {:.3f}'.format(feature, weight))
        
#function to fit the kmeans model
def fit_kmeans(data, centers):
    '''
    returns the kmeans score regarding SSE for points to centers
    INPUT:
        data - the dataset you want to fit kmeans to
        center - the number of centers you want (the k value)
    OUTPUT:
        score - the SSE score for the kmeans model fit to the data
    
    '''
    kmeans = KMeans(centers)
    model = kmeans.fit(data)
    
    # SSE score for kmeans model 
    score = np.abs(model.score(data))
    return score
        
        
#function to display elbow plot
def elbow_method(data):
    scores = []
    centers = list(range(1,15))
    i = 0
    for center in centers:
        i += 1
        print(i)
        scores.append(fit_kmeans(data, center))
        
    # Investigate the change in within-cluster distance across number of clusters.
    # Plot the original data with clusters
    f = plt.figure()
    plt.plot(centers, scores, linestyle='--', marker='o', color='b')
    plt.ylabel('SSE score')
    plt.xlabel('K')
    plt.title('SSE vs K')
    f.savefig('elbow.png', bbox_inches='tight', dpi=600)
    
    
def create_base_models():
    '''
    Creates base models.
    
    Args:
        None
    
    Returns:
        baseModels (list) - list containing base models.
    '''
    basedModels = []
    basedModels.append(('LR', LogisticRegression(solver='liblinear', random_state=SEED)))
    basedModels.append(('RF', RandomForestClassifier(n_estimators=250, random_state=SEED)))
    basedModels.append(('XGB', xgb.XGBClassifier(random_state=SEED)))
    basedModels.append(('LGBM', lgb.LGBMClassifier(random_state=SEED)))
    basedModels.append(('GB', GradientBoostingClassifier(random_state=SEED)))
    basedModels.append(('MLP', MLPClassifier(random_state=SEED)))
    
    return basedModels


def evaluate(features, response, models, curve=False):
    '''
    Evaluates models using X-Fold cross-validation. 
    Learning curve can also be plotted (optional).
    
    Args:
        features (dataframe) - dataset to be used for training.
        response (dataframe) - target variable
        models (list) - list of models to evaluated.
        curve (bool) - whether or not to plot learning curve.
        
    Returns:
        names (list) - list of models tested.
        results (list) - list of results for each model.
    '''
    results = []
    names = []
    for name, model in models:
        cv_results = cross_val_score(model, features, response, cv=skf, scoring='roc_auc', n_jobs=1)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
        if curve:
            train_sizes, train_scores, test_scores = learning_curve(
                model, features, response, cv=skf, scoring = 'roc_auc', train_sizes=np.linspace(.1, 1.0, 10), n_jobs=1)

            train_scores_mean = np.mean(train_scores, axis=1)
            test_scores_mean = np.mean(test_scores, axis=1)
            print("roc auc train score = {}".format(train_scores_mean[-1].round(2)))
            print("roc auc validation score = {}".format(test_scores_mean[-1].round(2)))
            plt.grid()

            plt.title("Learning Curve")
            plt.xlabel("% of training set")
            plt.ylabel("Score")

            plt.plot(np.linspace(.1, 1.0, 10)*100, train_scores_mean, 'o-', color="g",
                     label="Training score")
            plt.plot(np.linspace(.1, 1.0, 10)*100, test_scores_mean, 'o-', color="r",
                     label="Cross-validation score")

            plt.yticks(np.arange(0.45, 1.02, 0.05))
            plt.xticks(np.arange(0., 100.05, 10))
            plt.legend(loc="best")
            print("")
            plt.show()
        
        
    return names, results


def get_scaled_preprocess(type_of_scaler):
    '''
    Create machine learning pipeline with or without scaler.
    
    Args:
        type_of_scaler (str) - string value representing which scaler to use (if any).
        
    Returns (pipeline) - ml pipeline created.
    '''
    
    if type_of_scaler == 'standard':
        scaler = StandardScaler()
    elif type_of_scaler == 'minmax':
        scaler = MinMaxScaler()
        
    pipelines = []
    pipelines.append((type_of_scaler+'LR', Pipeline([('Scaler', scaler), ('LR', LogisticRegression(solver='liblinear', random_state=SEED))])))
    pipelines.append((type_of_scaler+'RF', Pipeline([('Scaler', scaler), ('RF', RandomForestClassifier(n_estimators=250, random_state=SEED))])))   
    pipelines.append((type_of_scaler+'XGB', Pipeline([('Scaler', scaler), ('XGB', xgb.XGBClassifier(random_state=SEED))])))
    pipelines.append((type_of_scaler+'LGBM', Pipeline([('Scaler', scaler), ('LGBM', lgb.LGBMClassifier(random_state=SEED))])))
    pipelines.append((type_of_scaler+'GB', Pipeline([('Scaler', scaler), ('GB', GradientBoostingClassifier(random_state=SEED))])))   
    pipelines.append((type_of_scaler+'MLP', Pipeline([('Scaler', scaler), ('MLP', MLPClassifier(random_state=SEED))])))   

    
    return pipelines


def create_score_df(names, results):
    '''
    Creates a dataframe containing model names and corresponding score.
    
    Args:
        names (list) - list of model names.
        results (list) - list of scores.
    '''
    def floatingDecimals(f_val, dec=3):
        prc = "{:."+str(dec)+"f}" 
    
        return float(prc.format(f_val))

    scores = []
    for r in results:
        scores.append(floatingDecimals(r.mean(),4))

    scoreDataFrame = pd.DataFrame({'Model':names, 'Score': scores})
    return scoreDataFrame



def plot_feature_importances(model, model_type, features, plot_n=10):
    '''
    Plots n most important features and importance.
    
    Args:
        model (classifier) - trained model.
        model_type (str) - type of model.
        features (list) - list of feature names.
        plot_n (int) - number of features to plot.
    '''
    feature_importance_values= np.zeros((len(model.feature_importances_)))
    
    feature_importance_values += model.feature_importances_

    feature_importances = pd.DataFrame({'feature': features, 'importance': feature_importance_values})

    # sort based on importance
    feature_importances = feature_importances.sort_values('importance', ascending = False).reset_index(drop = True)

    # normalize the feature importances to add up to one
    feature_importances['normalized_importance'] = feature_importances['importance'] / feature_importances['importance'].sum()
    feature_importances['cumulative_importance'] = np.cumsum(feature_importances['normalized_importance'])
    
    plt.figure(figsize=(10, 6))
    ax = plt.subplot()
    
    ax.barh(list(reversed(list(feature_importances.index[:plot_n]))), 
                feature_importances['normalized_importance'][:plot_n], 
                align = 'center', edgecolor = 'k')

    # Set ticks and labels
    ax.set_yticks(list(reversed(list(feature_importances.index[:plot_n]))))
    ax.set_yticklabels(feature_importances['feature'][:plot_n], size = 12)
    plt.xlabel('Normalized Importance', size = 15); plt.title(f'Feature Importances ({model_type})', size = 15) 
    
def plot_comparison_feature(column, df):
    '''
    Plots the distribution for a feature.
    Args:feature (string) - feature to plot.
        df (dataframe) - dataframe containing RESPONSE feature.
    '''
    responded = df[df['RESPONSE'] == 1]
    not_responded = df[df['RESPONSE'] == 0]

    sns.set(style="darkgrid")
    fig, (ax1, ax2) = plt.subplots(figsize=(12,4), ncols=2)
    sns.countplot(x=column, data=responded, ax=ax1, palette="Set2")
    ax1.set_xlabel('Value')
    ax1.set_title(f'Distribution for Responded = 1')
    sns.countplot(x=column, data=not_responded, ax=ax2, palette="Set2")
    ax2.set_xlabel('Value')
    ax2.set_title(f'Distribution for Responded = 0')
    fig.suptitle(f'Feature: {column}')