# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 10:50:02 2023

@author: carlo
"""
import pandas as pd
import numpy as np
import re, os, sys
from datetime import datetime
import config_carmax
import time
#from import_carmax_data_v2 import import_competitor_data, import_carmax_backend_data

import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from joblib import dump, load

# plots
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
output_path = os.path.abspath(os.path.dirname(__file__)) # current working dir
os.chdir(output_path)

#@st.cache_data
def import_carmax_data():
    '''
    Returns
    -------
    import_carmax_gsheet : pandas dataframe
    '''
    start_time = time.time()
    print ('Start data import')
    final_competitor_data = pd.read_csv('carmax_competitor_data.csv')
    print ('Imported competitor data')
    print("--- %s seconds ---" % (time.time() - start_time))
    # # competitor_data
    # competitor_data_dtime = datetime.fromtimestamp(os.path.getmtime(r'C:\Users\carlo\Documents\LICA Auto\carmax\scrapers\carmax_scrapers\spiders\carmax_competitor_data.csv')).date
    # # hist_data_dtime = datetime.fromtimestamp(os.path.getmtime('carmax_hist_data.csv')).date
    
    # if os.path.exists(r'C:\Users\carlo\Documents\LICA Auto\carmax\scrapers\carmax_scrapers\spiders\carmax_competitor_data.csv') and (datetime.today().date() == competitor_data_dtime):
    #     final_competitor_data = pd.read_csv(r'C:\Users\carlo\Documents\LICA Auto\carmax\scrapers\carmax_scrapers\spiders\carmax_competitor_data.csv')
        
    # else:
    #     competitor_data = import_competitor_data()
        
    #     # fuel type, transmission, mileage
    #     final_competitor_data = competitor_data[~(competitor_data['fuel_type'].isnull()) &
    #                                             ~(competitor_data['mileage'].isnull()) & 
    #                                             ~ (competitor_data['transmission'].isnull())]
    #     final_competitor_data = final_competitor_data.reset_index().drop('index', axis=1)
        
    #     # data cleaning
    #     final_competitor_data.loc[:, 'model'] = final_competitor_data.loc[:,'model'].apply(lambda x: re.sub('[AM(CV)](/)?T', '', x).strip())
    #     popular_models = list(final_competitor_data.model.value_counts()[final_competitor_data.model.value_counts() >= 3].index)
    #     final_competitor_data.loc[:,'model'] = final_competitor_data.model.apply(lambda x: cleanup_model(x, popular_models))
    #     final_competitor_data.loc[:,'make'] = final_competitor_data.make.apply(lambda x: x.upper())
    #     final_competitor_data.loc[:,'transmission'] = final_competitor_data.loc[:,'transmission'].apply(lambda x: x.upper())
    #     # save file
    #     final_competitor_data.to_csv('carmax_competitor_data.csv', index = 'False')
    
    # hist data
    # if os.path.exists('carmax_hist_data.csv') and (datetime.today().date() == hist_data_dtime):
    #     hist_data = pd.read_csv('carmax_hist_data.csv')
    # else:
    #     hist_data = import_hist_carmax_gsheet()
    #     hist_data = hist_data.reset_index().drop('index', axis=1)
    #     hist_data.to_csv('carmax_hist_data.csv', index = False)
    start_time = time.time()
    backend_data = config_carmax.import_backend_data(None)
    print('Imported backend data')
    print("TOTAL: %s seconds" % (time.time() - start_time))
    return final_competitor_data, backend_data


def feature_filter(data, upper = None, lower = 0):
    '''
    Applies quantiles for filtering numerical data from outliers

    Parameters
    ----------
    data : dataframe or series
    upper : float, optional
        Ceiling value to filter. The default is 0.
    lower : float, optional
        Floor value to filter. The default is 0.

    Returns
    -------
    Filtered series

    '''
    
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    result = data[(data >= (Q1 - 1.5*IQR)) & (data < (Q3 + 1.5*IQR))]
    if upper is not None:
        result = result[(result >= lower) & (result <= upper)]
    else:
        result = result[result >= lower]        
    
    return result
        
#@st.cache_data
def training_data_prep(df, test_size = 0.2, rs = 101):
    '''
    Prepare filtered data for training into models

    Parameters
    ----------
    df : filtered dataframe
    test_size : float, optional
        Ratio of test size compared to all data. The default is 0.2.
    rs : int, optional
        random state. The default is 101.

    Returns
    -------
    X_train, X_test, y_train, y_test

    '''
    # setup training data
    y = df['price']
    X = df.drop(['price', 'platform'], axis=1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size = test_size,
                                                        random_state = rs)
    
    # get object data types
    object_cols = X.select_dtypes(include=[object]).columns
    num_cols = X.select_dtypes(include=[np.number]).columns
    
    # OneHotEncoding
    enc = OneHotEncoder(handle_unknown = 'ignore')
    enc.fit(X[object_cols])
    
    # StandardScaler
    scaler = StandardScaler()
    ## construct scaled numerical features
    X_train_num = pd.DataFrame(scaler.fit_transform(X_train[num_cols]),
                               columns = num_cols)
    X_test_num = pd.DataFrame(scaler.transform(X_test[num_cols]),
                              columns = num_cols)
    
    # Construct one hot encoded features
    X_train_obj = pd.DataFrame(enc.transform(X_train[object_cols]).toarray(), 
                               columns = enc.get_feature_names_out())
    X_test_obj = pd.DataFrame(enc.transform(X_test[object_cols]).toarray(),
                              columns = enc.get_feature_names_out())
    
    # construct X_train and X_test
    X_train = pd.concat([X_train_obj, X_train_num], axis=1)
    X_test = pd.concat([X_test_obj, X_test_num], axis=1)
    
    return X_train, X_test, y_train, y_test
    
#@st.cache_resource
def rf_model(X_train, y_train, grid_search = False, random_state = 101):
    '''
    
    Parameters
    ----------
    X_train : dataframe
    y_train : dataframe
    grid_search: boolean, optional
        option to use grid search or not. Default option is False.
    random_state : int, optional
        random number generator seed. Default value is 101.
        
    Returns
    -------
    clf : trained RF model
    
    '''
    
    if grid_search:
        param_grid = {'n_estimators' : [125, 150, 200, 300],
                      'max_depth' : [9, 12, 15, 18]}
        
        model_rf = RandomForestRegressor(criterion = "squared_error",
                                         random_state = random_state)
        clf = GridSearchCV(estimator = model_rf,
                                   param_grid = param_grid,
                                   scoring = 'neg_mean_squared_error',
                                   cv = 5,
                                   verbose = 2,
                                   n_jobs = 4)
        dump(clf, 'carmax_rf_gridsearch.joblib')
        
    else:
        if os.path.exists('carmax_rf_gridsearch.joblib'):
            clf = load('carmax_rf_gridsearch.joblib')
        else:
            clf = RandomForestRegressor(random_state = 101)
        
    clf.fit(X_train, y_train)
        
    return clf

#@st.cache_resource
def xgb_model(X_train, y_train, grid_search = False, random_state = 101):
    '''
    
    Parameters
    ----------
    X_train : dataframe
    y_train : dataframe
    grid_search: boolean, optional
        option to use grid search or not. Default option is False.
    random_state : int, optional
        random number generator seed. Default value is 101.
        
    Returns
    -------
    clf : trained XGB model
    
    '''
    if grid_search:
        # XGBRegressor
        param_grid = {'n_estimators' : [150, 200, 300, 500, 1000],
                      'max_depth' : [4, 6, 8]}
        model_xgbreg = XGBRegressor(learning_rate = 0.2,
                                    tree_method = 'hist',
                                    objective = 'reg:squarederror',
                                    random_state = random_state)
        clf = GridSearchCV(estimator = model_xgbreg,
                                       param_grid = param_grid,
                                       scoring = 'neg_mean_squared_error',
                                       cv = 5,
                                       verbose = 2,
                                       n_jobs = 4)
        
        dump(clf, 'carmax_xgb_gridsearch.joblib')
                
    else:
        if os.path.exists('carmax_xgb_gridsearch.joblib'):
            clf = load('carmax_xgb_gridsearch.joblib')
        else:
            clf = XGBRegressor(objective = 'reg:squarederror',
                               random_state = random_state)
        
    clf.fit(X_train, y_train)
        
    return clf

#@st.cache_resource
def linear_model(X_train, y_train, grid_search = False, random_state = 101):
    '''
    
    Parameters
    ----------
    X_train : dataframe
    y_train : dataframe
    grid_search: boolean, optional
        option to use grid search or not. Default option is False.
    random_state : int, optional
        random number generator seed. Default value is 101.
        
    Returns
    -------
    clf : trained EL model
    
    '''
    if grid_search:
        base_elastic_model = ElasticNet(l1_ratio = 1, 
                                        fit_intercept = True)
        param_grid = {'alpha':[40, 50, 60, 75],
                      'l1_ratio': [0.1, 0.25, 0.5, 0.9, 0.95, 0.99, 1]}
        clf = GridSearchCV(estimator=base_elastic_model,
                              param_grid=param_grid,
                              scoring='neg_mean_squared_error',
                              cv=5,
                              verbose=2,
                              n_jobs = 4)   
        
        dump(clf, 'carmax_EL_gridsearch.joblib')
    else:
        if os.path.exists('carmax_EL_gridsearch.joblib'):
            clf = load('carmax_EL_gridsearch.joblib')
        else:
            clf = ElasticNet(l1_ratio = 1, 
                             fit_intercept = True)
        
    clf.fit(X_train, y_train)
    
    return clf

def go_scat(name, x, y, c = None, dash = 'solid'):
    # add dash lines
    if c is None:
        go_fig = go.Scatter(name = name,
                            x = x,
                            y = y,
                            mode = 'markers',
                            marker = dict(size=6),
                            )
    else:
        go_fig = go.Scatter(name = name,
                            x = x,
                            y = y,
                            mode = 'markers',
                            marker = dict(size=6),
                            line = dict(color = c)
                            )
                                        
    return go_fig


def use_plotly(y_true, y_preds):
    '''
    
    Parameters
    ----------
    y_true : array
        array of test/true y values to be predicted
    y_preds: list
        list of arrays to plot
    
    '''
    fig_1 = go.Figure(data = go_scat(name = 'Price',
                                        x = list(range(len(y_true))),
                                        y = y_true.to_numpy().ravel(),
                                        c = '#36454F'
                                        ))
    
    colors = ['#FF0000', '#0000FF', '#00FF00']
    for ndx, y_pred in enumerate(y_preds):
        fig_1.add_trace(go_scat(name = f'model {ndx}',
                                    x = list(range(len(y_pred))),
                                    y = y_pred,
                                    c = colors[ndx],
                                    ))
    
    fig_1.update_layout(xaxis_title = "X",
                        yaxis_title = "Price")
    
    return fig_1

def remove_outliers(x, col):
    data = x[col]
    q25 = data.quantile(0.25)
    q75 = data.quantile(0.75)
    iqr = q75 - q25
    data = data[data <= (q75 + 1.5*iqr)]
    data = data[data >= (q25 - 1.5*iqr)]
    return x.reindex(data.index)


def year_mileage(data):
    data_list = []
    for y in list(data['year'].unique()):
        temp = remove_outliers(data[data['year'] == y], 'mileage')
        data_list.append(temp)
    
    return pd.concat(data_list)
   
def cluster_carmax(data, clusters = 4):
    data = data.reset_index().drop('index', axis=1)
    cats = data.select_dtypes(include = object).columns
    if len(cats) == 0:
        X = data.copy()
    else:
        X = data[['days_on_hand', 'gp_%']]
        enc = OneHotEncoder(sparse = False)
        enc.fit(data[cats])
        X_enc = pd.DataFrame(enc.transform(data[cats]), columns = enc.get_feature_names_out())
        X = pd.concat([X, X_enc], axis=1)
    
    kmeans = KMeans(n_clusters = clusters)
    kmeans.fit(X)
    X['labels'] = kmeans.predict(X)
    return X

def appraisal(df1, df2, make, model, year):
    comp = df1[(df1.make == make) & (df1.model == model) & (df1.year == year)]
    be = df2[(df2.make == make) & (df2.model == model) & (df2.year == str(year))]
    
    # Statistical comparison with current competitor listings
    print('Competitor Market Values')
    avg_comp_stats = comp['price'].describe()
    print(avg_comp_stats[['min', 'max', 'mean', '50%']])
    
    # Statistical comparison with historical carmax transactions
    print('Carmax Historical Record')
    avg_cmax_stats = be[['po_value', 'selling_price', 'amount_sold']].describe()
    print(avg_cmax_stats)
    
    
    
    return comp, be



if __name__ == "__main__":
    #st.title('CarMax Used Car Appraisal')
    
    #1 import data
    df_comp, df_backend = import_carmax_data()
    
    make_col, model_col, year_col = st.columns(3)
    with make_col:
        make = st.selectbox('Make',
                            options = config_carmax.makes_list,
                            index = 0)
    with model_col:
        model = st.selectbox('Model',
                            options = config_carmax.models_list,
                            index = 0)
    with year_col:
        # year is kept as str datatype, list of year is generated from min year
        yr = sorted(list(map(str, range(int(df_backend.year.min()), 
                               datetime.today().year))), 
                                reverse = True)
        year = st.selectbox('Year',
                            options = yr,
                            index = 0)
        
    # market price
    selected = df_comp[(df_comp.make == make) & (df_comp.model == model) & (df_comp.year == int(year))]
    st.dataframe(selected[['make', 'model', 'year', 'transmission', 'mileage', 'fuel_type', 'price', 'url']])
    
    
    
    
    
    # df_sold = df_backend[df_backend.current_status == 'Sold']
    
    # #2 apply filters
    
    # # price > 75,000 (390 removed)
    # price_filter = feature_filter(df_comp['price'], 
    #                               lower = 50000)
    # filtered_data = df_comp.filter(items = price_filter.index, axis=0)
    
    # # year > 2010 (300 removed)
    # year_filter = feature_filter(filtered_data['year'], 
    #                               lower = 2000)
    # filtered_data = filtered_data.filter(items = year_filter.index, axis=0)
    
    # # mileage
    # mileage_filter = feature_filter(filtered_data['mileage'], 
    #                                 upper = 100000,
    #                                 lower = 100)
    # filtered_data = filtered_data.filter(items = mileage_filter.index, axis=0)
    
    # # make
    # make_filter = list(filtered_data.make.value_counts()[filtered_data.make.value_counts() < 10].index)
    # filtered_data = filtered_data[~filtered_data['make'].isin(make_filter)]
    
    # # # remove models with only 1 entry
    # # filtered_data = filtered_data[filtered_data.model.isin(filtered_data.model.value_counts()[filtered_data.model.value_counts() != 1].index)]
    
    # st.header('Filetered Carmax data')
    # # show filtered dataframe
    # st.dataframe(filtered_data)
    
    # # divide data into train and test
    # X_train, X_test, y_train, y_test = training_data_prep(filtered_data, 
    #                                                       test_size = 0.2,
    #                                                       rs = 101)
    
    # y_test = y_test.reset_index().drop('index', axis=1).sort_values(by = 'price')
    # X_test = X_test.reindex(y_test.index)
    
    # # setup models
    # # xgboost, random forest, nonlinear regression, 
    
    # # RANDOM FOREST
    # y_pred_rf = rf_model(X_train, y_train, grid_search = True).predict(X_test)
    
    # # XGBRegressor
    # y_pred_xgb = xgb_model(X_train, y_train, grid_search = True).predict(X_test)
    
    # # elastic net
    # y_pred_EL = linear_model(X_train, y_train, grid_search = True).predict(X_test)
    
    # pred_list = [y_pred_rf, y_pred_xgb, y_pred_EL]
    # RMSE = [np.sqrt(mean_squared_error(y_test.to_numpy().ravel(), y)) for y in pred_list]
    # MAE = [mean_absolute_error(y_test.to_numpy().ravel(), y) for y in pred_list]
    
    # # plot
    # st.plotly_chart(use_plotly(y_test, [pred_list[RMSE.index(min(RMSE))]]), use_container_width = True)
    
    # st.header('Model Error Comparison')
    # mae_col, rmse_col = st.columns(2)
    # with mae_col:
    #     st.subheader('MAE')
    #     df_MAE = pd.DataFrame(MAE, index = ['RF', 'XGB', 'ElasticNet']).reset_index().rename(columns = {'index': 'model',
    #                                                                                                   0 : 'MAE'})
    #     st.dataframe(df_MAE)
        
    # with rmse_col:
    #     st.subheader('RMSE')
    #     df_RMSE = pd.DataFrame(RMSE, index = ['RF', 'XGB', 'ElasticNet']).reset_index().rename(columns = {'index': 'model',
    #                                                                                                   0 : 'RMSE'})
    #     st.dataframe(df_RMSE)
    

    # print (f'Random Forest RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_rf))}')
    # print (f'XGB RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_xgb))}')
    # print (f'ElasticNet RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_EL))}')
    
    # sns.scatterplot(x = range(X_test.shape[0]), y = y_test.to_numpy().ravel())
    # sns.scatterplot(x = range(X_test.shape[0]), y = pd.Series(y_pred_EL).reindex(y_test.index).to_numpy().ravel())