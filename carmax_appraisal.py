# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 10:50:02 2023

@author: carlo
"""
import pandas as pd
import numpy as np
import re, os
from import_carmax_data import import_hist_carmax_gsheet, import_all_carmax_data

import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
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

def cleanup_model(s, ref):
    '''
    Cleans up model strings by using most popular models as reference
    
    Parameters
    ----------
    s : string
        model string
    ref: list
        popular_models
    
    Returns
    -------
    model : string
    
    '''
    
    model = s
    for m in ref:
        if m in s:
            model = m
            break
        else:
            continue
    
    return model

@st.experimental_memo
def import_carmax_data():
    '''
    Returns
    -------
    import_carmax_gsheet : pandas dataframe
    '''
    # competitor_data
    if os.path.exists('carmax_competitor_data.csv'):
        final_competitor_data = pd.read_csv('carmax_competitor_data.csv')
        
    else:
        competitor_data = import_all_carmax_data()
        
        # fuel type, transmission, mileage
        final_competitor_data = competitor_data[~(competitor_data['fuel_type'].isnull()) &
                                                ~(competitor_data['mileage'].isnull()) & 
                                                ~ (competitor_data['transmission'].isnull())]
        final_competitor_data = final_competitor_data.reset_index().drop('index', axis=1)
        
        # data cleaning
        final_competitor_data.loc[:, 'model'] = final_competitor_data.loc[:,'model'].apply(lambda x: re.sub('[AM(CV)](/)?T', '', x).strip())
        popular_models = list(final_competitor_data.model.value_counts()[final_competitor_data.model.value_counts() >= 3].index)
        final_competitor_data.loc[:,'model'] = final_competitor_data.model.apply(lambda x: cleanup_model(x, popular_models))
        final_competitor_data.loc[:,'make'] = final_competitor_data.make.apply(lambda x: x.upper())
        final_competitor_data.loc[:,'transmission'] = final_competitor_data.loc[:,'transmission'].apply(lambda x: x.upper())
        # save file
        final_competitor_data.to_csv('carmax_competitor_data.csv')
    
    # hist data
    if os.path.exists('carmax_hist_data.csv'):
        hist_data = pd.read_csv('carmax_hist_data.csv')
    else:
        hist_data = import_hist_carmax_gsheet()
        hist_data = hist_data.reset_index().drop('index', axis=1)
        hist_data.to_csv('carmax_hist_data.csv')
        
    return final_competitor_data, hist_data

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
        
@st.experimental_memo
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
    
@st.experimental_singleton
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

@st.experimental_singleton
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

@st.experimental_singleton
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




if __name__ == "__main__":
    st.title('CarMax Used Car Appraisal')
    
    #1 import data
    df_data, df_hist = import_carmax_data()
    
    
    #2 apply filters
    
    # price > 75,000 (390 removed)
    price_filter = feature_filter(df_data['price'], 
                                  lower = 75000)
    filtered_data = df_data.filter(items = price_filter.index, axis=0)
    
    # year > 2010 (300 removed)
    year_filter = feature_filter(filtered_data['year'], 
                                 lower = 2005)
    filtered_data = filtered_data.filter(items = year_filter.index, axis=0)
    
    # mileage
    mileage_filter = feature_filter(filtered_data['mileage'], 
                                    upper = 100000,
                                    lower = 100)
    filtered_data = filtered_data.filter(items = mileage_filter.index, axis=0)
    
    # make
    make_filter = list(filtered_data.make.value_counts()[filtered_data.make.value_counts() < 10].index)
    filtered_data = filtered_data[~filtered_data['make'].isin(make_filter)]
    
    # remove models with only 1 entry
    filtered_data = filtered_data[filtered_data.model.isin(filtered_data.model.value_counts()[filtered_data.model.value_counts() != 1].index)]
    
    st.header('Filetered Carmax data')
    # show filtered dataframe
    st.dataframe(filtered_data)
    
    # divide data into train and test
    X_train, X_test, y_train, y_test = training_data_prep(filtered_data, 
                                                          test_size = 0.1,
                                                          rs = 101)
    
    y_test = y_test.reset_index().drop('index', axis=1).sort_values(by = 'price')
    X_test = X_test.reindex(y_test.index)
    
    # setup models
    # xgboost, random forest, nonlinear regression, 
    
    # RANDOM FOREST
    y_pred_rf = rf_model(X_train, y_train, grid_search = True).predict(X_test)
    
    # XGBRegressor
    y_pred_xgb = xgb_model(X_train, y_train, grid_search = True).predict(X_test)
    
    # elastic net
    y_pred_EL = linear_model(X_train, y_train, grid_search = True).predict(X_test)
    
    pred_list = [y_pred_rf, y_pred_xgb, y_pred_EL]
    RMSE = [np.sqrt(mean_squared_error(y_test.to_numpy().ravel(), y)) for y in pred_list]
    MAE = [mean_absolute_error(y_test.to_numpy().ravel(), y) for y in pred_list]
    
    # plot
    st.plotly_chart(use_plotly(y_test, [pred_list[RMSE.index(min(RMSE))]]), use_container_width = True)
    
    st.header('Model Error Comparison')
    mae_col, rmse_col = st.columns(2)
    with mae_col:
        st.subheader('MAE')
        df_MAE = pd.DataFrame(MAE, index = ['RF', 'XGB', 'ElasticNet']).reset_index().rename(columns = {'index': 'model',
                                                                                                      0 : 'MAE'})
        st.dataframe(df_MAE)
        
    with rmse_col:
        st.subheader('RMSE')
        df_RMSE = pd.DataFrame(RMSE, index = ['RF', 'XGB', 'ElasticNet']).reset_index().rename(columns = {'index': 'model',
                                                                                                      0 : 'RMSE'})
        st.dataframe(df_RMSE)
    
    # print (f'Random Forest RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_rf))}')
    # print (f'XGB RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_xgb))}')
    # print (f'ElasticNet RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_EL))}')
    
    # sns.scatterplot(x = range(X_test.shape[0]), y = y_test.to_numpy().ravel())
    # sns.scatterplot(x = range(X_test.shape[0]), y = pd.Series(y_pred_EL).reindex(y_test.index).to_numpy().ravel())