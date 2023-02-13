# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 12:30:57 2023

@author: carlo
"""
import pandas as pd
import gspread, json, time
import requests, openpyxl
from io import BytesIO

def import_hist_carmax_gsheet():
    '''
    Imports historical carmax competitor data
    
    Parameters
    ----------
    
    Returns
    -------
    final_df : dataframe
    '''
    
    with open("C:\\Users\\carlo\\Documents\\LICA Auto\\carmax_\\google_service_account_credentials.txt") as f:
        data = f.read()
    credentials = json.loads(data)
    
    sheet_id = '1-vHbqVXA40iQ_7Rwg14wB6C7ZMQYgAJZjpG_PzuCE5U'
    # load google worksheet via gspread
    gc = gspread.service_account_from_dict(credentials)
    sh = gc.open_by_key(sheet_id)
    temp_list = []
    for i in range(len(sh.worksheets())):
        worksheet = sh.get_worksheet(i)
        df = pd.DataFrame(worksheet.get_all_records())
        df.loc[:,'date'] = pd.to_datetime(worksheet.title).date()
        temp_list.append(df)
        time.sleep(4) # prevent reaching rate limit
    
    final_df = pd.concat(temp_list)
    final_df = final_df[['date', 'model', 'make', 'transmission', 'year', 'mileage',
                         'fuel_type', 'price', 'platform']]
    
    return final_df

def import_all_carmax_data():
    '''
    Reads/imports carmax competitor excel file and converts to dataframe
    '''
    
    # read excel file
    file = pd.ExcelFile('C:\\Users\\carlo\\Documents\\LICA Auto\\carmax_\\carmax_data.xlsx')
    cols = ['make', 'model', 'year', 'transmission', 'fuel_type', 'mileage', 'price', 'platform']
    # autodeal: 
    # Date, Make, Model, year, Transmission, Fuel Type, Mileage, Price
    # automart
    # year, make, model, fuel_type, mileage, transmission, price
    # carmudi
    # make, model, year, mileage, price, Transmission, Fuel Type, Colors
    # philkotse
    # make, model, year, price, transmission, color, mileage, fuel_type
    # carsada
    # price, Mileage, Fuel Type, Year, Transmission, make, model, color
    df_list = []
    for i in [j for j in file.sheet_names if 'raw' in j]:
        print (i)
        file_ = pd.read_excel(file, i)
        file_.columns = ['_'.join(col.lower().split(' ')) for col in file_.columns]
        df_list.append(file_[cols])
        
    final_df = pd.concat(df_list)
    return final_df