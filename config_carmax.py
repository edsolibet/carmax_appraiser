# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 13:35:23 2023

@author: carlo
"""

import pandas as pd
import numpy as np
import re, os, json, sys
from datetime import datetime
import time
from string import punctuation
import doctest # run via doctest.testmod()
from fuzzywuzzy import fuzz, process

from scrapy.crawler import CrawlerProcess
import openai
from google.cloud import storage, bigquery
import scrapydo

from color_corrections import color_corrections
from address_corrections import cities, provinces
#from positions_corrections import positions_corrections
#from comments_corrections import comments_corrections



## PARAMETERS ===============================================================##
gpt_on = False # gpt engine use 
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
output_path = os.path.abspath(os.path.dirname(__file__))
os.chdir(output_path) # current working directory
service_account = '\\carmax-ph-6f27871fa49f.json'

## ==========================================================================##

def lev_dist(seq1, seq2):
    '''
    Calculates levenshtein distance between texts
    '''
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros ((size_x, size_y))
    for x in range(size_x):
        matrix [x, 0] = x
    for y in range(size_y):
        matrix [0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix [x,y] = min(
                    matrix[x-1, y] + 1,
                    matrix[x-1, y-1],
                    matrix[x, y-1] + 1
                )
            else:
                matrix [x,y] = min(
                    matrix[x-1,y] + 1,
                    matrix[x-1,y-1] + 1,
                    matrix[x,y-1] + 1
                )
    return (matrix[size_x - 1, size_y - 1])

def punctuation_removal(punctuation):
    '''
    Removes needed symbols from punctuation list (which will be removed from model strings)
    '''
    remove_list = ['&']
    
    for r in remove_list:
        try:
            ndx = punctuation.index(r)
            punctuation = punctuation[:ndx] + punctuation[ndx+1:]
        except:
            continue
    return punctuation

punctuation = punctuation_removal(punctuation)
##-----------------------------------------------------------------------------

def import_makes():
    '''
    Import list of makes
    '''
    # output_path = os.path.abspath(os.path.join(os.path.dirname(__file__),
    #                                                                '..'))
    with open(os.getcwd() + '\\makes.txt') as makes_file:
        makes = makes_file.readlines()
        
    makes = [re.sub('\n', '', m).strip() for m in makes]
    return makes

makes_list = import_makes()

def clean_makes(x, makes):
    '''
    Cleans carmax makes input
    
    Parameters
    ----------
    x : string
        makes string input
    makes: list of string
        list of reference makes with string datatype
    
    >>> clean_makes('PORSHE', makes_list)
    'PORSCHE'
    >>> clean_makes('PUEGEOT', makes_list)
    'PEUGEOT'
    >>> clean_makes('VW', makes_list)
    'VW'
    
    '''
    if pd.isna(x):
        return np.NaN
        
    else:
        x = x.strip().upper()
        if any((match := m) for m in makes if fuzz.partial_ratio(m, x) >= 95):
            return match
        elif process.extractOne(x, makes)[1] >= 75:
            return process.extractOne(x, makes)[0]
        else:
            return x
    

##-----------------------------------------------------------------------------

def import_models():
    '''
    Import list of makes
    '''
    with open(os.getcwd() + '\\models.txt') as models_file:
        models = models_file.readlines()
        
    models = [re.sub('\n', '', m).strip() for m in models]
    return models

models_list = import_models()

def askGPT(text):
    #openai.api_key = "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"
    openai.api_key = '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'
    response = openai.Completion.create(
        engine = "text-davinci-003",
        prompt = text,
        temperature = 0.6,
        max_tokens = 100
        )
    return response.choices[0].text

def clean_model(model, makes, models):
    '''
    Cleans carmax model string
    
    Parameters
    ----------
    model : string
    makes : list
    * model_corrections module with module_corrections dict of regex expressions as keys
    
    Returns
    -------
    cleaned model string
    
    '''
    # uppercase and remove unnecessary spaces
    model = re.sub('[(\t)(\n)]', ' ', model.upper().strip())

    matches = [(m, fuzz.partial_ratio(model, re.sub('(CLASS|SERIES)', '', m).strip())) for m in models if fuzz.partial_ratio(re.sub('(CLASS|SERIES)', '', m).strip(), model) >= 85]
    
    if len(matches):
        ratio_list = list(zip(*matches))[1]
        best_match = matches[ratio_list.index(max(ratio_list))][0]
        
        if ('CLASS' in best_match) or ('SERIES' in best_match):
            best_match = re.sub('(SERIES|CLASS)', '', best_match).strip()
            no_make = [re.sub(make, '', best_match).strip() for make in makes if fuzz.partial_ratio(make, best_match) >= 90]
            
            pattern = re.compile(f'(?<={no_make[0]})(\s)?[0-9]+')
            model_num = re.search(pattern, model)
            if model_num:
                best_match = best_match + f'{model_num[0].strip()}'
            else:
                pass
        
        if any((match_make := make) for make in makes if make in best_match):
            return re.sub(match_make, '', best_match).strip()
        else:
            return best_match.upper().strip()
        
    elif process.extractOne(model, models)[1] >= 85:
        match = process.extractOne(model, models)[0]
        
        if any((match_make := make) for make in makes if make in match):
            return re.sub(match_make, '', match).strip()
        else:
            return match.upper().strip()
    
    else:
        # proceed to askGPT
        return np.NaN

def import_vehicle_types():
    '''
    Import list of vehicle_types
    '''
    
    with open(os.getcwd() + "\\vehicle_type.txt") as types_file:
        types = types_file.readlines()
        
    types = [re.sub('\n', '', t).strip() for t in types]
    return types

vehicle_types_list = import_vehicle_types()

def clean_vehicle_types(x, vehicle_types):
    '''
    Clean vehicle/body types using fuzzywuzzy partial ratio (if needed)
    '''
    if pd.isna(x):
        return np.NaN
    
    else:
        # baseline correction
        x = x.strip().upper()
        
        if x in vehicle_types:
            # if x exactly matches in vehicle_types
            return x
        else:
            # no exact match, use fuzzy partial ratio
            fuzz_list = [fuzz.partial_ratio(x,t) for t in vehicle_types]
            return vehicle_types[fuzz_list.index(max(fuzz_list))]


def remove_emoji(text):
    emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", re.UNICODE)
    return re.sub(emoj, ' ', text).strip()

def clean_name(name):
  '''
  Fix names which are duplicated.
  Ex. "John Smith John Smith"
  
  Parameters:
  -----------
  name: str
    
  Returns:
  --------
  fixed name; str
  
  '''
  name_corrections = {'MERCEDESBENZ' : 'MERCEDES-BENZ',
                      'MERCEDES BENZ' : 'MERCEDES-BENZ',
                      'IXSFORALL INC.' : 'Marvin Mendoza (iXSforAll Inc.)',
                      'GEMPESAWMENDOZA' : 'GEMPESAW-MENDOZA',
                      'MIKE ROLAND HELBES' : 'MIKE HELBES'}
  name_list = list()
  # removes emojis and ascii characters (i.e. chinese chars)
  name = remove_emoji(name).encode('ascii', 'ignore').decode()
  # split name by spaces
  for n in name.split(' '):
    if n not in name_list:
    # check each character for punctuations minus '.' and ','
        name_list.append(''.join([ch for ch in n 
                                if ch not in punctuation.replace('.', '')]))
    else:
        continue
  name_ = ' '.join(name_list).strip().upper()
  for corr in name_corrections.keys():
      if re.search(corr, name_):
          return name_corrections[re.search(corr, name_)[0]]
      else:
          continue
  return name_

def clean_engine(x, description = None):
    '''
    Cleans engine string value from carmax entries
    '''
    if pd.isna(x):
        if description is not None:
            x = description.upper().strip()
            if (match := re.search('(?<![0-9])[0-9]\.[0-9]((?<=L)|(?<=-LITER))?', x)):
                return f'{float(match[0].strip())}L'
            else:
                return np.NaN
        else:
            return np.NaN
        
    else:
        # baseline correction
        x = str(x).upper().strip()
        if (match := re.search('(?<![0-9])[0-9]\.[0-9]((?<=L)|(?<=-LITER))?', x)):
            return f'{float(match[0].strip())}L'
        else:
            if description is not None:
                x = description.upper().strip()
                if (match := re.search('(?<![0-9])[0-9]\.[0-9]((?<=L)|(?<=-LITER))?', x)):
                    return f'{float(match[0].strip())}L'
                else:
                    return np.NaN
            else:
                return np.NaN
            

def clean_engine_disp(x):
    if pd.notna(x):
        return '{:.1f}'.format(round(float(x)/1000, 2)) + 'L'
    else:
        return np.NaN

def clean_transmission(x, variant = None):
    '''
    Cleans/Finds transmission string values from carmax data
    
    DOCTESTS:
    >>> clean_transmission('AT')
    'AUTOMATIC'
    >>> clean_transmission('MANUEL')
    'MANUAL'
    >>> clean_transmission('MT')
    'MANUAL'
    >>> clean_transmission('ELEC')
    'ELECTRIC'
    >>> clean_transmission('VARIABLE')
    'CVT'
    '''
    t_dict = {'AT' : 'AUTOMATIC',
              'MT' : 'MANUAL',
              'CVT' : 'CVT',
              'VARIABLE' : 'CVT',
              'DUAL-CLUTCH' : 'DCT',
              'AUTO' : 'AUTOMATIC',
              'ELECTRIC' : 'ELECTRIC'}
    
    # value is NaN
    if pd.isna(x):
        # check if car name has transmission
        if variant is not None:
            t = re.search('(A|M|CV)T', variant)
            if t is not None:
                return t_dict[t[0]]
            else:
                return np.NaN
        else:
            return np.NaN
    # value is not NaN
    else:
        # baseline correct
        x = x.upper().strip()
        try:
            # if exact match
            if x in t_dict.values():
                return x
            
            # subset match
            elif any((match := t) for t in t_dict.values() if fuzz.partial_ratio(x, t) >= 85):
                return match
            
            # subset match in keys
            elif any((match := v) for v in t_dict.keys() if fuzz.partial_ratio(x, v) >= 85):
                return t_dict[match]
            
            else:
                match = process.extractOne(x, t_dict.values())
                if match[1] < 80:
                    return x
                else:
                    return match[0]
        except:
            return x

def clean_fuel_type(x):
    '''
    Cleans fuel_type data
    
    Parameters
    ----------
    x : string
        fuel_type string input or NaN
    
    DOCTESTS:
    >>> clean_fuel_type('GAS')
    'GASOLINE'
    >>> clean_fuel_type('DEISEL')
    'DIESEL'
    >>> clean_fuel_type('GASSOLIN')
    'GASOLINE'
    '''
    fuel_list = ['GASOLINE', 
                 'DIESEL',
                 'PETROL',
                 'FLEX/E85']
    
    if pd.isna(x):
        return np.NaN
    else:
        x = x.upper().strip()
        try:
            # exact match
            if any((match := f) in x for f in fuel_list):
                return match
            # subset match
            elif any((match := f) for f in fuel_list if fuzz.partial_ratio(x, f) >= 85):
                return match
            # approximate match
            elif any((match := f) for f in fuel_list if fuzz.token_set_ratio(x, f) >= 85):
                return match
            else:
                match = process.extractOne(x, fuel_list)
                if match[1] < 80:
                    return x
                else:
                    return match[0]
        except:
            return x

def clean_price(x):
    '''
    Cleans price string values from carmax entries
    '''
    if pd.isna(x):
        return np.NaN
    else:
        # baseline correct
        x = str(x).upper().strip()
        # normal result
        try:
            if 'Million' in x or 'M' in x:
                match = re.search('[1-9](.)?[0-9]+((?<!MILLION)|(?<!M))', x)
                return float(match[0])*1E6
            else:
                match = re.search('((?<=P)|(?<=₱))?(\s)?[1-9]+(,)?[0-9]+(,)?[0-9]+(.)?[0-9]+',x)
                return float(''.join(match[0].strip().split(',')))
        # unexpected result
        except:
            # get all digits
            try:
                return float(''.join(re.findall('[0-9]', x)))
            # return cleaned string
            except:
                return np.NaN
            
def get_year(x):
    '''
    Finds the year in a string value
    '''
    if pd.isna(x):
        return x
    else:
        x = re.search('(19|20)[0-9]{2}', x)
        if x:
            return x[0]
        else:
            return np.NaN

def clean_mileage(x):
    '''
    Cleans mileage values
    '''
    if pd.isna(x):
        return np.NaN
    
    else:
        # baseline correction
        x = str(x).upper().strip()
        # normal result
        try:
            r = re.search('[0-9]+(,)?[0-9]+(?<!KM)', x)
            if r is not None:
                return float(''.join(r[0].split(',')))
            else:
                return np.NaN
        # unexpected result
        except:
            # get all digits
            try:
                return float(''.join(re.findall('[0-9]', x)))
            # return cleaned string
            except:
                return np.NaN

def clean_color(color):
    '''
    Cleans color value string of carmax entries
    '''
    # returns '' if NaN
    if pd.isna(color):
        return np.NaN
    else:
        # baseline correction
        color = color.upper().strip()
        
        # lookup if color matches key in color_corrections
        if any(re.search((match_color := c), color) for c in color_corrections.keys()):
            color = color_corrections[match_color]
        else:
            try:
                dist_list = [c for c in np.unique(list(color_corrections.values())) if lev_dist(color, c) <= 2]
                if dist_list:
                    return dist_list[0]
                else:
                    return color
            except:
                return color


            
# def clean_vehicle_type(v_type):
#     '''
#     Cleans vehicle_type value string of carmax entries
    
#     '''
#     # if null, most probably sedan
#     if pd.isna(v_type):
#         return 'SEDAN'
#     # baseline correct
#     v_type = ''.join(v_type.upper().split(' '))
#     if 'SUV' in v_type:
#         v_type = 'SUV'
#     elif v_type in ['TRUCK', 'PICKUP']:
#         v_type = 'PICKUP TRUCK'
#     elif 'CROSSOVER':
#         v_type = 'SUV'
#     else:
#         pass
#     return v_type

def clean_address(address, lev_dist_tol = 3):
    '''
    Cleans address string of carmax financing data with help from levenshtein distance calc
    '''
    if pd.isna(address):
        return np.NaN
    else:
        # baseline correction
        address = address.upper().strip()
        # list of unique cities list
        cities_list = list(np.unique(list(cities.values())))
        if any(re.search((match := city), address) for city in cities_list):
            return f'{match} CITY, METRO MANILA, PHILIPPINES'
        elif re.search('Q(UEZON)?.*C(ITY)?', address):
            return 'QUEZON CITY, METRO MANILA, PHILIPPINES'
        # no exact matches found, use levenshtein dist
        else:
            # calculate lev dist for each word in address comparing to each city value
            # if lev dist passes tolerance value, pick match
            dist_list = [city for city in cities_list for a in address.split(' ') if lev_dist(a, city) <= lev_dist_tol]
            if dist_list:
                return f'{dist_list[0]} CITY, METRO MANILA, PHILIPPINES'
            # no city passes lev tolerance, move to province list and repeat process
            else:
                provinces_list = list(np.unique(list(provinces.values())))
                if any(re.search((match := prov), address) for prov in provinces_list):
                    return f'{match}, PHILIPPINES'
                else:
                    dist_list = [prov for prov in provinces_list for a in address.split(' ') if lev_dist(a, prov) <= lev_dist_tol]
                    if dist_list:
                        return f'{dist_list[0]}, PHILIPPINES'
                    else:
                        return address
                        
def clean_birthdate(bd):
    '''
    Cleans birthdate string value of carmax entries
    '''
    try:
        # triggers except clause if NaT
        pd.to_datetime(bd)
        year = bd.split('/')[-1]
        if int(year) >= int(str(datetime.today().year)[2:]):
            bd = bd[:6] + '19' + bd[6:]
        else:
            bd = bd[:6] + '20' + bd[6:]
            
        birthdate = pd.to_datetime(bd).date()
    except:
        birthdate = pd.NaT
    return birthdate

# def clean_positions(pos):
#     '''
#     Cleans positions data in carmax financing data
#     '''
#     if pd.notna(pos):
#         pos = pos.upper().strip()
#         if pos in positions_corrections.keys():
#             pos = positions_corrections[pos]
#         if 'MANAGER' in pos:
#             pos = 'MANAGER'
#         if 'OWNER' in pos:
#             pos = 'OWNER'
#     else:
#         pos = np.NaN
#     return pos

# def clean_TD_comments(x):
#     '''
#     Cleans Test Drive data comments string values
#     '''
#     if pd.isna(x):
#         return ''
#     # baseline correct
#     x = x.upper().strip()
#     if any((re.search((match := key), x) for key in comments_corrections.keys())):
#         comments = comments_corrections[match]
#     else:
#         comments = x
#     return comments

def upload_to_gcloud(filename):
    """
    this function take a dictionary as input and uploads
    it in a google cloud storage bucket
    """          
    
    ## your service-account credentials as JSON file
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.getcwd() + service_account
    
    ## instane of the storage client
    storage_client = storage.Client()

    ## instance of a bucket in your google cloud storage
    bucket = storage_client.get_bucket("carmax_scrapers-lica")
    
    try:
        ## if there already exists a file
        blob = bucket.get_blob('spiders/' + filename + '.json')

        
    except:
        ## if you want to create a new file 
        blob = bucket.blob('spiders/' + filename + '.json')
        
    ## uploading data using upload_from_filename method
    ## json.dumps() serializes a dictionary object as string
    blob.upload_from_filename(filename + '.json')
    
def scrape_upload(scraper):
    name = scraper.name
    process = CrawlerProcess(settings={
            "FEEDS": {
                f"{name}.json" : {"format": "json",
                                  "overwrite":True}
            },
        })
    
    process.crawl(scraper)
    process.start()
    
    try:
        upload_to_gcloud(name)
        print (f'{name}.json result exported to GCS')
    except:
        print (f'{name}.json upload to GCS failed.')

def json_to_df(filename):
    with open(filename) as file:
        data = json.load(file)
    
    row_list = []
    for ndx, d in enumerate(data):
        temp_df = pd.DataFrame(d['value'], d['label']).T
        temp_df.index = [ndx]
        row_list.append(temp_df)
    df = pd.concat(row_list)
    
    return df

def import_backend_data(date_filter = '2021-02-15'):
    '''
    Import carmax data from redash query
    
    CMX - CARMAX - All Status (w/ Dealership Categories & Saleability)
    http://app.redash.licagroup.ph/queries/7
    '''
    start_time = time.time()
    data = pd.read_csv('http://app.redash.licagroup.ph/api/queries/7/results.csv?api_key=sSt3ILBkdxIbOFC5DqmQxQhBq7SiiKVZBc8FBtei')
    data.columns = ['_'.join(col.lower().split(' ')) for col in data.columns]
    data = data[data.loc[:, 'current_status'].isin(['Sold', 'Available'])]
    # filters
    data = data.drop(['days_before_sold', "pony's_age"], axis=1) # duplicate of days_on_hand
    data = data[~(data['selling_price'].isnull())]
    data = data[~(data['model'] == '')]
    data = data.drop(index = [384])
    data = data[data.model.str.lower() != 'TEST']
    
    print ('Import and intial filters')
    print("--- %s seconds ---" % (time.time() - start_time))
    
    start_time = time.time()
    
    # data cleaning & engineering
    data.loc[:, 'model_'] = data.loc[:,'model']
    data.loc[:, 'date_sold'] = pd.to_datetime(data.loc[:,'date_sold'])
    data.loc[:, 'day'] = data.date_sold.dt.day_name()
    data.loc[:, 'qtr'] = pd.Series(pd.PeriodIndex(data.date_sold, freq='Q'))
    
    # car features
    data.loc[:, 'make'] = data.apply(lambda x: clean_makes(x['make'], makes_list), axis=1)
    webpage_model = data.apply(lambda x: clean_model(' '.join(x['webpage'][5:].split('-')), makes_list, models_list), axis=1)
    data.loc[:, 'model'] = data.apply(lambda x: clean_model(x['model'], makes_list, models_list), axis=1).fillna(webpage_model)
    data.loc[:, 'body_type'] = data.apply(lambda x: clean_vehicle_types(x['vehicle_type'], vehicle_types_list), axis=1)
    data.loc[:, 'year'] = data.apply(lambda x: str(int(x['year'])), axis=1)
    data.loc[:, 'transmission'] = data.apply(lambda x: clean_transmission(x['transmission']), axis=1)
    data.loc[:, 'fuel_type'] = data.apply(lambda x: clean_fuel_type(x['fuel_type']), axis=1)
    data.loc[:, 'mileage'] = data.apply(lambda x: clean_mileage(x['mileage']), axis=1)
    # fill in missing mileage data with mean of cars with same year
    m = data.groupby('year')['mileage'].mean()
    data.loc[:,'mileage'] = data.mileage.fillna(data['year'].map(m))
    data.loc[:, 'mileage_pass'] = data.apply(lambda x: float(x['mileage']) < 7000*abs(float(x['year']) - datetime.today().year), axis=1)
    
    data.loc[:, 'po_date'] = pd.to_datetime(data.loc[:, 'po_date']).fillna(data.loc[:,'date_sold'])
    data.loc[:, 'po_date'] = data.apply(lambda x: x['date_sold'] if x['po_date'] > x['date_sold'] else x['po_date'], axis=1)
    data.loc[:, 'days_on_hand'] = (data.loc[:, 'date_sold'] - data.loc[:,'po_date']).dt.days.fillna((datetime.today() - data.loc[:,'po_date']).dt.days)
    
    # additional columns
    data.loc[:, 'markup_%'] = data.apply(lambda x: (x['selling_price']*100/x['po_value'])-100 if x['po_value'] > 0 else np.NaN, axis=1)
    data.loc[:, 'gp_%'] = data.apply(lambda x: (x['gp/unit']*100/x['po_value']) if x['po_value'] > 0 else np.NaN, axis=1)
    # data.loc[:, 'MMY'] = data.apply(lambda x: '_'.join([str(x['make']), str(x['model']), str(x['year'])]), axis=1)
    data.loc[:, 'sold_year-month'] = data.apply(lambda x: str(x['date_sold'].year) + '-' + str(x['date_sold'].month).zfill(2), axis=1)
    data.loc[:, 'po_year-month'] = data.apply(lambda x: x['po_date'].year, axis=1)
    
    data = data[~(data['gp_%'].isnull())]
    data = data.reset_index().drop('index', axis=1)
    
    print ('Cleaning and data engineering')
    print("--- %s seconds ---" % (time.time() - start_time))
    # c = data.groupby('model').agg(color = ('color', lambda x: x.mode()[0]))['color']
    
    # data.loc[:, 'color'] = data.loc[:,'color'].apply(lambda x: config_carmax.clean_color(x))
    # data.loc[:, 'color'] = data.color.fillna(data['model'].map(c))
    # data.loc[:, 'color'] = data.loc[:, 'color'].apply(lambda x: x.upper())
    
    # drop unnecessary columns/rows
    
    if date_filter is not None:
        data = data[data.po_date >= pd.to_datetime(date_filter)]
        
    return data

