import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
import os
from functools import reduce
import math
import numpy as np

#########################################################
############ ~ Used for County Graphs ~ #################
#########################################################

def get_test_positivity():

    path = os.getcwd()
    path_sheets_api = path + '/hdc_api/'

    scope = ["https://spreadsheets.google.com/feeds",
             'https://www.googleapis.com/auth/spreadsheets',
             "https://www.googleapis.com/auth/drive.file",
             "https://www.googleapis.com/auth/drive"]

    creds = ServiceAccountCredentials.from_json_keyfile_name((path_sheets_api + "creds.json"), scope)

    client = gspread.authorize(creds)

    sheet = client.open_by_url('https://docs.google.com/spreadsheets/d/1sd-L317Je9ZhiQh3_uH9jTkl3ckc_o3sgrVauShcwCk/edit#gid=0')

    worksheet = sheet.worksheet('Test Positivity Rate')

    date_reported = worksheet.col_values(1)
    county = worksheet.col_values(2)
    positivity_rate = worksheet.col_values(3)

    df = pd.DataFrame([date_reported, county, positivity_rate]).T
    df.columns = df.iloc[0]
    df = df[1:]

    df['% Pos'] = [float(i.strip('%'))*.01 for i in df['% Pos']]

    return  df

def get_rt():

    path = os.getcwd()
    path_sheets_api = path + '/hdc_api/'

    scope = ["https://spreadsheets.google.com/feeds",
             'https://www.googleapis.com/auth/spreadsheets',
             "https://www.googleapis.com/auth/drive.file",
             "https://www.googleapis.com/auth/drive"]

    creds = ServiceAccountCredentials.from_json_keyfile_name((path_sheets_api + "creds.json"), scope)

    client = gspread.authorize(creds)

    sheet = client.open_by_url('https://docs.google.com/spreadsheets/d/1sd-L317Je9ZhiQh3_uH9jTkl3ckc_o3sgrVauShcwCk/edit#gid=0')

    worksheet = sheet.worksheet('Rate of Transmission')

    date_reported = worksheet.col_values(1)
    rt = worksheet.col_values(4)
    rt_lower = worksheet.col_values(6)
    rt_upper = worksheet.col_values(7)

    df = pd.DataFrame([date_reported, rt, rt_lower, rt_upper]).T
    df.columns = df.iloc[0]
    df = df[1:]

    return  df

def get_cases():

    path = os.getcwd()
    path_sheets_api = path + '/hdc_api/'

    scope = ["https://spreadsheets.google.com/feeds",
             'https://www.googleapis.com/auth/spreadsheets',
             "https://www.googleapis.com/auth/drive.file",
             "https://www.googleapis.com/auth/drive"]

    creds = ServiceAccountCredentials.from_json_keyfile_name((path_sheets_api + "creds.json"), scope)

    client = gspread.authorize(creds)

    sheet = client.open_by_url('https://docs.google.com/spreadsheets/d/1sd-L317Je9ZhiQh3_uH9jTkl3ckc_o3sgrVauShcwCk/edit#gid=0')

    worksheet = sheet.worksheet('COVID Data')

    date_reported = worksheet.col_values(2)
    county = worksheet.col_values(1)
    cases_100k = worksheet.col_values(7)
    cases_new = worksheet.col_values(4)

    df = pd.DataFrame([date_reported, county, cases_100k, cases_new]).T
    df.columns = df.iloc[0]
    df = df[1:]

    return  df

#########################################################
############### ~ Used for Forecast ~ ###################
#########################################################

def get_hospital():

    path = os.getcwd()
    path_sheets_api = path + '/hdc_api/'

    scope = ["https://spreadsheets.google.com/feeds",
             'https://www.googleapis.com/auth/spreadsheets',
             "https://www.googleapis.com/auth/drive.file",
             "https://www.googleapis.com/auth/drive"]

    creds = ServiceAccountCredentials.from_json_keyfile_name((path_sheets_api + "creds.json"), scope)

    client = gspread.authorize(creds)

    sheet = client.open_by_url('https://docs.google.com/spreadsheets/d/1sd-L317Je9ZhiQh3_uH9jTkl3ckc_o3sgrVauShcwCk/edit#gid=0')

    worksheet = sheet.worksheet('Hospital Data')

    date_reported = worksheet.col_values(1)
    active_hospitalizations = worksheet.col_values(3)
    active_ICU = worksheet.col_values(7)

    df = pd.DataFrame([date_reported, active_hospitalizations, active_ICU]).T
    df.columns = df.iloc[0]
    df = df[1:]
    df.rename(columns={'Active Hospitalized':'Hospitalizations', 'ICU - COVID':'ICU'}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])


    return  df

def get_deaths():

    path = os.getcwd()
    path_sheets_api = path + '/hdc_api/'

    scope = ["https://spreadsheets.google.com/feeds",
             'https://www.googleapis.com/auth/spreadsheets',
             "https://www.googleapis.com/auth/drive.file",
             "https://www.googleapis.com/auth/drive"]

    creds = ServiceAccountCredentials.from_json_keyfile_name((path_sheets_api + "creds.json"), scope)

    client = gspread.authorize(creds)

    sheet = client.open_by_url('https://docs.google.com/spreadsheets/d/1sd-L317Je9ZhiQh3_uH9jTkl3ckc_o3sgrVauShcwCk/edit#gid=0')

    worksheet = sheet.worksheet('COVID Data')

    date_reported = worksheet.col_values(2)
    county = worksheet.col_values(1)
    deaths = worksheet.col_values(21)

    df = pd.DataFrame([date_reported, county, deaths]).T
    df.columns = df.iloc[0]
    df = df[1:]
    df.rename(columns={'Deaths_Tot':'Deaths'}, inplace=True)
    df = df[df['Region'] == 'State']
    df.drop(['Region'], axis=1, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])

    return  df

def get_cases_state():

    path = os.getcwd()
    path_sheets_api = path + '/hdc_api/'

    scope = ["https://spreadsheets.google.com/feeds",
             'https://www.googleapis.com/auth/spreadsheets',
             "https://www.googleapis.com/auth/drive.file",
             "https://www.googleapis.com/auth/drive"]

    creds = ServiceAccountCredentials.from_json_keyfile_name((path_sheets_api + "creds.json"), scope)

    client = gspread.authorize(creds)

    sheet = client.open_by_url('https://docs.google.com/spreadsheets/d/1sd-L317Je9ZhiQh3_uH9jTkl3ckc_o3sgrVauShcwCk/edit#gid=0')

    worksheet = sheet.worksheet('COVID Data')

    date_reported = worksheet.col_values(2)
    county = worksheet.col_values(1)
    cases_new = worksheet.col_values(4)
    cases_new2 = worksheet.col_values(9)

    df = pd.DataFrame([date_reported, county, cases_new, cases_new2]).T
    df.columns = df.iloc[0]
    df = df[1:]
    df['Cases'] = df['New Confirmed Cases'][0:3023].append(df['New Cases'][3029:])
    df = df[df['Region'] == 'State']
    df.drop(['Region', 'New Confirmed Cases', 'New Cases'], axis=1, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])

    return  df

def get_hdc_covid_data():

    hospital = get_hospital()
    deaths = get_deaths()
    cases_state = get_cases_state()

    hospital = hospital.set_index('Date', drop=True).rename_axis(None)
    deaths = deaths.set_index('Date', drop=True).rename_axis(None)
    cases_state = cases_state.set_index('Date', drop=True).rename_axis(None)


    df = pd.concat([deaths, cases_state, hospital], axis=1, join='outer')
    df.replace('', np.nan, inplace=True)

    for col in df:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.interpolate(method='linear', limit_direction='forward')
    df.replace(np.nan, 0, inplace=True)
    df = df.astype(int)
    df = df.reset_index().rename(columns={'index':'Date'})
    df.rename(columns={'New Confirmed Cases': 'Cases'}, inplace=True)

    return df

test_df = get_cases_state()

test_df
