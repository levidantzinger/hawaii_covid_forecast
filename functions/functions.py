import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from dd_model.model import run_scenario
from hdc_api.sheets import get_test_positivity, get_rt, get_cases, get_hdc_covid_data

#########################################################
########### ~ Initialize Code for Model Run ~ ###########
#########################################################

def ndays(date1, date2):
    date_format = "%m/%d/%Y"
    date1 = datetime.strptime(date1, date_format)
    date2 = datetime.strptime(date2, date_format)
    delta = date2 - date1
    return delta.days

def rt_ndays(rt_change_dates):
    """
    Get the number of days between each inputted period and the Model Begin
    """
    rt_change_ndays = []

    for i in rt_change_dates:
        rt_change_ndays.append(ndays(rt_change_dates[0], i))

    return rt_change_ndays

['3/2/2021', '3/8/2021', [4, 0], [3, 0]]

def infected_travelers(rt_change_dates, Model_End, travel_values):
    """
    Creates a list of the number of travelers expected to be traveling to Hawaii
    that are exposed or infected with COVID that will enter undetected
    """
    # Calculate number of days for each period
    travel_dates = rt_change_dates + [Model_End]
    travel_dates = [ndays(travel_dates[i], travel_dates[i + 1]) for i in range(len(travel_dates) - 1)]

    # Create Travel array
    Travel_expected = []
    for e, dates in enumerate(travel_dates):
        for i in range(dates):
            Travel_expected.append(travel_values[e])

    Travel_expected = [[e] + value for e, value in enumerate(Travel_expected)]

    return Travel_expected

def run_model(model_inputs, rt_change_dates, rt_change_values, travel_values, island, Model_End):
    """
    Runs model based from Model Begin to Model End with Rt changing at stated points throughout the run.
    Outputs a dataframe from today's date to Model End.
    """

    # Get the number of days between each inputted period and the Model Begin
    rt_change_ndays = rt_ndays(rt_change_dates)

    # Zip dates and Rt values together for future processing
    zipped_rt_ndays = list(zip(rt_change_ndays, rt_change_values))

    # Creates a list of the number of travelers expected to be traveling to Hawaii
    # that are exposed or infected with COVID that will enter undetected
    Travel_expected = infected_travelers(rt_change_dates, model_inputs[1], travel_values)

    # Runs model code in ./dd_model/model.py with previously stated parameters
    data = run_scenario(model_inputs[0],
                        model_inputs[1],
                        model_inputs[2],
                        model_inputs[3],
                        model_inputs[4],
                        model_inputs[5],  # Size of population
                        model_inputs[6],  # Number of initial infections
                        model_inputs[7],  # Length of incubation period (days) # !age distributable
                        model_inputs[8],  # Duration patient is infectious (days)
                        model_inputs[9],  # Time delay for severe cases (non-icu) to be hospitalized
                        model_inputs[10], # Length of hospital stay (recovery time for severe cases)
                        model_inputs[11], # Length of hospital stay for ICU cases
                        model_inputs[12], # Percent of infectious people who go to hospital
                        model_inputs[13], # Percent of Hospitalizations that flow to ICU (remainder go to Recovered (R))
                        model_inputs[14], # Percent of ICU that flow to back to Hospitalizations (remainder go to Fatal (F))
                        zipped_rt_ndays,
                        Travel_expected)

    # Create df from model output
    data_columns = ['Date', 'Cases', 'Hospitalizations', 'ICU', 'Deaths', 'Rt', 'Infected_Travels', 'Exposed_Travelers', 'Susceptible', 'Total_Infected', 'New_Hospitalizations']
    df = pd.DataFrame(data, columns = data_columns)
    df = df[['Date', 'Cases', 'Hospitalizations', 'ICU', 'Deaths', 'Rt', 'Susceptible', 'Total_Infected', 'New_Hospitalizations']]

    df['New_Hospitalizations'] = [i - df['New_Hospitalizations'].iloc[e-1] if e >= 1 else 0 for e, i in enumerate(df['New_Hospitalizations'])]

    return df

#########################################################
############# ~ Add Reported New Cases ~ ################
#########################################################

def add_reported_new_cases(scenario, df, cases_scale, shift_days, ndays_today_Model_Begin):
    """
    Calculates (using total infected and scaling back down according to cases_scale)
    and adds 'Reported_New_Cases' to each scenario
    """
    # Change date to datetime
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')

    # Create 'active cases' from infected individuals
    df['Cases'] = df['Cases'] / cases_scale

    # Create 'daily new cases' from 'total infected'
    df['Reported_New_Cases'] = np.concatenate((np.zeros(1+shift_days),np.diff(df['Total_Infected'] / cases_scale)))[:len(df['Date'])]

    # Create column labeling scenario
    df['Scenario'] = scenario

    # Start output at today's date
    df = df.loc[ndays_today_Model_Begin:]

    return df

#########################################################
################# ~ Add Active Cases ~ ##################
#########################################################

def get_active_cases(model_output, daily_reported_cases, scenario):
    """
    Concatenates model output data with reported cases, then applies a rolling sum (rolling_sum_days)
    reported cases to create 'Active_Cases'
    """
    model_output = model_output[['Date', 'Reported_New_Cases']]
    model_output.columns = ['Date', 'Cases']

    daily_reported_cases = daily_reported_cases[['Date', 'Cases']][:-1]

    active_df = pd.concat([daily_reported_cases, model_output])
    active_df['Cases'] = active_df['Cases'].round()
    active_df['Active_Cases'] = active_df['Cases'].rolling(rolling_sum_days).sum().astype(float)
    active_df = active_df.reset_index(drop=True)
    return active_df

def insert_active_cases(data_df, active_cases_df):
    """
    Inserts active cases into model output dataframes by matching dates between the model output df (data_df) & active cases df
    """
    active_cases_list = []
    for i in data_df['Date']:
        active_cases_list.append(float(active_cases_df['Active_Cases'][active_cases_df['Date'] == i].values[0]))

    data_df.insert(len(data_df.columns), 'Active_Cases', active_cases_list)

#########################################################
############# ~ Add Reported New Cases ~ ################
#########################################################

# Initialize cleaning function
def add_reported_new_cases(scenario, df, shift_days, cases_scale, ndays_today_Model_Begin):
    """
    Calculates (using total infected and scaling back down according to cases_scale)
    and adds 'Reported_New_Cases' to each scenario
    """
    # Change date to datetime
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')

    # Create 'active cases' from infected individuals
    df['Cases'] = df['Cases'] / cases_scale

    # Create 'daily new cases' from 'total infected'
    df['Reported_New_Cases'] = np.concatenate((np.zeros(1+shift_days),np.diff(df['Total_Infected'] / cases_scale)))[:len(df['Date'])]

    # Create column labeling scenario
    df['Scenario'] = scenario

    # Start output at today's date
    df = df.loc[ndays_today_Model_Begin:]

    return df

#########################################################
################# ~ Add Active Cases ~ ##################
#########################################################

def get_active_cases(model_output, hdc_covid_data_df, scenario, rolling_sum_days):
    """
    Concatenates model output data with reported cases, then applies a rolling sum (rolling_sum_days)
    reported cases to create 'Active_Cases'
    """
    model_output = model_output[['Date', 'Reported_New_Cases']]
    model_output.columns = ['Date', 'Cases']

    hdc_covid_data_df = hdc_covid_data_df[['Date', 'Cases']][:-1]

    active_df = pd.concat([hdc_covid_data_df, model_output])
    active_df['Cases'] = active_df['Cases'].round()
    active_df['Active_Cases'] = active_df['Cases'].rolling(rolling_sum_days).sum().astype(float)
    active_df = active_df.reset_index(drop=True)
    return active_df

#########################################################
############## ~ Create Graphs (Bokeh) ~ ################
#########################################################

def create_bokeh_graph_df(hdc_covid_data_df, rolling_mean_days, historical_days, pessimistic_active):
        # Create 7 day moving average for Daily Reported New Cases
        hdc_covid_data_df['Rolling_Average_Cases'] = hdc_covid_data_df['Cases'].rolling(rolling_mean_days).mean()

        # Reduce daily new cases df to length of selected period
        hdc_covid_data_df_len = len(hdc_covid_data_df.index)
        hdc_covid_data_df_historical = hdc_covid_data_df.loc[(hdc_covid_data_df_len - (historical_days + (rolling_mean_days - 1))):]

        # Clean df
        hdc_covid_data_df_historical_graph = hdc_covid_data_df_historical.rename(columns={'Rolling_Average_Cases': 'Reported_New_Cases'})
        hdc_covid_data_df_historical_graph['Scenario'] = 'Historical'

        # Reduce active cases df to length of selected periods
        pessimistic_active_len = len(pessimistic_active.index)
        active_historical = pessimistic_active.loc[hdc_covid_data_df_historical_graph.index[0]:hdc_covid_data_df_historical_graph.index[-1]]

        # Add active cases
        hdc_covid_data_df_historical_graph['Active_Cases'] = active_historical['Active_Cases']

        return hdc_covid_data_df_historical_graph, active_historical

#########################################################
################ ~ Current Situation ~ ##################
#########################################################

def get_start_date_index(df_date_series, hdc_covid_data_df_historical_graph):
    start_date_index = 0
    for e, i in enumerate(df_date_series):
        if i == hdc_covid_data_df_historical_graph['Date'].iloc[0]:
            start_date_index = e
            break
    return start_date_index

def load_hdc_data(df_name, hdc_covid_data_df_historical_graph):
    if df_name == 'positivity':
        hdc_data = get_test_positivity()
        hdc_data = hdc_data[(hdc_data['Region'] == 'Hawaii County') | (hdc_data['Region'] == 'Honolulu County') | (hdc_data['Region'] == 'Kauai County') | (hdc_data['Region'] == 'Maui County') | (hdc_data['Region'] == 'State')]
        hdc_data['Date'] = pd.to_datetime(hdc_data['Date'][630:]) # cut df due to errors in labels
    elif df_name == 'cases':
        hdc_data = get_cases()
        hdc_data = hdc_data[(hdc_data['Region'] == 'Hawaii County') | (hdc_data['Region'] == 'Honolulu County') | (hdc_data['Region'] == 'Kauai County') | (hdc_data['Region'] == 'Maui County') | (hdc_data['Region'] == 'State')]
        hdc_data['Date'] = pd.to_datetime(hdc_data['Date'])
    else:
        print("Incorrect name.")
    hdc_data = hdc_data[get_start_date_index(hdc_data['Date'], hdc_covid_data_df_historical_graph):]
    return hdc_data

#########################################################
################ ~ Email Generation ~ ###################
#########################################################
