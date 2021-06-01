#########################################################
############### ~ Import Libraries ~ ####################
#########################################################

import numpy as np
import pandas as pd
import scipy.integrate as integrate
from dd_model.model import run_scenario
from bokeh.plotting import figure, output_file, show
from bokeh.models import ColumnDataSource
from bokeh.models.tools import HoverTool
from datetime import datetime, date, timedelta
from dateutil.parser import parse
from bokeh.models import HoverTool
from bokeh.models.widgets import Tabs, Panel
import csv
import json
import time
from bokeh.resources import CDN
from bokeh.embed import file_html
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import chart_studio.plotly as py
import chart_studio
from hdc_api.sheets import get_test_positivity, get_rt, get_cases, get_hdc_covid_data
from functions.functions import ndays, rt_ndays, infected_travelers, run_model, add_reported_new_cases, get_active_cases, insert_active_cases, get_start_date_index, load_hdc_data, create_bokeh_graph_df
from functions.visualizations import initialize_plotting_function, forecast_graph, create_forecast_graphs, create_oahu_reopening_graph_plotly, create_case_situation_graph, create_positivity_situation_graph

#########################################################
################### ~ Load Dfs ~ ########################
#########################################################

# Formats dates to reflect the following example: 9/7/2020 or 2020-9-7 (# or - represents removing 0s)
# format_date_str = "%#m/%#d/%Y" # PC
format_date_str = "%-m/%-d/%Y" # Mac/Linux

# Load COVID data from HDC (scraped from DOH)
hdc_covid_data_df = get_hdc_covid_data()

#########################################################
##################### ~ Set Dates ~ #####################
#########################################################

# Used for JSON update due to potential lag from today's date to model begin (resulting from covidtracking.org not updating until 1pm)
todays_date = str(datetime.now().strftime(format_date_str))

# Use for CSV creation
todays_date_f_string = todays_date.replace('/', '.')

# Set's 'today' (when the forecast output begins based on available historic data) - used to initialize start of the forecast
data_lag = 0

if list(hdc_covid_data_df['Date'])[-1] != todays_date:
    # sets the 'today' to the beginning of historic data start to ensure no data gap
    data_lag = (pd.to_datetime(todays_date) - list(hdc_covid_data_df['Date'])[-1]).days
else:
    data_lag = 0

today = str((datetime.now() - timedelta(days = data_lag)).strftime(format_date_str))

# Set initialization days and length of the forecast (recommend keeping consistent and only change situationally for specific scenarios)
initialization_days = 15
forecast_length = 13

Model_Begin = str((datetime.now() - timedelta(days = initialization_days)).strftime(format_date_str))
Model_End = str((datetime.now() - timedelta(days = -forecast_length)).strftime(format_date_str))

# Calculates time difference between model start and current date (used in data cleaning function)
ndays_today_Model_Begin = (ndays(Model_Begin, today))

#########################################################
################# ~ Set Parameters ~ ####################
#########################################################

# Model parameters used to move from Reported New Cases to Estimated Number of Initial Infections
shift_days = 7
cases_scale = 7

# Populations:
oahu = 953207
all_islands = 1415872

# Set Rt values for initalization, pessimistic, and expected scenarios
rt_initialization = 2.0
rt_estimate_pessimistic = 1.04
rt_estimate_expected = 1.00

# Set parameters
incubation = 3
infectious_duration = 6
delay = 3
hosp_stay = 7
ICU_stay = 10
hospitalization_rate = 0.0118
hospitalization_ICU_rate = 0.197
ICU_hosp_rate = 0.001

# Set [Exposed, Infected] travelers for each day in respective range of dates
travel_values  = [[4, 0], # rt_initialization - rt_estimate
                  [3, 0]] # rt_estimate - Model_End

# Set how much historical data is included in df & number of rolling days for reported new cases average
historical_days = 30
rolling_mean_days = 7

# Set how many days of Reported New Cases are summed to get the Active Cases for Quarantine
rolling_sum_days = 14

#########################################################
##### ~ Get Values for Initial Compartment Vector ~ #####
#########################################################

def loop_through_model():
# To start calculation for Estimated Number of Initial Infections,
# get the first day in day range equal to range of duration of infectious period,
# which when summed will account for total persons in the I compartment (infected) based on the Model Begin date
    start_index = [e for e, i in enumerate(hdc_covid_data_df['Date']) if i == pd.to_datetime(Model_Begin) + timedelta(shift_days - infectious_duration)][0]

    # Sum Reported New Cases for duration of infection,
    # then scale by the cases_scale factor to estimate true number of infected.
    initial_infections = hdc_covid_data_df[start_index : start_index + (infectious_duration + 1)]['Cases'].sum() * cases_scale

    # Get initial values from historical data for hospitalizations, ICU, and deaths
    initial_hospitalizations = int(hdc_covid_data_df['Hospitalizations'][hdc_covid_data_df['Date'] == pd.to_datetime(Model_Begin)])
    initial_ICU = int(hdc_covid_data_df['ICU'][hdc_covid_data_df['Date'] == Model_Begin])
    initial_Deaths = int(hdc_covid_data_df['Deaths'][hdc_covid_data_df['Date'] == Model_Begin]) + -4

    #########################################################
    #################### ~ Run Model ~ ######################
    #########################################################

    # Date Rt for pessimistic / expected begins. Starts ~1 week prior to today's date to smooth curve
    rt_estimate_start = str((datetime.now() - timedelta(days = 9)).strftime(format_date_str))

    # Run pessimistic & expected scenarios
    pessimistic_14 = run_model([Model_Begin, Model_End, initial_hospitalizations, initial_ICU, initial_Deaths, all_islands, initial_infections, incubation, infectious_duration, delay, hosp_stay, ICU_stay, hospitalization_rate, hospitalization_ICU_rate, ICU_hosp_rate], # Select which population to use in simulation
                        [Model_Begin, rt_estimate_start], # Dates for Rt changes
                        [rt_initialization, rt_estimate_pessimistic], # Rt values beginning on above dates
                        travel_values,
                        all_islands,
                        Model_End)
    expected_14 = run_model([Model_Begin, Model_End, initial_hospitalizations, initial_ICU, initial_Deaths, all_islands, initial_infections, incubation, infectious_duration, delay, hosp_stay, ICU_stay, hospitalization_rate, hospitalization_ICU_rate, ICU_hosp_rate],
                           [Model_Begin, rt_estimate_start],
                           [rt_initialization, rt_estimate_expected],
                           travel_values,
                           all_islands,
                           Model_End)

    ############# ~ Add Reported New Cases ~ ################

    # Run add_reported_new_cases for both scenarios
    pessimistic_14 = add_reported_new_cases('Pessimistic', pessimistic_14, shift_days, cases_scale, ndays_today_Model_Begin)
    expected_14 = add_reported_new_cases('Expected', expected_14, shift_days, cases_scale, ndays_today_Model_Begin)

    #########################################################
    ################# ~ Add Active Cases ~ ##################
    #########################################################

    # Run get_active_cases for both scenarios
    pessimistic_active = get_active_cases(pessimistic_14, hdc_covid_data_df, 'Pessimistic', rolling_sum_days)
    expected_active = get_active_cases(expected_14, hdc_covid_data_df, 'Expected', rolling_sum_days)

    # Add active cases to forecast dfs
    # expected_14['Active_Cases'] = expected_active['Active_Cases'][-len(expected_14.index):].values
    # pessimistic_14['Active_Cases'] = pessimistic_active['Active_Cases'][-len(pessimistic_14.index):].values

    insert_active_cases(expected_14, expected_active)
    insert_active_cases(pessimistic_14, pessimistic_active)

    return expected_14, pessimistic_14, expected_active, pessimistic_active
#########################################################
########## ~ Create Forecast Graphs (Bokeh) ~ ###########
#########################################################

# first run for initialization
expected_14, pessimistic_14, expected_active, pessimistic_active = loop_through_model()

# Create df for graphs
hdc_covid_data_df_historical_graph, active_historical = create_bokeh_graph_df(hdc_covid_data_df, rolling_mean_days, historical_days, pessimistic_active)

# parameter optimization test
latest_cases, latest_hospitalizations, latest_ICU, latest_deaths = list(hdc_covid_data_df_historical_graph['Reported_New_Cases'])[-1], list(hdc_covid_data_df_historical_graph['Hospitalizations'])[-1], list(hdc_covid_data_df_historical_graph['ICU'])[-1], list(hdc_covid_data_df_historical_graph['Deaths'])[-1]
first_day_forecast_cases, first_day_forecast_hospitalizations, first_day_forecast_ICU, first_day_forecast_deaths = list(expected_14['Reported_New_Cases'])[0], list(expected_14['Hospitalizations'])[0], list(expected_14['ICU'])[0], list(expected_14['Deaths'])[0]

latest_list = [latest_cases, latest_hospitalizations, latest_ICU, latest_deaths]
first_day_forecast_list = [first_day_forecast_cases, first_day_forecast_hospitalizations, first_day_forecast_ICU, first_day_forecast_deaths]

for i, (latest, first_day_forecast) in enumerate(zip(latest_list, first_day_forecast_list)):
    print("Current variable: " + str(latest))
    if 0.995 < (latest / first_day_forecast) < 1.005:
        continue
    while (0.995 > (latest / first_day_forecast)) or ((latest / first_day_forecast) > 1.005):
        if (latest / first_day_forecast) < 1:
            if i == 0:
                rt_initialization = rt_initialization - (rt_initialization * 0.01)
                expected_14, pessimistic_14, expected_active, pessimistic_active = loop_through_model()
                first_day_forecast = list(expected_14['Reported_New_Cases'])[0]
            if i == 1:
                hospitalization_rate = hospitalization_rate - (hospitalization_rate * 0.01)
                expected_14, pessimistic_14, expected_active, pessimistic_active = loop_through_model()
                first_day_forecast = list(expected_14['Hospitalizations'])[0]
            if i == 2:
                hospitalization_ICU_rate = hospitalization_ICU_rate - (hospitalization_ICU_rate * 0.01)
                expected_14, pessimistic_14, expected_active, pessimistic_active = loop_through_model()
                first_day_forecast = list(expected_14['ICU'])[0]
            if i == 3:
                ICU_hosp_rate = ICU_hosp_rate - (ICU_hosp_rate * 0.01)
                expected_14, pessimistic_14, expected_active, pessimistic_active = loop_through_model()
                first_day_forecast = list(expected_14['Deaths'])[0]
            print(' < 1: ' + str(latest / first_day_forecast))
        elif (latest / first_day_forecast) > 1:
            if i == 0:
                rt_initialization = rt_initialization + (rt_initialization * 0.01)
                expected_14, pessimistic_14, expected_active, pessimistic_active = loop_through_model()
                first_day_forecast = list(expected_14['Reported_New_Cases'])[0]
            if i == 1:
                hospitalization_rate = hospitalization_rate + (hospitalization_rate * 0.01)
                expected_14, pessimistic_14, expected_active, pessimistic_active = loop_through_model()
                first_day_forecast = list(expected_14['Hospitalizations'])[0]
            if i == 2:
                hospitalization_ICU_rate = hospitalization_ICU_rate + (hospitalization_ICU_rate * 0.01)
                expected_14, pessimistic_14, expected_active, pessimistic_active = loop_through_model()
                first_day_forecast = list(expected_14['ICU'])[0]
            if i == 3:
                ICU_hosp_rate = ICU_hosp_rate + (ICU_hosp_rate * 0.01)
                expected_14, pessimistic_14, expected_active, pessimistic_active = loop_through_model()
                first_day_forecast = list(expected_14['Deaths'])[0]
            print(' > 1: ' + str(latest / first_day_forecast))

print('Rt_Initialization: ' + str(rt_initialization))
print('Hospital Flow Rate: ' + str(hospitalization_rate))
print('ICU Flow Rate: ' + str(hospitalization_ICU_rate))
print('Death Flow Rate: ' + str(ICU_hosp_rate))

# Set Y axis max
max_hosp = pd.concat([pessimistic_14['Hospitalizations'], hdc_covid_data_df_historical_graph['Hospitalizations']]).astype(int).max() * 1.1
max_ICU = pd.concat([pessimistic_14['ICU'], hdc_covid_data_df_historical_graph['ICU']]).astype(int).max() * 1.1
max_Deaths = pd.concat([pessimistic_14['Deaths'], hdc_covid_data_df_historical_graph['Deaths']]).astype(int).max() * 1.5
max_Reported_New_Cases = pd.concat([pessimistic_14['Reported_New_Cases'], hdc_covid_data_df_historical_graph['Reported_New_Cases']]).astype(int).max() * 1.1
max_Active_Cases = pd.concat([pessimistic_active['Active_Cases'][-15:], active_historical['Active_Cases']]).astype(int).max() * 1.1

# Display forecast graphs
show(forecast_graph(pessimistic_14, expected_14, hdc_covid_data_df_historical_graph, max_hosp, max_ICU, max_Deaths, max_Reported_New_Cases))

#########################################################
######### ~ Create Forecast Graphs (Plotly) ~ ###########
#########################################################

# Change push_to_site to 'Y' if you want the forecasts live, otherwise use 'N' to view in IDE for QA
push_to_site = 'N'

# create_forecast_graphs(cdc_metric, df, df_column, expected_14, pessimistic_14, legend_name, max_metric, chart_studio_name)
create_forecast_graphs('case', hdc_covid_data_df_historical_graph, 'Reported_New_Cases', expected_14, pessimistic_14, 'Cases', max_Reported_New_Cases, 'cases', push_to_site)
create_forecast_graphs('death', hdc_covid_data_df_historical_graph, 'Deaths', expected_14, pessimistic_14, 'Deaths', max_Deaths, 'death', push_to_site)
create_forecast_graphs('active_cases', active_historical, 'Active_Cases', expected_14, pessimistic_14, 'Active Cases', max_Active_Cases, 'active_cases', push_to_site)
create_forecast_graphs('', hdc_covid_data_df_historical_graph, 'Hospitalizations', expected_14, pessimistic_14, 'Hospitalizations', max_hosp, 'hospitalizations', push_to_site)
create_forecast_graphs('', hdc_covid_data_df_historical_graph, 'ICU', expected_14, pessimistic_14, 'ICU', max_ICU, 'ICU', push_to_site)

#########################################################
######## ~ Create Oahu Tier Graph (Plotly) ~ ############
#########################################################

oahu_7_day_avg_cases = [93, 73, 68.7, 80, 49, 71, 81, 71, 84, 60, 72, 89, 83, 62, 88, 130, 86, 83, 59, 52, 33, 28, 22, 27, 30, 40, 58]
oahu_7_day_avg_cases_color = ['orange', 'orange', 'orange', 'orange', 'gold', 'orange', 'orange', 'orange', 'orange', 'orange', 'orange', 'orange', 'orange', 'orange', 'orange', 'orange', 'orange', 'orange', 'orange', 'orange','gold', 'gold', 'gold', 'gold', 'gold', 'gold', 'orange']
oahu_test_positivity_rate = [0.04, 0.032, 0.034, 0.023, 0.02, 0.027, 0.031, 0.027, 0.025, 0.021, 0.022, 0.031, 0.028, 0.029, 0.042, 0.040, 0.031, 0.031, 0.024, 0.020, 0.013, 0.011, 0.009, 0.01, 0.01, 0.015, 0.022]
oahu_test_positivity_rate_color = ['orange', 'orange', 'orange', 'gold', 'gold', 'orange', 'orange', 'orange', 'orange', 'gold', 'gold', 'orange', 'orange', 'orange', 'orange', 'orange', 'orange', 'orange', 'gold', 'gold', 'gold', 'gold', 'lightgreen', 'gold', 'gold','gold', 'gold', 'orange']
oahu_dates = ['9/30', '10/7', '10/14', '10/21', '10/28', '11/04', '11/11', '11/18', '11/25', '12/02', '12/09', '12/16', '12/23', '12/30', '1/06', '1/13', '1/20', '1/27', '2/3', '2/10', '2/17', '2/24', '3/3', '3/10', '3/17', '3/24', '3/31']

push_to_site_oahu = 'N'

create_oahu_reopening_graph_plotly(oahu_7_day_avg_cases, oahu_test_positivity_rate, oahu_dates, oahu_7_day_avg_cases_color, oahu_test_positivity_rate_color, push_to_site_oahu)

#########################################################
##### ~ Create Current Situation Graphs (Plotly) ~ ######
#########################################################

hdc_positivity = load_hdc_data('positivity', hdc_covid_data_df_historical_graph)
hdc_cases = load_hdc_data('cases', hdc_covid_data_df_historical_graph)

push_to_site_current_situation = 'N'

create_positivity_situation_graph(hdc_positivity, push_to_site_current_situation)
create_case_situation_graph(hdc_cases, push_to_site_current_situation)

#########################################################
########### ~ Create CSV of Forecast Output ~ ###########
#########################################################

# Create dfs for csv output
hdc_covid_data_df_historical_csv = hdc_covid_data_df_historical_graph[['Date', 'Reported_New_Cases', 'Hospitalizations', 'ICU', 'Deaths', 'Scenario']].reset_index(drop=True)
pessimistic_14_csv = pessimistic_14[['Date', 'Reported_New_Cases', 'Hospitalizations', 'ICU', 'Deaths', 'Scenario']].reset_index(drop=True)
expected_14_csv = expected_14[['Date', 'Reported_New_Cases', 'Hospitalizations', 'ICU', 'Deaths', 'Scenario']].reset_index(drop=True)

insert_active_cases(hdc_covid_data_df_historical_csv, expected_active)
insert_active_cases(pessimistic_14_csv, pessimistic_active)
insert_active_cases(expected_14_csv, expected_active)

# Create CSV
data_df = pd.concat([pessimistic_14_csv, expected_14_csv, hdc_covid_data_df_historical_csv]) # Create final df for CSV
data_df.to_csv(f'./Model_Outputs/model_output_{todays_date_f_string}.csv')

#########################################################
###### ~ Create Historical Parameters Dictionary ~ ######
#########################################################

# Create historical record of model inputs
todays_input_dict = {'Todays_Date' : todays_date,
              'Initialization_Start' : Model_Begin,
              'Rt_Initialization' : rt_initialization,
              'Rt_Estimate_Start' : rt_estimate_start,
              'Rt_Estimate_Worst' : rt_estimate_pessimistic,
              'Rt_Estimate_Expected' : rt_estimate_expected,
              'Incubation' : incubation,
              'Infectious_Duration' : infectious_duration,
              'Delay' : delay,
              'Hosp_Stay' : hosp_stay,
              'ICU_stay' : ICU_stay,
              'Hospitalization_Rate' : hospitalization_rate,
              'Hospitalization_ICU_Rate' : hospitalization_ICU_rate,
              'ICU_Hosp_Rate' : ICU_hosp_rate}

nest_todays_input_dict = {todays_date : todays_input_dict}

with open('historical_input_dict.json') as f:
  historical_input_dict_data = json.load(f)

# Add today's inputs to dict
historical_input_dict_data.update(nest_todays_input_dict)
