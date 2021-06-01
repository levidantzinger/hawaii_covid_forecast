import numpy as np
import pandas as pd
from bokeh.plotting import figure, output_file, show
from bokeh.models import ColumnDataSource
from bokeh.models.tools import HoverTool
from bokeh.models import HoverTool
from bokeh.models.widgets import Tabs, Panel
from bokeh.resources import CDN
from bokeh.embed import file_html
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import chart_studio.plotly as py
import chart_studio
#########################################################
############## ~ Create Graphs (Bokeh) ~ ################
#########################################################

def initialize_plotting_function(y_metric, pessimistic_14, expected_14, hdc_covid_data_df_historical_graph, max_hosp, max_ICU, max_Deaths, max_Reported_New_Cases):
    """
    Initializing function for plotting historical data + model output data
    """
    # Set data sources
    source_pessimistic_14 = ColumnDataSource(pessimistic_14)
    source_expected_14 = ColumnDataSource(expected_14)

    source_daily_new_cases_historical = ColumnDataSource(hdc_covid_data_df_historical_graph)
    source_state_df_historical = ColumnDataSource(hdc_covid_data_df_historical_graph)

    # Creates interactive hover
    tooltips = [
            ('Scenario', '@Scenario'),
            (f'{y_metric}',f'@{y_metric}'),
            ('Date', '@Date{%F}')
           ]

    y_max = 0

    if y_metric == 'Hospitalizations':
        y_max = int(max_hosp)
    if y_metric == 'ICU':
        y_max = int(max_ICU)
    if y_metric == 'Deaths':
        y_max = int(max_Deaths)
    if y_metric == 'Reported_New_Cases':
        y_max = int(max_Reported_New_Cases)

    # Initalize plot foundation
    p = figure(x_axis_type = "datetime", y_range=(0, y_max))

    # Add historical lines
    if y_metric == 'Hospitalizations':
        historical_hosp_line = p.line(x='Date', y=f'{y_metric}',
             source=source_state_df_historical,
             line_width=2, color = 'grey')
        p.add_tools(HoverTool(renderers=[historical_hosp_line], tooltips=tooltips, mode='vline', formatters={'Date': 'datetime'}))
    if y_metric == 'Deaths':
        historical_death_line = p.line(x='Date', y=f'{y_metric}',
             source=source_state_df_historical,
             line_width=2, color = 'grey')
        p.add_tools(HoverTool(renderers=[historical_death_line], tooltips=tooltips, mode='vline', formatters={'Date': 'datetime'}))
    if y_metric == 'ICU':
        historical_ICU_line = p.line(x='Date', y=f'{y_metric}',
             source=source_state_df_historical,
             line_width=2, color = 'grey')
        p.add_tools(HoverTool(renderers=[historical_ICU_line], tooltips=tooltips, mode='vline', formatters={'Date': 'datetime'}))
    if y_metric == 'Reported_New_Cases':
        historical_daily_new_cases_line = p.line(x='Date', y=f'{y_metric}',
            source=source_daily_new_cases_historical,
            line_width=2, color = 'grey')
        p.add_tools(HoverTool(renderers=[historical_daily_new_cases_line], tooltips=tooltips, mode='vline', formatters={'Date': 'datetime'}))

    # Add forecast lines
    pessimistic_14_line = p.line(x='Date', y=f'{y_metric}',
             source=source_pessimistic_14,
             line_width=2, color='firebrick', legend='Pessimistic')
    expected_14_line = p.line(x='Date', y=f'{y_metric}',
             source=source_expected_14,
             line_width=2, color='steelblue', legend='Expected')

    p.add_tools(HoverTool(renderers=[expected_14_line], tooltips=tooltips, mode='vline', formatters={'Date': 'datetime'}))
    p.add_tools(HoverTool(renderers=[pessimistic_14_line], tooltips=tooltips, mode='vline', formatters={'Date': 'datetime'}))

    # Add Graph details
    p.title.text = f'Number of {y_metric}'
    p.xaxis.axis_label = 'Date'
    p.yaxis.axis_label = f'{y_metric}'

    # Sets graph size
    p.plot_width = 1200
    p.plot_height = 700

    # Sets legend
    p.legend.location = "top_left"
    p.legend.click_policy="hide"

    return p

def forecast_graph(pessimistic_14, expected_14, hdc_covid_data_df_historical_graph, max_hosp, max_ICU, max_Deaths, max_Reported_New_Cases):
    """
    Creates tabs for each metric and displays output to html
    """

    # Create panels for each tab
    cases_tab = Panel(child=initialize_plotting_function('Reported_New_Cases', pessimistic_14, expected_14, hdc_covid_data_df_historical_graph, max_hosp, max_ICU, max_Deaths, max_Reported_New_Cases), title='Reported New Cases')
    hospitalized_tab = Panel(child=initialize_plotting_function('Hospitalizations', pessimistic_14, expected_14, hdc_covid_data_df_historical_graph, max_hosp, max_ICU, max_Deaths, max_Reported_New_Cases), title='Hospitalizations')
    ICU_tab = Panel(child=initialize_plotting_function('ICU', pessimistic_14, expected_14, hdc_covid_data_df_historical_graph, max_hosp, max_ICU, max_Deaths, max_Reported_New_Cases), title='ICU')
    Deaths_tab = Panel(child=initialize_plotting_function('Deaths', pessimistic_14, expected_14, hdc_covid_data_df_historical_graph, max_hosp, max_ICU, max_Deaths, max_Reported_New_Cases), title='Deaths')
    Susceptible_tab = Panel(child=initialize_plotting_function('Susceptible', pessimistic_14, expected_14, hdc_covid_data_df_historical_graph, max_hosp, max_ICU, max_Deaths, max_Reported_New_Cases), title='Susceptible')

    # Assign the panels to Tabs
    tabs = Tabs(tabs=[Susceptible_tab, cases_tab, hospitalized_tab, ICU_tab, Deaths_tab])

    return tabs

#########################################################
############## ~ Create Graphs (Bokeh) ~ ################
#########################################################

def create_forecast_graphs(cdc_metric, df, df_column, expected_14, pessimistic_14, legend_name, max_metric, chart_studio_name, push_to_site):
    if (cdc_metric == 'case') or (cdc_metric == 'death'):

        cdc_forecast = pd.read_csv(f'./cdc_forecasts/{cdc_metric}_forecast.csv')
        cdc_forecast.rename(columns={cdc_forecast.iloc[:, 0].name : 'Date'}, inplace=True)
        cdc_forecast['Date'] = pd.to_datetime(cdc_forecast['Date'])

        # Filter CDC forecast to correct dates
        start_counter = 0
        for i in cdc_forecast['Date']:
            if i != pessimistic_14['Date'].iloc[0]:
                start_counter += 1
            else:
                break

        cdc_forecast = cdc_forecast.iloc[start_counter:start_counter+16]
        cdc_forecast = cdc_forecast.set_index('Date').rename_axis(None)

        # Add bools for ensemble buttons
        show_ensemble_lines = [True, True, True] # True's represent HIPAM lines and will always be displayed
        hide_ensemble_lines = [True, True, True] # True's represent HIPAM lines and will always be displayed

        for i in range(0, len(cdc_forecast.columns)):
            show_ensemble_lines.append(True)
            hide_ensemble_lines.append(False)

    # Function to create plotly figure & push to Chart Studio
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df['Date'], y=df[df_column], mode='lines', name=legend_name,
                             line=dict(color='lightgray', width=4)))
    fig.add_trace(go.Scatter(x=pessimistic_14['Date'], y=pessimistic_14[df_column], mode='lines', name=legend_name,
                             line=dict(color='lightcoral', width=4)))
    fig.add_trace(go.Scatter(x=expected_14['Date'], y=expected_14[df_column], mode='lines', name=legend_name,
                             line=dict(color='lightblue', width=4)))
    if (cdc_metric == 'case') or (cdc_metric == 'death'):
        for i in range(0, len(cdc_forecast.columns)):
            fig.add_trace(go.Scatter(x=pessimistic_14['Date'], y=cdc_forecast.iloc[:, i], mode='lines', name=legend_name,
                                     line=dict(color='lightgray', width=1)))
    fig.update_xaxes(showline=True,
                     showgrid=False,
                     showticklabels=True,
                     linecolor='rgb(140, 140, 140)',
                     linewidth=2,
                     ticks='outside',
                     tickfont=dict(
                        family='Arial',
                        size=16,
                        color='rgb(140, 140, 140)'
                        ))
    fig.update_yaxes(range=[0, max_metric],
                     showline=True,
                     showgrid=False,
                     showticklabels=True,
                     linecolor='rgb(140, 140, 140)',
                     linewidth=2,
                     ticks='outside',
                     tickfont=dict(
                        family='Arial',
                        size=16,
                        color='rgb(140, 140, 140)',
                        ))
    if (cdc_metric == 'case') or (cdc_metric == 'death'):
        fig.update_layout(autosize=False,
                          width=1500,
                          height=1000,
                          showlegend=False,
                          plot_bgcolor='white',
                          margin=dict(
                            autoexpand=False,
                            l=80,
                            r=80,
                            t=50
                            ),
                          title={'y':1},
                            updatemenus=[
                                    dict(
                                        type="buttons",
                                        bgcolor = 'rgb(205, 205, 205)',
                                        bordercolor = 'rgb(84, 84, 84)',
                                        font = dict(color='rgb(84, 84, 84)'),
                                        direction="right",
                                        active=-1,
                                        x=0.57,
                                        y=1.05,
                                        buttons=list([
                                            dict(label="Show Ensemble",
                                                 method="update",
                                                 args=[{"visible": show_ensemble_lines},
                                                       {"annotations": []}]),
                                            dict(label="Hide Ensemble",
                                                 method="update",
                                                 args=[{"visible": hide_ensemble_lines},
                                                       {"annotations": []}])
                                        ]),

                                    )
                                ])
    else:
            fig.update_layout(autosize=False,
                              width=1500,
                              height=1000,
                              showlegend=False,
                              plot_bgcolor='white',
                              margin=dict(
                                autoexpand=False,
                                l=80,
                                r=80,
                                t=50
                                ),
                              title={'y':1}
                              )

    if push_to_site == 'Y':
        username = '' # your username
        api_key = '' # your api key - go to profile > settings > regenerate key
        chart_studio.tools.set_credentials_file(username=username, api_key=api_key)
        py.plot(fig, filename = f'hipam_forecast_{chart_studio_name}', auto_open=True)

        fig.write_html(f"./chart_studio/file_{chart_studio_name}.html")
    else:
        fig.show()

#########################################################
########## ~ Create Oahu Graph (Plotly) ~ ###############
#########################################################

def create_oahu_reopening_graph_plotly(oahu_7_day_avg_cases, oahu_test_positivity_rate, oahu_dates, oahu_7_day_avg_cases_color, oahu_test_positivity_rate_color, push_to_site_oahu):
    oahu_stats = {'7 Day Avg. Cases' : oahu_7_day_avg_cases,
                  'Test Positivity Rate' : oahu_test_positivity_rate
                  }
    oahu_df = pd.DataFrame(oahu_stats)
    oahu_df.index = oahu_dates

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Bar(x=oahu_df.index, y=oahu_df['7 Day Avg. Cases'], name = 'Cases', marker_color = oahu_7_day_avg_cases_color),
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=oahu_df.index, y=oahu_df['Test Positivity Rate'], mode='markers', marker=dict(size=50, color=oahu_test_positivity_rate_color), marker_symbol='cross-dot', name='Test Positivity'),
                  secondary_y=True)

    fig.update_traces(marker_line_color='rgb(84,84,84)', marker_line_width=3, opacity=0.8)

    fig.update_xaxes(showline=True,
                     showgrid=False,
                     showticklabels=True,
                     linecolor='rgb(140, 140, 140)',
                     linewidth=2,
                     ticks='outside',
                     tickfont=dict(
                        family='Arial',
                        size=16,
                        color='rgb(140, 140, 140)',
                        ))
    fig.update_yaxes(range = [0, 140],
                     title_text='7 Day Avg. Cases',
                     title_font = {"size": 20, "color": 'rgb(140, 140, 140)'},
                     secondary_y=False,
                     showline=True,
                     showgrid=False,
                     showticklabels=True,
                     linecolor='rgb(140, 140, 140)',
                     linewidth=2,
                     ticks='outside',
                     tickfont=dict(
                        family='Arial',
                        size=16,
                        color='rgb(140, 140, 140)',
                        ))
    fig.update_yaxes(range = [0, 0.0625],
                     title_text='7 Day Avg. Test Positivity',
                     title_font = {"size": 20, "color": 'rgb(140, 140, 140)'},
                     secondary_y=True,
                     showline=True,
                     showgrid=False,
                     showticklabels=True,
                     linecolor='rgb(140, 140, 140)',
                     linewidth=2,
                     tickformat='.1%',
                     ticks='outside',
                     tickfont=dict(
                        family='Arial',
                        size=16,
                        color='rgb(140, 140, 140)',
                        ))
    fig.update_layout(autosize=False,
                      width=1500,
                      height=1000,
                      showlegend=False,
                      plot_bgcolor='white',
                      margin=dict(
                        autoexpand=False,
                        l=80,
                        r=80,
                        t=50
                        ),
                      title={'y':1},
                      shapes=[
                            dict(
                                type="rect",
                                # x-reference is assigned to the x-values
                                xref="paper",
                                # y-reference is assigned to the plot paper [0,1]
                                yref="paper",
                                x0=0,
                                y0=0,
                                x1=0.29,
                                y1=1,
                                fillcolor="FireBrick",
                                opacity=0.2,
                                layer="below",
                                line_width=0,
                            ),
                            dict(
                                type="rect",
                                # x-reference is assigned to the x-values
                                xref="paper",
                                # y-reference is assigned to the plot paper [0,1]
                                yref="paper",
                                x0=0.29,
                                y0=0,
                                x1=0.815,
                                y1=1,
                                fillcolor="orange",
                                opacity=0.2,
                                layer="below",
                                line_width=0,
                            ),
                            dict(
                                type="rect",
                                # x-reference is assigned to the x-values
                                xref="paper",
                                # y-reference is assigned to the plot paper [0,1]
                                yref="paper",
                                x0=0.815,
                                y0=0,
                                x1=0.92,
                                y1=1,
                                fillcolor="gold",
                                opacity=0.2,
                                layer="below",
                                line_width=0,
                            ),
                            {
                                'type': 'line',
                                'xref': 'paper',
                                'x0': 0.94,
                                'y0': 100, # use absolute value or variable here
                                'x1': 0,
                                'y1': 100, # ditto
                                'line': {
                                    'color': 'FireBrick',
                                    'width': 4,
                                    'dash': 'dash',
                                },
                            },
                            {
                                'type': 'line',
                                'xref': 'paper',
                                'x0': 0.94,
                                'y0': 50, # use absolute value or variable here
                                'x1': 0,
                                'y1': 50, # ditto
                                'line': {
                                    'color': 'Orange',
                                    'width': 4,
                                    'dash': 'dash',
                                },
                            },
                            {
                                'type': 'line',
                                'xref': 'paper',
                                'x0': 0.94,
                                'y0': 25, # use absolute value or variable here
                                'x1': 0,
                                'y1': 25, # ditto
                                'line': {
                                    'color': 'Gold',
                                    'width': 4,
                                    'dash': 'dash',
                                },
                            },

                        ],
                      )

    if push_to_site_oahu == 'Y':
        username = '' # your username
        api_key = '' # your api key - go to profile > settings > regenerate key
        chart_studio.tools.set_credentials_file(username=username, api_key=api_key)
        py.plot(fig, filename = 'current_oahu', auto_open=True)

        fig.write_html("./chart_studio/file_current_oahu.html")
    else:
        fig.show()

#########################################################
################ ~ Current Situation ~ ##################
#########################################################

############# ~ Cases ~ ##############

def create_case_situation_graph(hdc_cases, push_to_site_current_situation):
    hdc_cases['NewCases_Rate'] = hdc_cases['NewCases_Rate'].astype(float)
    hdc_cases['Region'] = ['All County' if i == 'State' else i for i in hdc_cases['Region']]

    low = 1
    medium = 10
    high = 25
    critical = 38

    def get_threshold_data(df, metric, region, threshold, ceiling):
        threshold_list = []
        for e, i in enumerate(df[f'{metric}'][df['Region'] == f'{region} County']):
            if (e > len((df[f'{metric}'][df['Region'] == f'{region} County']))-2) & (i >= threshold) & (i <= ceiling):
                threshold_list.append(i)
                break
            if (e > len((df[f'{metric}'][df['Region'] == f'{region} County']))-2):
                threshold_list.append(np.nan)
                break
            if df[f'{metric}'][df['Region'] == f'{region} County'].iloc[e+1] > threshold:
                threshold_list.append(i)
                continue
            if df[f'{metric}'][df['Region'] == f'{region} County'].iloc[e] > threshold:
                threshold_list.append(i)
                continue
            if (e != 0) & (df[f'{metric}'][df['Region'] == f'{region} County'].iloc[e-1] > threshold) & (i <= threshold):
                threshold_list.append(i)
            else:
                threshold_list.append(np.nan)
        return threshold_list

    hawaii_medium_cases = get_threshold_data(hdc_cases, 'NewCases_Rate', 'Hawaii', low, medium)
    hawaii_high_cases = get_threshold_data(hdc_cases, 'NewCases_Rate', 'Hawaii', medium, high)
    hawaii_critical_cases = get_threshold_data(hdc_cases, 'NewCases_Rate', 'Hawaii', high, critical)

    Honolulu_medium_cases = get_threshold_data(hdc_cases, 'NewCases_Rate', 'Honolulu', low, medium)
    Honolulu_high_cases = get_threshold_data(hdc_cases, 'NewCases_Rate', 'Honolulu', medium, high)
    Honolulu_critical_cases = get_threshold_data(hdc_cases, 'NewCases_Rate', 'Honolulu', high, critical)

    Kauai_medium_cases = get_threshold_data(hdc_cases, 'NewCases_Rate', 'Kauai', low, medium)
    Kauai_high_cases = get_threshold_data(hdc_cases, 'NewCases_Rate', 'Kauai', medium, high)
    Kauai_critical_cases = get_threshold_data(hdc_cases, 'NewCases_Rate', 'Kauai', high, critical)

    Maui_medium_cases = get_threshold_data(hdc_cases, 'NewCases_Rate', 'Maui', low, medium)
    Maui_high_cases = get_threshold_data(hdc_cases, 'NewCases_Rate', 'Maui', medium, high)
    Maui_critical_cases = get_threshold_data(hdc_cases, 'NewCases_Rate', 'Maui', high, critical)

    all_medium_cases = get_threshold_data(hdc_cases, 'NewCases_Rate', 'All', low, medium)
    all_high_cases = get_threshold_data(hdc_cases, 'NewCases_Rate', 'All', medium, high)
    all_critical_cases = get_threshold_data(hdc_cases, 'NewCases_Rate', 'All', high, critical)

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=hdc_cases['Date'][hdc_cases['Region'] == 'Hawaii County'], y=hdc_cases['NewCases_Rate'][hdc_cases['Region'] == 'Hawaii County'], mode='lines', name='Hawaii County',
                             line=dict(color='white', width=4)))
    fig.add_trace(go.Scatter(x=hdc_cases['Date'][hdc_cases['Region'] == 'Hawaii County'], y=hdc_cases['NewCases_Rate'][hdc_cases['Region'] == 'Hawaii County'].where(hdc_cases['NewCases_Rate'][hdc_cases['Region'] == 'Hawaii County'] <= low), mode='lines', name='Hawaii County',
                             line=dict(color='lightgreen', width=4)))
    fig.add_trace(go.Scatter(x=hdc_cases['Date'][hdc_cases['Region'] == 'Hawaii County'], y=hawaii_medium_cases, mode='lines', name='Hawaii County',
                             line=dict(color='#fce38a', width=4)))
    fig.add_trace(go.Scatter(x=hdc_cases['Date'][hdc_cases['Region'] == 'Hawaii County'], y=hawaii_high_cases, mode='lines', name='Hawaii County',
                             line=dict(color='#f38181', width=4)))
    fig.add_trace(go.Scatter(x=hdc_cases['Date'][hdc_cases['Region'] == 'Hawaii County'], y=hawaii_critical_cases, mode='lines', name='Hawaii County',
                             line=dict(color='firebrick', width=4)))

    fig.add_trace(go.Scatter(x=hdc_cases['Date'][hdc_cases['Region'] == 'Honolulu County'], y=hdc_cases['NewCases_Rate'][hdc_cases['Region'] == 'Honolulu County'], mode='lines', name='Honolulu County',
                             line=dict(color='white', width=4)))
    fig.add_trace(go.Scatter(x=hdc_cases['Date'][hdc_cases['Region'] == 'Honolulu County'], y=hdc_cases['NewCases_Rate'][hdc_cases['Region'] == 'Honolulu County'].where(hdc_cases['NewCases_Rate'][hdc_cases['Region'] == 'Honolulu County'] <= low), mode='lines', name='Honolulu County',
                             line=dict(color='lightgreen', width=4)))
    fig.add_trace(go.Scatter(x=hdc_cases['Date'][hdc_cases['Region'] == 'Honolulu County'], y=Honolulu_medium_cases, mode='lines', name='Honolulu County',
                             line=dict(color='#fce38a', width=4)))
    fig.add_trace(go.Scatter(x=hdc_cases['Date'][hdc_cases['Region'] == 'Honolulu County'], y=Honolulu_high_cases, mode='lines', name='Honolulu County',
                             line=dict(color='#f38181', width=4)))
    fig.add_trace(go.Scatter(x=hdc_cases['Date'][hdc_cases['Region'] == 'Honolulu County'], y=Honolulu_critical_cases, mode='lines', name='Honolulu County',
                             line=dict(color='firebrick', width=4)))

    fig.add_trace(go.Scatter(x=hdc_cases['Date'][hdc_cases['Region'] == 'Kauai County'], y=hdc_cases['NewCases_Rate'][hdc_cases['Region'] == 'Kauai County'], mode='lines', name='Kauai County',
                             line=dict(color='white', width=4)))
    fig.add_trace(go.Scatter(x=hdc_cases['Date'][hdc_cases['Region'] == 'Kauai County'], y=hdc_cases['NewCases_Rate'][hdc_cases['Region'] == 'Kauai County'].where(hdc_cases['NewCases_Rate'][hdc_cases['Region'] == 'Kauai County'] <= low), mode='lines', name='Kauai County',
                             line=dict(color='lightgreen', width=4)))
    fig.add_trace(go.Scatter(x=hdc_cases['Date'][hdc_cases['Region'] == 'Kauai County'], y=Kauai_medium_cases, mode='lines', name='Kauai County',
                             line=dict(color='#fce38a', width=4)))
    fig.add_trace(go.Scatter(x=hdc_cases['Date'][hdc_cases['Region'] == 'Kauai County'], y=Kauai_high_cases, mode='lines', name='Kauai County',
                             line=dict(color='#f38181', width=4)))
    fig.add_trace(go.Scatter(x=hdc_cases['Date'][hdc_cases['Region'] == 'Kauai County'], y=Kauai_critical_cases, mode='lines', name='Kauai County',
                             line=dict(color='firebrick', width=4)))

    fig.add_trace(go.Scatter(x=hdc_cases['Date'][hdc_cases['Region'] == 'Maui County'], y=hdc_cases['NewCases_Rate'][hdc_cases['Region'] == 'Maui County'], mode='lines', name='Maui County',
                             line=dict(color='white', width=4)))
    fig.add_trace(go.Scatter(x=hdc_cases['Date'][hdc_cases['Region'] == 'Maui County'], y=hdc_cases['NewCases_Rate'][hdc_cases['Region'] == 'Maui County'].where(hdc_cases['NewCases_Rate'][hdc_cases['Region'] == 'Maui County'] <= low), mode='lines', name='Maui County',
                             line=dict(color='lightgreen', width=4)))
    fig.add_trace(go.Scatter(x=hdc_cases['Date'][hdc_cases['Region'] == 'Maui County'], y=Maui_medium_cases, mode='lines', name='Maui County',
                             line=dict(color='#fce38a', width=4)))
    fig.add_trace(go.Scatter(x=hdc_cases['Date'][hdc_cases['Region'] == 'Maui County'], y=Maui_high_cases, mode='lines', name='Maui County',
                             line=dict(color='#f38181', width=4)))
    fig.add_trace(go.Scatter(x=hdc_cases['Date'][hdc_cases['Region'] == 'Maui County'], y=Maui_critical_cases, mode='lines', name='Maui County',
                             line=dict(color='firebrick', width=4)))

    fig.add_trace(go.Scatter(x=hdc_cases['Date'][hdc_cases['Region'] == 'All County'], y=hdc_cases['NewCases_Rate'][hdc_cases['Region'] == 'All County'], mode='lines', name='All County',
                             line=dict(color='white', width=4)))
    fig.add_trace(go.Scatter(x=hdc_cases['Date'][hdc_cases['Region'] == 'All County'], y=hdc_cases['NewCases_Rate'][hdc_cases['Region'] == 'All County'].where(hdc_cases['NewCases_Rate'][hdc_cases['Region'] == 'All County'] <= low), mode='lines', name='All County',
                             line=dict(color='lightgreen', width=4)))
    fig.add_trace(go.Scatter(x=hdc_cases['Date'][hdc_cases['Region'] == 'All County'], y=all_medium_cases, mode='lines', name='All County',
                             line=dict(color='#fce38a', width=4)))
    fig.add_trace(go.Scatter(x=hdc_cases['Date'][hdc_cases['Region'] == 'All County'], y=all_high_cases, mode='lines', name='All County',
                             line=dict(color='#f38181', width=4)))
    fig.add_trace(go.Scatter(x=hdc_cases['Date'][hdc_cases['Region'] == 'All County'], y=all_critical_cases, mode='lines', name='All County',
                             line=dict(color='firebrick', width=4)))


    fig.update_xaxes(showline=True,
                     showgrid=False,
                     showticklabels=True,
                     linecolor='rgb(140, 140, 140)',
                     linewidth=2,
                     ticks='outside',
                     tickfont=dict(
                        family='Arial',
                        size=16,
                        color='rgb(140, 140, 140)',
                        ))
    fig.update_yaxes(range=[0, hdc_cases['NewCases_Rate'].max()+1],
                     showline=True,
                     showgrid=False,
                     showticklabels=True,
                     linecolor='rgb(140, 140, 140)',
                     linewidth=2,
                     ticks='outside',
                     tickfont=dict(
                        family='Arial',
                        size=16,
                        color='rgb(140, 140, 140)',
                        ))

    annotations = []
    annotations.append(dict(xref='paper', x=1.01, y=hdc_cases['NewCases_Rate'][hdc_cases['Region'] == 'Hawaii County'].iloc[-1],
                                  xanchor='left', yanchor='middle',
                                  text='Hawaii'.format(color='rgb(140, 140, 140)'),
                                  font=dict(family='Arial',
                                            size=16,
                                            color='rgb(140, 140, 140)'),
                                  showarrow=False))
    annotations.append(dict(xref='paper', x=1.01, y=hdc_cases['NewCases_Rate'][hdc_cases['Region'] == 'Honolulu County'].iloc[-1],
                                  xanchor='left', yanchor='middle',
                                  text='Honolulu'.format(color='rgb(140, 140, 140)'),
                                  font=dict(family='Arial',
                                            size=16,
                                            color='rgb(140, 140, 140)'),
                                  showarrow=False))
    annotations.append(dict(xref='paper', x=1.01, y=hdc_cases['NewCases_Rate'][hdc_cases['Region'] == 'Kauai County'].iloc[-1],
                                  xanchor='left', yanchor='middle',
                                  text='Kauai'.format(color='rgb(140, 140, 140)'),
                                  font=dict(family='Arial',
                                            size=16,
                                            color='rgb(140, 140, 140)'),
                                  showarrow=False))
    annotations.append(dict(xref='paper', x=1.01, y=hdc_cases['NewCases_Rate'][hdc_cases['Region'] == 'Maui County'].iloc[-1],
                                  xanchor='left', yanchor='middle',
                                  text='Maui'.format(color='rgb(140, 140, 140)'),
                                  font=dict(family='Arial',
                                            size=16,
                                            color='rgb(140, 140, 140)'),
                                  showarrow=False))
    annotations.append(dict(xref='paper', x=1.01, y=hdc_cases['NewCases_Rate'][hdc_cases['Region'] == 'All County'].iloc[-1],
                                  xanchor='left', yanchor='middle',
                                  text='All'.format(color='rgb(140, 140, 140)'),
                                  font=dict(family='Arial',
                                            size=16,
                                            color='rgb(140, 140, 140)'),
                                  showarrow=False))
    fig.update_layout(autosize=False,
                      width=1500,
                      height=1000,
                      showlegend=False,
                      plot_bgcolor='white',
                      margin=dict(
                        autoexpand=False,
                        l=80,
                        r=80,
                        t=50
                        ),
                      title={'y':1},
                      annotations=annotations,
                      shapes=[
                            {
                                'type': 'line',
                                'xref': 'paper',
                                'x0': 1,
                                'y0': 10, # use absolute value or variable here
                                'x1': 0,
                                'y1': 10, # ditto
                                'line': {
                                    'color': '#f38181',
                                    'width': 4,
                                    'dash': 'dash',
                                },
                            },
                            {
                                'type': 'line',
                                'xref': 'paper',
                                'x0': 1,
                                'y0': 1, # use absolute value or variable here
                                'x1': 0,
                                'y1': 1, # ditto
                                'line': {
                                    'color': '#fce38a',
                                    'width': 4,
                                    'dash': 'dash',
                                },
                            }
                        ]
                      )

    if push_to_site_current_situation == 'Y':
        username = '' # your username
        api_key = '' # your api key - go to profile > settings > regenerate key
        chart_studio.tools.set_credentials_file(username=username, api_key=api_key)
        py.plot(fig, filename = 'all_counties_cases', auto_open=True)

        fig.write_html("./chart_studio/file_all_counties_cases.html")
    else:
        fig.show()

############# ~ Positivity ~ ##############

def create_positivity_situation_graph(hdc_positivity, push_to_site_current_situation):
    hdc_positivity['% Pos'] = hdc_positivity['% Pos'].astype(float)
    hdc_positivity['Region'] = ['All County' if i == 'State' else i for i in hdc_positivity['Region']]

    low = 0.03
    medium = 0.1
    high = 0.2
    critical = 0.31

    def get_threshold_data(df, metric, region, threshold, ceiling):
        threshold_list = []
        for e, i in enumerate(df[f'{metric}'][df['Region'] == f'{region} County']):
            if (e > len((df[f'{metric}'][df['Region'] == f'{region} County']))-2) & (i >= threshold) & (i <= ceiling):
                threshold_list.append(i)
            if (e > len((df[f'{metric}'][df['Region'] == f'{region} County']))-2):
                threshold_list.append(np.nan)
                break
            if df[f'{metric}'][df['Region'] == f'{region} County'].iloc[e+1] > threshold:
                threshold_list.append(i)
                continue
            if df[f'{metric}'][df['Region'] == f'{region} County'].iloc[e] > threshold:
                threshold_list.append(i)
                continue
            if (e != 0) & (df[f'{metric}'][df['Region'] == f'{region} County'].iloc[e-1] > threshold) & (i <= threshold):
                threshold_list.append(i)
            else:
                threshold_list.append(np.nan)
        return threshold_list

    hawaii_medium_cases = get_threshold_data(hdc_positivity, '% Pos', 'Hawaii', low, medium)
    hawaii_high_cases = get_threshold_data(hdc_positivity, '% Pos', 'Hawaii', medium, high)
    hawaii_critical_cases = get_threshold_data(hdc_positivity, '% Pos', 'Hawaii', high, critical)

    Honolulu_medium_cases = get_threshold_data(hdc_positivity, '% Pos', 'Honolulu', low, medium)
    Honolulu_high_cases = get_threshold_data(hdc_positivity, '% Pos', 'Honolulu', medium, high)
    Honolulu_critical_cases = get_threshold_data(hdc_positivity, '% Pos', 'Honolulu', high, critical)

    Kauai_medium_cases = get_threshold_data(hdc_positivity, '% Pos', 'Kauai', low, medium)
    Kauai_high_cases = get_threshold_data(hdc_positivity, '% Pos', 'Kauai', medium, high)
    Kauai_critical_cases = get_threshold_data(hdc_positivity, '% Pos', 'Kauai', high, critical)

    Maui_medium_cases = get_threshold_data(hdc_positivity, '% Pos', 'Maui', low, medium)
    Maui_high_cases = get_threshold_data(hdc_positivity, '% Pos', 'Maui', medium, high)
    Maui_critical_cases = get_threshold_data(hdc_positivity, '% Pos', 'Maui', high, critical)

    all_medium_cases = get_threshold_data(hdc_positivity, '% Pos', 'All', low, medium)
    all_high_cases = get_threshold_data(hdc_positivity, '% Pos', 'All', medium, high)
    all_critical_cases = get_threshold_data(hdc_positivity, '% Pos', 'All', high, critical)


    hdc_positivity['% Pos'][hdc_positivity['Region'] == 'Maui County'].where((hdc_positivity['% Pos'][hdc_positivity['Region'] == 'Maui County'] > low) & (hdc_positivity['% Pos'][hdc_positivity['Region'] == 'Maui County'] <= medium))


    fig = go.Figure()

    fig.add_trace(go.Scatter(x=hdc_positivity['Date'][hdc_positivity['Region'] == 'Hawaii County'], y=hdc_positivity['% Pos'][hdc_positivity['Region'] == 'Hawaii County'], mode='lines', name='Hawaii County',
                             line=dict(color='white', width=4)))
    fig.add_trace(go.Scatter(x=hdc_positivity['Date'][hdc_positivity['Region'] == 'Hawaii County'], y=hdc_positivity['% Pos'][hdc_positivity['Region'] == 'Hawaii County'].where(hdc_positivity['% Pos'][hdc_positivity['Region'] == 'Hawaii County'] <= low), mode='lines', name='Hawaii County',
                             line=dict(color='lightgreen', width=4)))
    fig.add_trace(go.Scatter(x=hdc_positivity['Date'][hdc_positivity['Region'] == 'Hawaii County'], y=hawaii_medium_cases, mode='lines', name='Hawaii County',
                             line=dict(color='#fce38a', width=4)))
    fig.add_trace(go.Scatter(x=hdc_positivity['Date'][hdc_positivity['Region'] == 'Hawaii County'], y=hawaii_high_cases, mode='lines', name='Hawaii County',
                             line=dict(color='#f38181', width=4)))
    fig.add_trace(go.Scatter(x=hdc_positivity['Date'][hdc_positivity['Region'] == 'Hawaii County'], y=hawaii_critical_cases, mode='lines', name='Hawaii County',
                             line=dict(color='firebrick', width=4)))

    fig.add_trace(go.Scatter(x=hdc_positivity['Date'][hdc_positivity['Region'] == 'Honolulu County'], y=hdc_positivity['% Pos'][hdc_positivity['Region'] == 'Honolulu County'], mode='lines', name='Honolulu County',
                             line=dict(color='white', width=4)))
    fig.add_trace(go.Scatter(x=hdc_positivity['Date'][hdc_positivity['Region'] == 'Honolulu County'], y=hdc_positivity['% Pos'][hdc_positivity['Region'] == 'Honolulu County'].where(hdc_positivity['% Pos'][hdc_positivity['Region'] == 'Honolulu County'] <= low), mode='lines', name='Honolulu County',
                             line=dict(color='lightgreen', width=4)))
    fig.add_trace(go.Scatter(x=hdc_positivity['Date'][hdc_positivity['Region'] == 'Honolulu County'], y=Honolulu_medium_cases, mode='lines', name='Honolulu County',
                             line=dict(color='#fce38a', width=4)))
    fig.add_trace(go.Scatter(x=hdc_positivity['Date'][hdc_positivity['Region'] == 'Honolulu County'], y=Honolulu_high_cases, mode='lines', name='Honolulu County',
                             line=dict(color='#f38181', width=4)))
    fig.add_trace(go.Scatter(x=hdc_positivity['Date'][hdc_positivity['Region'] == 'Honolulu County'], y=Honolulu_critical_cases, mode='lines', name='Honolulu County',
                             line=dict(color='firebrick', width=4)))

    fig.add_trace(go.Scatter(x=hdc_positivity['Date'][hdc_positivity['Region'] == 'Kauai County'], y=hdc_positivity['% Pos'][hdc_positivity['Region'] == 'Kauai County'], mode='lines', name='Kauai County',
                             line=dict(color='white', width=4)))
    fig.add_trace(go.Scatter(x=hdc_positivity['Date'][hdc_positivity['Region'] == 'Kauai County'], y=hdc_positivity['% Pos'][hdc_positivity['Region'] == 'Kauai County'].where(hdc_positivity['% Pos'][hdc_positivity['Region'] == 'Kauai County'] <= low), mode='lines', name='Kauai County',
                             line=dict(color='lightgreen', width=4)))
    fig.add_trace(go.Scatter(x=hdc_positivity['Date'][hdc_positivity['Region'] == 'Kauai County'], y=Kauai_medium_cases, mode='lines', name='Kauai County',
                             line=dict(color='#fce38a', width=4)))
    fig.add_trace(go.Scatter(x=hdc_positivity['Date'][hdc_positivity['Region'] == 'Kauai County'], y=Kauai_high_cases, mode='lines', name='Kauai County',
                             line=dict(color='#f38181', width=4)))
    fig.add_trace(go.Scatter(x=hdc_positivity['Date'][hdc_positivity['Region'] == 'Kauai County'], y=Kauai_critical_cases, mode='lines', name='Kauai County',
                             line=dict(color='firebrick', width=4)))

    fig.add_trace(go.Scatter(x=hdc_positivity['Date'][hdc_positivity['Region'] == 'Maui County'], y=hdc_positivity['% Pos'][hdc_positivity['Region'] == 'Maui County'], mode='lines', name='Maui County',
                             line=dict(color='white', width=4)))
    fig.add_trace(go.Scatter(x=hdc_positivity['Date'][hdc_positivity['Region'] == 'Maui County'], y=hdc_positivity['% Pos'][hdc_positivity['Region'] == 'Maui County'].where(hdc_positivity['% Pos'][hdc_positivity['Region'] == 'Maui County'] <= low), mode='lines', name='Maui County',
                             line=dict(color='lightgreen', width=4)))
    fig.add_trace(go.Scatter(x=hdc_positivity['Date'][hdc_positivity['Region'] == 'Maui County'], y=Maui_medium_cases, mode='lines', name='Maui County',
                             line=dict(color='#fce38a', width=4)))
    fig.add_trace(go.Scatter(x=hdc_positivity['Date'][hdc_positivity['Region'] == 'Maui County'], y=Maui_high_cases, mode='lines', name='Maui County',
                             line=dict(color='#f38181', width=4)))
    fig.add_trace(go.Scatter(x=hdc_positivity['Date'][hdc_positivity['Region'] == 'Maui County'], y=Maui_critical_cases, mode='lines', name='Maui County',
                             line=dict(color='firebrick', width=4)))

    fig.add_trace(go.Scatter(x=hdc_positivity['Date'][hdc_positivity['Region'] == 'All County'], y=hdc_positivity['% Pos'][hdc_positivity['Region'] == 'All County'], mode='lines', name='All County',
                             line=dict(color='white', width=4)))
    fig.add_trace(go.Scatter(x=hdc_positivity['Date'][hdc_positivity['Region'] == 'All County'], y=hdc_positivity['% Pos'][hdc_positivity['Region'] == 'All County'].where(hdc_positivity['% Pos'][hdc_positivity['Region'] == 'All County'] <= low), mode='lines', name='All County',
                             line=dict(color='lightgreen', width=4)))
    fig.add_trace(go.Scatter(x=hdc_positivity['Date'][hdc_positivity['Region'] == 'All County'], y=all_medium_cases, mode='lines', name='All County',
                             line=dict(color='#fce38a', width=4)))
    fig.add_trace(go.Scatter(x=hdc_positivity['Date'][hdc_positivity['Region'] == 'All County'], y=all_high_cases, mode='lines', name='All County',
                             line=dict(color='#f38181', width=4)))
    fig.add_trace(go.Scatter(x=hdc_positivity['Date'][hdc_positivity['Region'] == 'All County'], y=all_critical_cases, mode='lines', name='All County',
                             line=dict(color='firebrick', width=4)))


    fig.update_xaxes(showline=True,
                     showgrid=False,
                     showticklabels=True,
                     linecolor='rgb(140, 140, 140)',
                     linewidth=2,
                     ticks='outside',
                     tickfont=dict(
                        family='Arial',
                        size=16,
                        color='rgb(140, 140, 140)',
                        ))
    fig.update_yaxes(range=[0, hdc_positivity['% Pos'].max()+.01],
                     showline=True,
                     showgrid=False,
                     showticklabels=True,
                     linecolor='rgb(140, 140, 140)',
                     linewidth=2,
                     tickformat='.1%',
                     ticks='outside',
                     tickfont=dict(
                        family='Arial',
                        size=16,
                        color='rgb(140, 140, 140)',
                        ))

    annotations = []
    annotations.append(dict(xref='paper', x=1.01, y=hdc_positivity['% Pos'][hdc_positivity['Region'] == 'Hawaii County'].iloc[-1],
                                  xanchor='left', yanchor='middle',
                                  text='Hawaii'.format(color='rgb(140, 140, 140)'),
                                  font=dict(family='Arial',
                                            size=16,
                                            color='rgb(140, 140, 140)'),
                                  showarrow=False))
    annotations.append(dict(xref='paper', x=1.01, y=hdc_positivity['% Pos'][hdc_positivity['Region'] == 'Honolulu County'].iloc[-1],
                                  xanchor='left', yanchor='middle',
                                  text='Honolulu'.format(color='rgb(140, 140, 140)'),
                                  font=dict(family='Arial',
                                            size=16,
                                            color='rgb(140, 140, 140)'),
                                  showarrow=False))
    annotations.append(dict(xref='paper', x=1.01, y=hdc_positivity['% Pos'][hdc_positivity['Region'] == 'Kauai County'].iloc[-1],
                                  xanchor='left', yanchor='middle',
                                  text='Kauai'.format(color='rgb(140, 140, 140)'),
                                  font=dict(family='Arial',
                                            size=16,
                                            color='rgb(140, 140, 140)'),
                                  showarrow=False))
    annotations.append(dict(xref='paper', x=1.01, y=hdc_positivity['% Pos'][hdc_positivity['Region'] == 'Maui County'].iloc[-2],
                                  xanchor='left', yanchor='middle',
                                  text='Maui'.format(color='rgb(140, 140, 140)'),
                                  font=dict(family='Arial',
                                            size=16,
                                            color='rgb(140, 140, 140)'),
                                  showarrow=False))
    annotations.append(dict(xref='paper', x=1.01, y=hdc_positivity['% Pos'][hdc_positivity['Region'] == 'All County'].iloc[-2],
                                  xanchor='left', yanchor='middle',
                                  text='All'.format(color='rgb(140, 140, 140)'),
                                  font=dict(family='Arial',
                                            size=16,
                                            color='rgb(140, 140, 140)'),
                                  showarrow=False))
    fig.update_layout(autosize=False,
                      width=1500,
                      height=1000,
                      showlegend=False,
                      plot_bgcolor='white',
                      margin=dict(
                        autoexpand=False,
                        l=80,
                        r=80,
                        t=50
                        ),
                      title={'y':1},
                      annotations=annotations,
                      shapes=[
                            {
                                'type': 'line',
                                'xref': 'paper',
                                'x0': 1,
                                'y0': .1, # use absolute value or variable here
                                'x1': 0,
                                'y1': .1, # ditto
                                'line': {
                                    'color': '#f38181',
                                    'width': 4,
                                    'dash': 'dash',
                                },
                            },
                            {
                                'type': 'line',
                                'xref': 'paper',
                                'x0': 1,
                                'y0': 0.03, # use absolute value or variable here
                                'x1': 0,
                                'y1': 0.03, # ditto
                                'line': {
                                    'color': '#fce38a',
                                    'width': 4,
                                    'dash': 'dash',
                                },
                            }
                        ]
                      )

    if push_to_site_current_situation == 'Y':
        username = '' # your username
        api_key = '' # your api key - go to profile > settings > regenerate key
        chart_studio.tools.set_credentials_file(username=username, api_key=api_key)
        py.plot(fig, filename = 'all_counties_positivity', auto_open=True)

        fig.write_html("./chart_studio/file_all_counties_positivity.html")
    else:
        fig.show()
