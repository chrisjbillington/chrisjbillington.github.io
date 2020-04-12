import sys
import os
from scipy.optimize import curve_fit
import numpy as np
import datetime
import matplotlib.units as munits
import matplotlib.dates as mdates
from pathlib import Path
import subprocess
import pandas as pd
from uncertainties import ufloat

NBSP = u"\u00A0"
converter = mdates.ConciseDateConverter()
locator = mdates.AutoDateLocator(minticks=15, maxticks=15)

munits.registry[np.datetime64] = converter
munits.registry[datetime.date] = converter
munits.registry[datetime.datetime] = converter

import matplotlib

matplotlib.rc('legend', fontsize=9, handlelength=2, labelspacing=0.25)
matplotlib.rc('xtick', labelsize=9) 
matplotlib.rc('ytick', labelsize=9)
matplotlib.rc('axes', labelsize=9) 

import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'

CRITICAL_CASES = 0.05

N_DAYS_PROJECTION = 20

DATA_SOURCE = 'ulklc'
# DATA_SOURCE = 'JH'

# Whether to plot US state data instead. In this case, we use US states instead of
# countries:
US_STATES = 'US' in sys.argv



def estimate_recoveries(cases, deaths):
    from scipy.signal import convolve

    living_cases = cases - deaths
    t = np.arange(30)

    SEVERE = 0.15
    MILD = 1 - SEVERE
    mu_mild = 17
    sigma_mild = 4
    mu_severe = 32
    sigma_severe = 11

    mild_recovery_curve = np.exp(-((t - mu_mild) ** 2) / (2 * sigma_mild ** 2))
    mild_recovery_curve /= mild_recovery_curve.sum()

    severe_recovery_curve = np.exp(-((t - mu_severe) ** 2) / (2 * sigma_severe ** 2))
    severe_recovery_curve /= severe_recovery_curve.sum()

    recovery_curve = MILD * mild_recovery_curve + SEVERE * severe_recovery_curve

    return convolve(living_cases, recovery_curve)[: len(cases)].astype(int)


if US_STATES:
    # Clone or pull NYT data
    if not os.path.exists('covid-19-data'):
        subprocess.check_call(
            ['git', 'clone', 'https://github.com/nytimes/covid-19-data/']
        )
    else:
        subprocess.check_call(['git', 'pull'], cwd='covid-19-data')

    df = pd.read_csv('covid-19-data/us-states.csv')

    datestrings = list(sorted(set(df['date'])))[1:]
    cases = {}
    deaths = {}
    recoveries = {}

    IGNORE_STATES = [
        'Northern Mariana Islands',
        'Virgin Islands',
        'Guam',
        'American Samoa',
    ]

    for state in set(df['state']):
        if state in IGNORE_STATES:
            continue
        cases[state] = []
        deaths[state] = []
        subdf = df[df['state'] == state]
        for date in datestrings:
            rows = subdf[subdf['date'] == date]
            if len(rows):
                assert len(rows) == 1
                cases[state].append(rows['cases'].array[0])
                deaths[state].append(rows['deaths'].array[0])
            else:
                cases[state].append(0)
                deaths[state].append(0)

        cases[state] = np.array(cases[state])
        deaths[state] = np.array(deaths[state])
        recoveries[state] = estimate_recoveries(cases[state], deaths[state])

    dates = np.array(
        [
            np.datetime64(datetime.datetime.strptime(date, "%Y-%m-%d"), 'h')
            for date in datestrings
        ]
    )


elif DATA_SOURCE == 'ulklc':
    # Clone or pull ulklc repo:
    if not os.path.exists('covid19-timeseries'):
        subprocess.check_call(
            ['git', 'clone', 'https://github.com/ulklc/covid19-timeseries']
        )
    else:
        subprocess.check_call(['git', 'pull'], cwd='covid19-timeseries')


    DATA_DIR = Path('covid19-timeseries/countryReport/country')

    cases = {}
    deaths = {}
    recoveries = {}

    dates = None
    for csv_file in os.listdir(DATA_DIR):
        if not csv_file.endswith('.csv'):
            continue
        df = pd.read_csv(DATA_DIR / csv_file)
        country = df['countryName'][0]
        cases[country] = np.array(df['confirmed'])
        recoveries[country] = np.array(df['recovered'])
        deaths[country] = np.array(df['death'])
        if dates is None:
            dates = np.array(
                [
                    np.datetime64(datetime.datetime.strptime(date, "%Y/%m/%d"), 'h')
                    for date in df['day']
                ]
            )


elif DATA_SOURCE == 'JH':
    # Clone or pull JH repo:
    if not os.path.exists('COVID-19'):
        subprocess.check_call(
            ['git', 'clone', 'https://github.com/CSSEGISandData/COVID-19/']
        )
    else:
        subprocess.check_call(['git', 'pull'], cwd='COVID-19')

    DATA_DIR = Path('COVID-19/csse_covid_19_data/csse_covid_19_time_series/')

    # Translate JH country names to what we call them:
    COUNTRY_NAMES = {
        'Taiwan*': 'Taiwan',
        'US': 'United States',
        'Korea, South': 'South Korea',
    }
    
    
    def process_file(csv_file):
        COLS_TO_DROP = ['Province/State', 'Country/Region', 'Lat', 'Long']
        df = pd.read_csv(DATA_DIR / csv_file)
        dates = None
        data = {}
        for country, subdf in df.groupby('Country/Region'):
            country = COUNTRY_NAMES.get(country, country)
            subdf = subdf.drop(columns=COLS_TO_DROP)
            if dates is None:
                dates = np.array(
                [
                    np.datetime64(datetime.datetime.strptime(date, "%m/%d/%y"), 'h')
                    for date in subdf.columns
                ]
            )
            data[country] = np.array(subdf.sum())
        return dates, data
    
    dates, cases = process_file('time_series_covid19_confirmed_global.csv')
    _, deaths = process_file('time_series_covid19_deaths_global.csv')
    _, recoveries = process_file('time_series_covid19_recovered_global.csv')


if not US_STATES:
    cases['World'] = sum(cases.values())
    deaths['World'] = sum(deaths.values())
    recoveries['World'] = sum(recoveries.values())


if not US_STATES:
    # Country names and populations in millions:
    populations = {
        'United States': 327.2,
        'Australia': 24.6,
        'United Kingdom': 66.4,
        'Germany': 82.8,
        'Switzerland': 8.5,
        'Canada': 37.6,
        'Italy': 60.5,
        'Netherlands': 17.2,
        'Japan': 126.8,
        'France': 67,
        'Iran': 81.2,
        'South Korea': 51.5,
        'Spain': 46.7,
        'China': 1386,
        'Brazil': 209.3,
        'Iceland': 0.364,
        'Mexico': 129.2,
        'Norway': 5.368,
        'India': 1339,
        'Russia': 144.5,
        'Singapore': 5.6,
        'Taiwan': 23.8,
        'Malaysia': 31.6,
        'South Africa': 56.7,
        'Indonesia': 264,
        'Belgium': 11.4,
        'Austria': 8.8,
        'New Zealand': 4.8,
        'Thailand': 69,
        'World': 7800,
        'Czechia': 10.65,
        'Chile': 18.1,
        'Turkey': 80.8,
        'Portugal': 10.3,
        'Israel': 8.7,
        'Sweden': 10.1,
        'Ireland': 4.8,
        'Denmark': 5.6,
        'Finland': 5.5,
        'Poland': 38
    }
else:
    df = pd.read_csv("nst-est2019-01.csv", header=3, skipfooter=5, engine='python')
    df = df.rename(columns={'Unnamed: 0': 'State'})
    populations = {}
    IGNORE_ROWS = ['United States', 'Northeast', 'Midwest', 'South', 'West']
    for i, row in df.iterrows():
        state = row['State']
        if not isinstance(state, str):
            continue
        if state in IGNORE_ROWS:
            continue
        state = state.replace('.', '')
        populations[state] = row['2019'] / 1e6
    for state in cases:
        if state not in populations:
            print("missing", state)
            assert False

countries = list(populations.keys())

# # Print html for per-country links when adding a new country:
# links = []
# for country in sorted(countries, key=lambda c: '' if c == 'World' else c):
#     links.append(
#         f'{NBSP*4}<a href="COVID/{country.replace(" ", "_")}.svg">•{country}</a>'
#     )

# TABLE_NCOLS = 3
# TABLE_NROWS = int(np.ceil(len(links) / TABLE_NCOLS))

# table_rows = [links[i::TABLE_NROWS] for i in range(TABLE_NROWS)]

# links_html_lines = ['<table>\n']
# for table_row in table_rows:
#     links_html_lines.append('<tr>')
#     links_html_lines.append(' '.join(f'<td>{item}</td>' for item in table_row))
#     links_html_lines.append('</tr>\n')
# links_html_lines.append('</table>')

# print(''.join(links_html_lines))
# assert False

# ICU beds per 100_000 inhabitants, from
# https://en.wikipedia.org/wiki/List_of_countries_by_hospital_beds
# And:
# https://www.ncbi.nlm.nih.gov/pubmed/31923030

icu_beds = {
    'United States': 34.7,
    'Australia': 8.9,  # Absent from wikipedia, googled instead
    'United Kingdom': 6.6,
    'Germany': 29.2,
    'Switzerland': 11.0,
    'Canada': 13.5,  # Also just googled it
    'Italy': 12.5,
    'Netherlands': 6.4,
    'Japan': 7.3,
    'France': 11.6,
    'Iran': 4.6,
    'South Korea': 10.6,
    'Spain': 9.7,
    'China': 3.6,
    'Brazil': 25,  # Google
    'Iceland': 9.1,
    'Mexico': 2.3,  # Google
    'Norway': 8,
    'India': 2.3,  # Google
    'Russia': 8.3,
    'Singapore': 11.4,
    'Taiwan': 29.8,  # Google
    'Malaysia': 3.3,  # Google
    'South Africa': 9,
    'Indonesia': 2.7,
    'Belgium': 15.9,
    'Austria': 21.8,
    'New Zealand': 4.7,
    'Thailand': 10.4,
    'Czechia': 11.6,
    'Chile': np.nan,
    'Turkey': 47.1,
    'Portugal': 4.2,
    'Israel': 59.7,
    'Sweden': 5.8,
    'Ireland': 6.5,
    'Denmark': 6.7,
    'Finland': 6.1,
    'Poland': 6.9
}


def partial_derivatives(function, x, params, u_params):
    model_at_center = function(x, *params)
    partial_derivatives = []
    for i, (param, u_param) in enumerate(zip(params, u_params)):
        d_param = u_param / 1e6
        params_with_partial_differential = np.zeros(len(params))
        params_with_partial_differential[:] = params[:]
        params_with_partial_differential[i] = param + d_param
        model_at_partial_differential = function(x, *params_with_partial_differential)
        partial_derivative = (model_at_partial_differential - model_at_center) / d_param
        partial_derivatives.append(partial_derivative)
    return partial_derivatives


def model_uncertainty(function, x, params, covariance):
    u_params = [np.sqrt(abs(covariance[i, i])) for i in range(len(params))]
    derivs = partial_derivatives(function, x, params, u_params)
    squared_model_uncertainty = sum(
        derivs[i] * derivs[j] * covariance[i, j]
        for i in range(len(params))
        for j in range(len(params))
    )
    return np.sqrt(squared_model_uncertainty)


def logistic(t, L, tau, t0):
    exponent = -(t - t0) / tau
    exponent = exponent.clip(-100, 100)
    return L / (1 + np.exp(exponent))


def exponential(t, k, t0):
    exponent = k * (t - t0)
    exponent = exponent.clip(-100, 100)
    return np.exp(exponent)


COLS = 6
ROWS = int(np.ceil(len(countries) / 6))

model = exponential

FIT_PTS = 5

DATES_START_INDEX = 2

SUBPLOT_HEIGHT = 10.8 / 3 * 1.5
TOTAL_WIDTH = 18.5

import matplotlib.gridspec as gridspec

for SINGLE in [False, True]:
    if SINGLE:
        fig = plt.figure(figsize=(10, 10))
        gs = gridspec.GridSpec(ncols=1, nrows=20, figure=fig)
    else:
        fig = plt.figure(figsize=(TOTAL_WIDTH, ROWS * SUBPLOT_HEIGHT))
        gs = gridspec.GridSpec(ncols=COLS, nrows=20 * ROWS, figure=fig)

    for i, country in enumerate(
        sorted(countries, key=lambda c: -np.nanmax(cases[c] / populations[c]))
    ):
        if SINGLE:
            row = col = 0
            plt.clf()
        else:
            row, col = divmod(i, COLS)

        ax1 = fig.add_subplot(gs[20 * row : 20 * row + 12, col])
        ax2 = fig.add_subplot(gs[20 * row + 12 : 20 * row + 18, col])
        ax3 = ax2.twinx() # Solely to add an extra scale to ax2

        print(country)

        # recovered = recoveries[country]
        recovered = estimate_recoveries(cases[country], deaths[country])
        active = cases[country] - deaths[country] - recovered
        

        x_fit = dates.astype(float)


        k_arr = []
        u_k_arr = []

        for j in range(FIT_PTS, len(active)):

            t2 = x_fit[j]
            t1 = x_fit[j - FIT_PTS + 1]
            y2 = active[j]
            y1 = active[j - FIT_PTS + 1]
            if 0 in [y2, y1] or y1 == y2:
                k_arr.append(0)
                u_k_arr.append(0)
                params = None
            else:
                k_guess = np.log(y2 / y1) / (t2 - t1)
                t0_guess = t2 - np.log(y2) / k_guess

                params, covariance = curve_fit(
                    model,
                    x_fit[j - FIT_PTS + 1 : j + 1],
                    active[j - FIT_PTS + 1 : j + 1],
                    [k_guess, t0_guess],
                    maxfev=100000,
                )

                k_arr.append(24 * params[0])
                u_k_arr.append(24 * np.sqrt(covariance[0, 0]))

        k_arr = np.array(k_arr)
        u_k_arr = np.array(u_k_arr)

        r_arr = np.exp(k_arr) - 1
        u_r_arr = u_k_arr * np.exp(k_arr)

        tau_2 = np.log(2) * ufloat(1 / k_arr[-1], u_k_arr[-1] / k_arr[-1] ** 2)

        tau_2_deaths_arr = []
        u_tau_2_deaths_arr = []

        k_deaths_arr = []
        u_k_deaths_arr = []

        for j in range(2 * FIT_PTS, len(active)):
            recent_deaths = np.diff(deaths[country])[j - FIT_PTS + 1 : j + 1].sum()
            prev_deaths = np.diff(deaths[country])[
                j - 2 * FIT_PTS + 1 : j - FIT_PTS + 1
            ].sum()
            if 0 in [recent_deaths, prev_deaths] or recent_deaths == prev_deaths:
                tau_2_deaths_arr.append(np.inf)
                u_tau_2_deaths_arr.append(np.inf)
                k_deaths_arr.append(0)
                u_k_deaths_arr.append(0)
            else:
                k_deaths_arr.append(np.log(recent_deaths / prev_deaths) / FIT_PTS)
                u_k_deaths_arr.append(
                    np.sqrt(1 / recent_deaths + 1 / prev_deaths) / FIT_PTS
                )
                # tau_2_deaths_arr.append(
                #     (np.log(2) * FIT_PTS * 1 / np.log(recent_deaths / prev_deaths))
                # )
                # u_tau_2_deaths_arr.append(
                #     np.log(2)
                #     * FIT_PTS
                #     * np.sqrt(1 / prev_deaths + 1 / recent_deaths)
                #     / np.log(recent_deaths / prev_deaths) ** 2
                # )
                # r_deaths_arr.append((recent_deaths / prev_deaths) ** (1 / FIT_PTS) - 1)
                # u_r_deaths_arr.append(
                #     (recent_deaths / prev_deaths) ** (1 / FIT_PTS)
                #     * np.sqrt(1 / recent_deaths + 1 / prev_deaths)
                # )

        r_deaths_arr = np.exp(k_deaths_arr) - 1
        u_r_deaths_arr = u_k_deaths_arr * np.exp(k_deaths_arr)

        r_deaths_arr = np.array(r_deaths_arr)
        u_r_deaths_arr = np.array(u_r_deaths_arr)

        # tau_2_deaths_arr = np.array(tau_2_deaths_arr)
        # u_tau_2_deaths_arr = np.array(u_tau_2_deaths_arr)
        # tau_2_deaths = ufloat(tau_2_deaths_arr[-1], u_tau_2_deaths_arr[-1])
        if k_deaths_arr[-1] == 0:
            tau_2_deaths = ufloat(np.inf, np.inf)
        else:
            tau_2_deaths = np.log(2) * ufloat(
                1 / k_deaths_arr[-1], u_k_deaths_arr[-1] / k_deaths_arr[-1] ** 2
            )

        x_model = np.arange(
            dates[-FIT_PTS] - np.timedelta64(24, 'h'),
            dates[-1] + np.timedelta64(24 * N_DAYS_PROJECTION, 'h'),
        )
        x_model_float = x_model.astype(float)

        # ax1.yaxis.tick_left()
        # ax1.yaxis.set_label_position("left")
        # ax2.yaxis.tick_right()
        # ax2.yaxis.set_label_position("right")

        if not US_STATES:
            ax1.axhline(
                icu_beds.get(country, np.nan) * 10 / CRITICAL_CASES,  # ×10 is conversion to per million
                linestyle=':',
                color='r',
                label='Critical cases ≈ ICU beds',
            )

        # Plot a bunch of random projectioins by drawing from Gaussian with the parameter
        # covariance:
        NUM_SIMS = 50
        if params is not None:
            for _ in range(NUM_SIMS):
                scenario_params = np.random.multivariate_normal(params, covariance)
                ax1.plot(
                    x_model,
                    model(x_model_float, *scenario_params) / populations[country],
                    '-',
                    color='orange',
                    alpha=0.01,
                    linewidth=4,
                )

        # A dummy item to create the legend for the projection
        ax1.fill_between(
            [dates[0], dates[1]],
            1e-6,
            2e-6,
            facecolor='orange',
            alpha=0.5,
            label='Active (projected)',
        )

        deaths_percent = deaths[country][-1] / cases[country][-1] * 100
        recovered_percent = recovered[-1] / cases[country][-1] * 100

        ax1.semilogy(
            dates,
            cases[country] / populations[country],
            'D',
            markerfacecolor='deepskyblue',
            markeredgewidth=0.5,
            markeredgecolor='k',
            markersize=4,
            label=f'Total',
        )

        ax1.semilogy(
            dates,
            recovered / populations[country],
            's',
            markerfacecolor='mediumseagreen',
            markeredgewidth=0.5,
            markeredgecolor='k',
            markersize=5,
            label=f'Recovered',
        )

        ax1.semilogy(
            dates,
            active / populations[country],
            'o',
            markerfacecolor='orange',
            markeredgewidth=0.5,
            markeredgecolor='k',
            markersize=5,
            label=f'Active',
        )

        ax1.semilogy(
            dates,
            deaths[country] / populations[country],
            '^',
            markerfacecolor='orangered',
            markeredgewidth=0.5,
            markeredgecolor='k',
            markersize=5,
            label=f'Deaths',
        )

        ax1.grid(True, linestyle=':')
        ax2.grid(True, linestyle=':')
        if not SINGLE and i == 0:
            if US_STATES:
                plt.suptitle('US per-capita COVID-19 cases and exponential projections by state')
            else:
                plt.suptitle('Per-capita COVID-19 cases and exponential projections by country')
        elif SINGLE:
            plt.suptitle(f'{country} per-capita COVID-19 cases and exponential projection')
        if SINGLE or i % COLS == 0:
            ax1.set_ylabel('Cases per million inhabitants')
        for ax in [ax1, ax2]:
            ax.axis(
                xmin=dates[DATES_START_INDEX] - np.timedelta64(24, 'h'), xmax=x_model[-1]
            )
        ax1.axis(ymin=2e-2, ymax=1e6)

        if not SINGLE and i % COLS != 0:
            ax1.set_yticklabels([])


        valid = deaths[country][2 * FIT_PTS:] > 100

        # ax2.errorbar(
        #     dates[2 * FIT_PTS:][valid],
        #     100 * r_deaths_arr[valid],
        #     100 * u_r_deaths_arr[valid],
        #     fmt='o',
        #     markerfacecolor='magenta',
        #     markeredgecolor='k',
        #     capsize=2,
        #     markersize=3,
        #     elinewidth=0.5,
        #     ecolor='k',
        #     markeredgewidth=0.5,
        #     label='New deaths growth rate'
        # )

        # ax2.plot(
        #     dates[2 * FIT_PTS :][valid],
        #     100 * r_deaths_arr[valid],
        #     'o',
        #     markerfacecolor='magenta',
        #     markeredgecolor='k',
        #     markersize=5,
        #     markeredgewidth=0.5,
        #     label='New deaths growth rate',
        # )

        # ax2.fill_between(
        #     dates[2 * FIT_PTS:][valid],
        #     100 * (r_deaths_arr + u_r_deaths_arr)[valid],
        #     100 * (r_deaths_arr - u_r_deaths_arr)[valid],
        #     color='magenta',
        #     alpha=0.5,
        #     label='New deaths growth rate',
        # )


        valid = active[FIT_PTS:] > 2

        # k = [0]
        # var_k = [0]
        # alpha = 1 / FIT_PTS
        # for active_new, active_old in zip(active[1:], active[:-1]):
        #     if 0 in [active_old, active_new]:
        #         k_new = 0
        #         var_k_new = 0
        #     else:
        #         k_new = np.log(active_new / active_old)
        #         # var_k_new = 1 / active_new + 1 / active_old
        #         var_k_new = (k_new - k[-1]) ** 2 / FIT_PTS

        #     # var_k.append((1 - alpha) * (var_k[-1] + alpha * (k_new - k[-1]) ** 2))
        #     var_k.append(alpha * var_k_new + (1 - alpha) * var_k[-1])
        #     k.append(alpha * k_new + (1 - alpha) * k[-1])

        # k = np.array(k)
        # var_k = np.array(var_k)

        # r_arr = np.exp(k) - 1
        # u_r_arr = np.sqrt(var_k) * np.exp(k)

        # k_arr = (active[1:] / active[:-1] - 1)[FIT_PTS - 1 :]
        # u_k_arr = (active[1:] / active[:-1] * np.sqrt(1 / active[1:] + 1 / active[:-1]))[FIT_PTS - 1 :]

        # ax2.errorbar(
        #     dates[FIT_PTS:][valid],
        #     100 * r_arr[valid],
        #     100 * u_r_arr[valid],
        #     fmt='o',
        #     markerfacecolor='gray',
        #     markeredgecolor='k',
        #     capsize=2,
        #     markersize=3,
        #     elinewidth=0.5,
        #     ecolor='k',
        #     markeredgewidth=0.5,
        #     label='Active growth rate'
        # )

        # ax2.plot(
        #     dates[1:],
        #     100 * (active[1:] / active[:-1] - 1),
        #     'o',
        #     markerfacecolor='gray',
        #     markeredgecolor='k',
        #     markersize=5,
        #     markeredgewidth=0.5,
        #     label='Active growth rate',
        # )


        ax2.fill_between(
            dates[FIT_PTS:][valid],
            100 * (r_arr + u_r_arr)[valid],
            100 * (r_arr - u_r_arr)[valid],
            color='k',
            alpha=0.5,
            label='Active growth rate',
        )

        # ax2.fill_between(
        #     dates,
        #     100 * (r_arr + u_r_arr),
        #     100 * (r_arr - u_r_arr),
        #     color='k',
        #     alpha=0.5,
        #     label='Active growth rate',
        # )


        # ax2.fill_between(
        #     dates[FIT_PTS:][valid_doubling],
        #     100 * (tau_2_arr + u_tau_2_arr)[valid_doubling],
        #     100 * (tau_2_arr - u_tau_2_arr)[valid_doubling],
        #     color='k',
        #     alpha=0.5,
        #     label='Active growth rate',
        # )

        # with np.errstate(invalid='ignore'):
        #     valid_doubling = (
        #         (active[2 * FIT_PTS :] > 100)
        #         & (tau_2_deaths_arr > 0)
        #         & (tau_2_deaths_arr < 50)
        #     )

        # ax2.fill_between(
        #     dates[2 * FIT_PTS :][valid_doubling],
        #     (tau_2_deaths_arr + u_tau_2_deaths_arr)[valid_doubling],
        #     (tau_2_deaths_arr - u_tau_2_deaths_arr)[valid_doubling],
        #     color='purple',
        #     alpha=0.3,
        #     label='Δ deaths doubling time',
        # )

        ax2.axis(ymin=-30, ymax=50)
        ax3.axis(ymin=-30, ymax=50)

        growth_rate_labels = [-20, -10, 0, 10, 20, 30, 40]

        doubling_time_labels = [
            f'{np.log(2) / np.log(r / 100 + 1):.1f}' if r else '∞' for r in growth_rate_labels
        ]

        ax2.set_yticks(growth_rate_labels)
        ax3.set_yticks(growth_rate_labels)

        ax3.set_yticklabels(doubling_time_labels)
        ax2.axhline(0, color='k', linestyle='-')

        if SINGLE or (i % COLS == 0):
            ax2.set_ylabel('Growth rate (%/day)')
        else:
            ax2.set_yticklabels([])

        if SINGLE or (i % COLS == COLS - 1) or (i == len(countries) - 1):
            ax3.set_ylabel('Doubling time (days)')

        for ax in [ax1, ax2]:
            ax.xaxis.set_major_locator(locator)
            ax.get_xaxis().get_major_formatter().show_offset = False

        ax1.set_xticklabels([])


        num_digits_tau2_uncertainty = max(len(str(np.ceil(abs(tau_2.s))).split('.')[0]), 1)
        tau2_format_specifier = f":.{num_digits_tau2_uncertainty}uP"

        num_digits_tau2_deaths_uncertainty = max(
            len(str(np.ceil(abs(tau_2_deaths.s))).split('.')[0]), 1
        )
        tau2_deaths_format_specifier = f":.{num_digits_tau2_deaths_uncertainty}uP"

        tau_2_deaths_formatted = (
            abs(tau_2_deaths).format(tau2_deaths_format_specifier).replace('inf', '∞')
        )

        tau_2_formatted = abs(tau_2).format(tau2_format_specifier).replace('inf', '∞')


        # Excape spaces in country names for latex
        display_name = country.replace(" ", NBSP)


        ax1.text(
            0.02,
            0.98,
            '\n'.join(
                [
                    f'$\\bf {display_name} $',
                    f'Total: {cases[country][-1]}',
                    f'Active: {active[-1]} ({int(round(100 * r_arr[-1])):+.0f}%/day)',
                    (
                        f'{NBSP * 2} → {"doubling" if tau_2 > 0 else "halving"} in {tau_2_formatted} days'
                        if abs(tau_2) < 50
                        else f'{NBSP * 2} → unchanging'
                    ),
                    f'Deaths: {deaths[country][-1]} ({deaths_percent:.1f}%) (Δ:{int(round(100 * r_deaths_arr[-1])):+.0f}%/day)',
                    (
                        f'{NBSP * 2} → Δ: {"doubling" if tau_2_deaths > 0 else "halving"} in {tau_2_deaths_formatted} days'
                        if abs(tau_2_deaths) < 50
                        else f'{NBSP * 2} → Δ unchanging'
                    ),
                    f'Recovered: {recovered[-1]} ({recovered_percent:.1f}%)',
                ]
            ),
            transform=ax1.transAxes,
            fontsize=8,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='w', pad=0),
            va='top',
            # fontdict=dict(family='Ubuntu mono'),
        )

        if SINGLE:
            plt.subplots_adjust(left=0.08, bottom=0.01, right=0.93, top=0.95, wspace=0, hspace=0.0)

            handles1, labels1 = ax1.get_legend_handles_labels()
            handles2, labels2 = ax2.get_legend_handles_labels()

            ax1.legend(handles1 + handles2, labels1 + labels2, loc='upper right', ncol=3)

            if not os.path.exists('COVID'):
                os.mkdir('COVID')
            plt.savefig(f'COVID/{country.replace(" ", "_")}.svg')


    if not SINGLE:
        plt.subplots_adjust(left=0.04, bottom=0.05, right=0.96, top=0.95, wspace=0, hspace=0.0)

        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()

        plt.gcf().legend(handles1 + handles2, labels1 + labels2, loc='upper right', ncol=3)

        plt.savefig('COVID_US.svg' if US_STATES else 'COVID.svg')