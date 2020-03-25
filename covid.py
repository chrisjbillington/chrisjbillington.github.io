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

converter = mdates.ConciseDateConverter()
locator = mdates.AutoDateLocator(minticks=15, maxticks=15)

munits.registry[np.datetime64] = converter
munits.registry[datetime.date] = converter
munits.registry[datetime.datetime] = converter

import matplotlib

matplotlib.rc('legend', fontsize=9, handlelength=2, labelspacing=0.25)

import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'

N_DAYS_PROJECTION = 20

# Country names and populations in millions:
populations = {
    'US': 327.2,
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
    'Korea, South': 51.5,
    'Spain': 46.7,
    'China': 1386,
    'Brazil': 209.3,
    'Iceland': 0.364,
    'Mexico': 129.2,
    'Norway': 5.368,
    'India': 1339,
    'Russia': 144.5,
    'Singapore': 5.6,
    'Taiwan*': 23.8,
    'Malaysia': 31.6,
    'South Africa': 56.7,
    'Indonesia': 264,
    'Belgium': 11.4,
    'Austria': 8.8,
}

countries = list(populations.keys())

# ICU beds per 100_000 inhabitants, from
# https://en.wikipedia.org/wiki/List_of_countries_by_hospital_beds
icu_beds = {
    'US': 34.7,
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
    'Korea, South': 10.6,
    'Spain': 9.7,
    'China': 3.6,
    'Brazil': 25,  # Google
    'Iceland': 9.1,
    'Mexico': 2.3,  # Google
    'Norway': 8,
    'India': 2.3,  # Google
    'Russia': 8.3,
    'Singapore': 11.4,
    'Taiwan*': 29.8,  # Google
    'Malaysia': 3.3,  # Google
    'South Africa': 9,
    'Indonesia': 2.7,
    'Belgium': 15.9,
    'Austria': 21.8,
}

# Names we will use instead of those in the Johns Hopkins data:
display_names = {
    "United Kingdom": "UK",
    "Taiwan*": "Taiwan",
    "Korea, South": "South Korea",
}


# DATA_SOURCE = 'timeseries' # Johns Hopkins timeseries
# DATA_SOURCE = 'daily reports' # Johns Hopkins daily reports
DATA_SOURCE = 'ulklc' # github

if DATA_SOURCE in ['timeseries', 'daily reports']:
    # Clone or pull Johns HOpkins repo:
    if not os.path.exists('COVID-19'):
        subprocess.check_call(
            ['git', 'clone', 'https://github.com/CSSEGISandData/COVID-19']
        )
    else:
        subprocess.check_call(['git', 'pull'], cwd='COVID-19')
else:
    # Clone or pull ulklc repo:
    if not os.path.exists('covid19-timeseries'):
        subprocess.check_call(
            ['git', 'clone', 'https://github.com/ulklc/covid19-timeseries']
        )
    else:
        subprocess.check_call(['git', 'pull'], cwd='covid19-timeseries')

if DATA_SOURCE == 'timeseries':
    DATA_DIR = Path('COVID-19/csse_covid_19_data/csse_covid_19_time_series')
    NON_DATE_COLS = ['Province/State', 'Country/Region', 'Lat', 'Long']

    case_data = pd.read_csv(DATA_DIR / 'time_series_19-covid-Confirmed.csv')
    cases = {
        name: np.array(rows.drop(columns=NON_DATE_COLS).sum())
        for name, rows in case_data.groupby('Country/Region')
        if name in populations
    }

    dates = [
        np.datetime64(datetime.datetime.strptime(date, "%m/%d/%y"), 'h')
        for date in case_data.drop(columns=NON_DATE_COLS).columns
    ]

    deaths_data = pd.read_csv(DATA_DIR / 'time_series_19-covid-Deaths.csv')

    deaths = {
        name: np.array(rows.drop(columns=NON_DATE_COLS).sum())
        for name, rows in deaths_data.groupby('Country/Region')
        if name in populations
    }

    recoveries_data = pd.read_csv(DATA_DIR / 'time_series_19-covid-Recovered.csv')

    recoveries = {
        name: np.array(rows.drop(columns=NON_DATE_COLS).sum())
        for name, rows in recoveries_data.groupby('Country/Region')
        if name in populations
    }

    dates = np.array(dates)


elif DATA_SOURCE == 'daily reports':
    dates = []
    cases = {name: [] for name in countries}
    deaths = {name: [] for name in countries}
    recoveries = {name: [] for name in countries}

    DATA_DIR = Path('COVID-19/csse_covid_19_data/csse_covid_19_daily_reports')

    # Data files changed the name of the countries a few times:
    alternate_names = {
        'Korea, South': ['South Korea', 'Republic of Korea'],
        'United Kingdom': ['UK'],
        'China': ['Mainland China'],
        'Taiwan*': ['Taiwan', 'Taipei and environs'],
        'Russia': ['Russian Federation'],
        'Iran': ['Iran (Islamic Republic of)']
    }

    # The dates before which there was no data for certain countries:
    no_data_before = {
        'South Africa': np.datetime64('2020-03-05', 'h'),
        'Indonesia': np.datetime64('2020-03-02', 'h'),
        'Mexico': np.datetime64('2020-02-28', 'h'),
        'Iceland': np.datetime64('2020-02-28', 'h'),
        'Netherlands': np.datetime64('2020-02-27', 'h'),
        'Norway': np.datetime64('2020-02-26', 'h'),
        'Brazil': np.datetime64('2020-02-26', 'h'),
        'Austria': np.datetime64('2020-02-25', 'h'),
        'Switzerland': np.datetime64('2020-02-25', 'h'),
        'Iran': np.datetime64('2020-02-19', 'h'),
        'Belgium': np.datetime64('2020-02-04', 'h'),
        'Spain': np.datetime64('2020-02-01', 'h'),
        'Russia': np.datetime64('2020-01-31', 'h'),
        'Italy': np.datetime64('2020-01-31', 'h'),
        'United Kingdom': np.datetime64('2020-01-31', 'h'),
        'India': np.datetime64('2020-01-30', 'h'),
        'Germany': np.datetime64('2020-01-28', 'h'),
        'Malaysia': np.datetime64('2020-01-25', 'h'),
        'Australia': np.datetime64('2020-01-25', 'h'),
        'France': np.datetime64('2020-01-24', 'h'),
        'Canada': np.datetime64('2020-01-26', 'h'),
        'Singapore': np.datetime64('2020-01-23', 'h'),
    }

    files = [s for s in os.listdir(DATA_DIR) if s.endswith('.csv')]

    for file in sorted(files, key=lambda s: datetime.datetime.strptime(s, "%m-%d-%Y.csv")):
        print(file)
        date = np.datetime64(datetime.datetime.strptime(file, "%m-%d-%Y.csv"), 'h')
        dates.append(date)
        df = pd.read_csv(DATA_DIR / file)
        if 'Country_Region' in df.columns:
            country_col = 'Country_Region'
        else:
            country_col = 'Country/Region'
        subdfs = {name: rows for name, rows in df.groupby(country_col)}
        for name in countries:
            rows = None
            if name in subdfs:
                rows = subdfs[name]
            else:
                for alternate_name in alternate_names.get(name, []):
                    if alternate_name in subdfs:
                        rows = subdfs[alternate_name]
                        break
            if rows is None:
                if name not in no_data_before or date >= no_data_before[name]:
                    print(f"No data for {name} in {file}")
                cases[name].append(0)
                deaths[name].append(0)
                recoveries[name].append(0)
                
            else:
                country_cases = rows['Confirmed'].sum()
                country_deaths = rows['Deaths'].sum()
                country_recovered = rows['Recovered'].sum()
                cases[name].append(country_cases)
                deaths[name].append(country_deaths)
                recoveries[name].append(country_recovered)

    # Convert to arrays and sort by date:
    order = np.argsort(dates)
    dates = np.array(dates)[order]
    cases = {name: np.array(country_cases)[order] for name, country_cases in cases.items()}
    deaths = {name: np.array(country_deaths)[order] for name, country_deaths in deaths.items()}
    recoveries = {
        name: np.array(country_recovered)[order] for name, country_recovered in recoveries.items()
    }

elif DATA_SOURCE == 'ulklc':

    country_codes = {
        'US': 'US',
        'Australia': 'AU',
        'United Kingdom': 'GB',
        'Germany': 'DE',
        'Switzerland': 'CH',
        'Canada': 'CA',
        'Italy': 'IT',
        'Netherlands': 'NL',
        'Japan': 'JP',
        'France': 'FR',
        'Iran': 'IR',
        'Korea, South': 'KR',
        'Spain': 'ES',
        'China': 'CN',
        'Brazil': 'BR',
        'Iceland': 'IS',
        'Mexico': 'MX',
        'Norway': 'NO',
        'India': 'IN',
        'Russia': 'RU',
        'Singapore': 'SG',
        'Taiwan*': 'TW',
        'Malaysia': 'MY',
        'South Africa': 'SA',
        'Indonesia': 'ID',
        'Belgium': 'BE',
        'Austria': 'AT',
    }

    DATA_DIR = Path('covid19-timeseries/countryReport/country')

    cases = {}
    deaths = {}
    recoveries = {}

    dates = None
    for country, code in country_codes.items():
        df = pd.read_csv(DATA_DIR / f'{code}.csv')
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
        else:
            assert len(dates) == len(df['day'])
        
else:
    assert False


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


def exponential(t, tau, t0):
    exponent = (t - t0) / tau
    exponent = exponent.clip(-100, 100)
    return np.exp(exponent)


colours = plt.get_cmap("Dark2")
N_COLOURS = 8


COLS = 6
ROWS = np.ceil(len(countries) / 6)

model = exponential

FIT_PTS = 5

# DATES_START_INDEX = 24  # Feb 15th
DATES_START_INDEX = 2
ITALY_LOCKDOWN_DATE = 47  # March 9th

SUBPLOT_HEIGHT = 10.8 / 3
TOTAL_WIDTH = 18.5

plt.figure(figsize=(TOTAL_WIDTH, ROWS * SUBPLOT_HEIGHT))
for i, country in enumerate(
    sorted(cases, key=lambda c: -np.nanmax(cases[c] / populations[c]))
):
    plt.subplot(ROWS, COLS, i + 1)
    print(country)

    active = cases[country] - deaths[country] - recoveries[country]
    recovered = recoveries[country]


    # active = [cases[country][0] - deaths[country][0]]
    # tau_recovery = 60
    # for j, date in enumerate(dates[1:], start=1):
    #     new_cases = cases[country][j] - cases[country][j - 1]
    #     new_deaths = deaths[country][j] - deaths[country][j - 1]
    #     active.append((active[j-1] - new_deaths) * (1 - 1 / tau_recovery) + new_cases)

    # recovered = cases[country] - active

    # tau_recovery = 17
    # recovered = np.zeros(len(dates))
    # recovered[tau_recovery:] = (cases[country] - deaths[country])[:-tau_recovery]




    # def gaussian_blur(arr, pts):
    #     """gaussian blur an array by given number of points"""
    #     from scipy.signal import convolve
    #     x = np.arange(-4 * pts, 4 * pts + 1, 1)
    #     kernel = np.exp(-((x - tau_recovery) ** 2) / (2 * pts ** 2))
    #     kernel /= kernel.sum()
    #     return convolve(arr, kernel, mode='same')

    # from scipy.ndimage import gaussian_filter
    
    # recovered = gaussian_blur(cases[country], 8) - deaths[country]

    # active = cases[country] - deaths[country] - recovered



    # new = np.concatenate([[cases[country][0]], np.diff(cases[country])])

    # kernel = np.zeros(len(dates))
    # kernel[:17] += .95
    # kernel[:50] += 0.05

    # from scipy.signal import convolve
    # active = convolve(new, kernel, mode='full', method='direct')[:len(dates)]
    # recovered = cases[country] - active - deaths[country]
    # recovered[recovered < 0] = 0


    # new_cases = np.diff(cases[country])
    # new_deaths = np.diff(deaths[country])


    colour = colours(i % N_COLOURS)
    x_fit = dates.astype(float)

    t2 = x_fit[-1]
    t1 = x_fit[-FIT_PTS]
    y2 = active[-1]
    y1 = active[-FIT_PTS]
    tau_guess = (t2 - t1) / np.log(y2 / y1)
    t0_guess = t2 - tau_guess * np.log(y2)

    params, covariance = curve_fit(
        model,
        x_fit[-FIT_PTS:],
        active[-FIT_PTS:],
        [tau_guess, t0_guess],
        maxfev=10000,
    )

    tau_2 = np.log(2) * ufloat(params[0], np.sqrt(covariance[0, 0])) / 24

    # Do another fit for the preceding FIT_PTS to measure the change in growth rate over
    # that time:
    params_prev, _ = curve_fit(
        model,
        x_fit[-2 * FIT_PTS : -FIT_PTS],
        active[-2 * FIT_PTS : -FIT_PTS],
        [tau_guess, t0_guess],
        maxfev=10000,
    )

    CAN_COMPUTE_DEATH_GROWTH_RATE = np.count_nonzero(deaths[country]) >= FIT_PTS
    CAN_COMPUTE_DEATH_ACCEL = np.count_nonzero(deaths[country]) >= 2 * FIT_PTS

    if CAN_COMPUTE_DEATH_GROWTH_RATE:
        # Another fit to deaths:
        y2 = deaths[country][-1]
        y1 = deaths[country][-FIT_PTS]
        tau_guess = (t2 - t1) / np.log(y2 / y1)
        t0_guess = t2 - tau_guess * np.log(y2)

        params_deaths, covariance_deaths = curve_fit(
            model,
            x_fit[-FIT_PTS:],
            deaths[country][-FIT_PTS:],
            [tau_guess, t0_guess],
            maxfev=10000,
        )

        tau_2_deaths = (
            np.log(2) * ufloat(params_deaths[0], np.sqrt(covariance_deaths[0, 0])) / 24
        )

        if CAN_COMPUTE_DEATH_ACCEL:
            params_deaths_prev, _ = curve_fit(
                model,
                x_fit[-2 * FIT_PTS : -FIT_PTS],
                deaths[country][-2 * FIT_PTS : -FIT_PTS],
                [tau_guess, t0_guess],
                maxfev=10000,
            )

    x_model = np.arange(
        dates[-FIT_PTS] - np.timedelta64(24, 'h'),
        dates[-1] + np.timedelta64(24 * N_DAYS_PROJECTION, 'h'),
    )
    x_model_float = x_model.astype(float)

    CRITICAL_CASES = 0.05
    plt.axhline(
        icu_beds[country] * 10 / CRITICAL_CASES,  # ×10 is conversion to per million
        linestyle=':',
        color='r',
        label='Critical cases ≈ ICU beds',
    )

    # Plot a bunch of random projectioins by drawing from Gaussian with the parameter
    # covariance:
    NUM_SIMS = 50
    for _ in range(NUM_SIMS):
        scenario_params = np.random.multivariate_normal(params, covariance)
        plt.plot(
            x_model,
            model(x_model_float, *scenario_params) / populations[country],
            '-',
            color='orange',
            alpha=0.01,
            linewidth=4,
        )

    # A dummy item to create the legend for the projection
    plt.fill_between(
        [dates[0], dates[1]],
        1e-6,
        2e-6,
        facecolor='orange',
        alpha=0.5,
        label='Active (projected)',
    )

    # Plot uncertainties in model via linear uncertainty propagation:
    # y_model = model(x_model_float, *params)
    # u_y_model = model_uncertainty(model, x_model_float, params, covariance)

    # plt.fill_between(
    #     x_model,
    #     y_model - 2 * u_y_model,
    #     y_model + 2 * u_y_model,
    #     facecolor=colour,
    #     edgecolor=colour,
    #     alpha=0.3,
    # )

    # Compute the percent change per day from the model:
    Y0 = model(x_fit[-1], *params)
    Y1 = model(x_fit[-1] + 24, *params)
    growth_rate = (Y1 / Y0 - 1) * 100

    # Compute the percent change per day from older data:
    Y0_prev = model(x_fit[-1], *params_prev)
    Y1_prev = model(x_fit[-1] + 24, *params_prev)
    growth_rate_prev = (Y1_prev / Y0_prev - 1) * 100

    # Compute the ac(de)celeration:
    acceleration = (growth_rate - growth_rate_prev) / FIT_PTS

    if CAN_COMPUTE_DEATH_GROWTH_RATE:
        # Compute the percent change per day in deaths:
        Y0 = model(x_fit[-1], *params_deaths)
        Y1 = model(x_fit[-1] + 24, *params_deaths)
        growth_rate_deaths = (Y1 / Y0 - 1) * 100
    else:
        growth_rate_deaths = np.nan


    if CAN_COMPUTE_DEATH_ACCEL:
        # Compute the percent change per day in deaths from older data:
        Y0_prev = model(x_fit[-1], *params_deaths_prev)
        Y1_prev = model(x_fit[-1] + 24, *params_deaths_prev)
        growth_rate_deaths_prev = (Y1_prev / Y0_prev - 1) * 100
    
        acceleration_deaths = (growth_rate_deaths - growth_rate_deaths_prev) / FIT_PTS
    else:
        acceleration_deaths = np.nan

    deaths_percent = deaths[country][-1] / cases[country][-1] * 100
    recovered_percent = recovered[-1] / cases[country][-1] * 100

    plt.semilogy(
        dates,
        cases[country] / populations[country],
        'D',
        markerfacecolor='deepskyblue',
        # color='g',
        markeredgewidth=0.5,
        markeredgecolor='k',
        markersize=4,
        label=f'Total',
        # alpha=0.75
    )

    plt.semilogy(
        dates,
        recovered / populations[country],
        's',
        markerfacecolor='mediumseagreen',
        # color='g',
        markeredgewidth=0.5,
        markeredgecolor='k',
        markersize=5,
        label=f'Recovered',
        # alpha=0.75
    )

    plt.semilogy(
        dates,
        active / populations[country],
        'o',
        markerfacecolor='orange',
        # color='orange',
        markeredgewidth=0.5,
        markeredgecolor='k',
        markersize=5,
        label=f'Active',
    )

    plt.semilogy(
        dates,
        deaths[country] / populations[country],
        '^',
        markerfacecolor='orangered',
        # color='r',
        markeredgewidth=0.5,
        markeredgecolor='k',
        markersize=5,
        label=f'Deaths',
        # alpha=0.75
    )

    plt.grid(True, linestyle=':')
    # plt.legend(loc='lower right')
    if i == 0:
        plt.suptitle('per capita COVID-19 cases and exponential projections by country')
    if i % COLS == 0:
        plt.ylabel('cases per million inhabitants')
    plt.axis(xmin=dates[DATES_START_INDEX] - np.timedelta64(24, 'h'), xmax=x_model[-1])
    plt.axis(ymin=2e-2, ymax=2e5)

    # Excape spaces in country names for latex
    display_name = display_names.get(country, country).replace(" ", "\\ ")

    growth_rate_deaths = (
        '-' if np.isnan(growth_rate_deaths) else f'{growth_rate_deaths:+.0f}'
    )
    
    acceleration_deaths = (
        '-' if np.isnan(acceleration_deaths) else f'{acceleration_deaths:+.1f}'
    )

    num_digits_tau2_uncertainty = max(len(str(abs(tau_2.s)).split('.')[0]), 1)
    tau2_format_specifier = f":.{num_digits_tau2_uncertainty}uLS"

    num_digits_tau2_deaths_uncertainty = max(
        len(str(abs(tau_2_deaths.s)).split('.')[0]), 1
    )
    tau2_deaths_format_specifier = f":.{num_digits_tau2_deaths_uncertainty}uLS"

    plt.text(
        0.03,
        0.97,
        '\n'.join(
            [
                f'$\\bf {display_name} $',
                f'Total: {cases[country][-1]}',
                f'Active: {active[-1]}, {"×" if tau_2 > 0 else "÷"}2 in ${abs(tau_2).format(tau2_format_specifier)}$ days',
                f'Recovered: {recovered[-1]} ({recovered_percent:.1f}%)',
                f'Deaths: {deaths[country][-1]} ({deaths_percent:.1f}%), {"×" if tau_2 > 0 else "÷"}2 in ${abs(tau_2_deaths).format(tau2_deaths_format_specifier)}$ days',
                # f'Δ active growth rate: {acceleration:+.1f}%/day²',
                # f'Δ death growth rate: {acceleration_deaths}%/day²',
            ]
        ),
        transform=plt.gca().transAxes,
        fontsize=8,
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='w', pad=0),
        va='top',
        # fontdict=dict(family='Ubuntu mono')
    )

    if i % COLS != 0:
        plt.gca().set_yticklabels([])

    plt.gca().xaxis.set_major_locator(locator)
    plt.gca().get_xaxis().get_major_formatter().show_offset = False

plt.subplots_adjust(
    left=0.04, bottom=0.05, right=0.995, top=0.95, wspace=0, hspace=0.1
)

handles, labels = plt.gca().get_legend_handles_labels()
plt.gcf().legend(handles, labels, loc='upper right', ncol=3)

plt.savefig('COVID_new.svg')
