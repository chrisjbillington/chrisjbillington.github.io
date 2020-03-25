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


COLS = 6
ROWS = np.ceil(len(countries) / 6)

model = exponential

FIT_PTS = 5

DATES_START_INDEX = 2

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

    x_fit = dates.astype(float)

    tau_2_arr = []
    u_tau_2_arr = []

    for j in range(FIT_PTS, len(active)):

        t2 = x_fit[j]
        t1 = x_fit[j - FIT_PTS]
        y2 = active[j]
        y1 = active[j - FIT_PTS]
        if 0 in [y2, y1] or y1 == y2:
            tau_2_arr.append(np.nan)
            u_tau_2_arr.append(np.nan)
        else:
            tau_guess = (t2 - t1) / np.log(y2 / y1)
            t0_guess = t2 - tau_guess * np.log(y2)

            params, covariance = curve_fit(
                model,
                x_fit[j - FIT_PTS : j],
                active[j - FIT_PTS : j],
                [tau_guess, t0_guess],
                maxfev=10000,
            )

            tau_2_arr.append(np.log(2) * params[0] / 24)
            u_tau_2_arr.append(np.log(2) * np.sqrt(covariance[0, 0]) / 24)

    tau_2_arr = np.array(tau_2_arr)
    u_tau_2_arr = np.array(u_tau_2_arr)
    tau_2 = ufloat(tau_2_arr[-1], u_tau_2_arr[-1])

    tau_2_deaths_arr = []
    u_tau_2_deaths_arr = []
    for j in range(2 * FIT_PTS, len(active)):
        recent_deaths = np.diff(deaths[country])[j - FIT_PTS :j].sum()
        prev_deaths = np.diff(deaths[country])[j - 2 * FIT_PTS : j - FIT_PTS].sum()
        if 0 in [recent_deaths, prev_deaths] or recent_deaths == prev_deaths:
            tau_2_deaths_arr.append(np.inf)
            u_tau_2_deaths_arr.append(np.inf)
        else:
            tau_2_deaths_arr.append(
                (np.log(2) * FIT_PTS * 1 / np.log(recent_deaths / prev_deaths))
            )
            u_tau_2_deaths_arr.append(
                np.log(2)
                * FIT_PTS
                * np.sqrt(1 / prev_deaths + 1 / recent_deaths)
                / np.log(recent_deaths / prev_deaths) ** 2
            )

    tau_2_deaths_arr = np.array(tau_2_deaths_arr)
    u_tau_2_deaths_arr = np.array(u_tau_2_deaths_arr)
    tau_2_deaths = ufloat(tau_2_deaths_arr[-1], u_tau_2_deaths_arr[-1])

    x_model = np.arange(
        dates[-FIT_PTS] - np.timedelta64(24, 'h'),
        dates[-1] + np.timedelta64(24 * N_DAYS_PROJECTION, 'h'),
    )
    x_model_float = x_model.astype(float)

    ax1 = plt.gca()
    ax2 = plt.gca().twinx()

    # ax1.yaxis.tick_left()
    # ax1.yaxis.set_label_position("left")
    # ax2.yaxis.tick_right()
    # ax2.yaxis.set_label_position("right")


    CRITICAL_CASES = 0.05
    ax1.axhline(
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
    if i == 0:
        plt.suptitle('per capita COVID-19 cases and exponential projections by country')
    if i % COLS == 0:
        ax1.set_ylabel('cases per million inhabitants')
    ax1.axis(xmin=dates[DATES_START_INDEX] - np.timedelta64(24, 'h'), xmax=x_model[-1])
    ax1.axis(ymin=1e-2, ymax=1e6)

    if i % COLS != 0:
        ax1.set_yticklabels([])

    with np.errstate(invalid='ignore'):
        valid_doubling = (active[FIT_PTS:] > 100) & (tau_2_arr > 0) & (tau_2_arr < 50)

    ax2.fill_between(
        dates[FIT_PTS:][valid_doubling],
        (tau_2_arr + u_tau_2_arr)[valid_doubling],
        (tau_2_arr - u_tau_2_arr)[valid_doubling],
        color='k',
        alpha=0.5,
        label='Active doubling time',
    )

    with np.errstate(invalid='ignore'):
        valid_halving = (active[FIT_PTS:] > 100) & (tau_2_arr < 0) & (tau_2_arr > -50)

    ax2.fill_between(
        dates[FIT_PTS:][valid_halving],
        np.abs(tau_2_arr + u_tau_2_arr)[valid_halving],
        np.abs(tau_2_arr - u_tau_2_arr)[valid_halving],
        color='grey',
        alpha=0.5,
        label='Active halving time',
    )

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

    ax2.axis(ymin=0, ymax=24)
    ax2.set_yticks([0, 3, 6, 9, 12, 15, 18, 21, 24])

    if (i % COLS != COLS - 1) and (i < len(countries) - 1):
        ax2.set_yticklabels([])
    else:
        ax2.set_ylabel('doubling/halving time (days)')

    for ax in [ax1, ax2]:
        ax.xaxis.set_major_locator(locator)
        ax.get_xaxis().get_major_formatter().show_offset = False



    # Excape spaces in country names for latex
    display_name = display_names.get(country, country).replace(" ", "\\ ")

    num_digits_tau2_uncertainty = max(len(str(abs(tau_2.s)).split('.')[0]), 1)
    tau2_format_specifier = f":.{num_digits_tau2_uncertainty}uP"

    num_digits_tau2_deaths_uncertainty = max(
        len(str(abs(tau_2_deaths.s)).split('.')[0]), 1
    )
    tau2_deaths_format_specifier = f":.{num_digits_tau2_deaths_uncertainty}uP"

    tau_2_deaths_formatted = (
        abs(tau_2_deaths).format(tau2_deaths_format_specifier).replace('inf', '∞')
    )

    tau_2_formatted = abs(tau_2).format(tau2_format_specifier).replace('inf', '∞')

    NBSP = u"\u00A0"
    plt.text(
        0.02,
        0.98,
        '\n'.join(
            [
                f'$\\bf {display_name} $',
                f'Total: {cases[country][-1]}',
                f'Active: {active[-1]}',
                (
                    f'{NBSP * 2} → {"doubling" if tau_2 > 0 else "halving"} in {tau_2_formatted} days'
                    if abs(tau_2) < 50
                    else f'{NBSP * 2} → unchanging'
                ),
                f'Deaths: {deaths[country][-1]} ({deaths_percent:.1f}%)',
                (
                    f'{NBSP * 2} → Δ {"doubling" if tau_2_deaths > 0 else "halving"} in {tau_2_deaths_formatted} days'
                    if abs(tau_2_deaths) < 50
                    else f'{NBSP * 2} → Δ unchanging'
                ),
                f'Recovered: {recovered[-1]} ({recovered_percent:.1f}%)',
            ]
        ),
        transform=plt.gca().transAxes,
        fontsize=8,
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='w', pad=0),
        va='top',
        # fontdict=dict(family='Ubuntu mono'),
    )


plt.subplots_adjust(
    left=0.04, bottom=0.05, right=0.96, top=0.95, wspace=0, hspace=0.1
)

handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()

plt.gcf().legend(handles1 + handles2, labels1 + labels2, loc='upper right', ncol=3)

plt.savefig('COVID.svg')
