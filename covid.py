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
}

countries = list(populations.keys())

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
    'World': np.nan,
}


# Clone or pull ulklc repo:
if not os.path.exists('covid19-timeseries'):
    subprocess.check_call(
        ['git', 'clone', 'https://github.com/ulklc/covid19-timeseries']
    )
else:
    subprocess.check_call(['git', 'pull'], cwd='covid19-timeseries')

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
    'New Zealand': 'NZ',
    'Thailand': 'TH',
    'World': None,
}

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

cases['World'] = sum(cases.values())
deaths['World'] = sum(deaths.values())
recoveries['World'] = sum(recoveries.values())


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
    sorted(countries, key=lambda c: -np.nanmax(cases[c] / populations[c]))
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

    ax2.axis(ymin=0, ymax=16)
    ax2.set_yticks([0, 2, 4, 6, 8, 10, 12, 14, 16])

    if (i % COLS != COLS - 1) and (i < len(countries) - 1):
        ax2.set_yticklabels([])
    else:
        ax2.set_ylabel('doubling/halving time (days)')

    for ax in [ax1, ax2]:
        ax.xaxis.set_major_locator(locator)
        ax.get_xaxis().get_major_formatter().show_offset = False



    # Excape spaces in country names for latex
    display_name = country.replace(" ", "\\ ")

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
