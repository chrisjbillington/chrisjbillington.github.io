import sys
import os
from scipy.optimize import curve_fit
import numpy as np
import datetime
import matplotlib.units as munits
import matplotlib.dates as mdates
from pathlib import Path
import pandas as pd

NBSP = u"\u00A0"
converter = mdates.ConciseDateConverter()
locator = mdates.DayLocator([1])

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

# Whether to plot US state data instead. In this case, we use US states instead of
# countries:
US_STATES = 'US' in sys.argv


def exponential_smoothing(arr, tau):
    k = 1 / tau
    result = np.zeros_like(arr)
    result[0] = arr[0]
    for i, y in enumerate(arr[1:], start=1):
        result[i] = k * y + (1 - k) * result[i - 1]
    return result

def estimate_recoveries(cases, deaths, clip_to_living=True):
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

    result = convolve(living_cases, recovery_curve)[: len(cases)].astype(int)
    if clip_to_living:
        result = result.clip(0, cases - deaths)

    return result


if US_STATES:

    # NYT repo url and directory we're interested in:
    REPO_URL = "https://raw.githubusercontent.com/nytimes/covid-19-data/master"
    df = pd.read_csv(f"{REPO_URL}/us-states.csv")

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

    # Vaccine repo url and directory we're interested in:
    REPO_URL = "https://raw.githubusercontent.com/govex/COVID-19/master"
    DATA_DIR = "data_tables/vaccine_data/raw_data"
    df = pd.read_csv(f"{REPO_URL}/{DATA_DIR}/vaccine_data_us_state_timeline.csv")

    vax_data = {}
    for state, subdf in df.groupby('Province_State'):
        vax_data[state] = {
            'dates': np.array(
                [
                    np.datetime64(datetime.datetime.strptime(date, "%m/%d/%Y"), 'h')
                    for date in subdf['date']
                ]
            ),
            'vaccinated': np.array(
                [
                    x.replace("\u202c", "") if isinstance(x, str) else x
                    for x in subdf['people_total']
                ],
                dtype=float,  # Work around an errant unicode character in data
            ),
        }


else:

    # JH repo location and subdirectory we're interested in:
    REPO_URL = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master"
    DATA_DIR = "csse_covid_19_data/csse_covid_19_time_series"

    # Translate JH country names to what we call them:
    COUNTRY_NAMES = {
        'Taiwan*': 'Taiwan',
        'US': 'United States',
        'Korea, South': 'South Korea',
    }
    
    PROVINCES_TO_TREAT_AS_COUNTRIES = ['Hong Kong']

    def process_file(csv_file):
        COLS_TO_DROP = ['Province/State', 'Country/Region', 'Lat', 'Long']
        df = pd.read_csv(f"{REPO_URL}/{DATA_DIR}/{csv_file}")
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

        for country, subdf in df.groupby('Province/State'):
            if country in PROVINCES_TO_TREAT_AS_COUNTRIES:
                country = COUNTRY_NAMES.get(country, country)
                subdf = subdf.drop(columns=COLS_TO_DROP)
                data[country] = np.array(subdf.sum())

        return dates, data
    

    dates, cases = process_file('time_series_covid19_confirmed_global.csv')
    _, deaths = process_file('time_series_covid19_deaths_global.csv')
    _, recoveries = process_file('time_series_covid19_recovered_global.csv')

    cases['World'] = sum(cases.values())
    deaths['World'] = sum(deaths.values())
    recoveries['World'] = sum(recoveries.values())

    # OWID repo location and subdirectory we're interested in:
    REPO_URL = "https://raw.githubusercontent.com/owid/covid-19-data/master"
    DATA_DIR = "public/data/vaccinations"
    df = pd.read_csv(f"{REPO_URL}/{DATA_DIR}/vaccinations.csv")

    NOT_REAL_COUNTRIES = ['Scotland', 'Northern Ireland', 'England', 'Wales']
    vax_data = {}
    for country, subdf in df.groupby('location'):
        if country in NOT_REAL_COUNTRIES:
            continue
        vax_data[country] = {
            'dates': np.array(
                [
                    np.datetime64(datetime.datetime.strptime(date, "%Y-%m-%d"), 'h')
                    for date in subdf['date']
                ]
            ),
            'vaccinated': np.array(subdf['people_vaccinated']),
        }


if US_STATES:
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

else:
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
        'Poland': 38,
        'Hong Kong': 7.45,
        'Peru': 32.0,
        'Ecuador': 17.1,
        'Romania': 19.4,
        'Saudi Arabia': 33.7,
        'Pakistan': 212.2,
        'United Arab Emirates': 9.6,
        'Greece': 10.7,
        'Egypt': 98.4,
        'Colombia': 49.7,
        'Qatar': 2.8,
        'Bangladesh': 161.4,
        'Belarus': 9.5,
        'Kuwait': 4.13,
        'Ukraine': 42,
        'Philippines': 106.7,
        'Argentina': 44.5,
        'Hungary': 9.77,
        'Vietnam': 95.54,
        'Slovakia': 5.45,
        'Croatia': 4.08,
        'Bahrain': 1.57,
        'Bulgaria': 7.0,
        'Costa Rica': 5.0,
        'Estonia': 1.325,
        'Latvia': 1.92,
        'Lithuania': 2.794,
        'Luxembourg': 0.614,
        'Malta': 0.494,
        'Oman': 4.83,
        'Slovenia': 2.081,
        'Cyprus': 0.876,
        'Guinea': 12.41,
        'Serbia': 6.964,
        'Seychelles': 0.097625,
        'Panama': 4.246,
}


countries = list(populations.keys())

for country in vax_data:
    # Make sure the country names are the same as what we are calling them:
    if country not in countries:
        print(country)


# Fix up the vaccination data a bit, add zero entries for countries not included:
for country in countries:
    if country in vax_data:
        vaccinated = vax_data[country]['vaccinated']
        vax_dates = vax_data[country]['dates']
        vax_dates = vax_dates[~np.isnan(vaccinated)]
        vaccinated = vaccinated[~np.isnan(vaccinated)]

        if len(vaccinated) > 0:
            # Prepend a zero:
            vaccinated = np.insert(vaccinated, 0, 0.0)
            vax_dates = np.insert(vax_dates, 0, vax_dates[0] - 24)
        else:
            # no non-nan data
            vax_dates = dates
            vaccinated = np.full(len(dates), -1e6)

    else:
        vax_dates = dates
        vaccinated = np.full(len(dates), -1e6)

    vax_data[country] = {
        'dates': vax_dates,
        'vaccinated': np.array(vaccinated, dtype=int),
    }


# Print html for per-country links when adding a new country:
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
    'Poland': 6.9,
    'Hong Kong': 7.1,
    'Peru': np.nan,
    'Ecuador': np.nan,
    'Romania': 10.3,
    'Saudi Arabia': 22.8,
    'Pakistan': 1.5,
    'United Arab Emirates': np.nan,
    'Greece': 6,
    'Egypt': np.nan,
    'Hungary': 13.8,
    'Vietnam': np.nan,
    'Slovakia': np.nan,
    'Croatia': np.nan,
    'Bahrain': np.nan,
    'Bulgaria': np.nan,
    'Costa Rica': np.nan,
    'Estonia': np.nan,
    'Latvia': np.nan,
    'Lithuania': np.nan,
    'Luxembourg': np.nan,
    'Malta': np.nan,
    'Oman': np.nan,
    'Slovenia': np.nan,
    'Cyprus': np.nan,
    'Guinea': np.nan,
    'Serbia': np.nan,
    'Seychelles': np.nan,
    'Panama': np.nan,
}


def make_exponential(t0):
    # When k ~ 0, fitting an exponential becomes very uncertain because t0 is very far
    # away from the data. Instead, treat t0 as fixed at today's date and fit A. This
    # makes for projections whose uncertainty doesn't blow up as k becomes close to
    # zero.
    def exponential(t, k, A):
        exponent = k * (t - t0)
        exponent = exponent.clip(-100, 100)
        return A * np.exp(exponent)

    return exponential

COLS = 5
ROWS = int(np.ceil(len(countries) / COLS))

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
        sorted(countries, key=lambda c: -np.nanmax(deaths[c] / populations[c]))
    ):
        if SINGLE:
            row = col = 0
            plt.clf()
        else:
            row, col = divmod(i, COLS)

        ax1 = fig.add_subplot(gs[20 * row : 20 * row + 12, col])
        ax2 = fig.add_subplot(gs[20 * row + 12 : 20 * row + 18, col])
        ax3 = ax2.twinx() # Solely to add an extra scale to ax2
        ax4 = ax1.twinx()

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
            if 0 in [y2, y1] or y1 == y2 or y1 < 0 or y2 < 0:
                k_arr.append(0)
                u_k_arr.append(0)
                params = None
            else:
                k_guess = np.log(y2 / y1) / (t2 - t1)
                A_guess = active[-1]

                params, covariance = curve_fit(
                    make_exponential(t2),
                    x_fit[j - FIT_PTS + 1 : j + 1],
                    active[j - FIT_PTS + 1 : j + 1],
                    [k_guess, A_guess],
                    maxfev=100000,
                )

                k_arr.append(24 * params[0])
                u_k_arr.append(24 * np.sqrt(covariance[0, 0]))

        k_arr = np.array(k_arr)
        u_k_arr = np.array(u_k_arr)

        r_arr = np.exp(k_arr) - 1
        u_r_arr = u_k_arr * np.exp(k_arr)

        x_model = np.arange(
            dates[-FIT_PTS] - np.timedelta64(24, 'h'),
            dates[-1] + np.timedelta64(24 * N_DAYS_PROJECTION, 'h'),
        )
        x_model_float = x_model.astype(float)

        if not US_STATES:
            ax1.axhline(
                icu_beds.get(country, np.nan) * 10 / CRITICAL_CASES,  # ×10 is conversion to per million
                linestyle=':',
                color='r',
                label='Critical cases ≈ ICU beds',
            )

        # Plot a bunch of random projections by drawing from Gaussian with the parameter
        # covariance:
        NUM_SIMS = 50
        if params is not None:
            for _ in range(NUM_SIMS):
                scenario_params = np.random.multivariate_normal(params, covariance)
                ax1.plot(
                    x_model,
                    make_exponential(x_fit[-1])(x_model_float, *scenario_params) / populations[country],
                    '-',
                    color='orange',
                    alpha=0.02,
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
            label='Total cases',
        )

        ax4.step(
            vax_data[country]['dates'],
            100 * vax_data[country]['vaccinated'] / (1e6 * populations[country]),
            color='mediumseagreen',
            label='Vaccinated',
            linewidth=3,
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
            label=f'Total deaths',
        )

        ax1.step(
            dates[1:],
            exponential_smoothing(np.diff(deaths[country] / populations[country]), 5),
            color='orangered',
            label='Daily deaths',
        )

        ax1.step(
            dates[1:],
            exponential_smoothing(np.diff(cases[country] / populations[country]), 5),
            color='deepskyblue',
            label='Daily cases',
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
        ax1.axis(ymin=1e-2, ymax=1e6)
        ax4.axis(ymin=0, ymax=100)

        if not SINGLE and i % COLS != 0:
            ax1.set_yticklabels([])


        valid = active[FIT_PTS:] > 2
        ax2.fill_between(
            dates[FIT_PTS:][valid],
            100 * (r_arr + u_r_arr)[valid],
            100 * (r_arr - u_r_arr)[valid],
            color='k',
            alpha=0.5,
            label='Active growth rate',
        )

        ax2.axis(ymin=-30, ymax=50)
        ax3.axis(ymin=-30, ymax=50)

        growth_rate_labels = [-20, -10, 0, 10, 20, 30, 40]

        doubling_time_labels = [
            f'{np.log(2) / np.log(r / 100 + 1):.1f}' if r else '∞' for r in growth_rate_labels
        ]

        ax2.set_yticks(growth_rate_labels)
        ax3.set_yticks(growth_rate_labels)
        ax4.set_yticks([25, 50, 75, 100])

        ax3.set_yticklabels(doubling_time_labels)
        ax2.axhline(0, color='k', linestyle='-')

        if SINGLE or (i % COLS == 0):
            ax2.set_ylabel('Growth rate (%/day)')
        else:
            ax2.set_yticklabels([])

        if SINGLE or (i % COLS == COLS - 1) or (i == len(countries) - 1):
            ax3.set_ylabel('Doubling time (days)')
            ax4.set_ylabel('Percent vaccinated')
        else:
            ax4.set_yticklabels([])

        for ax in [ax1, ax2]:
            ax.xaxis.set_major_locator(locator)
            ax.get_xaxis().get_major_formatter().show_offset = False

        ax1.set_xticklabels([])
        ax2.tick_params(axis='x', rotation=90)

        # Escape spaces in country names for latex
        display_name = country.replace(" ", NBSP)

        num_vaxed = vax_data[country]["vaccinated"][-1]
        num_vaxed_percent = f'{100 * num_vaxed / (1e6 * populations[country]):.1f}'

        if num_vaxed < 0:
            num_vaxed = '0'
            num_vaxed_percent = '0.0%'

        lines = [
            f'$\\bf {display_name} $',
            f'Total: {cases[country][-1]}',
            f'Active: {active[-1]} ({int(round(100 * r_arr[-1])):+.0f}%/day)',
            f'Deaths: {deaths[country][-1]} ({deaths_percent:.1f}% of cases)',
            f'Vaccinated: {num_vaxed} ({num_vaxed_percent}%)'
        ]

        ax1.text(
            0.02,
            0.98,
            '\n'.join(lines),
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
            handles4, labels4 = ax4.get_legend_handles_labels()

            ax1.legend(
                handles1 + handles2 + handles4,
                labels1 + labels2 + labels4,
                loc='upper right',
                ncol=3,
            )

            if not os.path.exists('COVID'):
                os.mkdir('COVID')
            plt.savefig(f'COVID/{country.replace(" ", "_")}.svg')


    if not SINGLE:
        plt.subplots_adjust(left=0.04, bottom=0.05, right=0.96, top=0.95, wspace=0, hspace=0.0)

        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        handles4, labels4 = ax4.get_legend_handles_labels()

        plt.gcf().legend(
            handles1 + handles2 + handles4,
            labels1 + labels2 + labels4,
            loc='upper right',
            ncol=3,
        )

        plt.tight_layout()
        if US_STATES:
            plt.savefig('COVID_US.svg')
        else:
            plt.savefig('COVID.svg')

        # Update the date in the HTML
        html_file = 'COVID_US.html' if US_STATES else 'COVID.html'
        html_lines = Path(html_file).read_text().splitlines()
        now = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d-%H:%M')
        for i, line in enumerate(html_lines):
            if 'Last updated' in line:
                html_lines[i] = f'    Last updated: {now} UTC'
        Path(html_file).write_text('\n'.join(html_lines) + '\n')
