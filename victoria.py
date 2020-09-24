import os
from datetime import datetime
from pytz import timezone
from pathlib import Path
import io
import zipfile
import tempfile

from scipy.optimize import curve_fit
from scipy.signal import convolve
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.units as munits
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import pandas as pd
import pantab
import requests

converter = mdates.ConciseDateConverter()

munits.registry[np.datetime64] = converter
munits.registry[datetime.date] = converter
munits.registry[datetime] = converter


def gaussian_smoothing(data, pts):
    """gaussian smooth an array by given number of points"""
    x = np.arange(-4 * pts, 4 * pts + 1, 1)
    kernel = np.exp(-(x ** 2) / (2 * pts ** 2))
    smoothed = convolve(data, kernel, mode='same')
    normalisation = convolve(np.ones_like(data), kernel, mode='same')
    return smoothed / normalisation


def fourteen_day_average(data):
    ret = np.cumsum(data, dtype=float)
    ret[14:] = ret[14:] - ret[:-14]
    return ret / 14


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


url = "https://public.tableau.com/workbooks/Cases_15982342702770.twb"
dbname = "Data/dash-charts/vic_detailed_prep Extract_daily-pubextract.hyper"
workbook_data = requests.get(url).content
workbook = zipfile.ZipFile(io.BytesIO(workbook_data))
with tempfile.TemporaryDirectory() as tempdir:
    dbpath = workbook.extract(dbname)
    name, df = pantab.frames_from_hyper(dbname).popitem()

data = []
for cases, date in zip(df['Cases'], df['Date']):
    try:
        cases = float(cases)
    except TypeError:
        cases = 0
    date = np.datetime64(date, 'h') + 24
    data.append((date, cases))

data.sort()
dates, new = [np.array(a) for a in zip(*data)]

new[np.isnan(new)] = 0

covidlive_data = pd.read_html("https://covidlive.com.au/vic")
latest_date = np.datetime64(
    datetime.strptime(covidlive_data[4]['DATE'][0] + ' 2020', "%d %b %Y"), 'h'
)

# If DHHS data not yet updated for today, use covidlive gross case number:
if dates[-1] != latest_date:
    dates = np.append(dates, [latest_date])
    df = covidlive_data[1]
    gross = list(df[df['CATEGORY'] == 'New Cases']['TOTAL'])
    net = list(df[df['CATEGORY'] == 'Cases']['NET'])
    if gross:
        # After gross numbers known today, this row exists in the table:
        new = np.append(new, [int(gross[0])])
    else:
        # Before gross numbers known today, net number is gross number:
        new = np.append(new, [int(net[0])])

START_IX = 35

all_new = new
all_dates = dates

ANIMATE = False


if ANIMATE:
    os.makedirs('VIC-animated', exist_ok=True)
    LOOP_START = START_IX + 10
else:
    LOOP_START = len(dates)

for j in range(LOOP_START, len(dates) + 1):
    dates = all_dates[:j]
    new = all_new[:j]

    SMOOTHING = 4
    new_padded = np.zeros(len(new) + 3 * SMOOTHING)
    new_padded[: -3 * SMOOTHING] = new


    def exponential(x, A, k):
        return A * np.exp(k * x)

    tau = 5  # reproductive time of the virus in days

    # Smoothing requires padding to give sensible results at the right edge. Compute an
    # exponential fit to daily cases over the last week or so, and pad the data with the fit
    # results prior to smoothing. Also pad with an upper and lower uncertainty bounds of
    # the fit in order to establish an uncertainty range for R.

    FIT_PTS = 20
    x0 = -14
    delta_x = 1
    fit_x = np.arange(-FIT_PTS, 0)
    fit_weights = 1 / (1 + np.exp(-(fit_x - x0) / delta_x))
    pad_x = np.arange(3 * SMOOTHING)

    params, cov = curve_fit(exponential, fit_x, new[-FIT_PTS:], sigma=1/fit_weights)
    fit = exponential(pad_x, *params).clip(0, None)
    new_padded[-3 * SMOOTHING :] = fit
    new_smoothed = gaussian_smoothing(new_padded, SMOOTHING)[: -3 * SMOOTHING]
    R = (new_smoothed[1:] / new_smoothed[:-1]) ** tau

    N_monte_carlo = 1000
    variance_R = np.zeros_like(R)
    # Monte-carlo of the above with noise to compute an uncertainty in R:
    u_new = np.sqrt((0.2 * new) ** 2 + new)  # sqrt(N) and 20%, added in quadrature
    for i in range(N_monte_carlo):
        new_with_noise = np.random.normal(new, u_new)
        params, cov = curve_fit(
            exponential,
            fit_x,
            new_with_noise[-FIT_PTS:],
            sigma=1 / fit_weights,
            maxfev=20000,
        )
        scenario_params = np.random.multivariate_normal(params, cov)
        fit = exponential(pad_x, *scenario_params).clip(0, None)
        new_padded[: -3 * SMOOTHING] = new_with_noise
        new_padded[-3 * SMOOTHING :] = fit
        new_smoothed_noisy = gaussian_smoothing(new_padded, SMOOTHING)[: -3 * SMOOTHING]
        R_noisy = (new_smoothed_noisy[1:] / new_smoothed_noisy[:-1]) ** tau
        variance_R += (R_noisy - R) ** 2 / N_monte_carlo

    u_R = np.sqrt(variance_R)

    R_upper = R + u_R
    R_lower = R - u_R

    R_upper = R_upper.clip(0, 10)
    R_lower = R_lower.clip(0, 10)
    R = R.clip(0, None)


    START_PLOT = np.datetime64('2020-03-01', 'h')
    END_PLOT = np.datetime64('2020-12-31', 'h')

    # Projection of daily case numbers:
    days_projection = (END_PLOT - dates[-1]).astype(int) // 24
    t_projection = np.linspace(0, days_projection, days_projection + 1)
    new_projection = new_smoothed[-1] * (R[-1] ** (1 / tau)) ** t_projection
    new_projection_upper = new_smoothed[-1] * (R_upper[-1] ** (1 / tau)) ** t_projection
    new_projection_lower = new_smoothed[-1] * (R_lower[-1] ** (1 / tau)) ** t_projection

    # # Examining whether the smoothing and uncertainty look decent
    # plt.bar(dates, new)
    # plt.fill_between(
    #     dates,
    #     new_smoothed_lower,
    #     new_smoothed_upper,
    #     color='orange',
    #     alpha=0.5,
    #     zorder=5,
    #     linewidth=0,
    # )
    # plt.plot(dates, new_smoothed, color='orange', zorder=6)
    # plt.plot(
    #     dates[-1] + 24 * t_projection.astype('timedelta64[h]'),
    #     new_projection,
    #     color='orange',
    #     zorder=6,
    # )
    # plt.fill_between(
    #     dates[-1] + 24 * t_projection.astype('timedelta64[h]'),
    #     new_projection_lower,
    #     new_projection_upper,
    #     color='orange',
    #     alpha=0.5,
    #     zorder=5,
    #     linewidth=0,
    # )
    # plt.grid(True)
    # plt.axis(
    #     xmin=np.datetime64('2020-06-15', 'h'), xmax=np.datetime64('2020-10-01', 'h')
    # )


    STAGE_ONE = np.datetime64('2020-03-23', 'h')
    STAGE_TWO = np.datetime64('2020-03-26', 'h')
    STAGE_THREE = np.datetime64('2020-03-31', 'h')
    STAGE_TWO_II = np.datetime64('2020-06-01', 'h')
    POSTCODE_STAGE_3 = np.datetime64('2020-07-02', 'h')
    STAGE_THREE_II = np.datetime64('2020-07-08', 'h')
    MASKS = np.datetime64('2020-07-23', 'h')
    STAGE_FOUR = np.datetime64('2020-08-02', 'h')
    FIRST_STEP = np.datetime64('2020-09-14', 'h')
    SECOND_STEP = np.datetime64('2020-09-28', 'h')
    THIRD_STEP = np.datetime64('2020-10-26', 'h')
    LAST_STEP = np.datetime64('2020-11-23', 'h')


    fig1 = plt.figure(figsize=(18, 6))
    plt.fill_betweenx(
        [-10, 10],
        [STAGE_ONE, STAGE_ONE],
        [STAGE_TWO, STAGE_TWO],
        color="green",
        alpha=0.5,
        linewidth=0,
        label="Stage 1 / Last step",
    )

    plt.fill_betweenx(
        [-10, 10],
        [STAGE_TWO, STAGE_TWO],
        [STAGE_THREE, STAGE_THREE],
        color="yellow",
        alpha=0.5,
        linewidth=0,
        label="Stage 2 / Third step",
    )

    plt.fill_betweenx(
        [-10, 10],
        [STAGE_THREE, STAGE_THREE],
        [STAGE_TWO_II, STAGE_TWO_II],
        color="orange",
        alpha=0.5,
        linewidth=0,
        label="Stage 3 / Second step",
    )

    plt.fill_betweenx(
        [-10, 10],
        [STAGE_TWO_II, STAGE_TWO_II],
        [POSTCODE_STAGE_3, POSTCODE_STAGE_3],
        color="yellow",
        alpha=0.5,
        linewidth=0,
    )


    plt.fill_betweenx(
        [-10, 10],
        [POSTCODE_STAGE_3, POSTCODE_STAGE_3],
        [STAGE_THREE_II, STAGE_THREE_II],
        color="yellow",
        edgecolor="orange",
        alpha=0.5,
        linewidth=0,
        hatch="//////",
        label="Postcode Stage 3",
    )


    plt.fill_betweenx(
        [-10, 10],
        [STAGE_THREE_II, STAGE_THREE_II],
        [MASKS, MASKS],
        color="orange",
        alpha=0.5,
        linewidth=0,
    )

    plt.fill_betweenx(
        [-10, 10],
        [MASKS, MASKS],
        [STAGE_FOUR, STAGE_FOUR],
        color="orange",
        edgecolor="red",
        alpha=0.5,
        linewidth=0,
        hatch="//////",
        label="Masks introduced",
    )


    plt.fill_betweenx(
        [-10, 10],
        [STAGE_FOUR, STAGE_FOUR],
        [SECOND_STEP, SECOND_STEP],
        color="red",
        alpha=0.5,
        linewidth=0,
        label="Stage 4 / First step",
    )

    plt.fill_betweenx(
        [-10, 10],
        [SECOND_STEP, SECOND_STEP],
        [THIRD_STEP, THIRD_STEP],
        color="orange",
        alpha=0.5,
        linewidth=0,
    )

    plt.fill_betweenx(
        [-10, 10],
        [THIRD_STEP, THIRD_STEP],
        [LAST_STEP, LAST_STEP],
        color="yellow",
        alpha=0.5,
        linewidth=0,
    )

    plt.fill_betweenx(
        [-10, 10],
        [LAST_STEP, LAST_STEP],
        [END_PLOT, END_PLOT],
        color="green",
        alpha=0.5,
        linewidth=0,
    )

    # for i in range(10):
    #     plt.fill_betweenx(
    #         [-10, 10],
    #         [END_STAGE_THREE_PLUS + 24 * i, END_STAGE_THREE_PLUS + 24 * i],
    #         [END_STAGE_THREE_PLUS + 24 * (i + 1), END_STAGE_THREE_PLUS + 24 * (i + 1)],
    #         color=ORANGERED,
    #         alpha=0.5 * (10 - i) / 10,
    #         linewidth=0,
    #     )

    plt.fill_between(dates[1:] + 24, R, label=R"$R_\mathrm{eff}$", step='pre', color='C0')

    plt.fill_between(
        dates[1:] + 24,
        R_lower,
        R_upper,
        label=R"$R_\mathrm{eff}$ uncertainty",
        color='cyan',
        edgecolor='blue',
        alpha=0.2,
        step='pre',
        zorder=2,
        # linewidth=0,
        hatch="////",
    )

    # # Reff values on given dates according to Dan Andrews infographic posted on facebook
    # # https://www.facebook.com/DanielAndrewsMP/photos/a.149185875145957/3350150198382826
    # gov_dates, gov_Reff = zip(
    #     *[
    #         ('2020-06-22', 1.72),
    #         ('2020-06-29', 1.61),
    #         ('2020-07-06', 1.33),
    #         ('2020-07-13', 1.26),
    #         ('2020-07-20', 1.17),
    #         ('2020-07-27', 0.97),
    #         ('2020-08-03', 0.86),
    #     ]
    # )
    # gov_dates = np.array([np.datetime64(d, 'h') for d in gov_dates])
    # plt.plot(gov_dates, gov_Reff, 'ro')

    plt.axhline(1.0, color='k', linewidth=1)
    plt.axis(
        xmin=START_PLOT, xmax=END_PLOT, ymin=0, ymax=3
    )
    plt.grid(True, linestyle=":", color='k', alpha=0.5)

    handles, labels = plt.gca().get_legend_handles_labels()

    plt.ylabel(R"$R_\mathrm{eff}$")

    u_R_latest = (R_upper[-1] - R_lower[-1]) / 2

    plt.title(
        "$R_\\mathrm{eff}$ in Victoria with Melbourne restriction levels and daily cases\n"
        + fR"Latest estimate: $R_\mathrm{{eff}}={R[-1]:.02f} \pm {u_R_latest:.02f}$"
    )

    plt.gca().yaxis.set_major_locator(mticker.MultipleLocator(0.25))
    ax2 = plt.twinx()
    plt.step(dates + 24, new, color='purple', label='Daily cases')
    plt.semilogy(
        dates + 12, new_smoothed, color='magenta', label='Daily cases (smoothed)'
    )

    # plt.fill_between(
    #     dates + 12,
    #     new_smoothed_lower,
    #     new_smoothed_upper,
    #     color='magenta',
    #     alpha=0.3,
    #     linewidth=0,
    # )
    plt.plot(
        dates[-1] + 12 + 24 * t_projection.astype('timedelta64[h]'),
        new_projection,
        color='magenta',
        linestyle='--',
        label='Daily cases (projected)',
    )
    plt.fill_between(
        dates[-1] + 12 + 24 * t_projection.astype('timedelta64[h]'),
        new_projection_lower,
        new_projection_upper,
        color='magenta',
        alpha=0.3,
        linewidth=0,
        label='Projection uncertainty',
    )
    plt.axvline(
        dates[-1] + 24,
        linestyle='--',
        color='k',
        label=f'Today ({dates[-1].tolist().strftime("%b %d")})',
    )
    plt.axis(ymin=1, ymax=1000)
    plt.ylabel("Daily confirmed cases")
    plt.tight_layout()

    handles2, labels2 = plt.gca().get_legend_handles_labels()

    handles += handles2
    labels += labels2

    order = [6, 7, 8, 0, 1, 3, 2, 4, 5, 9, 10, 12, 11]
    plt.legend(
        [handles[idx] for idx in order],
        [labels[idx] for idx in order],
        loc='upper right',
        ncol=2,
    )

    plt.gca().yaxis.set_major_formatter(mticker.ScalarFormatter())
    plt.gca().yaxis.set_minor_formatter(mticker.ScalarFormatter())
    plt.gca().tick_params(axis='y', which='minor', labelsize='x-small')
    plt.setp(plt.gca().get_yminorticklabels()[1::2], visible=False)
    plt.gca().xaxis.set_major_locator(mdates.DayLocator([1, 15]))

    fig2 = plt.figure(figsize=(10.8, 6))

    unknowns_last_14d_dates, unknowns_last_14d = zip(
        *[
            ('2020-09-04', 200),
            ('2020-09-05', 200),
            ('2020-09-06', 194),
            ('2020-09-10', 104),
            ('2020-09-11', 94),
            ('2020-09-12', 83),
            ('2020-09-13', 82),
            ('2020-09-14', 84),
            ('2020-09-15', 73),
            ('2020-09-16', 64),
            ('2020-09-17', 52),
            ('2020-09-18', 47),
            ('2020-09-19', 45),
            ('2020-09-20', 41),
            ('2020-09-21', 37),
        ]
    )
    unknowns_last_14d_dates = np.array(
        [np.datetime64(d, 'h') for d in unknowns_last_14d_dates]
    )

    cases_and_projection = np.concatenate((new, new_projection[1:]))
    cases_and_projection_upper = np.concatenate((new, new_projection_upper[1:]))
    cases_and_projection_lower = np.concatenate((new, new_projection_lower[1:]))
    average_cases = fourteen_day_average(cases_and_projection)
    average_projection_upper = fourteen_day_average(cases_and_projection_upper)
    average_projection_lower = fourteen_day_average(cases_and_projection_lower)

    all_dates = np.concatenate(
        (dates, dates[-1] + 24 * t_projection.astype('timedelta64[h]')[1:])
    )

    # plt.step(all_dates, cases_and_projection)

    plt.step(
        unknowns_last_14d_dates + 24,
        unknowns_last_14d,
        color='blue',
        label='14d total mystery cases* (DHHS)',
    )
    text = plt.figtext(
        0.57,
        0.69,
        "* 14d mystery cases must be below 5 to move to third step",
        fontsize='x-small',
    )
    text.set_bbox(dict(facecolor='white', alpha=0.8, linewidth=0))


    plt.step(dates + 24, average_cases[: len(dates)], color='grey', label='14d average daily cases')
    plt.plot(
        dates[-1] + 12 + 24 * t_projection.astype('timedelta64[h]'),
        average_cases[-len(t_projection) :],
        color='grey',
        linestyle='--',
        label='14d average (projected)',
    )

    plt.fill_between(
        dates[-1] + 12 + 24 * t_projection.astype('timedelta64[h]'),
        average_projection_lower[-len(t_projection):],
        average_projection_upper[-len(t_projection):],
        color='grey',
        alpha=0.5,
        linewidth=0,
        label='Projection uncertainty',
    )

    plt.axvline(
        dates[-1] + 24,
        linestyle='--',
        color='k',
        label=f'Today ({dates[-1].tolist().strftime("%b %d")})',
    )
    plt.yscale('log')
    plt.axis(xmin=np.datetime64('2020-07-01', 'h'), xmax=END_PLOT, ymin=1, ymax=1000)
    plt.grid(True, linestyle=":", color='k', alpha=0.5)
    plt.grid(True, linestyle=":", color='k', alpha=0.25, which='minor')
    plt.ylabel("Cases")

    STEP_ONE = np.datetime64('2020-09-14')
    plt.fill_betweenx(
        [0, 1000],
        [FIRST_STEP, FIRST_STEP],
        [SECOND_STEP, SECOND_STEP],
        color="red",
        alpha=0.5,
        linewidth=0,
        label="First step"
    )

    plt.fill_betweenx(
        [0, 1000],
        [SECOND_STEP, SECOND_STEP],
        [THIRD_STEP, THIRD_STEP],
        color="orange",
        alpha=0.5,
        linewidth=0,
        label="Second step"
    )

    plt.fill_betweenx(
        [0, 1000],
        [THIRD_STEP, THIRD_STEP],
        [LAST_STEP, LAST_STEP],
        color="yellow",
        alpha=0.5,
        linewidth=0,
        label="Third step"
    )

    plt.fill_betweenx(
        [-10, 1000],
        [LAST_STEP, LAST_STEP],
        [END_PLOT, END_PLOT],
        color="green",
        alpha=0.5,
        linewidth=0,
        label="Last step"
    )


    plt.step(
        [FIRST_STEP, SECOND_STEP, THIRD_STEP, LAST_STEP],
        [2000, 50, 5, 0],
        where='post',
        color='k',
        linewidth=2,
        label='Required target'
    )

    handles, labels = plt.gca().get_legend_handles_labels()

    order = [1, 2, 5, 4, 0, 6, 7, 8, 9, 3]
    plt.legend(
        [handles[idx] for idx in order],
        [labels[idx] for idx in order],
        loc='upper right',
        ncol=2,
    )
    plt.title(
        "VIC 14 day average with Melbourne reopening targets\n"
        + f"Current average: {average_cases[len(dates) - 1]:.1f} cases per day. Fortnightly mystery cases: {unknowns_last_14d[-1]}"
    )
    plt.tight_layout()
    plt.gca().yaxis.set_major_formatter(mticker.ScalarFormatter())
    plt.gca().yaxis.set_minor_formatter(mticker.ScalarFormatter())
    plt.gca().tick_params(axis='y', which='minor', labelsize='x-small')
    plt.setp(plt.gca().get_yminorticklabels()[1::2], visible=False)
    plt.gca().xaxis.set_major_locator(mdates.DayLocator([1, 15]))

    if ANIMATE:
        print(j)
        fig1.savefig(Path('VIC-animated', f'reff_{j:04d}.png'))
        fig2.savefig(Path('VIC-animated', f'reopening_{j:04d}.png'))
        plt.close(fig1)
        plt.close(fig2)
    else:
        fig1.savefig('COVID_VIC.svg')
        fig2.savefig('COVID_VIC_reopening.svg')
        plt.show()

        # Update the date in the HTML
        html_file = 'COVID_VIC.html'
        html_lines = Path(html_file).read_text().splitlines()
        now = datetime.now(timezone('Australia/Melbourne')).strftime('%Y-%m-%d-%H:%M')
        for i, line in enumerate(html_lines):
            if 'Last updated' in line:
                html_lines[i] = f'    Last updated: {now} Melbourne time'
        Path(html_file).write_text('\n'.join(html_lines) + '\n')
