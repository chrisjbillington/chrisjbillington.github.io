import datetime

from scipy.optimize import curve_fit
from scipy.signal import convolve
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.units as munits
import matplotlib.dates as mdates


converter = mdates.ConciseDateConverter()
locator = mdates.AutoDateLocator(minticks=3, maxticks=3)

munits.registry[np.datetime64] = converter
munits.registry[datetime.date] = converter
munits.registry[datetime.datetime] = converter


def gaussian_smoothing(data, pts):
    """gaussian smooth an array by given number of points"""
    x = np.arange(-4 * pts, 4 * pts + 1, 1)
    kernel = np.exp(-(x ** 2) / (2 * pts ** 2))
    kernel /= kernel.sum()
    return convolve(data, kernel, mode='same')


import pickle

df1 = pd.read_html('https://covidlive.com.au/report/daily-cases/vic')[1]
# with open('cases.pickle', 'wb') as f:
#     pickle.dump(df1, f)
# with open('cases.pickle', 'rb') as f:
#     df1 = pickle.load(f)

cases = np.array(df1['CASES'])


dates = np.array(
    [
        np.datetime64(
            datetime.datetime.strptime(date.split(' ', 1)[1] + ' 2020', "%d %b %Y"),
            'h',
        )
        for date in df1['DATE']
    ]
)

SMOOTHING = 4
DAYS_PROJECTION = 60

new = np.diff(cases, prepend=0)
new_padded = np.zeros(len(new) + 3 * SMOOTHING)
new_padded[: -3 * SMOOTHING] = new


def linear(x, m, c):
    return m * x + c


# Smoothing requires padding to give sensible results at the right edge. Compute a
# linear fit to daily cases over the last week, and pad the data with the fit results
# prior to smoothing. Also pad with an upper and lower uncertainty bounds of the fit in
# order to establish an uncertainty range for R.

FIT_PTS = 7
x_data = np.arange(-FIT_PTS, 0)
params, cov = curve_fit(linear, x_data, new[-FIT_PTS:])

x = np.arange(3 * SMOOTHING)
fit = linear(x, *params).clip(0, None)
if np.isinf(cov).any():
    u_fit = np.sqrt(fit)
else:
    u_fit = np.sqrt(cov[0, 0] * x ** 2 + 2 * cov[0, 1] * x + cov[1, 1])
    u_fit = u_fit.clip(np.sqrt(fit), None)

new_padded[-3 * SMOOTHING :] = fit
new_smoothed = gaussian_smoothing(new_padded, SMOOTHING)[: -3 * SMOOTHING]

new_padded[-3 * SMOOTHING :] = fit + u_fit
new_smoothed_upper = gaussian_smoothing(new_padded, SMOOTHING)[: -3 * SMOOTHING]

new_padded[-3 * SMOOTHING :] = fit - u_fit
new_smoothed_lower = gaussian_smoothing(new_padded, SMOOTHING)[: -3 * SMOOTHING]

# plt.bar(dates, new)
# plt.plot(dates, new_smoothed_upper)
# plt.plot(dates, new_smoothed)
# plt.plot(dates, new_smoothed_lower)
# plt.show()

tau = 5  # reproductive time of the virus in days
R_upper = (new_smoothed_upper[1:] / new_smoothed_upper[:-1]) ** tau
R_lower = (new_smoothed_lower[1:] / new_smoothed_lower[:-1]) ** tau
R = (new_smoothed[1:] / new_smoothed[:-1]) ** tau

# Other than the uncertainty caused by the padding, there is sqrt(N)/N uncertainty in R
# so clip the uncertainty to at least that much:
# u_R = R * np.sqrt(new_smoothed[1:]) / new_smoothed[1:]
# R_upper = R_upper.clip(R + u_R, None)
# R_lower = R_lower.clip(None, R - u_R)

# Projection of daily case numbers:
t_projection = np.linspace(0, DAYS_PROJECTION, DAYS_PROJECTION + 1)
new_projection = new_smoothed[-1] * (R[-1] ** (1 / tau)) ** t_projection
new_projection_upper = (
    new_smoothed_upper[-1] * (R_upper[-1] ** (1 / tau)) ** t_projection
)
new_projection_lower = (
    new_smoothed_lower[-1] * (R_lower[-1] ** (1 / tau)) ** t_projection
)

STAGE_ONE = dates[58]  # March 23
STAGE_TWO = dates[61]  # March 26
STAGE_THREE = dates[66]  # March 31
STAGE_TWO_AGAIN = dates[128]  # June 1
POSTCODE_STAGE_3 = dates[159]  # July 2
STAGE_THREE_AGAIN = dates[165]  # July 8
MASKS = dates[180]  # July 23
STAGE_FOUR = dates[190]  # Aug 2
END_STAGE_4 = np.datetime64('2020-09-13', 'h')

plt.figure(figsize=(18, 6))
plt.fill_betweenx(
    [-10, 10],
    [STAGE_ONE, STAGE_ONE],
    [STAGE_TWO, STAGE_TWO],
    color="green",
    alpha=0.5,
    linewidth=0,
    label="Stage 1",
)

plt.fill_betweenx(
    [-10, 10],
    [STAGE_TWO, STAGE_TWO],
    [STAGE_THREE, STAGE_THREE],
    color="yellow",
    alpha=0.5,
    linewidth=0,
    label="Stage 2",
)

plt.fill_betweenx(
    [-10, 10],
    [STAGE_THREE, STAGE_THREE],
    [STAGE_TWO_AGAIN, STAGE_TWO_AGAIN],
    color="orange",
    alpha=0.5,
    linewidth=0,
    label="Stage 3",
)

plt.fill_betweenx(
    [-10, 10],
    [STAGE_TWO_AGAIN, STAGE_TWO_AGAIN],
    [POSTCODE_STAGE_3, POSTCODE_STAGE_3],
    color="yellow",
    alpha=0.5,
    linewidth=0,
)


plt.fill_betweenx(
    [-10, 10],
    [POSTCODE_STAGE_3, POSTCODE_STAGE_3],
    [STAGE_THREE_AGAIN, STAGE_THREE_AGAIN],
    color="yellow",
    edgecolor="orange",
    alpha=0.5,
    linewidth=0,
    hatch="//////",
    label="Postcode Stage 3",
)


plt.fill_betweenx(
    [-10, 10],
    [STAGE_THREE_AGAIN, STAGE_THREE_AGAIN],
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
    label="Stage 3 + masks",
)


plt.fill_betweenx(
    [-10, 10],
    [STAGE_FOUR, STAGE_FOUR],
    [END_STAGE_4, END_STAGE_4],
    color="red",
    alpha=0.5,
    linewidth=0,
    label="Stage 4",
)

for i in range(10):
    plt.fill_betweenx(
        [-10, 10],
        [END_STAGE_4 + 24 * i, END_STAGE_4 + 24 * i],
        [END_STAGE_4 + 24 * (i + 1), END_STAGE_4 + 24 * (i + 1)],
        color="red",
        alpha=0.5 * (10 - i) / 10,
        linewidth=0,
    )

START_IX = 35
plt.fill_between(dates[1:] + 24, R, label=R"$R_\mathrm{eff}$", step='pre', color='C0')

plt.fill_between(
    dates[1:] + 24,
    R_lower,
    R_upper,
    label=R"$R_\mathrm{eff}$ uncertainty",
    color='cyan',
    alpha=0.3,
    step='pre',
    zorder=2,
    linewidth=0,
)

plt.axhline(1.0, color='k', linewidth=1)
plt.axis(
    xmin=dates[START_IX + 1], xmax=dates[-1] + DAYS_PROJECTION * 24, ymin=0, ymax=3
)
plt.grid(True, linestyle=":")

handles, labels = plt.gca().get_legend_handles_labels()

plt.ylabel("$R_{eff}$")

u_R_latest = (R_upper[-1] - R_lower[-1]) / 2

plt.title(
    "$R_{{eff}}$ in Victoria with Melbourne restriction levels and daily cases\n"
    + f"Latest estimate: $R_{{eff}}={R[-1]:.02f} \\pm {u_R_latest:.02f}$"
)

ax2 = plt.twinx()
plt.semilogy(dates, new_smoothed, color='magenta', label='Daily cases (smoothed)')
plt.fill_between(
    dates,
    new_smoothed_lower,
    new_smoothed_upper,
    color='magenta',
    alpha=0.3,
    linewidth=0,
)
plt.plot(
    dates[-1] + 24 * t_projection.astype('timedelta64[h]'),
    new_projection,
    color='magenta',
    linestyle='--',
    label='Daily cases (projected)',
)
plt.fill_between(
    dates[-1] + 24 * t_projection.astype('timedelta64[h]'),
    new_projection_lower,
    new_projection_upper,
    color='magenta',
    alpha=0.3,
    linewidth=0,
    label='Daily cases uncertainty',
)
plt.axvline(dates[-1] + 24, linestyle='--', color='k', label='Today')
plt.axis(ymin=1, ymax=1000)
plt.ylabel("Daily confirmed cases")
plt.tight_layout()

handles2, labels2 = plt.gca().get_legend_handles_labels()

handles += handles2
labels += labels2

order = [6, 7, 0, 1, 3, 2, 4, 5, 8, 9, 11, 10]
plt.legend(
    [handles[idx] for idx in order],
    [labels[idx] for idx in order],
    loc='upper left',
    ncol=3,
)

plt.savefig('COVID_VIC.svg')
# plt.show()

# Update the date in the HTML
from pathlib import Path
html_file = 'COVID_VIC.html'
html_lines = Path(html_file).read_text().splitlines()
now = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d-%H:%M')
for i, line in enumerate(html_lines):
    if 'Last updated' in line:
        html_lines[i] = f'    Last updated: {now} UTC'
Path(html_file).write_text('\n'.join(html_lines) + '\n')
