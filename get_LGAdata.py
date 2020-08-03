import datetime

import numpy as np
import pandas as pd
import pickle

LGAs = [
    "Hume",
    "Wyndham",
    "Melbourne",
    "Brimbank",
    "Moonee Valley",
    "Moreland",
    "Banyule",
    "Whittlesea",
    "Casey",
    "Melton",
    "Darebin",
    "Stonnington",
    "Yarra",
    "Maribyrnong",
    "Boroondara",
    "Monash",
    "Hobsons Bay",
    "Port Phillip",
    "Glen Eira",
    "Manningham",
    "Mornington Peninsula",
    "Whitehorse",
    "Kingston",
    "Frankston",
    "Bayside",
    "Nillumbik",
    "Greater Dandenong",
    "Cardinia",
    "Yarra Ranges",
    "Knox",
    "Mitchell",
    "Maroondah",
]

URL_PREFIx = "https://covidlive.com.au/vic/"

dates = None
cases = {}

for LGA in LGAs:
    print(LGA)
    df = pd.read_html(f"{URL_PREFIx}{LGA.lower().replace(' ' ,'-')}")[1]
    cases[LGA] = np.array(df['CASES'])[::-1]
    if dates is None:
        dates = np.array(
            [
                np.datetime64(
                    datetime.datetime.strptime(
                        date + ' 2020', "%d %b %Y"
                    ),
                    'h',
                )
                for date in df['DATE']
            ]
        )[::-1]

with open('LGAdata.pickle', 'wb') as f:
    pickle.dump((dates, cases), f)
