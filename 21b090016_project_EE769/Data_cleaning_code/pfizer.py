#!/usr/bin/env python

import pandas as pd
import numpy as np

d_dict = {}

for i in range(31):
    d = pd.read_csv("{}.csv".format(i+1))
    d = d.set_index("Unnamed: 0")
    d.dropna(inplace=True)
    d = d.dropna(thresh=1)
    d = d.T
    d = d.dropna(thresh=1)
    d = d.replace(',', '', regex=True)
    d = d.replace({'\$': ''}, regex=True)
    d = d.apply(pd.to_numeric)
    d_dict[i] = d



for i in range (31):
    d_dict[i].to_csv("quat{}.csv".format(i))

# v.to_csv("quat{}.csv".format(i))