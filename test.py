from scipy.stats import ttest_ind, levene
import pandas as pd


xf1 = [0.94, 0.96, 0.88, 0.87, 0.90, 0.82, 0.93, 0.91, 0.84, 0.86]
yf1 = [0.97, 0.98, 0.95, 0.94, 0.96, 0.95, 0.97, 0.96, 0.94, 0.94]

print(levene(xf1, yf1))  # pvalue=0.003585 pvalue < 0.05 方差不齐性

print(ttest_ind(xf1, yf1, equal_var=False))  # pvalue=0.001215, pvalue < 0.05, statistically significant difference

