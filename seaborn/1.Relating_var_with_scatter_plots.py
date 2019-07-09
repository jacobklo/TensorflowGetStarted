import numpy as np
import pandas
from matplotlib import pyplot
import seaborn

tips = seaborn.load_dataset("tips")
print(tips)

# basic scatter plot
seaborn.relplot(x='total_bill', y='tip', data=tips)
pyplot.show()

# third variable describe as color
seaborn.relplot(x='total_bill', y='tip', hue='sex', data=tips)
pyplot.show()

# third variable describe with different marker
seaborn.relplot(x='total_bill', y='tip', hue='sex', style='sex', data=tips)
pyplot.show()

# four variable with both
seaborn.relplot(x='total_bill', y='tip', hue='sex', style='time', data=tips)
pyplot.show()

# qualitative value show as color
seaborn.relplot(x='total_bill', y='tip', hue='size', data=tips)
pyplot.show()

# customize color
seaborn.relplot(x='total_bill', y='tip', hue='size', palette='ch:r=-0.5,l=0.75', data=tips)
pyplot.show()

# size of each point
seaborn.relplot(x='total_bill', y='tip', size='size', data=tips)
pyplot.show()

# change size of dots
seaborn.relplot(x='total_bill', y='tip', size='size', sizes=(1, 500), data=tips)
pyplot.show()
