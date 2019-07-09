import numpy as np
import seaborn
import pandas
from matplotlib import pyplot
from matplotlib import colors

df = pandas.DataFrame( dict(time=np.arange(500),
                            value=np.random.randn(500).cumsum()))
print(df)
graph = seaborn.relplot(x='time', y='value', kind='line', data=df)
graph.fig.autofmt_xdate()
pyplot.show()


df = pandas.DataFrame( np.random.randn(500, 2).cumsum(axis=0), columns=['x', 'y'])
seaborn.relplot(x='x', y='y', sort=False, kind='line', data=df)
pyplot.show()


# If having multiple set of values
fmri = seaborn.load_dataset('fmri')
print(fmri)
seaborn.relplot(x='timepoint', y='signal', kind='line', data=fmri)
pyplot.show()

# disable UI for aggregate to speed up
seaborn.relplot(x='timepoint', y='signal', ci=None, kind='line', data=fmri)
pyplot.show()

# spread distribution at each timepoint by standard deviation
seaborn.relplot(x='timepoint', y='signal', ci="sd", kind='line', data=fmri)
pyplot.show()

# turn off compute aggragation
seaborn.relplot(x='timepoint', y='signal', estimator=None, kind='line', data=fmri)
pyplot.show()

# 3 variable using color
seaborn.relplot(x='timepoint', y='signal', hue='event', kind='line', data=fmri)
pyplot.show()

# 3 variable using color and dotted lines
seaborn.relplot(x='timepoint', y='signal', hue='event', style='event', kind='line', data=fmri)
pyplot.show()

# 4 variables using color and dotted lines
seaborn.relplot(x='timepoint', y='signal', hue='region', style='event', kind='line', data=fmri)
pyplot.show()


dots = seaborn.load_dataset("dots").query("align == 'dots'")
print(dots)

# separate variable by color
seaborn.relplot(x="time", y="firing_rate",
            hue="coherence", style="choice",
            kind="line", data=dots)
pyplot.show()

# custom color
palette = seaborn.cubehelix_palette(light=0.1, n_colors=6)
seaborn.relplot(x="time", y="firing_rate",
            hue="coherence", style="choice",
            palette=palette,
            kind="line", data=dots)
pyplot.show()

# change the way color shift
seaborn.relplot(x="time", y="firing_rate",
            hue="coherence", style="choice",
                hue_norm=colors.LogNorm(),
            kind="line", data=dots)
pyplot.show()


# separate variable by size of line
seaborn.relplot(x="time", y="firing_rate",
            size='coherence', style='choice',
            kind="line", data=dots)
pyplot.show()



# plot with date
df = pandas.DataFrame( dict(time=pandas.date_range('2017-1-1', periods=500),
                            value=np.random.randn(500).cumsum()))
print(df)
g = seaborn.relplot(x='time', y='value', kind='line', data=df)
# make x label sideway
g.fig.autofmt_xdate()
pyplot.show()


