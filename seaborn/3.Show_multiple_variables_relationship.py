import numpy as np
import seaborn
import pandas
from matplotlib import pyplot

tips = seaborn.load_dataset("tips")
print(tips)

# separate graph base on a class variable
seaborn.relplot(x="total_bill", y="tip", hue="smoker",
            col="time", data=tips)
pyplot.show()



fmri = seaborn.load_dataset('fmri')
print(fmri)
# separate graph base on two class variables, interact each other
seaborn.relplot(x="timepoint", y="signal", hue="subject",
            col="region", row="event", height=3,
            kind="line", estimator=None, data=fmri);
pyplot.show()


# separate graph with a class variable with multiple values
seaborn.relplot(x="timepoint", y="signal", hue="event", style="event",
            col="subject", col_wrap=5,
            height=3, aspect=.75, linewidth=2.5,
            kind="line", data=fmri.query("region == 'frontal'"))
pyplot.show()
