# import this

'''
In Windows muss man so die packages installieren
py -m pip install pandas
'''
import datetime as dt
from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import pandas_datareader.data as pdr

style.use('ggplot')

start = dt.datetime(2015, 1, 1)
end = dt.datetime.now()

