from math import sqrt
import numpy as np
import pandas as pd

from AlgorithmImports import *


class TimeSeriesMomentum(QCAlgorithm):
    def initialize(self):
        self.SetStartDate(2000, 1, 1)
        self.SetCash(10000000)