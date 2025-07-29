import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

with open("./simulation-data/sim1.csv", "r", newline="") as file:
    data = file.read()