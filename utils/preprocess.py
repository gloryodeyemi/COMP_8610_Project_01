import numpy as np
import pandas as pd
from sklearn import preprocessing

def preprocess_dataset(data):
    # get dataset info
    print("Info:")
    print("-----")
    print(data.info())
    print("")

    # get shape
    print("Shape:")
    print("------")
    print(data.shape)
    print("")

    # statistical summary of the dat
    print("Statistical summary:")
    print("--------------------")
    print(data.describe())
    print("")

    # check for missing values
    print("Missing values:")
    print("---------------")
    print(data.isnull().any())
    print("")

    # check for duplicate data
    print("Duplicate data:")
    print("---------------")
    duplicate = data.duplicated().sum()
    print(duplicate)
    print("")

    # remove duplicate data (if any)
    if (duplicate > 0):
        print("Removing duplicate data:")
        print("------------------------")
        data.drop_duplicates(keep='first', inplace=True)
        print("Duplicate data removed.")
        print("")

    # count target variable values
    print("Target variable percentage, Yes=1, No=0:\n----------------------------------------"+
          "\n", data['target'].value_counts(normalize=True) * 100)
    print("")
    print("Target variable count, Yes=1, No=0:\n----------------------------------"+
          "\n", data['target'].value_counts())
    print("")

    return data

