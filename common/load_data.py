import pandas as pd
import numpy as np
import os

def load_and_check_data(data_file_path, dropna=True, drop_objects=True):
    """
    Loads dataset and checks if there are any NaN values and non numerical data
    Returns an np array of the values in the dataframe, and a list of labels

    file_path: file from where to extract the data
    dropna: bool to indicate if we drop all NaN values. default: False
    drop_objects: bool to indicate if we drop all non numerical data. default: False

    Supports txt files, xlsx files and csv files

    """

    file_extension = os.path.splitext(data_file_path)[-1]
    if file_extension == ".csv":
        data=pd.read_csv(data_file_path)
    elif file_extension == ".txt":
        data=pd.read_csv(data_file_path, sep = " ")
    elif file_extension == ".xlsx":
        data=pd.read_excel(data_file_path)
    cols_containing_nan = []

    # Check for NaN values in each column
    for col in data.columns:
        if data[col].isnull().any():
            cols_containing_nan.append(col)
    if len(cols_containing_nan) !=0:
        print("Columns with missing values:", cols_containing_nan)
        if dropna==True:
            missing_values_per_row = data.isnull().sum(axis=1)
            # Count how many rows have missing values
            rows_with_missing_values = len(missing_values_per_row[missing_values_per_row > 0])
            print("Dropping:"+rows_with_missing_values+ "rows containing missing values")
            data=data.dropna()

    #Check for non numerical data:

    object_columns = data.select_dtypes(include=['object']).columns
    if len(object_columns) > 0:
        print("Columns of object type found:", object_columns)
        if drop_objects==False:
            print("Please remove non numerical data, or set drop_objects to True")
        else:
            print("Dropping object type columns:", object_columns)
            data=data.drop(columns=object_columns)

    return data, data.values, data.columns