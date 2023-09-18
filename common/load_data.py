import pandas as pd
import numpy as np

def load_and_check_data(data_file_path, dropna=False, drop_objects=False):
  """
  Loads dataset and checks if there are any NaN values and non numerical data
  Returns an np array of the values in the dataframe, and a list of labels

  file_path: file from where to extract the data
  dropna: bool to indicate if we drop all NaN values. default: False
  drop_objects: bool to indicate if we drop all non numerical data. default: False

  """

  data=pd.read_csv(data_file_path)
  cols_containing_nan = []

  # Check for NaN values in each column
  for col in data.columns:
      if data[col].isnull().any():
          cols_containing_nan.append(col)
  if len(cols_containing_nan) !=0:
    print("Columns with missing values:", cols_containing_nan)
    if dropna==False:
      print("Please remove missing values, or set dropna to True")
      return None
    else:
      data=data.dropna()

  #Check for non numerical data:

  object_columns = data.select_dtypes(include=['object']).columns
  if len(object_columns) > 0:
    print("Columns of object type found:", object_columns)
    if drop_objects==False:
      print("Please remove non numerical data, or set drop_objects to True")
    else:
      data=data.drop(columns=object_columns)

  return data.values, data.columns