# -*- coding: utf-8 -*-
"""Test_CausalLearn_GES.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/13PSB3V1OvmW1PsprogVTIcb9S9UvS_nW

# Imports
"""

from causallearn.search.ScoreBased.GES import ges
from causallearn.utils.GraphUtils import GraphUtils
import pandas as pd
import numpy as np
import graphviz
import pydot
from IPython.display import Image, display
from causallearn.utils.cit import fisherz, mv_fisherz
from causallearn.graph.GraphClass import CausalGraph
from causallearn.graph.GraphNode import GraphNode

import sys

def load_and_check_data(file_path, dropna=False, drop_objects=False):
  """
  Loads dataset and checks if there are any NaN values and non numerical data
  Returns an np array of the values in the dataframe, and a list of labels

  file_path: file from where to extract the data
  dropna: bool to indicate if we drop all NaN values. default: False
  drop_objects: bool to indicate if we drop all non numerical data. default: False

  """

  data=pd.read_csv(file_path,index_col=0)
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

def run_ges(data, score_func = "local_score_BIC", maxP = None, parameters = None):

  """
  data: numpy.ndarray, shape (n_samples, n_features). Data, where n_samples is the number of samples and n_features is the number of features.

  score_func: The score function you would like to use, including (see score_functions.). Default: ‘local_score_BIC’.
    “local_score_BIC”: BIC score.

    “local_score_BDeu”: BDeu score.

    “local_score_cv_general”: Generalized score with cross validation for data with single-dimensional variables.

    “local_score_marginal_general”: Generalized score with marginal likelihood for data with single-dimensional variables.

    “local_score_cv_multi”: Generalized score with cross validation for data with multi-dimensional variables.

    “local_score_marginal_multi”: Generalized score with marginal likelihood for data with multi-dimensional variables.

  maxP: Allowed maximum number of parents when searching the graph. Default: None.

  parameters: Needed when using CV likelihood. Default: None.
    parameters[‘kfold’]: k-fold cross validation.

    parameters[‘lambda’]: regularization parameter.

    parameters[‘dlabel’]: for variables with multi-dimensions, indicate which dimensions belong to the i-th variable.

"""

  # Run the algorithm

  cg = ges(data, score_func = score_func, maxP = maxP, parameters = parameters)

  return cg

def draw_graph_ges(graph, labels, filename=None):
  """
  Draw the pydot graph

  graph: CausalGraph object
  labels: data.columns
  filename: if you wish to save the file
  """


  pyd = GraphUtils.to_pydot(graph['G'], labels = labels)
  if filename==None:
    png_data = pyd.create_png()
    display(Image(png_data))
  else:
    if filename[-3:]=="png":
      pyd.write_png(filename)
    elif filename[-4:]=="jpeg":
      pyd.write_jpeg(filename)
    elif filename[-3:]=="pdf":
      pyd.write_png(filename)
    else:
      pyd.write_png(filename+".png")
"""
def delete_path_ges(graph, labels, path_to_delete):

  
  deletes a selected path

  graph is a CausalGraph object

  labels: labels of the dataset

  path_to_delete: a list containing two nodes. For example ["Sex","Race"]. This will delete the path from "Sex" to "Race"

  

  column_to_index = {col: i for i, col in enumerate(labels)}
  if path_to_delete[0] not in column_to_index.keys() or path_to_delete[1] not in column_to_index.keys():
    print("Error: given path is not in the labels for the data")
    return graph

  # Find the indices of the nodes
  node1_index = column_to_index[path_to_delete[0]]
  node2_index = column_to_index[path_to_delete[1]]

  # Remove the edge from node1 to node2
  graph["G"].graph[node1_index, node2_index] = 0

  return graph

def add_path_ges(graph, labels, path_to_add):

  
  adds a new path

  graph is a CausalGraph object

  labels: labels of the dataset

  path_to_add: a list containing two nodes. For example ["Sex","Race"]. This will add the path from "Sex" to "Race"

  

  column_to_index = {col: i for i, col in enumerate(labels)}
  if path_to_add[0] not in column_to_index.keys() or path_to_add[1] not in column_to_index.keys():
    print("Error: given path is not in the labels for the data")
    return graph

  # Find the indices of the nodes
  node1_index = column_to_index[path_to_add[0]]
  node2_index = column_to_index[path_to_add[1]]

  # Add the edge from node1 to node2
  graph["G"].graph[node2_index, node1_index] = 1
  graph["G"].graph[node1_index, node2_index] = -1


  return graph
"""
def run_ges_and_draw(file_path_data, score_func = "local_score_BIC", maxP = None, parameters = None, file_path_graph = None):

  """
  loads the data, runs the pc algorithm and draws the graph

  """

  dataset, labels = load_and_check_data(file_path_data)
  g=run_ges(dataset, score_func = score_func, maxP = maxP, parameters = parameters)
  draw_graph_ges(g,labels, file_path_graph)
  return g

"""Example: Adult_cleaned_bin"""

file_path= sys.argv[1]
data, labels = load_and_check_data(file_path)
Record = ges(data)
draw_graph_ges(Record, labels, "static\image.png")

#g = delete_path(Record, labels, ["native_country", "race"])

#draw_graph_ges(g, labels)

#g = add_path_ges(g, labels, ["race", "native_country"])

#draw_graph_ges(g, labels)