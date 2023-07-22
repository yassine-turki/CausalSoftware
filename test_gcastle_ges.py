# -*- coding: utf-8 -*-
"""test_gcastle_ges.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1KiBA7lhrg_EBSkbAVEmDMb7VqDxSauT5

# Imports
"""

import os
os.environ['CASTLE_BACKEND'] = 'pytorch'

from collections import OrderedDict

import numpy as np
import networkx as nx
import pandas as pd
import pydot
from IPython.display import Image, display

import castle
from castle.common import GraphDAG
from castle.metrics import MetricsDAG
from castle.datasets import IIDSimulation, DAG
from castle.algorithms import PC, GES, ICALiNGAM, GOLEM
from castle.common.priori_knowledge import PrioriKnowledge

import matplotlib.pyplot as plt
import pygraphviz
import sys

"""Useful functions"""

def load_and_check_data(file_path, dropna=False, drop_objects=False):
  """
  Loads dataset and checks if there are any NaN values and non numerical data
  Returns an np array of the values in the dataframe, and a list of labels
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

def run_ges(data, criterion='bic', method='scatter', k=0.001, N=10):

  """
  data: a numpy array

  Parameters
    ----------
    criterion: str for DecomposableScore object
        scoring criterion, one of ['bic', 'bdeu'].

        Notes:
            1. 'bdeu' just for discrete variable.
            2. if you want to customize criterion, you must create a class
            and inherit the base class `DecomposableScore` in module
            `ges.score.local_scores`
    method: str
        effective when `criterion='bic'`, one of ['r2', 'scatter'].
    k: float, default: 0.001
        structure prior, effective when `criterion='bdeu'`.
    N: int, default: 10
        prior equivalent sample size, effective when `criterion='bdeu'`
  """

  ges = GES(criterion = criterion, method = method, k = k, N = N)
  ges.learn(data)
  return ges

def draw_graph(graph, labels, filename=None):
    learned_graph = nx.DiGraph(graph.causal_matrix)
    learned_graph.add_nodes_from(learned_graph.nodes())
    learned_graph.add_edges_from(learned_graph.edges())

    mapping = {i: str(labels[i]) for i in learned_graph.nodes()}  # Convert labels to strings
    learned_graph = nx.relabel_nodes(learned_graph, mapping)
    pos = nx.circular_layout(learned_graph)

    agraph = nx.drawing.nx_agraph.to_agraph(learned_graph)
    png_data = agraph.draw(format="png", prog="dot")

    if filename is None:
        display(Image(png_data))
    else:
        if not filename.lower().endswith((".png", ".jpeg", ".pdf")):
            filename += ".png"
        with open(filename, "wb") as f:
            f.write(png_data)
"""
def delete_path(graph, labels, path_to_delete):

  

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

  graph.causal_matrix[node1_index, node2_index] = 0

  return graph

def add_path(graph, labels, path_to_add):

  

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

  graph.causal_matrix[node1_index, node2_index] = 1
  # Remove edge from node2 to node1, if it exists (uncomment the line below if you want that)
  # pc.causal_matrix[node2_to_disconnect, node1_to_disconnect] = 0

  return graph
"""
"""Example : Adult_cleaned_bin"""

file_path= sys.argv[1] 
data, labels = load_and_check_data(file_path)
ges = run_ges(data)
draw_graph(ges, labels, "ges.png")

draw_graph(algo,labels)

ges = add_path(ges, labels, ["workclass", "relationship"])

draw_graph(ges, labels)