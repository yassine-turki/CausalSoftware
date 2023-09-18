"""# Imports"""

import os
os.environ['CASTLE_BACKEND'] = 'pytorch'

from collections import OrderedDict
from common.load_data import load_and_check_data
import numpy as np
import networkx as nx
import pandas as pd
import pydot

import castle
from castle.common import GraphDAG
from castle.metrics import MetricsDAG
from castle.datasets import IIDSimulation, DAG
from castle.algorithms import PC, GES, ICALiNGAM, GOLEM
from castle.common.priori_knowledge import PrioriKnowledge

import matplotlib.pyplot as plt
from graphviz import Digraph


"""#Functions for PC"""


def run_pc_gcastle(data, labels, variant="original", alpha=0.05, ci_test="fisherz", priori_knowledge=None):
  """
  data: a numpy array

  variant : str
  A variant of PC-algorithm, one of [`original`, `stable`, `parallel`].

  alpha: float, default 0.05  Significance level.

  ci_test : str, callable ci_test method, if str, must be one of [`fisherz`, `g2`, `chi2`]
  See more: `castle.common.independence_tests.CITest`

  priori_knowledge: PrioriKnowledge: a class object PrioriKnowledge. In this case, we will just use a list of lists, symbolizing tiers.

  For example: [["income", "relationship"],["Sex", "workclass"]]. The first sublist is tier 1, the second tier 2. No path from tier 2 to tier 1 can exist.

  """
  if priori_knowledge is not None:
    node_map = {}
    for i, label in enumerate(labels):
        node_map[label] = i
    priori = PrioriKnowledge(data.shape[1])

    for i in range(len(priori_knowledge) - 1, 0, -1):  # iterating through tiers in reverse order
        for j in range(len(priori_knowledge[i])):  # iterating through the nodes in the current tier
            for k in range(i - 1, -1, -1):  # iterate over the previous tiers in reverse order
                for p in range(len(priori_knowledge[k])):  # iterating through the previous tier nodes
                    priori.add_forbidden_edge(node_map[priori_knowledge[i][j]], node_map[priori_knowledge[k][p]])

    # Adding forbidden edges within the same tier
    for i in range(len(priori_knowledge)):
        for j in range(len(priori_knowledge[i]) - 1):
            for k in range(j + 1, len(priori_knowledge[i])):
                priori.add_forbidden_edge(node_map[priori_knowledge[i][j]], node_map[priori_knowledge[i][k]])
                priori.add_forbidden_edge(node_map[priori_knowledge[i][k]], node_map[priori_knowledge[i][j]])

  else:
    priori = None


  pc = PC(variant = variant, alpha = alpha, ci_test = ci_test, priori_knowledge = priori)
  pc.learn(data)
  return pc

