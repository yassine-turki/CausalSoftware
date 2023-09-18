from common.load_data import load_and_check_data

import os
os.environ['CASTLE_BACKEND'] = 'pytorch'

from collections import OrderedDict

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
import dowhy
from dowhy import CausalModel
from graphviz import Digraph



def run_ges_gcastle(data, criterion='bic', method='scatter', k=0.001, N=10):

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


