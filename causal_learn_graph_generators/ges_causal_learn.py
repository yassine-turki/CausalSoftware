

# Imports

from common.load_data import load_and_check_data
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
from causallearn.search.ConstraintBased.PC import pc
from causallearn.graph.GraphClass import CausalGraph
from causallearn.graph.GraphNode import GraphNode
import networkx as nx
from graphviz import Digraph




def run_ges_causal_learn(data, score_func = "local_score_BIC", maxP = None, parameters = None):

  """ Comments taken directly from causal learn library
  data: numpy.ndarray, shape (n_samples, n_features). Data, where n_samples is the number of samples and n_features is the number of features.

  score_func: The score function you would like to use, including (see score_functions.). Default: 'local_score_BIC'.
    "local_score_BIC": BIC score.

    "local_score_BDeu": BDeu score.

    "local_score_CV_general": Generalized score with cross validation for data with single-dimensional variables.

    "local_score_marginal_general": Generalized score with marginal likelihood for data with single-dimensional variables.

    "local_score_CV_multi": Generalized score with cross validation for data with multi-dimensional variables.

    "local_score_marginal_multi": Generalized score with marginal likelihood for data with multi-dimensional variables.

  maxP: Allowed maximum number of parents when searching the graph. Default: None.

  parameters: Needed when using CV likelihood. Default: None.
    parameters['kfold']: k-fold cross validation.

    parameters['lambda']: regularization parameter.

    parameters['dlabel']: for variables with multi-dimensions, indicate which dimensions belong to the i-th variable.

    For example, parameters = {"kfold":10, "lambda":0.01}

"""

  # Run the algorithm

  cg = ges(data, score_func = score_func, maxP = maxP, parameters = parameters)

  return cg

