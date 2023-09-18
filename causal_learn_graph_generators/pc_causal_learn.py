from common.load_data import load_and_check_data

from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.GraphUtils import GraphUtils
import pandas as pd
import numpy as np
import graphviz
import pydot
from causallearn.utils.cit import fisherz, mv_fisherz
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.utils.PCUtils.BackgroundKnowledgeOrientUtils import \
    orient_by_background_knowledge
from causallearn.graph.GraphClass import CausalGraph
from causallearn.graph.GraphNode import GraphNode
import networkx as nx
import dowhy
from dowhy import CausalModel
import dowhy.datasets
from graphviz import Digraph



def run_pc_causal_learn(data,labels,alpha=0.05, indep_test='fisherz', stable=True, uc_rule=0, uc_priority=2, mvpc=False, correction_name='MV_Crtn_Fisher_Z', background_knowledge=None, verbose=False, show_progress=True):

  """
  data: numpy.ndarray, shape (n_samples, n_features). Data, where n_samples is the number of samples and n_features is the number of features.

  alpha: desired significance level (float) in (0, 1). Default: 0.05.

  indep_test: string, name of the independence test method. Default: ‘fisherz’.
    “fisherz”: Fisher’s Z conditional independence test.

    “chisq”: Chi-squared conditional independence test.

    “gsq”: G-squared conditional independence test.

    “kci”: kernel-based conditional independence test. (As a kernel method, its complexity is cubic in the sample size, so it might be slow if the same size is not small.)

    “mv_fisherz”: Missing-value Fisher’s Z conditional independence test.

  stable: run stabilized skeleton discovery if True. Default: True.

  uc_rule: how unshielded colliders are oriented. Default: 0.
    0: run uc_sepset.

    1: run maxP. Orient an unshielded triple X-Y-Z as a collider with an additional CI test.

    2: run definiteMaxP. Orient only the definite colliders in the skeleton and keep track of all the definite non-colliders as well.

  uc_priority: rule of resolving conflicts between unshielded colliders. Default: 2.
    -1: whatever is default in uc_rule.

    0: overwrite.

    1: orient bi-directed.

    2: prioritize existing colliders.

    3: prioritize stronger colliders.

    4: prioritize stronger* colliders.

  mvpc: use missing-value PC or not. Default: False.

  correction_name. Missing value correction if using missing-value PC. Default: ‘MV_Crtn_Fisher_Z’

  background_knowledge: class BackgroundKnowledge. Add prior edges according to assigned causal connections. Default: None. For detailed usage, please kindly refer to its usage example.

  verbose: True iff verbose output should be printed. Default: False.

  show_progress: True iff the algorithm progress should be show in console. Default: True.

  return cg : a CausalGraph object, where cg.G.graph[j,i]=1 and cg.G.graph[i,j]=-1 indicate i –> j; cg.G.graph[i,j] = cg.G.graph[j,i] = -1 indicate i — j; cg.G.graph[i,j] = cg.G.graph[j,i] = 1 indicates i <-> j.


"""

  if background_knowledge is not None: # Create a background knowledge object

  # Run the algorithm once without background knowledge to get the nodes (to be changed later)
    cg_without_background_knowledge=pc(data,alpha=alpha, indep_test=indep_test, stable=stable, uc_rule=uc_rule, uc_priority=uc_priority, mvpc=mvpc, correction_name=correction_name, background_knowledge=None, verbose=False, show_progress=False)
    nodes = cg_without_background_knowledge.G.get_nodes()

    node_map = {} #Provide a mapping between names and nodes
    for i, label in enumerate(labels):
        node_map[label] = nodes[i]

    bk = BackgroundKnowledge()
    for i, sublist in enumerate(background_knowledge): #create tiers
      for j in range(len(sublist)):
        node = node_map[sublist[j]]
        bk.add_node_to_tier(node, i)

  else:
    bk = None

  # Run the algorithm

  cg = pc(data,alpha=alpha, indep_test=indep_test, stable=stable, uc_rule=uc_rule, uc_priority=uc_priority, mvpc=mvpc, correction_name=correction_name, background_knowledge=bk, verbose=verbose, show_progress=show_progress)

  return cg
