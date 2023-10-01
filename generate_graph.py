#global imports
import sys
import json
import pandas as pd
import numpy as np
import graphviz
import pydot
from IPython.display import Image, display
import networkx as nx
from graphviz import Digraph
import matplotlib.pyplot as plt
import pickle


#imports for causallearn

from causallearn.search.ScoreBased.GES import ges
from causallearn.utils.GraphUtils import GraphUtils
from causallearn.utils.cit import fisherz, mv_fisherz
from causallearn.graph.GraphClass import CausalGraph
from causallearn.graph.GraphNode import GraphNode
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.utils.PCUtils.BackgroundKnowledgeOrientUtils import \
    orient_by_background_knowledge

#imports for gcastle
import os
os.environ['CASTLE_BACKEND'] = 'pytorch'

import traceback

from collections import OrderedDict
import castle
from castle.common import GraphDAG
from castle.metrics import MetricsDAG
from castle.datasets import IIDSimulation, DAG
from castle.algorithms import PC, GES, ICALiNGAM, GOLEM
from castle.common.priori_knowledge import PrioriKnowledge

#import load_data function

from common.load_data import load_and_check_data


from gcastle_graph_generators.pc_gcastle import run_pc_gcastle
from gcastle_graph_generators.ges_gcastle import run_ges_gcastle
from causal_learn_graph_generators.pc_causal_learn import run_pc_causal_learn
from causal_learn_graph_generators.ges_causal_learn import run_ges_causal_learn


def draw_graph(graph_list, labels, filename=None):
    """
    Draw the pydot graph
    graph_list: a list containing a string and CausalGraph object 
    labels: data.columns
    filename: if you wish to save the file
    """
    algorithm_used = graph_list[0]
    graph = graph_list[1]

    graph_operations_list = []
    if len(sys.argv) > 3 and len(sys.argv[3]) > 0:
        graph_operations_list = json.loads(sys.argv[3])

    if len(graph_operations_list) > 0:
        print(graph_operations_list)
        for i in range(len(graph_operations_list)): 
            if graph_operations_list[i]["op"] == "add":
                add_path(graph_list, labels, [graph_operations_list[i]["start"], graph_operations_list[i]["end"]])
            if graph_operations_list[i]["op"] == "delete":
                delete_path(graph_list, labels, [graph_operations_list[i]["start"], graph_operations_list[i]["end"]])

    
    learned_graph = nx.DiGraph()
    undirected_paths = set() # Set to check if we have undirected_paths

    if algorithm_used == "pc_gcastle" or algorithm_used == "ges_gcastle":
        adjacency_matrix = graph.causal_matrix
        
        for i in range(len(adjacency_matrix)):
            for j in range(len(adjacency_matrix[0])):
                if adjacency_matrix[i][j] == 1 and adjacency_matrix[j][i] == 0:
                    learned_graph.add_edge(labels[i], labels[j])
                elif (adjacency_matrix[i][j] == 1 and adjacency_matrix[j][i] == 1) and ((labels[i],labels[j]) not in undirected_paths and (labels[j],labels[i]) not in undirected_paths):
                    learned_graph.add_edge(labels[i], labels[j], style="dashed", arrowhead="none")
                    undirected_paths.add((labels[i],labels[j]))
    

    elif algorithm_used == "pc_causal" or algorithm_used == "ges_causal":
        if algorithm_used == "pc_causal":
            adjacency_matrix = graph.G.graph
        elif algorithm_used == "ges_causal":
            adjacency_matrix = graph['G'].graph

        for i in range(len(adjacency_matrix)):
            for j in range(len(adjacency_matrix[0])):
                if adjacency_matrix[i][j] == -1 and adjacency_matrix[j][i] == 1:
                    learned_graph.add_edge(labels[i], labels[j])
                elif ((adjacency_matrix[i][j] == -1 and adjacency_matrix[j][i] == -1) or (adjacency_matrix[i][j] == 1 and adjacency_matrix[j][i] == 1)) and ((labels[i],labels[j]) not in undirected_paths and (labels[j],labels[i]) not in undirected_paths):
                    learned_graph.add_edge(labels[i], labels[j], style="dashed", arrowhead="none")
                    undirected_paths.add((labels[i],labels[j]))

    pos = nx.circular_layout(learned_graph)
    pydot_graph = nx.drawing.nx_pydot.to_pydot(learned_graph)

    png_data = pydot_graph.create_png()
    if filename is None:
        with open("graph.png", "wb") as f:
            f.write(png_data)
    else:
        if not filename.lower().endswith((".png", ".jpeg", ".pdf")):
            filename += ".png"
        with open(filename, "wb") as f:
            f.write(png_data)






def add_path(graph_list, labels, path_to_add):
    """
    graph_list: a list containing a string and CausalGraph object
    labels: labels of the dataset
    path_to_add: a list containing two nodes. For example ["Sex","Race"]. This will add the path from "Sex" to "Race"
    """
    algorithm_used = graph_list[0]
    graph = graph_list[1]

    column_to_index = {col: i for i, col in enumerate(labels)}
    if path_to_add[0] not in column_to_index.keys() or path_to_add[1] not in column_to_index.keys():
        print("Error: given path is not in the labels for the data")
        return graph

    node1_index = column_to_index[path_to_add[0]]
    node2_index = column_to_index[path_to_add[1]]

    if algorithm_used == "pc_gcastle" or algorithm_used == "ges_gcastle":
        graph.causal_matrix[node1_index, node2_index] = 1
    elif algorithm_used == "pc_causal":
        graph.G.graph[node2_index, node1_index] = 1
        graph.G.graph[node1_index, node2_index] = -1
    elif algorithm_used == "ges_causal":
        graph["G"].graph[node2_index, node1_index] = 1
        graph["G"].graph[node1_index, node2_index] = -1

    return graph


    
def delete_path(graph_list, labels, path_to_delete):

    """
    graph_list: a list containing a string and CausalGraph object

    labels: labels of the dataset

    path_to_delete: a list containing two nodes. For example ["Sex","Race"]. This will delete the path from "Sex" to "Race"

    """
    algorithm_used = graph_list[0]
    graph = graph_list[1]

    column_to_index = {col: i for i, col in enumerate(labels)}

    if path_to_delete[0] not in column_to_index.keys() or path_to_delete[1] not in column_to_index.keys():
        print("Error: given path is not in the labels for the data")
        return graph

    node1_index = column_to_index[path_to_delete[0]]
    node2_index = column_to_index[path_to_delete[1]]

    if algorithm_used == "pc_gcastle" or algorithm_used == "ges_gcastle":
        graph.causal_matrix[node1_index, node2_index] = 0
    elif algorithm_used == "pc_causal":
        graph.G.graph[node1_index, node2_index] = 0
    elif algorithm_used == "ges_causal":
        graph["G"].graph[node1_index, node2_index] = 0

    return graph



data_file_path = sys.argv[1]
algorithm_selected = sys.argv[2]

dataframe, data, labels = load_and_check_data(data_file_path, dropna = False, drop_objects = False)

try:
    # Algorithm selection and graph drawing
    if algorithm_selected == "pc_gcastle":
        graph = run_pc_gcastle(data, labels, variant = sys.argv[4], alpha = float(sys.argv[5]), ci_test = sys.argv[6], priori_knowledge = json.loads(sys.argv[7]))

    elif algorithm_selected == "pc_causal": 
        graph = run_pc_causal_learn(data, labels, alpha = float(sys.argv[4]), indep_test = sys.argv[5], stable = bool(json.loads(sys.argv[6])), uc_rule = int(sys.argv[7]), uc_priority = int(sys.argv[8]), mvpc = bool(json.loads(sys.argv[9])), correction_name = sys.argv[10], background_knowledge = json.loads(sys.argv[11]), verbose=False, show_progress=True)

    elif algorithm_selected == "ges_gcastle":
        graph = run_ges_gcastle(data, criterion = sys.argv[4], method = sys.argv[5], k = float(sys.argv[6]), N = int(sys.argv[7]))

    elif algorithm_selected == "ges_causal":

        if json.loads(sys.argv[5]) != None:
            parameter_maxP = int(json.loads(sys.argv[5]))

        optional_parameters = {}
        if json.loads(sys.argv[6]) != None:
            optional_parameters["kfold"] = int(json.loads(sys.argv[6]))

        if json.loads(sys.argv[7]) != None:
            optional_parameters["lambda"] = float(json.loads(sys.argv[7]))

        if json.loads(sys.argv[8]) != None:
            optional_parameters["dlabel"] = int(json.loads(sys.argv[8]))

        graph = run_ges_causal_learn(data, score_func = sys.argv[4], maxP = parameter_maxP, parameters = optional_parameters)

    graph_list = [algorithm_selected, graph]
    draw_graph(graph_list, labels, "static\image.png")

except Exception as e:
    traceback.print_exc()
    with open("error.txt", "w") as error_file:
        error_file.write("-"+str(e) + "\n")

# Save graph_list to a pickle file
with open('graph_list.pkl', 'wb') as pickle_file:
    pickle.dump(graph_list, pickle_file)