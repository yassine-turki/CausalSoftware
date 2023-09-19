
from common.load_data import load_and_check_data

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import pandas as pd
import pydot
import sys
import json
import pickle

import dowhy
from dowhy import CausalModel
from dowhy.causal_estimators import (
    causalml,
    distance_matching_estimator,
    econml,
    generalized_linear_model_estimator,
    instrumental_variable_estimator,
    linear_regression_estimator,
    propensity_score_estimator,
    propensity_score_matching_estimator,
    propensity_score_stratification_estimator,
    propensity_score_weighting_estimator,
    regression_discontinuity_estimator,
    regression_estimator,
    two_stage_regression_estimator
)
import statsmodels.api as sm





def adjacency_matrix_to_gml(graph_list, labels):

    """
    Given a graph_list (list with a string and a causal graph) and the labels corresponding to the labels of nodes, return the same graph in gml format

    """
    algorithm_used = graph_list[0]
    graph = graph_list[1] 

    if algorithm_used == "pc_gcastle" or algorithm_used == "ges_gcastle":
        matrix = graph.causal_matrix
        G = nx.DiGraph()
        for n in labels:
            G.add_node(n)
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if matrix[i][j]==1 and matrix[j][i]==1:
                    continue
                elif matrix[i][j]==1:
                    G.add_edge(labels[i],labels[j])
        gml = list(nx.generate_gml(G))
        return gml
    
    elif algorithm_used == "pc_causal" or algorithm_used == "ges_causal":
        if algorithm_used == "pc_causal":
            matrix = graph.G.graph
        elif algorithm_used == "ges_causal":
            matrix = graph['G'].graph 
        G = nx.DiGraph()
        for n in labels:
            G.add_node(n)
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if (matrix[i][j]== 1 and matrix[j][i]== 1) or (matrix[i][j]== -1 and matrix[j][i]== -1):
                    continue
                elif matrix[i][j]== -1 and matrix[j][i]== 1:
                    G.add_edge(labels[i],labels[j])
        gml = list(nx.generate_gml(G))
        return gml


def create_model(graph_list, df, treatment, outcome, common_causes = None, instruments = None, effect_modifiers = None, estimand_type = "nonparametric-ate", proceed_when_unidentifiable = True, missing_nodes_as_confounders = False, identify_vars = False):

    """
            :param graph_list : a list containing a string and CausalGraph object
            :param df: a pandas dataframe containing treatment, outcome and other variables.
            :param treatment: name of the treatment variable
            :param outcome: name of the outcome variable
            :param common_causes: names of common causes of treatment and _outcome. Only used when graph is None.
            :param instruments: names of instrumental variables for the effect of treatment on outcome. Only used when graph is None.
            :param effect_modifiers: names of variables that can modify the treatment effect. If not provided, then the causal graph is used to find the effect modifiers. Estimators will return multiple different estimates based on each value of effect_modifiers.
            :param estimand_type: the type of estimand requested (currently only "nonparametric-ate" is supported). In the future, may support other specific parametric forms of identification.
            :param proceed_when_unidentifiable: does the identification proceed by ignoring potential unobserved confounders. Binary flag.
            :param missing_nodes_as_confounders: Binary flag indicating whether variables in the dataframe that are not included in the causal graph, should be  automatically included as confounder nodes.
            :param identify_vars: Variable deciding whether to compute common causes, instruments and effect modifiers while initializing the class. identify_vars should be set to False when user is providing common_causes, instruments or effect modifiers on their own(otherwise the identify_vars code can override the user provided values). Also it does not make sense if no graph is given.
            :returns: an instance of CausalModel class

    """
    labels = df.columns
    if treatment not in labels:
        print(treatment, "not in dataset. Please try another treatment variable name")
    if outcome not in labels:
        print(outcome, "not in dataset. Please try another outcome variable name")
 
    gml = adjacency_matrix_to_gml(graph_list, labels)
    model=CausalModel(
            data = df,
            treatment=[treatment],
            outcome=[outcome],
            graph="".join(gml),
            common_causes=common_causes,
            instruments=instruments,
            effect_modifiers= effect_modifiers,
            estimand_type= estimand_type,
            proceed_when_unidentifiable=proceed_when_unidentifiable,
            missing_nodes_as_confounders=missing_nodes_as_confounders,
            identify_vars=identify_vars,
            )
    return model


def find_effect(model, estimand_type=None, method_name="default", proceed_when_unidentifiable=True, optimize_backdoor=False):

    """Identify the causal effect to be estimated, using properties of the causal graph.

    :param model: A dowhy model class instance
    :param method_name: Method name for identification algorithm. ("id-algorithm" or "default")
    :param proceed_when_unidentifiable: Binary flag indicating whether identification should proceed in the presence of (potential) unobserved confounders.
    :returns: a probability expression (estimand) for the causal effect if identified, else NULL

    """
    return model.identify_effect(estimand_type = estimand_type, method_name = method_name, proceed_when_unidentifiable = proceed_when_unidentifiable, optimize_backdoor = optimize_backdoor)


def compute_effect_dowhy(model, identified_estimand,
                   continuous_outcome=False,
                   method_name=None,
                   control_value=0,
                   treatment_value=1,
                   test_significance=None,
                   evaluate_effect_strength=False,
                   confidence_intervals=False,
                   target_units="ate",
                   effect_modifiers=None,
                   fit_estimator=True,
                   method_params=None):

    """Estimate the identified causal effect.

    Currently requires an explicit method name to be specified. Method names follow the convention of identification method followed by the specific estimation method: "[backdoor/iv].estimation_method_name". Following methods are supported.
        * Propensity Score Matching: "backdoor.propensity_score_matching"
        * Propensity Score Stratification: "backdoor.propensity_score_stratification"
        * Propensity Score-based Inverse Weighting: "backdoor.propensity_score_weighting"
        * Linear Regression: "backdoor.linear_regression"
        * Generalized Linear Models (e.g., logistic regression): "backdoor.generalized_linear_model"
        * Instrumental Variables: "iv.instrumental_variable"
        * Regression Discontinuity: "iv.regression_discontinuity"

    In addition, you can directly call any of the EconML estimation methods. The convention is "backdoor.econml.path-to-estimator-class". For example, for the double machine learning estimator ("DML" class) that is located inside "dml" module of EconML, you can use the method name, "backdoor.econml.dml.DML". CausalML estimators can also be called. See this demo notebook: https://py-why.github.io/dowhy/example_notebooks/dowhy-conditional-treatment-effects.html.

    :param model: A dowhy_model.
    :param identified_estimand: A probability expression that represents the effect to be estimated. Output of CausalModel.identify_effect method.
    :param continuous_outcome: Bool to specify if the outcome is continuous.
    :param method_name: Name of the estimation method to be used. If not specified:
        - If backdoor is available:
            - If the outcome is continuous, method_name = "backdoor.linear_regression"
            - If the outcome is binary, method_name = "backdoor.propensity_score_stratification"
        - If no backdoor estimand:
            - If IV exists, method_name = "iv.instrumental_variable"
    :param control_value: Value of the treatment in the control group, for effect estimation. If treatment is multi-variate, this can be a list.
    :param treatment_value: Value of the treatment in the treated group, for effect estimation. If treatment is multi-variate, this can be a list.
    :param test_significance: Binary flag on whether to additionally do a statistical significance test for the estimate.
    :param evaluate_effect_strength: (Experimental) Binary flag on whether to estimate the relative strength of the treatment's effect. This measure can be used to compare different treatments for the same outcome (by running this method with different treatments sequentially).
    :param confidence_intervals: (Experimental) Binary flag indicating whether confidence intervals should be computed.
    :param target_units: (Experimental) The units for which the treatment effect should be estimated. This can be of three types. (1) A string for common specifications of target units (namely, "ate", "att", and "atc"), (2) a lambda function that can be used as an index for the data (pandas DataFrame), or (3) a new DataFrame that contains values of the effect_modifiers and effect will be estimated only for this new data.
    :param effect_modifiers: Names of effect modifier variables can be (optionally) specified here too, since they do not affect identification. If None, the effect_modifiers from the CausalModel are used.
    :param fit_estimator: Boolean flag on whether to fit the estimator. Setting it to False is useful to estimate the effect on new data using a previously fitted estimator.
    :param method_params: Dictionary containing any method-specific parameters. These are passed directly to the estimating method. See the docs for each estimation method for allowed method-specific params.
    :returns: An instance of the CausalEstimate class, containing the causal effect estimate and other method-dependent information.
    """
    if method_name is None:
        if len(identified_estimand.backdoor_variables) != 0:  # If we have backdoor estimand
            if continuous_outcome:
                method_name = "backdoor.linear_regression"
            else:
                method_name = "backdoor.propensity_score_stratification"
        else:  # If no backdoor estimands
            if len(identified_estimand.instrumental_variables) != 0:  # If we have IV
                method_name = "iv.instrumental_variable"
            else:
                raise ValueError("Please specify method name")
    if method_name == "backdoor.generalized_linear_model":
      method_params = {"glm_family":sm.families.Binomial()}

    return model.estimate_effect(
        identified_estimand=identified_estimand,
        method_name=method_name,
        control_value=control_value,
        treatment_value=treatment_value,
        test_significance=test_significance,
        evaluate_effect_strength=evaluate_effect_strength,
        confidence_intervals=confidence_intervals,
        target_units=target_units,
        effect_modifiers=effect_modifiers,
        fit_estimator=fit_estimator,
        method_params=method_params
    )

def compute_estimates_dowhy(graph_list, data, treatment, outcome, method_name):
    """
    graph_list : a list of a string and a causalGraph object
    data: a pandas dataframe
    treatment : treatment variable name
    outcome : outcome variable name
    method_name : from compute_effect function :
    * Propensity Score Matching: "backdoor.propensity_score_matching"
    * Propensity Score Stratification: "backdoor.propensity_score_stratification"
    * Propensity Score-based Inverse Weighting: "backdoor.propensity_score_weighting"
    * Linear Regression: "backdoor.linear_regression"
    * Generalized Linear Models (e.g., logistic regression): "backdoor.generalized_linear_model"
    * Instrumental Variables: "iv.instrumental_variable"
    * Regression Discontinuity: "iv.regression_discontinuity"

    """

    dict_of_estimates = {"ATT":0, "ATC":0, "ATE":0}
    #ATT
    model = create_model(graph_list, data, treatment = treatment, outcome = outcome, estimand_type = "nonparametric-ate")
    id = find_effect(model)
    estimate_att = compute_effect_dowhy(model, id, method_name = method_name, target_units = "att")
    dict_of_estimates["ATT"]=estimate_att.value
    #ATC
    estimate_atc = compute_effect_dowhy(model, id, method_name = method_name, target_units = "atc")
    dict_of_estimates["ATC"]=estimate_atc.value
    #ATE
    estimate_ate = compute_effect_dowhy(model, id, method_name = method_name, target_units = "ate")
    dict_of_estimates["ATE"]=estimate_ate.value

    return dict_of_estimates


def compute_direct_effect_dowhy(graph_list, data, treatment, outcome, estimator = "linear_regression_estimator"):

    """ Computes the direct effect from the treatment to the outcome
    # graph_list : a list of a string and a causalgraph object
    # data : a pandas dataframe
    # treatment : the treatment variable
    # outcome : the outcome variable
    # estimator : estimator to compute the direct effect. Default: linear regression

    Other choices :
        - distance_matching_estimator :Simple matching estimator for binary treatments based on a distance
        metric.

        - generalized_linear_model_estimator : Compute effect of treatment using a generalized linear model such as logistic regression.
        Implementation uses statsmodels.api.GLM.
        Needs an additional parameter, "glm_family" to be specified in method_params.
        The value of this parameter can be any valid statsmodels.api families object.
        For example, to use logistic regression, specify "glm_family" as statsmodels.api.families.Binomial().

        - instrumental_variable_estimator : Compute effect of treatment using the instrumental variables method.

        - linear_regression_estimator : Compute effect of treatment using linear regression.
        Fits a regression model for estimating the outcome using treatment(s) and confounders. For a univariate treatment, the treatment effect is equivalent to the coefficient of the treatment variable.
        Simple method to show the implementation of a causal inference method that can handle multiple treatments and heterogeneity in treatment. Requires a strong assumption that all relationships from (T, W) to Y are linear.

        - propensity_score_matching_estimator : Estimate effect of treatment by finding matching treated and control
        units based on propensity score.
        Straightforward application of the back-door criterion.

        - propensity_score_stratification_estimator : Estimate effect of treatment by stratifying the data into bins with
        identical common causes.
        Straightforward application of the back-door criterion.

        - propensity_score_weighting_estimator : Estimate effect of treatment by weighing the data by
        inverse probability of occurrence.
        Straightforward application of the back-door criterion.

        - regression_discontinuity_estimator : Compute effect of treatment using the regression discontinuity method.
        Estimates effect by transforming the problem to an instrumental variables
        problem.

        - regression_estimator : Compute effect of treatment using some regression function.
        Fits a regression model for estimating the outcome using treatment(s) and
        confounders.
    """
    model = create_model(graph_list, data, treatment = treatment, outcome = outcome, estimand_type = "nonparametric-nde")

    identified_estimand_nde = model.identify_effect(estimand_type="nonparametric-nde",
                                                proceed_when_unidentifiable=True)
    if len(identified_estimand_nde.get_mediator_variables()) == 0 :  #If there is no mediator
        print("No mediator between treatment and outcome found, NDE is equal to ATE")
        return 
                                                

    if estimator == "distance_matching_estimator":
        causal_estimator = dowhy.causal_estimators.distance_matching_estimator.DistanceMatchingEstimator
    elif estimator == "generalized_linear_model_estimator":
        causal_estimator = dowhy.causal_estimators.generalized_linear_model_estimator.GeneralizedLinearModelEstimator
        causal_estimate_nde = model.estimate_effect(identified_estimand_nde,
                                            method_name="mediation.two_stage_regression",
                                            confidence_intervals=False,
                                            test_significance=False,
                                            method_params = {
                                                'first_stage_model': causal_estimator,
                                                'second_stage_model': causal_estimator,
                                                "glm_family":sm.families.Binomial()
                                            }
                                            )
        return causal_estimate_nde.value

    elif estimator == "instrumental_variable_estimator":
        causal_estimator = dowhy.causal_estimators.instrumental_variable_estimator.InstrumentalVariableEstimator

    elif estimator == "linear_regression_estimator":
        causal_estimator = dowhy.causal_estimators.linear_regression_estimator.LinearRegressionEstimator

    elif estimator == "propensity_score_matching_estimator":
        causal_estimator = dowhy.causal_estimators.propensity_score_matching_estimator.PropensityScoreMatchingEstimator

    elif estimator == "propensity_score_stratification_estimator":
        causal_estimator = dowhy.causal_estimators.propensity_score_stratification_estimator.PropensityScoreStratificationEstimator

    elif estimator == "propensity_score_weighting_estimator":
        causal_estimator = dowhy.causal_estimators.propensity_score_weighting_estimator.PropensityScoreWeightingEstimator

    elif estimator == "regression_discontinuity_estimator":
        causal_estimator = dowhy.causal_estimators.regression_discontinuity_estimator.RegressionDiscontinuityEstimator

    elif estimator == "regression_estimator":
        causal_estimator = dowhy.causal_estimators.regression_estimator.RegressionEstimator

    else:
        raise ValueError("Invalid estimator choice. Please choose one of the valid options.")


    causal_estimate_nde = model.estimate_effect(identified_estimand_nde,
                                            method_name="mediation.two_stage_regression",
                                            confidence_intervals=False,
                                            test_significance=False,
                                            method_params = {
                                                'first_stage_model': causal_estimator,
                                                'second_stage_model': causal_estimator
                                            }
                                            )
    return causal_estimate_nde.value


def compute_indirect_effect_dowhy(graph_list, data, treatment, outcome, estimator = "linear_regression_estimator"):

    """ Computes the direct effect from the treatment to the outcome
    # graph_list : a list with a string and a causal graph Object
    # data : a pandas dataframe
    # treatment : the treatment variable
    # outcome : the outcome variable
        # estimator : estimator to compute the direct effect. Default: linear regression

    Other choices :
        - distance_matching_estimator :Simple matching estimator for binary treatments based on a distance
        metric.

        - generalized_linear_model_estimator : Compute effect of treatment using a generalized linear model such as logistic regression.
        Implementation uses statsmodels.api.GLM.
        Needs an additional parameter, "glm_family" to be specified in method_params.
        The value of this parameter can be any valid statsmodels.api families object.
        For example, to use logistic regression, specify "glm_family" as statsmodels.api.families.Binomial().

        - instrumental_variable_estimator : Compute effect of treatment using the instrumental variables method.

        - linear_regression_estimator : Compute effect of treatment using linear regression.
        Fits a regression model for estimating the outcome using treatment(s) and confounders. For a univariate treatment, the treatment effect is equivalent to the coefficient of the treatment variable.
        Simple method to show the implementation of a causal inference method that can handle multiple treatments and heterogeneity in treatment. Requires a strong assumption that all relationships from (T, W) to Y are linear.

        - propensity_score_matching_estimator : Estimate effect of treatment by finding matching treated and control
        units based on propensity score.
        Straightforward application of the back-door criterion.

        - propensity_score_stratification_estimator : Estimate effect of treatment by stratifying the data into bins with
        identical common causes.
        Straightforward application of the back-door criterion.

        - propensity_score_weighting_estimator : Estimate effect of treatment by weighing the data by
        inverse probability of occurrence.
        Straightforward application of the back-door criterion.

        - regression_discontinuity_estimator : Compute effect of treatment using the regression discontinuity method.
        Estimates effect by transforming the problem to an instrumental variables
        problem.

        - regression_estimator : Compute effect of treatment using some regression function.
        Fits a regression model for estimating the outcome using treatment(s) and
        confounders.

    """
    model = create_model(graph_list, data, treatment = treatment, outcome = outcome, estimand_type = "nonparametric-nie")

    identified_estimand_nie = model.identify_effect(estimand_type="nonparametric-nie",
                                                proceed_when_unidentifiable=True)
    if estimator == "distance_matching_estimator":
        causal_estimator = dowhy.causal_estimators.distance_matching_estimator.DistanceMatchingEstimator
    elif estimator == "generalized_linear_model_estimator":
        causal_estimator = dowhy.causal_estimators.generalized_linear_model_estimator.GeneralizedLinearModelEstimator
        causal_estimate_nie = model.estimate_effect(identified_estimand_nie,
                                            method_name="mediation.two_stage_regression",
                                            confidence_intervals=False,
                                            test_significance=False,
                                            method_params = {
                                                'first_stage_model': causal_estimator,
                                                'second_stage_model': causal_estimator,
                                                "glm_family":sm.families.Binomial()
                                            }
                                            )
        return causal_estimate_nie.value

    elif estimator == "instrumental_variable_estimator":
        causal_estimator = dowhy.causal_estimators.instrumental_variable_estimator.InstrumentalVariableEstimator

    elif estimator == "linear_regression_estimator":
        causal_estimator = dowhy.causal_estimators.linear_regression_estimator.LinearRegressionEstimator

    elif estimator == "propensity_score_matching_estimator":
        causal_estimator = dowhy.causal_estimators.propensity_score_matching_estimator.PropensityScoreMatchingEstimator

    elif estimator == "propensity_score_stratification_estimator":
        causal_estimator = dowhy.causal_estimators.propensity_score_stratification_estimator.PropensityScoreStratificationEstimator

    elif estimator == "propensity_score_weighting_estimator":
        causal_estimator = dowhy.causal_estimators.propensity_score_weighting_estimator.PropensityScoreWeightingEstimator

    elif estimator == "regression_discontinuity_estimator":
        causal_estimator = dowhy.causal_estimators.regression_discontinuity_estimator.RegressionDiscontinuityEstimator

    elif estimator == "regression_estimator":
        causal_estimator = dowhy.causal_estimators.regression_estimator.RegressionEstimator

    else:
        raise ValueError("Invalid estimator choice. Please choose one of the valid options.")

    causal_estimate_nie = model.estimate_effect(identified_estimand_nie,
                                            method_name="mediation.two_stage_regression",
                                            confidence_intervals=False,
                                            test_significance=False,
                                            method_params = {
                                                'first_stage_model': causal_estimator,
                                                'second_stage_model': causal_estimator
                                            }
                                            )
    return causal_estimate_nie.value


# Load graph_list from the pickle file
with open('graph_list.pkl', 'rb') as pickle_file:
    graph_list = pickle.load(pickle_file)
file_path = sys.argv[1]
dowhy_data = pd.read_csv(file_path)
# Compute ATE, ATC, ATT
dict_of_estimates = compute_estimates_dowhy(graph_list,dowhy_data, treatment = sys.argv[2], outcome = sys.argv[3], method_name = sys.argv[5])
print(dict_of_estimates)
#Compute Direct Effect
direct_effect = compute_direct_effect_dowhy(graph_list, dowhy_data, treatment = sys.argv[2], outcome = sys.argv[3], estimator = sys.argv[4])
print("Direct effect from treatment to outcome =", direct_effect)
#Compute Indirect Effect
indirect_effect = compute_indirect_effect_dowhy(graph_list, dowhy_data, treatment = sys.argv[2], outcome = sys.argv[3], estimator = sys.argv[4])
print("Indirect effect from treatment to outcome =", indirect_effect)
