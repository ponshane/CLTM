import pandas as pd
import numpy as np
from sklearn.model_selection import ParameterGrid
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import parfit.parfit as pf
import os
import pickle

def normalize(v):
    norm=np.linalg.norm(v, ord=1)
    if norm==0:
        norm=np.finfo(v.dtype).eps
    return v/norm

def read_theta(theta_file):
    delimiter = " "
    thetas = []
    file = open(theta_file, "r")
    for each_theta in file.readlines():
        thetas.append(each_theta.strip("\n").split(delimiter)[:-1])
    return thetas

def transform_pd_to_numpy_data(Corpus, language):

    '''
    data_count = pd_Corpus.groupby('Data_type').size()
    training_size = data_count["train"]
    testing_size = data_count["test"]
    dev_size = data_count["dev"]
    input_feature_size = len(pd_Corpus["theta"][0])
    '''

    class_dictionary = {}
    class_count = Corpus.groupby('Class').size()
    for key, _ in class_count.iteritems():
        class_dictionary[key] = len(class_dictionary)

    class_size = len(class_dictionary)

    Data_Object = {}
    Data_Object["x_train"] = np.array(Corpus[(Corpus["Data_type"] == "train") &
                                            (Corpus["language"] == "English")]["theta"].values.tolist(), dtype="float")
    Data_Object["y_train"] = np.array(Corpus[(Corpus["Data_type"] == "train") &
                                            (Corpus["language"] == "English")]["Class"])
    Data_Object["x_dev"] = np.array(Corpus[(Corpus["Data_type"] == "dev") &
                                            (Corpus["language"] == "English")]["theta"].values.tolist(), dtype="float")
    Data_Object["y_dev"] = np.array(Corpus[(Corpus["Data_type"] == "dev") &
                                            (Corpus["language"] == "English")]["Class"])
    '''
    Data_Object["x_test"] = np.array(Corpus[Corpus["Data_type"] == "test"]["theta"].values.tolist())
    Data_Object["y_test"] = keras.utils.to_categorical(np.array(Corpus[Corpus["Data_type"] == "test"]["Class"].apply(
                                lambda x: class_dictionary[x]).tolist()), num_classes=class_size)
    '''
    Data_Object["x_test"] = np.array(Corpus[(Corpus["Data_type"] == "test") &
                                            (Corpus["language"] == language)]["theta"].values.tolist(), dtype="float")
    Data_Object["y_test"] = np.array(Corpus[(Corpus["Data_type"] == "test") &
                                                                       (Corpus["language"] == language)]["Class"])

    return Data_Object, class_dictionary

def evaluate_theta_of_CLTM_on_MLDoc(corpus, language, theta_file_list):
    
    """
    theta_file_list = ["./out/Experiment_results/CLTM/2018-11-27/10dim-MLDoc-engAndchi-LFLDA10T100I1e-1beta.theta",
                      "./out/Experiment_results/CLTM/2018-11-27/20dim-MLDoc-engAndchi-LFLDA10T100I1e-1beta.theta",
                      "./out/Experiment_results/CLTM/2018-11-27/30dim-MLDoc-engAndchi-LFLDA10T100I1e-1beta.theta",
                      "./out/Experiment_results/CLTM/2018-11-27/40dim-MLDoc-engAndchi-LFLDA10T100I1e-1beta.theta",
                      "./out/Experiment_results/CLTM/2018-11-27/50dim-MLDoc-engAndchi-LFLDA10T100I1e-1beta.theta",
                      "./out/Experiment_results/CLTM/2018-11-27/60dim-MLDoc-engAndchi-LFLDA10T100I1e-1beta.theta",
                      "./out/Experiment_results/CLTM/2018-11-27/70dim-MLDoc-engAndchi-LFLDA10T100I1e-1beta.theta",
                      "./out/Experiment_results/CLTM/2018-11-27/80dim-MLDoc-engAndchi-LFLDA10T100I1e-1beta.theta",
                      "./out/Experiment_results/CLTM/2018-11-27/90dim-MLDoc-engAndchi-LFLDA10T100I1e-1beta.theta"]
    """

    # Corpus load back and  describe class column
    Corpus = pd.read_pickle(corpus)

    grid = {
        'C': [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1e0],
        'penalty': ['l2'],
        'n_jobs': [-1],
        'solver': ['lbfgs'],
        'multi_class': ['ovr']
    }

    paramGrid = ParameterGrid(grid)
    results = []

    for each_theta_file in theta_file_list:
        each_corpus = Corpus.copy()

        CLTM_theta = read_theta(each_theta_file)
        each_corpus["theta"] = CLTM_theta
        Data_Object, _ = transform_pd_to_numpy_data(each_corpus, language)

        bestModel, bestScore, _, _ = pf.bestFit(LogisticRegression, paramGrid,
                   Data_Object["x_train"], Data_Object["y_train"], Data_Object["x_dev"], Data_Object["y_dev"],
                   metric = accuracy_score, scoreLabel = "Accuracy", showPlot=False)

        #print(bestModel, bestScore)
        acc = bestModel.score(X=Data_Object["x_test"], y=Data_Object["y_test"])
        _, file = os.path.split(each_theta_file)
        results.append({"file":file, "dev_acc":bestScore, "test_acc":acc})
    
    return results
    
    """
    df = pd.DataFrame.from_dict(results)
    df.to_csv(output)
    """
    
    '''
    TODO:
    1.define a panda dataframe to keep result,
    then export to_csv for visualization
    csv_formats: "topic_num", "input_dim", "test_acc"

    2. variablize language, corpus pd file, output file
    '''

def evaluate_theta_of_JointLDA_on_MLDoc(corpus, language, theta_file_list, dictionary):

    # Corpus load back and  describe class column
    Corpus = pd.read_pickle(corpus)

    grid = {
        'C': [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1e0],
        'penalty': ['l2'],
        'n_jobs': [-1],
        'solver': ['lbfgs'],
        'multi_class': ['ovr']
    }

    paramGrid = ParameterGrid(grid)
    results = []

    for each_theta_file in theta_file_list:
        each_corpus = Corpus.copy()
        
        with open(each_theta_file, 'rb') as handle:
            model = pickle.load(handle)
            
        with open(dictionary, 'rb') as handle:
            source_idx, target_idx = pickle.load(handle)
        
        thetas = []
        for idx, row in each_corpus.iterrows():
            if idx in source_idx.keys():
                thetas.append(normalize(model.DT[source_idx[idx], :]))
            elif idx in target_idx.keys():
                thetas.append(normalize(model.DT[(len(source_idx) + target_idx[idx]), :]))
        
        each_corpus["theta"] = thetas
        Data_Object, _ = transform_pd_to_numpy_data(each_corpus, language)

        bestModel, bestScore, _, _ = pf.bestFit(LogisticRegression, paramGrid,
                   Data_Object["x_train"], Data_Object["y_train"], Data_Object["x_dev"], Data_Object["y_dev"],
                   metric = accuracy_score, scoreLabel = "Accuracy", showPlot=False)

        #print(bestModel, bestScore)
        acc = bestModel.score(X=Data_Object["x_test"], y=Data_Object["y_test"])
        _, file = os.path.split(each_theta_file)
        results.append({"file":file, "dev_acc":bestScore, "test_acc":acc})
    
    return results

def evaluate_theta_of_PMLDA_on_MLDoc(corpus, language, theta_file_list, dictionary):

    # Corpus load back and  describe class column
    Corpus = pd.read_pickle(corpus)

    grid = {
        'C': [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1e0],
        'penalty': ['l2'],
        'n_jobs': [-1],
        'solver': ['lbfgs'],
        'multi_class': ['ovr']
    }

    paramGrid = ParameterGrid(grid)
    results = []

    for each_theta_file in theta_file_list:
        each_corpus = Corpus.copy()
        
        with open(each_theta_file, 'rb') as handle:
            _, source_matrix, target_matrix = pickle.load(handle)
            
        with open(dictionary, 'rb') as handle:
            source_idx, target_idx = pickle.load(handle)
        
        thetas = []
        for idx, row in each_corpus.iterrows():
            if idx in source_idx.keys():
                thetas.append(normalize(source_matrix[source_idx[idx]]))
            elif idx in target_idx.keys():
                thetas.append(normalize(target_matrix[target_idx[idx]]))
        
        each_corpus["theta"] = thetas
        Data_Object, _ = transform_pd_to_numpy_data(each_corpus, language)

        bestModel, bestScore, _, _ = pf.bestFit(LogisticRegression, paramGrid,
                   Data_Object["x_train"], Data_Object["y_train"], Data_Object["x_dev"], Data_Object["y_dev"],
                   metric = accuracy_score, scoreLabel = "Accuracy", showPlot=False)

        #print(bestModel, bestScore)
        acc = bestModel.score(X=Data_Object["x_test"], y=Data_Object["y_test"])
        _, file = os.path.split(each_theta_file)
        results.append({"file":file, "dev_acc":bestScore, "test_acc":acc})
    
    return results

def evaluate_theta_of_MTAnchor_on_MLDoc(corpus, language, theta_file_list, dictionary):

    # Corpus load back and  describe class column
    Corpus = pd.read_pickle(corpus)

    grid = {
        'C': [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1e0],
        'penalty': ['l2'],
        'n_jobs': [-1],
        'solver': ['lbfgs'],
        'multi_class': ['ovr']
    }

    paramGrid = ParameterGrid(grid)
    results = []

    for each_theta_file in theta_file_list:
        each_corpus = Corpus.copy()
        
        with open(each_theta_file, 'rb') as handle:
            _, _, source_matrix, target_matrix = pickle.load(handle)
            
        with open(dictionary, 'rb') as handle:
            source_idx, target_idx = pickle.load(handle)
        
        thetas = []
        for idx, row in each_corpus.iterrows():
            if idx in source_idx.keys():
                thetas.append(normalize(source_matrix[source_idx[idx]]))
            elif idx in target_idx.keys():
                thetas.append(normalize(target_matrix[target_idx[idx]]))
        
        each_corpus["theta"] = thetas
        Data_Object, _ = transform_pd_to_numpy_data(each_corpus, language)

        bestModel, bestScore, _, _ = pf.bestFit(LogisticRegression, paramGrid,
                   Data_Object["x_train"], Data_Object["y_train"], Data_Object["x_dev"], Data_Object["y_dev"],
                   metric = accuracy_score, scoreLabel = "Accuracy", showPlot=False)

        #print(bestModel, bestScore)
        acc = bestModel.score(X=Data_Object["x_test"], y=Data_Object["y_test"])
        _, file = os.path.split(each_theta_file)
        results.append({"file":file, "dev_acc":bestScore, "test_acc":acc})
    
    return results

if __name__ == '__main__':
    pass
