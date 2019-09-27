import argparse
import time
from codebase.topic_evaluator import *

def str2bool(v):
    #susendberg's function
    return v.lower() in ("yes","true","t","1")

parser = argparse.ArgumentParser()
parser.add_argument('--is_pickle', type=str2bool, required=True,
                    help='Do documents store in pickle?')
parser.add_argument('--model_type', type=str, required=True,
                    help='Are models created by LFTM(True)?')
parser.add_argument('--documents_path', type=str, required=True,
                    help='the location of documents file')
parser.add_argument('--models_path', type=str, required=True,
                    help='the location of models')
parser.add_argument('--topic_num', type=int, required=True,
                    help='the number of topic')
parser.add_argument('--start_top_n', type=int, required=True,
                    help='range of evaluating topic')
parser.add_argument('--end_top_n', type=int, required=True,
                    help='range of evaluating topic')
parser.add_argument('--step_size', type=int, default=5,
                    help='the step size of evaluating range')
parser.add_argument('--coherence_method', type=str, default="umass",
                    help='which coherence method?')
args = parser.parse_args()

is_pickle = args.is_pickle
model_type = args.model_type
documents_path = args.documents_path
models_path = args.models_path
topic_num = args.topic_num
start_top_n = args.start_top_n
end_top_n = args.end_top_n
step_size = args.step_size
coherence_method = args.coherence_method

cooccurence_matrix, _, compound_dictionary, num_of_documents = documents_to_cooccurence_matrix(file_path=documents_path,
    is_pickle=is_pickle)

model_lists = loop_to_get_each_model(models_path)

for each_model_path in model_lists:

    print(each_model_path)

    if model_type == "LFTM":
        loaded_cluster = load_LFTM_cluster(file_path= each_model_path, cluster_num=topic_num)
        splitted_cluster = split_language(loaded_cluster)
    elif model_type == "Tou":
        loaded_cluster = load_method_cluster(file_path= each_model_path, method_cluster_number=topic_num)
        splitted_cluster = split_language(loaded_cluster)
    elif model_type == "PLDA":
        splitted_cluster = load_plda_cluster(file_path = each_model_path, cluster_num=topic_num)
    elif model_type == "PMLDA":
        splitted_cluster = load_pmlda_cluster(file_path = each_model_path, cluster_num=topic_num)

    ch_topics = [ topic[0] for topic in splitted_cluster if len(topic[0]) > 0 and len(topic[1]) > 0 ]
    en_topics = [ topic[1] for topic in splitted_cluster if len(topic[0]) > 0 and len(topic[1]) > 0 ]

    print("total # of cross-lingual topics:", len(ch_topics))
    
    for each_variation in range(start_top_n, end_top_n, step_size):
        if coherence_method == "umass":
            average_coherence_score = coherence_score(cn_topic=ch_topics, en_topic=en_topics, topk=each_variation,
                cooccurence_matrix=cooccurence_matrix, compound_dictionary=compound_dictionary,
                coherence_method="umass", num_of_documents=num_of_documents)
        elif coherence_method == "npmi":
            average_coherence_score = coherence_score(cn_topic=ch_topics, en_topic=en_topics, topk=each_variation,
            cooccurence_matrix=cooccurence_matrix, compound_dictionary=compound_dictionary,
            coherence_method="npmi", num_of_documents=num_of_documents)
        jcd_score = avg_jaccard_similarity_between_topics(ch_topics, en_topics, each_variation)
        print(each_variation, average_coherence_score, jcd_score)
