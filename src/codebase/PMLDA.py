from codebase.LanguageEmbedding import LanguageEmbedding

from gensim import corpora, models
import numpy as np
import pandas as pd
import pickle

import logging
logging.basicConfig(level=logging.INFO)

from sklearn.cluster import AgglomerativeClustering
from gensim.models.wrappers import LdaMallet

def select_representative_word_list(lda_model, top_n_representative_words):
    """
    for representating topic model, we select top P most contributed words in
    each topic and these topic words in a model

    Args:
        lda_model: gensim's lda model
        top_n_representative_words: top_n word to represent a topic
    Returns:
        representative_word_list: a list of list, each inner list contains
            top_n_representative_words of each topic.
    """

    topic_num = lda_model.num_topics

    representative_word_list = list()
    for topic_idx in range(0, topic_num):
        each_topic_word = list()

        contribution_pairs = lda_model.show_topic(topic_idx,
        topn = top_n_representative_words)

        for each_tuple in contribution_pairs:
            each_topic_word.append(each_tuple[0])

        representative_word_list.append(each_topic_word)
    return representative_word_list

def calculated_centroid_of_topics(word_vectors, representative_word_list):
    """
    calculating centroid by averaging vectors of topic word list

    Args:
        word_vectors: a cross-lingual word embedding
        representative_word_list: a list of list, each inner list contains
            top_n_representative_words of each topic..
    Returns:
        centroid_list: list, each element is a numpy object and represent
            centroids of topics
    """

    centroid_list = list()
    dim = word_vectors.vector_size

    # to loop each topic in the representative_word_list
    for topic_idx in range(0, len(representative_word_list)):

        # first create a matrix for this topic.
        FeatureMatrix = np.zeros((len(representative_word_list[topic_idx]),
                                  dim),dtype="float32")

        # this matrix is responsible for storing vector of each word.
        for idx, word in enumerate(representative_word_list[topic_idx]):
            FeatureMatrix[idx] = word_vectors.word_vec(word)

        # taking average to get a centroid of this topic
        centroid = np.mean(FeatureMatrix, axis=0, keepdims=False)
        centroid_list.append(centroid)

    return centroid_list

def hierarchical_clustering(source_centroid_list, target_centroid_list,
 num_of_topic):
    """
    feed into two mono-lingual centroid lists, then apply hierarchical cluster
    to concatenate similar but cross-lingual topics.

    Args:
        source_centroid_list: source language的各主題向量中心
        target_centroid_list: target language的各主題向量中心
        num_of_topic: 想要得到的最後群數
    Return:
        member_of_clusters: to record memberships of mapping clusters
            Key: mapping 後群的id
            Value: 對應的成員(from source centroid or target centroid)
    """

    # to remember the source of centroid
    # "C" for source_centroid_list
    # "E" for target_centroid_list
    node_labels = []
    # to store all centroids
    centers = []

    # loop source centroids and target centroids
    for idx, center in enumerate(source_centroid_list):
        centers.append(center)
        node_labels.append("C"+str(idx))

    for idx, center in enumerate(target_centroid_list):
        centers.append(center)
        node_labels.append("E"+str(idx))

    # Compute AgglomerativeClustering
    # take centers(nnumpy object) as input data
    agg = AgglomerativeClustering(n_clusters=num_of_topic, affinity="cosine",
                                linkage="complete").fit(centers)
    # take results of AgglomerativeClustering
    labels = agg.labels_

    # Number of clusters in labels, ignoring noise if present.
    # 中文：就是階層式分群最後的可能結果，通常是 num_of_topic
    # 但以防萬一才有後段的判斷式
    num_of_mapping_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    # to record memberships of mapping clusters
    # Key: mapping 後群的id
    # Value: 對應的成員(from source centroid or target centroid)
    member_of_clusters = dict()
    total_count = 0

    # loop n_clusters_ to collect mapping member
    for idx in range(num_of_mapping_clusters):
        # idx => 尋訪 mapping 後的每一群
        # node_list => 紀錄 mapping 後的每一群有哪些 mapping 前的主題
        node_list = []

        # cluster_idx => AgglomerativeClustering 判斷屬於 mapping 後的哪一群
        # 例如：idx = 1 (分群後的第2個聯合主題，尋訪 labels 中是第2個聯合主題的成員)
        for l_idx, cluster_idx in enumerate(labels):
            if cluster_idx == idx:
                node_list.append(node_labels[l_idx])

        # 紀錄成員
        member_of_clusters[idx] = node_list
        total_count += len(node_list)

    assert total_count == len(source_centroid_list) + len(target_centroid_list)

    #count how many cluster containing more than 2 node list
    compound_cluster_num = [ len(x) for x in list(member_of_clusters.values())
     if len(x) > 1 ]
    print(compound_cluster_num)

    return member_of_clusters

def merge_word_list(model, topic_list):
    """
    此函式負責將數個同語言的主題字分佈，按字加總機率
    並降冪排序後輸出，並以 pandas series 實做
    """

    word_series = pd.Series()

    # loop every topic in the list
    for topic_idx in topic_list:
        word_idx = []
        probs = []
        for each_word, each_prob in model.show_topic(topicid=topic_idx,
         topn=model.num_terms):
            word_idx.append(each_word)
            probs.append(each_prob)

        # if this is the first topic, creating pandas series
        if word_series.empty:
            word_series = pd.Series(probs, index=word_idx)
        # add probs by words
        else:
            word_series = word_series.add(pd.Series(probs, index=word_idx))

    return word_series.sort_values(ascending=False)

def recalculate_phi_of_each_topic(source_ldamodel, target_ldamodel,
 mapping_results, top_n):
    """
    此函式負責將 hierarchical clustering 的結果(多語言主題結果)，重新計算每一個主題的
    字分佈，並選擇 topN 個字回傳，格式如下：
    [([source language words],[target language words]), .....]

    Args:
        source_ldamodel: lda model(gensim) of source language
        target_ldamodel: lda model(gensim) of target language
        mapping_results: hierarchical clustering 的結果(多語言主題結果)
        top_n: how many words in final word list?
    Returns:
        resultant_topic_word_list: list of list
        #ex. [([source language words],[target language words]), .....]
    """
    resultant_topic_word_list = []
    for topic_idx, member_list in mapping_results.items():
        source_language_topics = [] # precede by C
        target_language_topics = [] # precede by E

        for each_member in member_list:
            if "C" in each_member:
                source_language_topics.append(int(each_member.strip("C")))
            elif "E" in each_member:
                target_language_topics.append(int(each_member.strip("E")))

        source_topic_word = merge_word_list(source_ldamodel,
         source_language_topics)
        target_topic_word = merge_word_list(target_ldamodel,
         target_language_topics)

        # below transformation is needed
        # because we hope to transform pd.Series to list, and
        # indexes are words
        resultant_topic_word_list.append((list(source_topic_word.index[:top_n]),
                                         list(target_topic_word.index[:top_n])))

    return resultant_topic_word_list

def recalculate_theta(mapping_dictionary, monolingual_mallet_model):
    
    all_transformed_theta = []
    
    # load_document_topics() will read doctopics.txt from /tmp
    # we will not reboot the computer, so basically it is safe
    orginal_theta = monolingual_mallet_model.load_document_topics()
    for each_theta in orginal_theta:
        
        transformed_theta = [0]*len(mapping_dictionary)
        assert len(each_theta) == len(mapping_dictionary)
        
        for idx, topic_tuple in enumerate(each_theta):
            transformed_theta[mapping_dictionary[idx]] += topic_tuple[1]
        
        all_transformed_theta.append(transformed_theta)
        
    return all_transformed_theta

def save(resultant_topic_word_list, resultant_source_doctopics, resultant_target_doctopics, out_path):
    f = open(out_path, "wb")
    pickle.dump((resultant_topic_word_list, resultant_source_doctopics, resultant_target_doctopics), f)
    f.close()

class PMLDA(object):
    """docstring for ."""
    def __init__(self, source_model_path, target_model_path, vector_path):
        logging.info('Start to initialize PMLDA')
        self.source_language_model = LdaMallet.load(source_model_path)
        self.target_language_model = LdaMallet.load(target_model_path)
        self.cross_lingual_word_vector = \
        LanguageEmbedding.read_from_KeyedVectors(vector_path)

    def train(self, top_n_representative_words, num_of_topic):
        logging.info('Start to train model')
        logging.info('1) Select representative words')
        source_word_list = select_representative_word_list(
        self.source_language_model,
         top_n_representative_words = top_n_representative_words)
        target_word_list = select_representative_word_list(
        self.target_language_model,
         top_n_representative_words = top_n_representative_words)

        logging.info('2) Calculate centroid of topics')
        source_centroid_list = calculated_centroid_of_topics(
        word_vectors=self.cross_lingual_word_vector,
        representative_word_list=source_word_list)
        target_centroid_list = calculated_centroid_of_topics(
        word_vectors=self.cross_lingual_word_vector,
        representative_word_list=target_word_list)

        logging.info('3) Start to do hierarchical clustering')
        self.member_of_clusters = hierarchical_clustering(source_centroid_list,
         target_centroid_list,
         num_of_topic=num_of_topic)
        
        self.source_topic_mapping_dictionary = {}
        self.target_topic_mapping_dictionary = {}
        for cross_index, original_indexes in self.member_of_clusters.items():
            for each_member in original_indexes:
                if "C" in each_member:
                    self.source_topic_mapping_dictionary[int(each_member.strip("C"))] = cross_index
                elif "E" in each_member:
                    self.target_topic_mapping_dictionary[int(each_member.strip("E"))] = cross_index

    def export(self, top_n, output_path):
        logging.info('Export the remapping results')
        resultant_topic_word_list = recalculate_phi_of_each_topic(
        self.source_language_model, self.target_language_model,
         self.member_of_clusters, top_n= top_n)
        resultant_source_doctopics = recalculate_theta(self.source_topic_mapping_dictionary,
                                                       self.source_language_model)
        resultant_target_doctopics =recalculate_theta(self.target_topic_mapping_dictionary,
                                                      self.target_language_model)
        ###
        # recalculate theta here.
        ###
        save(resultant_topic_word_list, resultant_source_doctopics,
             resultant_target_doctopics, output_path)

def main():

    pm = PMLDA(source_model_path="../out/Mallet_Mono_LDA/50K-doc-iter500-alpha01-cn.model",
    target_model_path="../out/Mallet_Mono_LDA/50K-doc-iter500-alpha01-en.model",
    vector_path="/home/ponshane/jupyter_working_dir/cross-lingual-topic-analysis/UM_Corpus_vectors/2018-09-27-ponshane-um-concatenate-wordvec-mikolov-100d.vec")

    pm.train(top_n_representative_words=100, num_of_topic=20)
    pm.export(top_n=20, output_path="/home/ponshane/Desktop/test.pkl")

if __name__ == "__main__":
    main()
