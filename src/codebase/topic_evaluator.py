import pickle
from gensim import corpora
from gensim.matutils import corpus2csc
from sklearn.metrics import jaccard_similarity_score
from scipy.sparse import isspmatrix_csr
from math import sqrt
from joblib import Parallel, delayed
from ptm.utils import get_top_words
from scipy.stats import entropy

import csv
import re
import math
import numpy as np

'''
following functions are used for evaluating coherence and diversity performance
'''

def load_pmlda_cluster(file_path):
    f = open(file_path, 'rb')
    return(pickle.load(f))

def load_method_cluster(file_path, method_cluster_number):
    """
    Input: file_path(csv位置), method_cluster_number(分群數)
    Output: 巢狀list => 第一層記錄各群，第二層紀錄每群的主題字
    Purpose: 讀取跨語言分群csv，並轉換其格式至巢狀list，方便衡量
    """
    f = open(file_path, 'r', encoding='utf-8')
    method_file = [[] for x in range(0,method_cluster_number)]
    cid = 0
    for row in csv.reader(f, delimiter=','):
        # every topic are seperated by -
        if row[0] == '-':
            cid +=1
            continue
        else:
            # [0] indicates word type
            method_file[cid].append(row[0])

    return method_file

def load_plda_cluster(file_path, cluster_num):

    # open the cluster files
    with open(file_path, encoding='utf-8') as input:
        topic_keys_lines = input.readlines()

    topic_words = []

    for line in topic_keys_lines:
        # ignore information of phi
        if len(line.split('\t')) > 2:
            language, num_word, beta, words = line.split('\t')  # tab-separated
            words = words.rstrip().split(' ')  # remove the trailing '\n'
            topic_words.append(words)

    all_topic_pairs = []
    for idx, each_topic_words in enumerate(topic_words):
        if idx % 2 == 0:
            each_pair = []
            each_pair.append(each_topic_words)
        else:
            each_pair.append(each_topic_words)
            all_topic_pairs.append(each_pair)

    assert len(all_topic_pairs) == cluster_num
    return all_topic_pairs

def load_LFTM_cluster(file_path, cluster_num):
    topic_file = open(file_path)
    topic_file_strings = topic_file.read()
    topic_file.close()

    regex = r"Topic[\d]+:\s(.*)"

    matches = re.finditer(regex, topic_file_strings, re.MULTILINE)

    topic_words = []
    for _, match in enumerate(matches):
        # by split 會讓結果為 list of list 與 load_method_cluster 輸出格式一致
        topic_words.append(match.group(1).split(" "))

    assert cluster_num == len(topic_words)
    return topic_words

def calculate_proportion_of_cross_lingual_topics_of_CLTM(clusters, topN=100, criteria_prop = 0.7):
    
    single_cluster_counter = 0 # used for count how many cluster is single lingual
    for cluster in clusters:
        temp = 0 
        for word in cluster[:topN]:
            # 以正規表示法判斷語言
            try:
                # 可以解碼為 ascii 的為英文單字
                word.encode(encoding='utf-8').decode('ascii')
                temp += 1
            except UnicodeDecodeError:
                continue
        
        prop = temp/topN
        if prop <= (1-criteria_prop):
            single_cluster_counter +=1
        elif prop >= criteria_prop:
            single_cluster_counter +=1
        
    return 1 - (single_cluster_counter/len(clusters)) # the proportion of cross-lingual topics

def split_language(method_cluster):
    """
    Input: 巢狀list => 第一層記錄各群，第二層紀錄每群的主題字
    Output: 巢狀list => 第一層紀錄各群，第二層紀錄長度固定為2（中、英文），第三層記錄各語言的主題字
    Purpose: 將巢狀list，在往下增加一層以利區分中英文
    Reference:
        1) detect chinese word => https://blog.csdn.net/dcrmg/article/details/79228465
    Issue:
        1) 解碼部份會造成不同語言的 word_list 長度不同
    """

    # 用來除存後續分離的中文、英文各主題主題字
    splited_rank_cluster = list()

    for cluster in method_cluster:
        # 每一群在細分中英文主題字
        cn_list = list()
        en_list = list()

        for word in cluster:
            # 以正規表示法判斷語言
            try:
                # 可以解碼為 ascii 的為英文單字
                word.encode(encoding='utf-8').decode('ascii')
                en_list.append(word)
            except UnicodeDecodeError:
                # 為什麼還有一正規判斷?
                if re.search(u'[\u4e00-\u9fff]', word):
                    cn_list.append(word)
                else:
                    continue

        splited_rank_cluster.append([cn_list, en_list])

    return splited_rank_cluster

def documents_to_cooccurence_matrix(file_path, is_pickle=True):
    """
    Input: pickle file 存中英文文本(以雙層巢狀list儲存) or text_file 多數連續型LDA的文本輸入格式, is_pickle Boolean
    Output: co-occurence matrix (word by word)
    References:
        1) https://stackoverflow.com/questions/49431270/word-co-occurrence-matrix-from-gensim
        2) https://stackoverflow.com/questions/19766757/replacing-numpy-elements-if-condition-is-met
        3) https://stackoverflow.com/questions/3845423/remove-empty-strings-from-a-list-of-strings
    """
    if is_pickle:
        f = open(file_path, "rb")
        source_language_documents, target_language_documents, _ = pickle.load(f)
        f.close()

        # 確保中英文文章數一致
        assert len(source_language_documents) == len(target_language_documents)

        compound_documents = [doc_in_source + doc_in_target for doc_in_source,\
        doc_in_target in zip(source_language_documents, target_language_documents)]
    else:
        f = open(file_path, "r")
        line_list = f.readlines()
        f.close()

        # 確保中英文文章數一致
        assert len(line_list) % 2 == 0

        compound_documents = list()
        temp = ""
        for idx, text in enumerate(line_list):

            # readlines or readline function 都會殘留換行符號\n，故需要替代掉
            temp += text.replace("\n", " ")

            # 由 0 開始，故讓偶數都是一對翻譯文章的第一篇
            # 奇數則是一對翻譯文章的第二篇
            if idx %2 != 0:
                # filter 是為了解決 replace 所留下的 empty element
                # list 包住 filter 是將 iterator 轉回 list
                compound_documents.append(list(filter(None, temp.split(" "))))
                # 為下一對翻譯文章準備，故清空
                temp = ""

    # turn into gensim's corpora
    compound_dictionary = corpora.Dictionary(compound_documents)
    compund_corpus = [compound_dictionary.doc2bow(text) for text in \
    compound_documents]

    # transform into term_document matrix, each element represents as frequency
    term_document_matrix = corpus2csc(compund_corpus)

    # 利用 corpus2csc 轉換後每個元素為該詞於該篇的詞頻(會大於1)，但 umass score 需要的是 the count of documents containing the word
    # 因此得利用 np.where 重新轉換矩陣，使每個元素單純標記該詞是否出現於該篇(1 or 0)
    # np.where 無法在 csc matrix 故使用以下解決
    term_document_matrix[term_document_matrix >= 1] = 1
    cooccurence_matrix = term_document_matrix @ term_document_matrix.T

    print(type(cooccurence_matrix))
    print(term_document_matrix.shape, cooccurence_matrix.shape)

    return cooccurence_matrix, term_document_matrix, compound_dictionary, len(compund_corpus)

def NPMI(cooccurence_matrix, word_i, word_j, num_of_documents):
    epsilon = 1e-12
    co_count = cooccurence_matrix[word_i, word_j] / num_of_documents
    single_count_i = cooccurence_matrix[word_i, word_i] / num_of_documents
    single_count_j = cooccurence_matrix[word_j, word_j] / num_of_documents
    pmi = math.log((co_count+epsilon)/(single_count_i*single_count_j))
    return pmi / (math.log(co_count+epsilon)*(-1))

def coherence_score(cn_topic, en_topic, topk, cooccurence_matrix, compound_dictionary, num_of_documents, coherence_method):
    """
    Input: list of list: cn_topic(中文分群結果), en_topic(英文分群結果); scalar: topk(衡量topk個字); matrix: cooccurence_matrix
    (儲存每個字出現次數和兩兩個字的共同出現次數 － 以篇為單位); compound_dictionary: gensim 的字典
    Output: umass, npmi coherence score
    Reference:
        1) http://qpleple.com/topic-coherence-to-evaluate-topic-models/
        2) Mimno, D., Wallach, H. M., Talley, E., Leenders, M., & McCallum, A. (2011, July). Optimizing semantic coherence in topic models.
           #In Proceedings of the conference on empirical methods in natural language processing (pp. 262-272). Association for Computational Linguistics.
    Issue:
        1) [Solve!] Original metric uses count of documents containing the words
    """
    each_topic_coher = []
    for ctopic, etopic in zip(cn_topic, en_topic):

        # below two assertion is very important because
        # 1) minor problem split_language method is a risky method because it may strips some words
        # 2) continue LDAs can not promise to produce the same vocabularies size across languages,
        #    and be a extreme imbalance distribution. (單語言主題群，僅有少數跨語言詞彙)

        assert len(ctopic) >= topk
        assert len(etopic) >= topk

        cn_idx = [ compound_dictionary.token2id[cn] for cn in ctopic[:topk] if cn in compound_dictionary.token2id]
        en_idx = [ compound_dictionary.token2id[en] for en in etopic[:topk] if en in compound_dictionary.token2id]

        '''
        debug line
        print(ctopic[:topk])
        print(etopic[:topk])
        '''

        coherences = []
        for each_cn in cn_idx:
            for each_en in en_idx:
                if coherence_method == "umass":
                    # calculate_umass_score_between_two_words
                    co_count = cooccurence_matrix[each_cn, each_en]
                    single_count = cooccurence_matrix[each_en, each_en]
                    pmi = math.log((co_count+1)/single_count)
                    coherences.append(pmi)
                elif coherence_method == "npmi":
                    npmi = NPMI(cooccurence_matrix, each_cn, each_en, num_of_documents)
                    coherences.append(npmi)

        each_topic_coher.append(sum(coherences)/len(coherences))

    return sum(each_topic_coher)/len(each_topic_coher)

def avg_jaccard_similarity_between_topics(ch_topic_clusters, en_topic_clusters, top_n_words):
    # jaccard_similarity_score(d_2_LFTM_cluster[0][:5], d_2_LFTM_cluster[6][:5])
    assert len(ch_topic_clusters) == len(en_topic_clusters)
    topic_num = len(ch_topic_clusters)

    jaccard_similarities = []
    for anchor_idx in range(0, topic_num-1):
        anchor_word_list = ch_topic_clusters[anchor_idx][:top_n_words]
        en_anchor_word_list = en_topic_clusters[anchor_idx][:top_n_words]
        for loop_idx in range(anchor_idx+1, topic_num):
            loop_word_list = ch_topic_clusters[loop_idx][:top_n_words]
            en_loop_word_list = en_topic_clusters[loop_idx][:top_n_words]
            jaccard_similarities.append(jaccard_similarity_score(anchor_word_list, loop_word_list))
            jaccard_similarities.append(jaccard_similarity_score(en_anchor_word_list, en_loop_word_list))

    return 1-(sum(jaccard_similarities) / len(jaccard_similarities)), stddev(jaccard_similarities)

def unique_words_proportion_between_topics(ch_topic_clusters, en_topic_clusters, top_n_words):

    assert len(ch_topic_clusters) == len(en_topic_clusters)
    topic_num = len(ch_topic_clusters)

    chinese_unique_words_counts = []
    english_unique_words_counts = []
    for anchor_idx in range(0, topic_num-1):
        anchor_word_list = ch_topic_clusters[anchor_idx][:top_n_words]
        en_anchor_word_list = en_topic_clusters[anchor_idx][:top_n_words]
        for loop_idx in range(anchor_idx+1, topic_num):
            loop_word_list = ch_topic_clusters[loop_idx][:top_n_words]
            en_loop_word_list = en_topic_clusters[loop_idx][:top_n_words]
            chinese_unique_words_counts.append(len(set(anchor_word_list) -  set(loop_word_list)))
            english_unique_words_counts.append(len(set(en_anchor_word_list) -  set(en_loop_word_list)))

    return min(chinese_unique_words_counts)/top_n_words, min(english_unique_words_counts)/top_n_words

def loop_to_get_each_model(models_path):
    models = []
    f = open(models_path, 'r', encoding='utf-8')
    for model_path in f.readlines():
        models.append(model_path.strip("\n"))
    f.close()
    return models

def stddev(lst):
    mean = float(sum(lst)) / len(lst)
    return sqrt(sum((x - mean)**2 for x in lst) / len(lst))

def REF_NPMI(word_tuple, tdm, epsilon = 1e-12):
    docSize = tdm.shape[1]
    wordi, wordj = word_tuple
    cocount = ((tdm[wordi,:] @ tdm[wordj,:].T).toarray()[0,0] + epsilon) / docSize
    ci = (tdm[wordi,:] @ tdm[wordi,:].T).toarray()[0,0] / docSize
    cj = (tdm[wordj,:] @ tdm[wordj,:].T).toarray()[0,0] / docSize
    pmi = math.log(cocount / (ci*cj))
    return pmi / (math.log(cocount)*(-1))
    
def REF_coherence_score(cn_topic, en_topic, topk, tdm, dictionary):
    """
    to be added
    """
    
    # makesure this is a Compressed Sparse Row matrix
    # this will dramatically affect the speed........
    assert isspmatrix_csr(tdm) == True
    
    cn_vocabs_coverage = []
    en_vocabs_coverage = []
    tuple_args = []
    
    for ctopic, etopic in zip(cn_topic, en_topic):

        # below two assertion is very important because
        # 1) minor problem split_language method is a risky method because it may strips some words
        # 2) continue LDAs can not promise to produce the same vocabularies size across languages,
        #    and be a extreme imbalance distribution. (單語言主題群，僅有少數跨語言詞彙)

        assert len(ctopic) >= topk
        assert len(etopic) >= topk

        cn_idx = [ dictionary.token2id[cn] for cn in ctopic[:topk] if cn in dictionary.token2id]
        en_idx = [ dictionary.token2id[en] for en in etopic[:topk] if en in dictionary.token2id]
        
        cn_vocabs_coverage.append(len(cn_idx) / topk)
        en_vocabs_coverage.append(len(en_idx) / topk)

        for each_cn in cn_idx:
            for each_en in en_idx:
                tuple_args.append((each_cn, each_en))
    
    #print("Scanned over the {} tuple_args.".format(len(tuple_args)))
    
    coherences = Parallel(n_jobs=-1, verbose=0, backend="threading")\
    (delayed(REF_NPMI)(each_tuple, tdm) for each_tuple in tuple_args)

    return float(sum(coherences)) / len(coherences), stddev(coherences),\
            float(sum(cn_vocabs_coverage)) / len(cn_vocabs_coverage),\
            float(sum(en_vocabs_coverage)) / len(en_vocabs_coverage)

def split_topics_by_languages(n_topic, n_words, model, corpus):
    """
    for JointLDA
    """
    source_topic_list = []
    target_topic_list = []
    for ti in range(n_topic):
        top_words = get_top_words(model.TW, corpus.reconcatenate_dict, ti, n_words=n_words)
        source_temp = []
        target_temp = []
        for word in top_words:
            if isinstance(word, tuple):
                source_temp.append(word[0])
                target_temp.append(word[1])
            elif isinstance(word, str):
                try:
                    word.encode("ascii")
                    source_temp.append(word)
                except UnicodeEncodeError:
                    target_temp.append(word)
        source_topic_list.append(source_temp)
        target_topic_list.append(target_temp)
    return source_topic_list, target_topic_list

'''
following functions are used for evaluating theta performance on JSD
'''

def jsd(p, q, base=2):
    '''
        Implementation of pairwise `jsd` based on  
        https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence
        # reference from https://gist.github.com/zhiyzuo/f80e2b1cfb493a5711330d271a228a3d
    '''
    ## convert to np.array
    p, q = np.asarray(p), np.asarray(q)
    ## normalize p, q to probabilities
    p, q = p/p.sum(), q/q.sum()
    m = (1./2)*(p + q)
    return entropy(p,m, base=base)/2. + entropy(q, m, base=base)/2.

def calculate_jsd_of_theta_from_CLTM(file_path):
    with open(file_path, encoding='utf-8') as handler:
        thetas = handler.readlines()

    # if input are not parallel corpus, this function can not act normally
    assert len(thetas) %2 == 0

    jsd_array = []
    for idx in range(0, len(thetas)-1, 2):
        try:
            jsd_array.append(jsd(list(map(float, thetas[idx].split())),
                                 list(map(float, thetas[idx+1].split()))))
        except TypeError:
            print("array does not meet the requirement!")
            break

    return (1 - sum(jsd_array) / len(jsd_array)), stddev(jsd_array)

def calculate_jsd_of_theta_from_PLTM(file_path, another_file_path):
    with open(file_path, encoding='utf-8') as handler:
        thetas = handler.readlines()

    with open(another_file_path, encoding='utf-8') as handler:
        another_thetas = handler.readlines()

    assert len(thetas) == len(another_thetas)

    jsd_array = []
    for idx in range(1, len(thetas)):
        try:
            jsd_array.append(jsd(list(map(float, thetas[idx].split("\t")))[2:],
                                 list(map(float, another_thetas[idx].split("\t")))[2:]))
        except TypeError:
            print("array does not meet the requirement!")
            break

    return (1 - sum(jsd_array) / len(jsd_array)), stddev(jsd_array)

def calculate_jsd_of_theta_from_PMLDA(file_path):
    _, source_thetas, target_thetas = load_pmlda_cluster(file_path)
    assert len(source_thetas) == len(target_thetas)
    
    jsd_array = []
    for idx in range(1, len(source_thetas)):
        try:
            jsd_array.append(jsd(source_thetas[idx], target_thetas[idx]))
        except TypeError:
            print("array does not meet the requirement!")
            break

    return (1 - sum(jsd_array) / len(jsd_array)), stddev(jsd_array)

def calculate_jsd_of_theta_from_JointLDA(file_path):
    
    with open(file_path, 'rb') as handle:
        JointLDA_model = pickle.load(handle)
        
    doc_size = JointLDA_model.DT.shape[0]
    assert doc_size % 2 == 0
    
    target_corpus_start_index = doc_size//2
    
    jsd_array = []
    for idx in range(0, target_corpus_start_index):
        try:
            jsd_array.append(jsd(JointLDA_model.DT[idx,:],
                                 JointLDA_model.DT[target_corpus_start_index + idx,:]))
        except TypeError:
            print("array does not meet the requirement!")
            break

    return (1 - sum(jsd_array) / len(jsd_array)), stddev(jsd_array)

def main():
    pass

if __name__ == "__main__":
    main()
