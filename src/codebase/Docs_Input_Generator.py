import configparser
from datetime import datetime
from pymongo import MongoClient
from LanguageEmbedding import LanguageEmbedding
import pandas as pd
from sklearn.utils import shuffle
import pickle
import re
import os

### init and read config
config = configparser.ConfigParser()
config.read('/home/ponshane/work_dir/CLTM/src/config.ini')

MongoDB = config["Mongo"]["Database"]
MongoUser = config["Mongo"]["User"]
MongoPW = config["Mongo"]["PW"]

###連接MONGO
uri = "mongodb://" + MongoUser + ":" + MongoPW + "@140.117.69.70:30241/" + MongoDB + "?authMechanism=SCRAM-SHA-1"

client = MongoClient(uri)
db = client.MIRDC

def export_selected_documents(word_vector_path, output_path, doc_num):
    """
    this function is used to export selective corpus e.g., UM-Corpus for CLTM, PLTM, JointLDA and PMLDA.

    Args:
        word_vector_path (str): the location of word vector
        output_path (str): the location of output file
        doc_num (int): # of documents u want to get

    Returns:
        None but export files
    """

    start_time = datetime.now()

    target_collection = db.UM_Corpus

    query_documents = target_collection.find({"type": "News"},
                                             {"chi_result":1, "eng_result":1}, no_cursor_timeout=True)

    concatenate_word_vectors = LanguageEmbedding.read_from_KeyedVectors(word_vector_path)
    word_list = concatenate_word_vectors.index2word

    # for efficiency
    # https://stackoverflow.com/questions/7571635/fastest-way-to-check-if-a-value-exist-in-a-list
    # original performance => top 1,000, Time elapsed (hh:mm:ss.ms) 0:03:36.006745
    # IMPROVED performance => top 1,000, Time elapsed (hh:mm:ss.ms) 0:00:03.524681

    word_set = set(word_list)

    # list of list, contains all documents
    chinese_document_list = list()
    english_document_list = list()
    
    # words apper in selected corpus
    show_word_dictionary = dict()

    index = 0

    for each_document in query_documents:

        # for saving each document's all sentences' tokens
        eng_raw_loop_document = list()
        chi_raw_loop_document = list()

        for each_sentence in each_document["eng_result"]["tokens"]:
            for word in each_sentence:
                if word in word_set:
                    show_word_dictionary[word] = True
                    eng_raw_loop_document.append(word)

        for each_sentence in each_document["chi_result"]["tokens"]:
            for word in each_sentence:
                if word in word_set:
                    show_word_dictionary[word] = True
                    chi_raw_loop_document.append(word)

        if len(eng_raw_loop_document) == 0 or len(chi_raw_loop_document) == 0:
            continue

        english_document_list.append(eng_raw_loop_document)
        chinese_document_list.append(chi_raw_loop_document)

        # 隨意拿英文或中文來記
        # mongoid_dictionary[each_document["_id"]] = len(chinese_document_list) - 1

        index += 1

        if(index % 2500 == 0):
            print("Already process %d documents" % index)
            time_elapsed = datetime.now() - start_time
            print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))

        if(index % doc_num == 0):
            print("Already process %d documents" % index)
            time_elapsed = datetime.now() - start_time
            print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))
            break

    query_documents.close()
    
    assert len(chinese_document_list) == len( english_document_list)
    
    """
    for CLTM, e.g., selected50KDos.txt
    """
    out = open(output_path+"selected"+str(int((doc_num*2)/1000))+"KDos.txt",'w')
    for each_cn, each_en in zip(chinese_document_list, english_document_list):
        out.write(" ".join(each_cn) + "\n")
        out.write(" ".join(each_en) + "\n")

    out.close()
    
    """
    for PMLDA, JointLDA e.g., 50K_Chinese_UM_Corpus.txt, 50K_English_UM_Corpus.txt
    """
    cnout = open(output_path+str(int((doc_num*2)/1000))+"K_Chinese_UM_Corpus.txt",'w')
    enout = open(output_path+str(int((doc_num*2)/1000))+"K_English_UM_Corpus.txt",'w')
    
    for each_cn, each_en in zip(chinese_document_list, english_document_list):
        cnout.write(" ".join(each_cn) + "\n")
        enout.write(" ".join(each_en) + "\n")

    cnout.close()
    enout.close()
    
    """
    for PLTM e.g., 50K-cn-docs.txt, 50K-en-docs.txt
    """
    cnout = open(output_path+str(int((doc_num*2)/1000))+"K-cn-docs.txt",'w')
    enout = open(output_path+str(int((doc_num*2)/1000))+"K-en-docs.txt",'w')
    
    for idx, (each_cn, each_en) in enumerate(zip(chinese_document_list, english_document_list)):
        cnout.write(str(idx) + "\tCN\t" + " ".join(each_cn) + "\n")
        enout.write(str(idx) + "\tEN\t" + " ".join(each_en) + "\n")

    cnout.close()
    enout.close()

    return show_word_dictionary

def export_shuffled_documents_from_pd(pd_path, output_path, random_state):
    """
    this function is used to export shuffled labeled corpus e.g., MLDoc for CLTM, JointLDA and PMLDA.

    Args:
        pd_path (str): the location of corpus dataframe
        output_path (str): the location of output folder --> file1 for CLTM, file2(file3) for JointLDA and PMLDA
        random_state (int): random state

    Returns:
        None but export files
    """

    df = pd.read_pickle(pd_path)
    shuffled_df = shuffle(df, random_state=random_state)

    word_dictionary = {}
    
    file1 = "CLTM-MLDoc.txt"
    file2 = "MLDoc-Chinese.txt"
    file3 = "MLDoc-English.txt"
    out = open(output_path+file1, 'w')
    out2 = open(output_path+file2, 'w')
    out3 = open(output_path+file3, 'w')
    
    # source_idx and target_idx 用來記得 pd_index and corpus idx 的對應
    source_idx = dict()
    target_idx = dict()

    # export shuffled documents and word dictionary
    for idx, each_row in shuffled_df.iterrows():

        # write document file
        
        out.write(' '.join(each_row["extracted_text"]) + "\n")
        for word in each_row["extracted_text"]:
            # record the presence of word
            word_dictionary[word] = True
        
        if each_row["language"] == "English":
            source_idx[idx] = len(source_idx)
            out3.write(" ".join(each_row["extracted_text"]) + "\n")
        elif each_row["language"] == "Chinese":
            target_idx[idx] = len(target_idx)
            out2.write(" ".join(each_row["extracted_text"]) + "\n")
        
    out.close()
    out2.close()
    out3.close()
    
    assert len(source_idx) + len(target_idx) == shuffled_df.shape[0]
    print("Corpus stores in:", output_path)

    shuffled_df.to_pickle(output_path+"Shuffled_RS"+str(random_state)+"_tagged_englishAndchinese_corpus_pd.pkl")
    print("Shuffled corpus stores in:", output_path)
    
    out4 = open(output_path+"inverse-index-mapping-dict.pickle", 'wb')
    pickle.dump((source_idx, target_idx), out4)
    out4.close()
    print("inverse-index-mapping-dict stores in:", output_path)
    
    return word_dictionary

def export_shuffled_documents_from_en_jp_pd(pd_path, output_path, random_state):
    """
    This function is used to export shuffled labeled corpus e.g., MLDoc for CLTM, JointLDA and PMLDA.
    BUT ONLY for EN <-> JP Case
     
    Args:
        pd_path (str): the location of corpus dataframe
        output_path (str): the location of output folder --> file1 for CLTM, file2(file3) for JointLDA and PMLDA
        random_state (int): random state

    Returns:
        None but export files
    """

    df = pd.read_pickle(pd_path)
    shuffled_df = shuffle(df, random_state=random_state)

    word_dictionary = {}
    
    file1 = "CLTM-EN-JP-MLDoc.txt"
    file2 = "MLDoc-Jpanese.txt"
    file3 = "MLDoc-English.txt"
    out = open(output_path+file1, 'w')
    out2 = open(output_path+file2, 'w')
    out3 = open(output_path+file3, 'w')
    
    # source_idx and target_idx 用來記得 pd_index and corpus idx 的對應
    source_idx = dict()
    target_idx = dict()

    # export shuffled documents and word dictionary
    for idx, each_row in shuffled_df.iterrows():

        # write document file
        
        out.write(' '.join(each_row["extracted_text"]) + "\n")
        for word in each_row["extracted_text"]:
            # record the presence of word
            word_dictionary[word] = True
        
        if each_row["language"] == "English":
            source_idx[idx] = len(source_idx)
            out3.write(" ".join(each_row["extracted_text"]) + "\n")
        elif each_row["language"] == "Japanese":
            target_idx[idx] = len(target_idx)
            out2.write(" ".join(each_row["extracted_text"]) + "\n")
        
    out.close()
    out2.close()
    out3.close()
    
    assert len(source_idx) + len(target_idx) == shuffled_df.shape[0]
    print("Corpus stores in:", output_path)

    shuffled_df.to_pickle(output_path+"Shuffled_RS"+str(random_state)+"_tagged_englishAndjapanese_corpus_pd.pkl")
    print("Shuffled corpus stores in:", output_path)
    
    out4 = open(output_path+"en-jp-inverse-index-mapping-dict.pickle", 'wb')
    pickle.dump((source_idx, target_idx), out4)
    out4.close()
    print("en-jp-inverse-index-mapping-dict stores in:", output_path)
    
    return word_dictionary

def export_selected_word_space(vector_path, word_dictionary, outfilename):
    """
    Because the resultant corpus of export_selected_documents function
    will not contains all vocabularies in pre-trained word vector, this
    function can help to fix this word-no-show problem.

    Args:
        pre_check_vector_list_path (list): a list of the location of
            pre_check word spaces
        word_dictionary (dict): a dictionary shows which word appears in
            resultant corpus

    Returns:
        None but export files
    """
    #f = open(pre_check_vector_list_path, 'r', encoding='utf-8')
    #for vector_path in f.readlines():
    
    each_model = LanguageEmbedding.read_from_KeyedVectors(vector_path)

    # create new file
    # below regex can replace by os built-in function
    
    """regex = r"\/(.+)\/(.+)"
    matches = re.search(regex, vector_path, re.DOTALL)
    if matches:
        selected_word_space_out_path = "/"+matches.group(1)+"/selected"+matches.group(2)"""
    
    out = open(outfilename, 'w')
    # star cleaning
    for each_word in word_dictionary.keys():
        out.write(each_word + " ")
        out.write(' '.join(map(str, each_model[each_word])) + "\n")
    out.close()

    #f.close()
