from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import coo_matrix
import numpy
import anchor_topic.topics

import argparse
import pickle

def get_data(file_):
    """ generate word-doc matrix
    """
    with open(file_, 'r') as f:
        docs = f.read().splitlines()
    cv = CountVectorizer()
    doc_word = cv.fit_transform(doc for doc in docs)
    doc_word_sparse = coo_matrix(doc_word)
    word_doc_sparse = doc_word_sparse.T
    word_doc = word_doc_sparse.tocsc()
    vocab = cv.get_feature_names()
    return word_doc, vocab, doc_word

def code_dictionary(dict_path, src_dict, tgt_dict, sep):
    """ generate bilingual dictionary 
    """
    with open(dict_path, 'r') as f:
        entries = f.read().splitlines()
    dictionary = []
    for entry in entries:
        ent = entry.split(sep)
        w1 = ent[0].lower()
        w2 = ent[1]
        try:
            dictionary.append([src_dict.index(w1), tgt_dict.index(w2)])
        except ValueError:
            continue
    print(f"The final size of dictionary: {len(dictionary)}")
    return dictionary

def convert_2dlist(lst, index):
    new_lst = []
    for row in lst:
        new_row = []
        for entry in row:
            new_row.append(index[entry])
        new_lst.append(new_row)
    return new_lst

def get_top_topic_words(A, n, vocab=None):
    """Return top [n] words for every topic with information about probability
    distribution provided by [A].
    If [vocab] is not None, convert indices to words.
    """  
    assert n <= A.shape[0], \
        'Number of words requested greater than model\'s number of words'
    topic_words = A.T

    # sort words based on probabilities
    sort_words = numpy.argsort(topic_words, axis=1)
    # reverse so that the higher probabilities come first
    rev_words = numpy.flip(sort_words, axis=1)
    # retrieve top n words
    top_words = rev_words[:,:n]

    if vocab is None:
        return top_words
    else:
        top_words = convert_2dlist(top_words, vocab)
        return top_words

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--source_corpus', type=str, required=True,
                        help='corpus of source language')
    parser.add_argument('--target_corpus', type=str, required=True,
                        help='corpus of target language')
    parser.add_argument('--dictionary', type=str, required=True,
                        help='bilingual dictionary')
    parser.add_argument('--num_of_topic', type=int, nargs="+", required=True,
                        help='the number of cross-lingual topic')
    parser.add_argument('--prefix_output_path', type=str, required=True,
                        help='prefix output path')
    args = parser.parse_args()

    # read parameters
    source_corpus = args.source_corpus
    target_corpus = args.target_corpus
    dictionary = args.dictionary
    num_of_topic = args.num_of_topic
    prefix_output_path = args.prefix_output_path

    # (1) preprocessing inputs
    src_word_doc, src_vocab, src_d2w = get_data(source_corpus)
    tgt_word_doc, tgt_vocab, tgt_d2w = get_data(target_corpus)
    dictionary = code_dictionary(dict_path=dictionary,
    src_dict=src_vocab,tgt_dict=tgt_vocab,sep=",")

    for n_topic in num_of_topic:

        # (2) train model
        print(f"Start to train {n_topic} topics.\n")
        A1, A2, Q1, Q2, anchors1, anchors2 = anchor_topic.topics.model_multi_topics(M1=src_word_doc,\
            M2=tgt_word_doc, k=n_topic,\
                threshold1=0.001, threshold2=0.001, dictionary=dictionary)
        # threshold1, threshold2 = 0.05, 0,05 for MLDoc

        # (3) calculate phi and theta
        topic_words1 = get_top_topic_words(A1, 100, src_vocab)
        # anchor_words1 = convert_2dlist(anchors1, src_vocab)
        topic_words2 = get_top_topic_words(A2, 100, tgt_vocab)
        # anchor_words2 = convert_2dlist(anchors2, tgt_vocab)

        theta1 = src_d2w.dot(A1)
        theta2 = tgt_d2w.dot(A2)

        # (4) serialize results
        with open(f"{prefix_output_path}MTAnchor-{n_topic}-topics.pkl", "wb") as wf:
            pickle.dump((topic_words1, topic_words2, theta1, theta2), wf)