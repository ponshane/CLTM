from sklearn.feature_extraction.text import CountVectorizer
import numpy
import os
import argparse

def encode_corpus(file_):
    """ generate encoded corpus and dictionary
    """
    with open(file_, 'r') as f:
        docs = f.read().splitlines()
    cv = CountVectorizer()
    doc_word = cv.fit_transform(doc for doc in docs)
    vocab = cv.get_feature_names()
    return doc_word, vocab

def write_formatted_corpus(doc_word, out):

    wf = open(out, "w")

    for row_index in range(doc_word.shape[0]):
        doc = doc_word[row_index]
        I, J = numpy.nonzero(doc)
        vals = doc[I, J]
        temp = []
        total = 0
        for v_position, j in numpy.ndenumerate(J):
            word_index = j
            cnt = vals[0, v_position][0,0] # [0,0] for unpack the matrix
            temp.append(f"{word_index}:{cnt}")
            total += cnt
        wf.write(f"{total} {' '.join(temp)}\n")

    wf.close()

def write_vocab(vocabs, out):
    wf = open(out, "w")
    for v in vocabs:
        wf.write(f"{v}\n")
    wf.close()

def format_dictionary(dictionary):
    dir_path, filename = os.path.split(dictionary)
    wf = open(os.path.join(dir_path, f"MTM-{filename}"), "w")
    with open(dictionary, "r") as rf:
        for line in rf:
            src_w, tgt_w = line.strip().split(",")
            wf.write(f"0\t{src_w}\t1\t{tgt_w}\n")
    wf.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--source_corpus', type=str, required=True,
                        help='corpus of source language')
    parser.add_argument('--target_corpus', type=str, required=True,
                        help='corpus of target language')
    parser.add_argument('--dictionary', type=str, required=True,
                        help='bilingual dictionary')
    args = parser.parse_args()

    source_corpus = args.source_corpus
    target_corpus = args.target_corpus
    dictionary = args.dictionary

    # source corpus
    src_doc_word, src_vocab = encode_corpus(file_=source_corpus)
    dir_path, filename = os.path.split(source_corpus)
    write_formatted_corpus(src_doc_word, os.path.join(dir_path, f"MTM-{filename}"))
    write_vocab(src_vocab, os.path.join(dir_path,"MTM-src-vocabs"))

    # target corpus
    tgt_doc_word, tgt_vocab = encode_corpus(file_=target_corpus)
    dir_path, filename = os.path.split(target_corpus)
    write_formatted_corpus(tgt_doc_word, os.path.join(dir_path, f"MTM-{filename}"))
    write_vocab(tgt_vocab, os.path.join(dir_path,"MTM-tgt-vocabs"))

    # dictionay
    format_dictionary(dictionary)