import logging
import argparse

from ptm import JointGibbsLDA, JointCorpus
import pickle

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

    # prepare corpus
    corpus = JointCorpus(source_corpus_file=source_corpus,
                        target_corpus_file=target_corpus)
    corpus.update_doctionary(dictionary)
    corpus.convert_raw_corpus_to_trainable_corpus()

    for n_topic in num_of_topic:
        model = JointGibbsLDA(n_doc=len(corpus.docs), n_concept=corpus.n_concept, n_s_vocab=corpus.n_s_vocab,
                          n_t_vocab=corpus.n_t_vocab, n_topic=n_topic)
        model.fit(corpus.docs, corpus.language_flags, max_iter=1000)
        
        file_name = f"JointLDA-{n_topic}topics.pickle"
        with open(f"{prefix_output_path}{file_name}", 'wb') as handle:
            pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)