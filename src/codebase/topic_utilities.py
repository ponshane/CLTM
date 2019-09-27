import re
import matplotlib.pyplot as plt

def export_lda_topic_word_dist(output_path, lda_model, topic_num, top_n_words):
    #new_path = "./LDA_Outputs/2018-01-29-ponshane-um-corpus-translation-method-50000Chunk-20topics-1pass-200words-topic-word-dist.txt"
    out = open(output_path,'w')

    for topic_idx in range(0, topic_num):
        contribution_pairs = lda_model.show_topic(topic_idx, topn = top_n_words)
        out.write("Topic: " + str(topic_idx) + "\n")
        for each_tuple in contribution_pairs:
            out.write("\t")
            out.write(each_tuple[0] + " ")
        out.write("\n\n")
    out.close()

def export_sample_doc_topic_dist(output_path, lda_model, corpus, raw_documents, raw_english_documents, sample_amounts=1000):

    out = open(output_path,'w')
    
    for document_idx in range(0, sample_amounts):
        if document_idx % 2 == 0:
            # document_idx % 2 == 0 are chinese documents
            out.write("Chinese Doc: " + str(document_idx) + "\t" + ', '.join(raw_documents[document_idx]) + "\n")
        else:
            # for those english documents
            out.write("Original English Doc: " + str(document_idx) + "\t" + ', '.join(raw_english_documents[document_idx]) + "\n")
            out.write("Translated English Doc: " + str(document_idx) + "\t" + ', '.join(raw_documents[document_idx]) + "\n")

        out.write("Topic distribution: " + str(lda_model[corpus[document_idx]]) + "\n")
        out.write("\n")
        
    out.close()


def make_perplexity_plots(log_path, output_path, eval_every):
    # https://stackoverflow.com/questions/37570696/how-to-monitor-convergence-of-gensim-lda-model

    p = re.compile("(-*\d+\.\d+) per-word .* (\d+\.\d+) perplexity")
    matches = [p.findall(line) for line in open(log_path)]

    # not every line has perplexity information
    matches = [m for m in matches if len(m) > 0]

    # pick match tuple "There is only one match in one line"
    tuples = [t[0] for t in matches]

    # furthermore to extract likelihood and perplexity
    liklihood = [float(t[0]) for t in tuples]
    perplexity = [float(t[1]) for t in tuples]

    iters = list(range(0,len(tuples)))
    plt.plot(iters,perplexity,c="black")
    plt.ylabel("perplexity")
    plt.xlabel("iteration")
    plt.title("Topic Model Convergence")
    plt.grid()
    plt.savefig(output_path)
    plt.close()
