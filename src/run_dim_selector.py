import argparse
import time
from codebase.LanguageEmbedding import LanguageEmbedding

parser = argparse.ArgumentParser()

parser.add_argument('--embedding_file', type=str, required=True,
                    help='the location of word embedding file')
parser.add_argument('--file_path', type=str, required=True,
                    help='path of output file')
parser.add_argument('--postfix', type=str, required=True,
                    help='postfix of filename')
parser.add_argument('--start_dim', type=int, default=5,
                    help='the start number for masking')
parser.add_argument('--end_dim', type=int, default=50,
                    help='the end number for masking')
parser.add_argument('--step_size', type=int, default=5,
                    help='the step_size of masking dim')
parser.add_argument('--classifier', type=str, default="logistic",
                    help='the classifier used for estimating language dims')
args = parser.parse_args()

embedding_file = args.embedding_file
file_path = args.file_path
postfix = args.postfix
start_dim = args.start_dim
end_dim = args.end_dim
step_size = args.step_size
classifier = args.classifier

concatenate_word_vectors = LanguageEmbedding.read_from_KeyedVectors(embedding_file)
# en_num, chi_num, error_word_idx = LanguageEmbedding.check_vocabulary_type_of_space(concatenate_word_vectors)

print("Note that I diretly set the en_num & chi_num for jp-en case!")
print("SHOULD check when in en-zh case")
"""
12894 for jp words
19982 for en words
"""
en_num, chi_num, error_word_idx = 12894, 19982, list()

if len(error_word_idx) != 0:
    print("Please clean your word embeddings first, there are some words can't recognize!")
    sys.exit()

coef = LanguageEmbedding.estimate_coefficient_of_language_dimension(concatenate_word_vectors, en_num, chi_num, classifier)

num_of_dim = concatenate_word_vectors.wv.vectors.shape[1]

for each_mask_dim in range(start_dim, end_dim, step_size):
    LanguageEmbedding.mask_language_dim_and_export_to_CLTM(coef, each_mask_dim, concatenate_word_vectors.index2word,
     concatenate_word_vectors.wv.vectors, file_path+str(num_of_dim - each_mask_dim*2)+postfix, classifier)
