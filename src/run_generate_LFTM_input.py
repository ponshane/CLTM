import argparse
import time
from codebase.LFTM_Input_Generator import *

parser = argparse.ArgumentParser()

parser.add_argument('--output_path', type=str, required=True,
                    help='path of output file')

parser.add_argument('--pre_check_vector_list_path', type=str, required=True,
                    help='list of pre-check word embedding file')

####################
# for export_selected_documents
####################
parser.add_argument('--word_vector_path', type=str,
                    help='the location of word embedding file')

parser.add_argument('--doc_num', type=int,
                    help='how many documents u want to sample?')

####################
# for export_shuffled_documents_from_pd
####################
parser.add_argument('--pd_path', type=str,
                    help='the location of corpus pd file')

parser.add_argument('--random_state', type=int,
                    help='random state in shuffle function')

args = parser.parse_args()

word_vector_path = args.word_vector_path
output_path = args.output_path
pre_check_vector_list_path = args.pre_check_vector_list_path
doc_num = args.doc_num
pd_path = args.pd_path
random_state = args.random_state

if word_vector_path is not None and pd_path is None:
    word_dict = export_selected_documents(word_vector_path,
     output_path, doc_num)
    export_selected_word_space(pre_check_vector_list_path,
     word_dictionary=word_dict)
elif word_vector_path is None and pd_path is not None:
    word_dict = export_shuffled_documents_from_pd(pd_path,
     output_path, random_state)
    export_selected_word_space(pre_check_vector_list_path,
     word_dictionary=word_dict)
else:
    print("Cannot accept word_vector_path and pd_path in the same time!")
