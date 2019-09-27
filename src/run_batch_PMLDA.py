import argparse
import time
from codebase.PMLDA import *

parser = argparse.ArgumentParser()

parser.add_argument('--source_model_path', type=str, required=True,
                    help='source_lda_model_path')
parser.add_argument('--target_model_path', type=str, required=True,
                    help='target_lda_model_path')
parser.add_argument('--vector_path', type=str, required=True,
                    help='the location of cross-lingual vectors')
parser.add_argument('--num_of_topic', type=int, required=True,
                    help='the number of cross-lingual topic')
parser.add_argument('--top_n', type=int, required=True,
                    help='how many topical words of each topic hope to export?')
parser.add_argument('--prefix_output_path', type=str, required=True,
                    help='prefix_output_path')
"""
parser.add_argument('--end_top_n', type=int, required=True,
                    help='range of evaluating topic')
parser.add_argument('--step_size', type=int, default=5,
                    help='the step size of evaluating range')
"""

args = parser.parse_args()

source_model_path = args.source_model_path
target_model_path = args.target_model_path
vector_path = args.vector_path
num_of_topic = args.num_of_topic
top_n = args.top_n
prefix_output_path = args.prefix_output_path

pm = PMLDA(source_model_path=source_model_path,
target_model_path=target_model_path,
vector_path=vector_path)

for each_representative_words in range(10, 15, 5):
    pm.train(top_n_representative_words = each_representative_words,
     num_of_topic=num_of_topic)
    pm.export(top_n=top_n,
     output_path= prefix_output_path + str(each_representative_words) + "rep.pkl")
