#!/bin/bash
for i in 20 30 40 50
do
python run_batch_PMLDA.py \
--source_model_path /home/ponshane/work_dir/CLTM-Experiments/Data/UM-Corpus/800K-sampled-docs/monolingual-lda/"$i"topics-cn.model \
--target_model_path /home/ponshane/work_dir/CLTM-Experiments/Data/UM-Corpus/800K-sampled-docs/monolingual-lda/"$i"topics-en.model \
--vector_path /home/ponshane/work_dir/CLTM-Experiments/Data/UM-Corpus/word-vectors/2018-09-27-ponshane-um-concatenate-wordvec-mikolov-100d.vec \
--num_of_topic "$i" --top_n 1000 \
--prefix_output_path /home/ponshane/work_dir/CLTM-Experiments/Results/UM-Corpus-800K-sampled-docs/PMLDA-"$i"topics-
done
