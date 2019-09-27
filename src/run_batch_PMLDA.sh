#!/bin/bash
for i in 10 20 30 40 50
do
   python run_batch_PMLDA.py --source_model_path /home/ponshane/work_dir/CLTM-Experiments/Data/MLDoc/monolingual-lda/"$i"topics-en.model --target_model_path /home/ponshane/work_dir/CLTM-Experiments/Data/MLDoc/monolingual-lda/"$i"topics-cn.model --vector_path /home/ponshane/work_dir/CLTM-Experiments/Data/MLDoc/word-vectors/Chinese_English_wordvectors.vec --num_of_topic "$i" --top_n 1000 --prefix_output_path /home/ponshane/work_dir/CLTM-Experiments/Results/MLDoc/model-comparison/PMLDA-"$i"topics-
done
