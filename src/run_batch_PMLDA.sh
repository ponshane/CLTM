for i in 10 20 30 40 50
do
    python run_batch_PMLDA.py \
--source_model_path /home/ponshane/work_dir/CLTM-Experiments/Data/EN-JP-MLDoc/monolingual-lda/"$i"topics-en.model \
--target_model_path /home/ponshane/work_dir/CLTM-Experiments/Data/EN-JP-MLDoc/monolingual-lda/"$i"topics-jp.model \
--vector_path /home/ponshane/work_dir/CLTM-Experiments/Data/EN-JP-MLDoc/word-vectors/Japanese_English_wordvectors.vec \
--num_of_topic "$i" --top_n 1000 \
--prefix_output_path /home/ponshane/work_dir/CLTM-Experiments/Results/EN-JP-MLDoc/model-comparison/PMLDA-"$i"topics-
done
