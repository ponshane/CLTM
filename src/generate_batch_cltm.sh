#!/bin/bash
for i in 10 20 30 40 50 60 70 80 90
do
   java -jar jar/LFTM.jar -model LFLDA -corpus Experiments/2018-12-29/tagged_englishAndjapanese_corpus_pd.txt -vectors Experiments/2018-12-29/selected"$i"dim-test-run.txt -ntopics 10 -alpha 0.1 -beta 0.1 -lambda 1 -initers 0 -niters 100 -name "$i"dim-MLDoc-engAndjap-LFLDA10T100I1e-1beta -twords 50
done

