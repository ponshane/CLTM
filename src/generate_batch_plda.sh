#!/bin/bash
for i in 10 25 50 75 90
do
   bin/mallet run cc.mallet.topics.PolylingualTopicModel --language-inputs experiments/ver1/"$i"percKeep-50K-cn-docs.sequences experiments/ver1/"$i"percKeep-50K-en-docs.sequences --num-topics 20 --alpha 0.1 --optimize-interval 10 --optimize-burn-in 20 --num-top-words 500 --output-topic-keys experiments/ver1/"$i"percKeep-50K-K20-Top500-phi.txt
done
