#!/bin/bash
for i in 10 25 50 75 90
do
   #bin/mallet import-file --input experiments/"$i"percKeep-50K-cn-docs.txt --output experiments/"$i"percKeep-50K-cn-docs.sequences --keep-sequence
   bin/mallet import-file --input experiments/"$i"percKeep-50K-en-docs.txt --output experiments/"$i"percKeep-50K-en-docs.sequences --keep-sequence
done
