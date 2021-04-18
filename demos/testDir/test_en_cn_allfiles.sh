#!/bin/bash
cd ../../srcMutiLang/
for lang in en_en en_fr fr_fr
do
  echo $lang
  for i in {0..23}
  do
   python3 multi_test.py --candidate /Users/zhou/Dropbox/Travail/MAThesis/VGNSL/output_multi/en_fr_0403/${i}.pth.tar --langs ${lang} >> /Users/zhou/Dropbox/Travail/MAThesis/VGNSL/demos/testDir/en_fr_res.txt 
  done
done
