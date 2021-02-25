from LAC import LAC
import pkuseg
import jieba
import os

cn_data_path = "./demo_data/cn/demo.txt"
output_data_path = "./demo_data/cn/seg_train_caps.txt"

bseg = LAC(mode='seg') # baidu
pseg = pkuseg.pkuseg() # pku

with open (cn_data_path, "r") as f:
    for line in f.readlines():
        print(f'bseg: {bseg.run(line)}')
        print(f'pseg: {pseg.cut(line)}')
        print(f'jieba: {list(jieba.cut(line))}')
        print("===========")

