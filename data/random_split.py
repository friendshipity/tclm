# coding=utf-8
import random
files = ['OCEMOTION_train.csv','OCNLI_train.csv','TNEWS_train.csv']
split_rate = 0.9
for file in files:
    with open(file,'r',encoding='utf-8') as f:
        lines = f.readlines()
    f.close()
    random.shuffle(lines)
    train_nums = int(len(lines)*split_rate)
    with open(file.replace('.csv','_s.csv'),'w',encoding='utf-8') as f:
        for line in lines[:train_nums]:
            f.write(line)
    f.close()
    with open(file.replace('_train.csv','_dev.csv'), 'w', encoding='utf-8') as f:
        for line in lines[train_nums:]:
            f.write(line)
    f.close()
